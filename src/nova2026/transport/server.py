"""One same-origin FastAPI app: REST commands, the live socket and the SPA.

The demo is a single process on loopback (decision §3.2): ``/api/*`` commands,
``/ws/live`` delivery and the built frontend's ``dist/`` are served from the same
origin, so there is no CORS policy to loosen and no second server to keep in sync.
The event loop only *pushes*: every computation happens on the producer's own
thread, and the synchronous handlers run in the threadpool FastAPI gives them.

Delivery is bounded live delivery, not durable storage. A socket first receives
the packets of one atomic snapshot, then every event strictly after the
snapshot's high-water sequence, which is what closes the subscribe race. There is
no unbounded per-client queue: a socket that overruns the 256-packet ring is
closed with code **1013** and reconnects for a fresh snapshot, and so is a socket
whose send does not complete within the send timeout. Losing history is explicit;
a silent gap in the sequence would be worse, because the frontend counts protocol
violations and refuses to guess.

``create_app(producer_factory=...)`` is the integration boundary: the attention
producer is injected there (planned step 8). Without a factory the app still
serves health, state and media, and ``POST /api/session/start`` fails with an
explicit message instead of inventing a producer - there is no mock data in this
repository that could be mistaken for a measurement.

Security posture, deliberately small:

* Runtime configuration (media file, static directory) comes from the environment
  at start-up; **no** filesystem path is ever accepted over HTTP, and the media
  descriptor exposes only ``/api/media/file``.
* Nothing here logs packet payloads. Payloads can carry participant data, so the
  transport never writes them to stdout, disk or an error message.
* Bind loopback only: :func:`serve` refuses any other host.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .media import MediaTimeline
from .protocol import VERSION
from .publisher import LaggedSubscriber, Publisher
from .sessions import Producer, Sessions

DEFAULT_SEND_TIMEOUT = 5.0
"""Seconds one WebSocket send may take before the socket is closed with 1013."""

DEFAULT_POLL_INTERVAL = 0.02
"""Seconds between event-ring polls for a connected socket."""

CLOSE_LAGGED = 1013
"""Close code for a client that fell behind; it must reconnect for a snapshot."""

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
"""Hosts :func:`serve` accepts; anything else would expose the demo."""


async def _receive_ignored(socket: WebSocket, stopping: asyncio.Event) -> None:
    """Consume client messages until the socket disconnects or the app stops.

    This is a server-to-client stream: client text and binary frames are not
    commands (commands belong to REST), so they are read and dropped only to
    observe the disconnect.
    """

    while not stopping.is_set():
        message = await socket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def stream(
    socket: WebSocket,
    publisher: Publisher,
    stopping: asyncio.Event,
    *,
    send_timeout: float = DEFAULT_SEND_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> None:
    """Deliver the latest snapshot, then every later event, to one socket.

    Args:
        socket: Accepted WebSocket. Only ``send_json``, ``receive`` and ``close``
            are used, which keeps the delivery loop testable without a port.
        publisher: Source of snapshots and events.
        stopping: Set by the app's lifespan when shutdown begins.
        send_timeout: Seconds one send may take before the socket is closed 1013.
        poll_interval: Seconds between polls of the event ring.

    Raises:
        Nothing. A lagging or broken client is closed with code 1013 (see
        :data:`CLOSE_LAGGED`); a plain disconnect simply ends the stream.
    """

    async def send_packet(packet: dict[str, Any]) -> None:
        await asyncio.wait_for(socket.send_json(packet), timeout=send_timeout)

    async def send_events() -> None:
        # The snapshot and its cursor are read atomically, so no event published
        # between the two can be lost or delivered twice.
        snapshot = publisher.snapshot()
        cursor = snapshot["sequence"]
        for packet in snapshot["packets"]:
            await send_packet(packet)
        while not stopping.is_set():
            for packet in publisher.events_after(cursor):
                await send_packet(packet)
                cursor = packet["sequence"]
            await asyncio.sleep(poll_interval)

    tasks = [
        asyncio.create_task(send_events()),
        asyncio.create_task(_receive_ignored(socket, stopping)),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (LaggedSubscriber, asyncio.TimeoutError):
        await socket.close(code=CLOSE_LAGGED, reason="Reconnect for latest snapshot")
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        # A quiet disconnect or an already-closing socket is not an error.
        pass
    finally:
        for task in tasks:
            task.cancel()
        # Shielding keeps cancellation of this task from abandoning the children.
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)


def create_app(
    producer_factory: Callable[[], Producer] | None = None,
    publisher: Publisher | None = None,
    media: MediaTimeline | None = None,
    static_dir: str | Path | None = None,
    *,
    send_timeout: float = DEFAULT_SEND_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> FastAPI:
    """Build the transport app.

    Args:
        producer_factory: Builds the producer a session runs. This is the only
            way a producer enters the process.
        publisher: Publisher to use; a default bounded one is created otherwise.
        media: Optional media timeline served at ``/api/media*``.
        static_dir: Optional directory of a built frontend to serve at ``/``.
            It must exist; a typo fails loudly instead of serving nothing.
        send_timeout: Seconds one WebSocket send may take before close 1013.
        poll_interval: Seconds between event-ring polls per socket.

    Returns:
        The configured application.

    Raises:
        ValueError: If ``static_dir`` is not an existing directory.
    """

    if not isinstance(send_timeout, (int, float)) or send_timeout <= 0:
        raise ValueError("send_timeout must be a positive number of seconds.")
    if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
        raise ValueError("poll_interval must be a positive number of seconds.")

    static_root: Path | None = None
    if static_dir is not None:
        static_root = Path(static_dir).resolve()
        if not static_root.is_dir():
            raise ValueError(f"static_dir must be an existing directory: {static_root}")

    publisher = publisher if publisher is not None else Publisher()
    sessions = Sessions(publisher, producer_factory=producer_factory)
    sockets: set[WebSocket] = set()
    stopping = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            stopping.set()
            await asyncio.gather(
                *(socket.close(code=1001) for socket in list(sockets)),
                return_exceptions=True,
            )
            # close() joins the worker; a worker that ignores cancellation raises
            # here rather than being silently abandoned.
            await asyncio.to_thread(sessions.close)

    app = FastAPI(lifespan=lifespan)
    app.state.publisher = publisher
    app.state.sessions = sessions
    app.state.sockets = sockets
    app.state.media = media
    app.state.static_dir = static_root

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Transport health, independent of whether any model is available."""

        return {"status": "ok", "protocol_version": VERSION}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        """Latest packet per retained stream, with the high-water sequence."""

        return publisher.snapshot()

    @app.get("/api/sessions")
    def session_list() -> list[dict[str, Any]]:
        """In-memory session summaries, oldest first."""

        return sessions.list()

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str) -> dict[str, Any]:
        """One session summary by id."""

        for record in sessions.list():
            if record["id"] == session_id:
                return record
        raise HTTPException(404, "unknown session")

    @app.post("/api/session/start")
    def start() -> dict[str, Any]:
        """Start the producer session, or return the running one unchanged."""

        try:
            record = sessions.start()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            # The factory's own failure text may name local paths or credentials.
            raise HTTPException(503, "producer could not start") from error
        if media is not None:
            media.bind(record["id"])
        return record

    @app.post("/api/session/stop")
    def stop() -> dict[str, Any] | None:
        """Stop the session and join the worker; idempotent."""

        try:
            if media is not None:
                media.stop()
            return sessions.stop()
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.get("/api/media")
    def media_config() -> dict[str, Any] | None:
        """Asset identity for the controller, or ``None`` when unconfigured."""

        return media.descriptor() if media is not None else None

    @app.get("/api/media/file")
    def media_file() -> FileResponse:
        """Serve the configured asset; the path is server-side configuration only."""

        if media is None or not media.path.is_file():
            raise HTTPException(404, "Media unavailable")
        return FileResponse(media.path, headers={"Cache-Control": "no-store"})

    @app.post("/api/media/control")
    def media_control(data: dict[str, Any]) -> dict[str, Any]:
        """Accept one controller command for the running session's timeline."""

        if media is None:
            raise HTTPException(404, "Media unavailable")
        with sessions.command_lock:
            records = sessions.list()
            if (
                not records
                or records[-1]["id"] != data.get("session_id")
                or records[-1]["status"] != "running"
            ):
                raise HTTPException(409, "Session inactive")
            media.bind(data["session_id"])
            try:
                return media.control(data)
            except (ValueError, TypeError) as error:
                raise HTTPException(
                    409, "Media command rejected; stop and prepare again"
                ) from error

    @app.websocket("/ws/live")
    async def live(socket: WebSocket) -> None:
        """Stream packets to one client until it disconnects or falls behind."""

        await socket.accept()
        sockets.add(socket)
        try:
            await stream(
                socket,
                publisher,
                stopping,
                send_timeout=send_timeout,
                poll_interval=poll_interval,
            )
        finally:
            sockets.discard(socket)

    if static_root is not None:
        # Mounted last so /api/* and /ws/live keep precedence over the SPA.
        app.mount("/", StaticFiles(directory=static_root, html=True), name="spa")

    return app


def serve(
    app: FastAPI,
    *,
    host: str = "127.0.0.1",
    port: int = 8001,
    log_level: str = "info",
) -> None:
    """Run the app with one uvicorn worker on a loopback port.

    Args:
        app: Application from :func:`create_app`.
        host: Loopback host. Any other value is refused: this transport has no
            authentication and must not be reachable from the network.
        port: TCP port to bind.
        log_level: uvicorn log level. Access logs record request paths only; the
            transport never logs packet payloads.

    Raises:
        ValueError: If ``host`` is not a loopback host.
    """

    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"the transport binds loopback only; refusing host {host!r}.")
    # Imported here so the contract tests never need a server runtime.
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=log_level, ws="websockets")


def _app_from_environment() -> FastAPI:
    """Build the app from environment configuration, for ``uvicorn ...:app``.

    Only two settings are read, and neither accepts a producer: the producer has
    to be injected in code, because a value that selects executable behaviour from
    the environment would be a hole, not a convenience.
    """

    media_path = os.environ.get("NOVA_TRANSPORT_MEDIA_FILE")
    media = (
        MediaTimeline(media_path, os.environ.get("NOVA_TRANSPORT_MEDIA_TITLE", "Demo Audio"))
        if media_path
        else None
    )
    return create_app(media=media, static_dir=os.environ.get("NOVA_TRANSPORT_STATIC_DIR"))


app = _app_from_environment()
