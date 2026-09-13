"""Local transport app. Producer factory injection is the integration boundary."""
import asyncio
import os
from pathlib import Path
import anyio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from .publisher import Publisher, LaggedSubscriber
from .sessions import Sessions
from .media import MediaTimeline
from backend.adapters.mock import MockProducer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def create_app(producer_factory=None, publisher=None, media=None):
    publisher = publisher if publisher is not None else Publisher()
    if media is not None and producer_factory is None:
        producer_factory = lambda: MockProducer(timeline=media)
    sessions = Sessions(publisher, **({'producer_factory': producer_factory} if producer_factory else {}))
    sockets = set()
    stopping = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            stopping.set()
            await asyncio.gather(*(socket.close(code=1001) for socket in list(sockets)),
                                 return_exceptions=True)
            await asyncio.to_thread(sessions.close)

    app = FastAPI(lifespan=lifespan)
    app.state.publisher = publisher
    app.state.sessions = sessions
    app.state.sockets = sockets
    app.state.media = media

    @app.get('/api/media')
    def media_config():
        return media.descriptor() if media else None

    @app.get('/api/media/file')
    def media_file():
        if media is None or not media.path.is_file():
            raise HTTPException(404, 'Media unavailable')
        return FileResponse(media.path, headers={'Cache-Control': 'no-store'})

    @app.post('/api/media/control')
    def media_control(data: dict):
        if media is None:
            raise HTTPException(404, 'Media unavailable')
        with sessions.command_lock:
            records = sessions.list()
            if not records or records[-1]['id'] != data.get('session_id') or records[-1]['status'] != 'running':
                raise HTTPException(409, 'Session inactive')
            media.bind(data['session_id'])
            try:
                return media.control(data)
            except (ValueError, TypeError):
                raise HTTPException(409, 'Media command rejected; stop and prepare again')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'protocol_version': 1}

    @app.get('/api/state')
    def state():
        return publisher.snapshot()

    @app.get('/api/sessions')
    def session_list():
        return sessions.list()

    @app.get('/api/sessions/{session_id}')
    def session_detail(session_id: str):
        for record in sessions.list():
            if record['id'] == session_id:
                return record
        raise HTTPException(404, 'unknown session')

    @app.post('/api/session/start')
    def start():
        try:
            record = sessions.start()
            if media is not None:
                media.bind(record['id'])
            return record
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, 'producer could not start') from exc

    @app.post('/api/session/stop')
    def stop():
        try:
            if media is not None:
                media.stop()
            return sessions.stop()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.websocket('/ws/live')
    async def live(socket: WebSocket):
        await socket.accept()
        sockets.add(socket)

        async def send_packet(packet):
            await asyncio.wait_for(socket.send_json(packet), timeout=5)

        async def send():
            # Atomic snapshot + high-water cursor avoids a subscribe/snapshot race.
            snapshot = publisher.snapshot()
            cursor = snapshot['sequence']
            for packet in snapshot['packets']:
                await send_packet(packet)
            while not stopping.is_set():
                for packet in publisher.events_after(cursor):
                    await send_packet(packet)
                    cursor = packet['sequence']
                await asyncio.sleep(.02)

        async def receive():
            # Server-to-client stream; client messages are not commands.
            while not stopping.is_set():
                message = await socket.receive()
                if message['type'] == 'websocket.disconnect':
                    return

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (LaggedSubscriber, asyncio.TimeoutError):
            await socket.close(code=1013, reason='Reconnect for latest snapshot')
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
            sockets.discard(socket)

    # The production UI belongs to this checkout. API and WebSocket routes
    # are registered first so the static root cannot intercept them.
    frontend = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
    if frontend.is_dir():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
    return app


# Runtime configuration stays outside packets; never accept filesystem paths over HTTP.
_media_path = os.environ.get('ATTUNE_MEDIA_FILE')
def configured_app():
    trial = os.environ.get('ATTUNE_NOVA_TRIAL')
    if trial:
        from backend.adapters.nova_live import make_coordinator
        from backend.adapters.results import ResultProducer
        import json
        from pathlib import Path
        profile = os.environ.get('ATTUNE_NOVA_TIMING_PROFILE')
        coordinator = make_coordinator(trial, os.environ['ATTUNE_NOVA_MODEL'],
            os.environ['ATTUNE_NOVA_STREAM'],
            presentation=os.environ.get('ATTUNE_NOVA_PRESENTATION', 'dichotic'),
            output=os.environ.get('ATTUNE_NOVA_OUTPUT', 'wav'),
            timing_profile=json.loads(Path(profile).read_text()) if profile else None,
            check_channels=os.environ.get('ATTUNE_NOVA_CHANNEL_POLICY', 'strict') != 'record-only',
            report_dir=os.environ.get('ATTUNE_NOVA_REPORT_DIR'),
            player_raw_path=os.environ.get('ATTUNE_NOVA_PLAYER_RAW'),
            marker_name=os.environ.get('ATTUNE_NOVA_MARKERS'))
        return create_app(producer_factory=lambda: ResultProducer(coordinator, simulated=False))
    return create_app(media=MediaTimeline(_media_path, os.environ.get('ATTUNE_MEDIA_TITLE', 'Demo Audio')) if _media_path else None)


app = configured_app()
