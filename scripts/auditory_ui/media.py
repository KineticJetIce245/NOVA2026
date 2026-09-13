"""A browser, minus the browser: drive ``/api/media/control`` the way the UI does.

The vendored frontend's ``mediaController.js`` is the only client whose behaviour
matters, and it is not reachable from this repository's test path (no browser).
What *is* reachable is its protocol, and that protocol is small enough to obey
exactly:

* ``prepare`` at position 0 before anything else, once per playback session;
* then ``playing`` at position 0, so the transport learns the timeline is running;
* then ``report`` every 250 ms - the interval ``MediaPlayback.js`` sets its timer
  to - carrying ``element.currentTime`` for a playing element, which is wall-clock
  elapsed time since play began;
* a monotonically increasing ``request_id`` within one ``client_id``;
* every acknowledgement validated field by field, and a rejection treated as a
  lost playback session rather than something to retry blindly.

Two timing rules are the client's, and both matter for the ``|Δt|`` this step has
to report:

* the position reported is the position the *client* measured, never re-derived
  from the server's answer (plan section 2.2, decision D-02);
* the client measures its own position at the same instant it sends, so the
  difference between what the server acknowledges and what the client would report
  is exactly the request's round trip - which is what makes a network delay show
  up as a timing gap instead of hiding inside a timestamp.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

MEDIA_TICK_SECONDS = 0.25
"""Report cadence; ``MediaPlayback.js`` drives ``controller.tick()`` at 250 ms."""

CONTROL_TIMEOUT_SECONDS = 1.5
"""``mediaController.send`` aborts a control request after 1500 ms."""

ACKNOWLEDGEMENT_TIMEOUT_SECONDS = 1500.0
"""Milliseconds a round trip may take before the client declares playback lost."""


@dataclass
class ControlExchange:
    """One controller command and what the transport answered."""

    index: int
    action: str
    request_id: int
    media_time_s: float
    duration_s: float
    wall_seconds: float
    round_trip_seconds: float
    status: int
    acknowledged_revision: int | None = None
    acknowledged_time_s: float | None = None
    sync_status: str | None = None
    playback_state: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe record; the exchange log is step 9's primary evidence."""

        return {
            "index": self.index,
            "action": self.action,
            "request_id": self.request_id,
            "media_time_s": round(self.media_time_s, 6),
            "duration_s": round(self.duration_s, 6),
            "wall_seconds": round(self.wall_seconds, 6),
            "round_trip_seconds": round(self.round_trip_seconds, 6),
            "status": self.status,
            "acknowledged_revision": self.acknowledged_revision,
            "acknowledged_time_s": self.acknowledged_time_s,
            "sync_status": self.sync_status,
            "playback_state": self.playback_state,
            "detail": self.detail,
        }


@dataclass
class MediaClientResult:
    """What one simulated playback session did."""

    descriptor: dict[str, Any]
    duration_s: float
    client_id: str
    started_playback: bool
    prepared: bool
    exchanges: list[ControlExchange] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe copy, with the exchange log in order."""

        return {
            "client_id": self.client_id,
            "descriptor": self.descriptor,
            "duration_s": round(self.duration_s, 6),
            "started_playback": self.started_playback,
            "prepared": self.prepared,
            "exchanges": [exchange.to_dict() for exchange in self.exchanges],
            "failures": list(self.failures),
        }


class SimulatedMediaClient:
    """Play the media timeline over REST, exactly as ``mediaController.js`` would.

    Args:
        base_url: Loopback origin of the running transport.
        client_id: Controller identity sent with every command. One session uses
            one id, which is what makes ``request_id`` monotone a real constraint
            rather than a formality.
        duration_s: Media duration to report; a browser reads it from the loaded
            element, and the transport refuses a command without it.
        clip_seconds: Stop reporting after this many seconds of playback; ``None``
            plays the whole asset.
        tick_seconds: Report interval.
        timeout: Per-request timeout, as in the controller.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client_id: str,
        duration_s: float,
        clip_seconds: float | None = None,
        tick_seconds: float = MEDIA_TICK_SECONDS,
        timeout: float = CONTROL_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.duration_s = float(duration_s)
        self.clip_seconds = None if clip_seconds is None else float(clip_seconds)
        self.tick_seconds = float(tick_seconds)
        self.timeout = float(timeout)
        self.request_id = 0
        self.session_id = ""
        self.media_id = ""
        # The playback clock, published for whoever is watching the packet stream:
        # a gain packet is compared against the position the browser held when it
        # arrived, not against the position it carries.
        self.state: dict[str, Any] = {
            "prepared": False,
            "started_playback": False,
            "playback_started_at": None,
            "last_ack": None,
            "revision": None,
            "error": None,
        }

    def playback_snapshot(self) -> dict[str, Any]:
        """The client's playback state right now, in ``mediaController``'s shape.

        ``time`` is the position this client would report at this instant - the
        same construction ``play()`` uses - and ``ready`` is the controller's own
        rule, ``prepared && now - lastAck <= 1.5 s``.
        """

        started = self.state["playback_started_at"]
        last_ack = self.state["last_ack"]
        return {
            "mediaId": self.media_id,
            "revision": self.state["revision"],
            "time": 0.0 if started is None else max(0.0, time.perf_counter() - started),
            "duration": self.duration_s,
            "playbackState": "playing" if started is not None else ("paused" if self.state["prepared"] else "stopped"),
            "ready": bool(
                self.state["prepared"]
                and last_ack is not None
                and time.perf_counter() - last_ack <= ACKNOWLEDGEMENT_TIMEOUT_SECONDS / 1000.0
            ),
            "error": self.state["error"],
            "busy": False,
            "mode": "attune",
        }

    async def play(self, stop: asyncio.Event) -> MediaClientResult:
        """Run one playback session until ``stop`` is set or the clip ends.

        The stop event is shared with the packet collector: playback ends when the
        session's terminal packet arrives, so the two timelines are the same run
        rather than two loosely synchronised ones.
        """

        import httpx

        result = MediaClientResult(
            descriptor={},
            duration_s=self.duration_s,
            client_id=self.client_id,
            started_playback=False,
            prepared=False,
        )
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get("/api/media")
            if response.status_code != 200:
                result.failures.append(f"GET /api/media answered {response.status_code}")
                return result
            result.descriptor = response.json()
            if not isinstance(result.descriptor, dict) or "media_id" not in result.descriptor:
                result.failures.append("GET /api/media returned no media_id")
                return result

            began = time.perf_counter()
            # Playing: position 0, then real time. A browser's currentTime while
            # playing advances with the audio clock; under the deterministic
            # replay this client is part of, that clock is wall time.
            playback_started: float | None = None
            index = 0
            while not stop.is_set():
                position = 0.0 if playback_started is None else time.perf_counter() - playback_started
                if self.clip_seconds is not None and position >= self.clip_seconds:
                    break
                action = "prepare" if not result.prepared else ("playing" if not result.started_playback else "report")
                exchange = await self._send(client, index, action, position, began)
                result.exchanges.append(exchange)
                if exchange.status == 200:
                    if action == "prepare":
                        result.prepared = True
                        self.state["prepared"] = True
                    elif action == "playing":
                        result.started_playback = True
                        self.state["started_playback"] = True
                        playback_started = time.perf_counter()
                        self.state["playback_started_at"] = playback_started
                    self.state["last_ack"] = time.perf_counter()
                    self.state["revision"] = exchange.acknowledged_revision
                    self.state["error"] = None
                else:
                    result.failures.append(
                        f"{action} answered {exchange.status}: {exchange.detail}"
                    )
                    self.state["error"] = exchange.detail or f"status {exchange.status}"
                    # A rejected command is a lost playback session: the
                    # controller pauses and warns instead of retrying in a loop.
                    break
                if action != "playing":
                    index += 1
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=self.tick_seconds)
                    except asyncio.TimeoutError:
                        pass
            return result

    async def _send(self, client, index: int, action: str, position: float, began: float) -> ControlExchange:
        """Send one command and validate the acknowledgement as the frontend does."""

        self.request_id += 1
        body = {
            "session_id": self.session_id,
            "media_id": self.media_id,
            "client_id": self.client_id,
            "request_id": self.request_id,
            "action": action,
            "media_time_s": 0.0 if action in ("prepare", "stopped") else position,
            "duration_s": self.duration_s,
        }
        started = time.perf_counter()
        response = await client.post("/api/media/control", json=body)
        trip = time.perf_counter() - started
        record = ControlExchange(
            index=index,
            action=action,
            request_id=self.request_id,
            media_time_s=body["media_time_s"],
            duration_s=self.duration_s,
            wall_seconds=time.perf_counter() - began,
            round_trip_seconds=trip,
            status=response.status_code,
        )
        if response.status_code != 200:
            record.detail = _short(response.text)
            return record
        payload = response.json()
        error = self._validate(action, payload, trip)
        record.acknowledged_revision = payload.get("revision")
        record.acknowledged_time_s = payload.get("media_time_s")
        record.sync_status = payload.get("sync_status")
        record.playback_state = payload.get("playback_state")
        record.detail = error
        if error:
            record.status = 0
        return record

    def _validate(self, action: str, payload: dict[str, Any], trip: float) -> str:
        """The controller's own field-by-field check, as a returned reason."""

        expected = "desynchronized" if action == "stopped" else "observed"
        if payload.get("session_id") != self.session_id:
            return "acknowledgement named another session"
        if payload.get("media_id") != self.media_id:
            return "acknowledgement named another media asset"
        if not isinstance(payload.get("revision"), int) or isinstance(payload.get("revision"), bool):
            return "acknowledgement carried no integer revision"
        if payload.get("sync_status") != expected:
            return f"sync_status was {payload.get('sync_status')!r}, expected {expected!r}"
        if trip * 1000.0 > ACKNOWLEDGEMENT_TIMEOUT_SECONDS:
            return f"round trip {trip:.3f}s exceeded the controller's 1500 ms budget"
        return ""

    session_id: str
    """Session the controller commands; set before :meth:`play` via :meth:`configure`."""

    media_id: str
    """Asset id from the descriptor; set before :meth:`play` via :meth:`configure`."""

    def configure(self, session_id: str, media_id: str) -> None:
        """Bind this client to one session and one media identity."""

        self.session_id = str(session_id)
        self.media_id = str(media_id)


def _short(text: str, limit: int = 200) -> str:
    """One line of an error body, bounded; the log is evidence, not a dump."""

    collapsed = " ".join(str(text).split())
    return collapsed[:limit]


def load_control_log(path) -> dict[str, Any]:
    """Read an exchange log written by the media runner (used by tests and reviews)."""

    return json.loads(path.read_text())


__all__ = [
    "ACKNOWLEDGEMENT_TIMEOUT_SECONDS",
    "CONTROL_TIMEOUT_SECONDS",
    "MEDIA_TICK_SECONDS",
    "ControlExchange",
    "MediaClientResult",
    "SimulatedMediaClient",
    "load_control_log",
]
