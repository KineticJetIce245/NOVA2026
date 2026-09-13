"""Thread-safe packet publication — a test double.

The vendored frontend test drives a publisher through the adapter layer and
feeds the resulting packets into the browser-side validators, so this keeps the
observable contract: a monotonic sequence, a session identifier, and events
readable after a cursor. It is deliberately **not** the transport
implementation; ``src/nova2026/transport`` owns that, including the ring buffer,
the snapshot, and the 1013 slow-client policy.
"""

from __future__ import annotations

from threading import Lock

from .protocol import make_packet


class Publisher:
    """Assign sequences, wrap payloads in version-1 envelopes, keep a log."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sequence = 0
        self._session_id: str | None = None
        self._events: list[dict] = []

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def begin(self, session_id: str) -> str:
        """Start a session; a new session clears the retained events."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("A session needs a nonempty identifier.")
        with self._lock:
            self._session_id = session_id
            self._events = []
        return session_id

    def publish(self, packet_type: str, timestamp: float, source: str, session_id: str, payload: dict) -> dict:
        """Wrap and retain one packet, assigning the next sequence number."""

        with self._lock:
            if self._session_id is None:
                raise RuntimeError("Call begin() before publishing.")
            if session_id != self._session_id:
                raise ValueError("A packet must carry the active session identifier.")
            self._sequence += 1
            packet = make_packet(
                packet_type, float(timestamp), self._sequence, session_id, source, payload
            )
            self._events.append(packet)
            return packet

    def events_after(self, cursor: int) -> list[dict]:
        """Every retained packet with a sequence above ``cursor``."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer.")
        with self._lock:
            return [dict(event) for event in self._events if event["sequence"] > cursor]
