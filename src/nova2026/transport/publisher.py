"""Thread-safe publication, the latest-state snapshot and the bounded event ring.

A producer runs on its own thread and never waits for a client:
:meth:`Publisher.publish` validates a packet, refreshes one latest packet per
``(source, type)`` stream, appends it to a bounded ring and returns. Readers - the
REST snapshot and each WebSocket's cursor - get defensive copies under the same
lock, so a producer can never rewrite a packet a client already holds, and a
snapshot plus its high-water sequence are read atomically.

The two bounds *are* the memory guarantee: at most ``state_limit`` retained
streams (128 by default) and ``event_limit`` retained events (256, the ring).
Starting a session clears the latest streams but keeps the ring, so clients still
observe the session boundary. A reader whose cursor fell behind the ring is told
with :class:`LaggedSubscriber` instead of being handed a silent gap; the server
turns that into WebSocket close code 1013 and the client takes a fresh snapshot.

``sequence`` increases once per accepted publication and never resets inside one
server process - it is process-wide, not per session. Session identity travels in
``session_id``, which is why two sessions are never compared by sequence.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from threading import RLock
from typing import Any

from .protocol import MAX_TEXT, make_packet

DEFAULT_EVENT_LIMIT = 256
"""Retained events across session boundaries - the accepted-packet ring."""

DEFAULT_STATE_LIMIT = 128
"""Retained ``(source, type)`` streams in the latest-state snapshot."""


class LaggedSubscriber(Exception):
    """A reader fell behind the event ring and must take a fresh snapshot."""


class Publisher:
    """Assign sequences, retain bounded state, and hand readers detached copies.

    Args:
        event_limit: Retained events in the ring. Older events are dropped first.
        state_limit: Retained ``(source, type)`` streams in the latest snapshot.
            Least recently updated streams are evicted first.

    Attributes:
        lock: Reentrant lock protecting every attribute below.
        sequence: Last assigned sequence; increases once per published packet.
        session_id: Session whose packets are currently accepted, or ``None``.
        accepting: Whether :meth:`publish` currently accepts packets.
        latest: Latest packet per ``(source, type)``, oldest update first.
        events: The bounded ring of published packets, oldest first.
        state_limit: Stream bound, as passed in.
    """

    def __init__(
        self,
        event_limit: int = DEFAULT_EVENT_LIMIT,
        state_limit: int = DEFAULT_STATE_LIMIT,
    ) -> None:
        """Validate the bounds and start with no session and no retained state."""

        if isinstance(event_limit, bool) or not isinstance(event_limit, int) or event_limit < 1:
            raise ValueError("event_limit must be a positive integer.")
        if isinstance(state_limit, bool) or not isinstance(state_limit, int) or state_limit < 1:
            raise ValueError("state_limit must be a positive integer.")

        self.lock = RLock()
        self.sequence = 0
        self.session_id: str | None = None
        self.accepting = False
        self.latest: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self.events: deque[dict[str, Any]] = deque(maxlen=event_limit)
        self.state_limit = state_limit

    def begin(self, session_id: str) -> None:
        """Accept packets for ``session_id`` and clear the latest-state streams.

        The event ring is deliberately *not* cleared: connected clients must still
        see the session boundary that the start packet marks.

        Raises:
            ValueError: If ``session_id`` is not a nonempty string of at most
                :data:`nova2026.transport.protocol.MAX_TEXT` characters.
        """

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string.")
        if len(session_id) > MAX_TEXT:
            raise ValueError(f"session_id must be at most {MAX_TEXT} characters.")
        with self.lock:
            self.session_id = session_id
            self.accepting = True
            self.latest.clear()

    def end(self, session_id: str) -> None:
        """Stop accepting packets for ``session_id``; other sessions are ignored."""

        with self.lock:
            if session_id == self.session_id:
                self.accepting = False

    def publish(
        self,
        kind: str,
        timestamp: float,
        source: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and retain one packet, assigning the next sequence number.

        Args:
            kind: Packet type.
            timestamp: Session-relative seconds; nondecreasing per retained
                ``(source, type)`` stream.
            source: Producing stream. ``server`` is reserved for the transport.
            session_id: Must be the session :meth:`begin` activated.
            payload: JSON object, not interpreted here.

        Returns:
            A detached copy of the published packet.

        Raises:
            ValueError: If no session accepts this packet, if the timestamp moved
                backwards within its stream, or if the packet is invalid.

        Notes:
            Validation happens before the sequence is incremented, so a rejected
            packet leaves the publisher exactly as it was.
        """

        with self.lock:
            if session_id != self.session_id or not self.accepting:
                raise ValueError("publish requires the active, accepting session.")
            packet = make_packet(kind, timestamp, self.sequence + 1, source, session_id, payload)
            key = (source, kind)
            previous = self.latest.get(key)
            if previous is not None and timestamp < previous["timestamp"]:
                raise ValueError(
                    f"timestamp regressed within stream ({source!r}, {kind!r}); "
                    "session-relative time must not move backwards."
                )
            self.sequence += 1
            self.latest[key] = packet
            self.latest.move_to_end(key)
            while len(self.latest) > self.state_limit:
                self.latest.popitem(last=False)
            self.events.append(packet)
            return deepcopy(packet)

    def snapshot(self) -> dict[str, Any]:
        """Return ``{sequence, session_id, packets}`` with packets by sequence.

        The sequence is the high-water cursor of this snapshot: every packet with
        a higher sequence is strictly newer than everything returned here, which is
        what lets a socket subscribe without a race (see ``server.stream``).
        """

        with self.lock:
            packets = sorted(self.latest.values(), key=lambda packet: packet["sequence"])
            return {
                "sequence": self.sequence,
                "session_id": self.session_id,
                "packets": deepcopy(packets),
            }

    def events_after(self, cursor: int) -> list[dict[str, Any]]:
        """Return every retained packet with a sequence above ``cursor``.

        Args:
            cursor: Last sequence the caller has already seen.

        Returns:
            Detached copies, oldest first.

        Raises:
            LaggedSubscriber: If the ring no longer reaches back to ``cursor``, so
                the caller cannot be given a gap-free continuation and must take a
                fresh snapshot instead.
        """

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer.")
        with self.lock:
            if self.events and cursor < self.events[0]["sequence"] - 1:
                raise LaggedSubscriber("event buffer overrun")
            return deepcopy(
                [packet for packet in self.events if packet["sequence"] > cursor]
            )
