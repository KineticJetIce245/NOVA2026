"""Serialized session commands and the producer thread they own, without HTTP.

Start and stop are commands, not requests to a thread: they take one command lock,
so a start cannot race a stop, and a second start while the worker is alive
returns the running session instead of launching a second producer. The producer
itself runs on a background daemon thread, outside every request handler, and is
handed exactly two things - a ``publish`` callable and a ``stop`` event.

Failure stays explicit on this boundary:

* A producer that raises ends its session with ``status='error'`` and the fixed
  code ``producer_failed``. The exception's text, type and traceback are
  deliberately not copied into the record: a transport is not a log sink, and an
  exception message can carry participant data, local paths or credentials.
* Once the producer has returned, the session stops accepting packets, so a late
  writer cannot append results to a session that has already been reported closed.
* A worker that ignores cancellation is reported after the stop deadline rather
  than being silently abandoned; Python cannot forcibly stop such a thread, so the
  caller learns that the process still owns a live worker.
* Source ``server`` is reserved for the transport's own lifecycle packets, so a
  producer cannot forge them.

This module is transport-only: it computes nothing and imports nothing from the
scientific stack.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from .publisher import Publisher

RESERVED_SOURCE = "server"
"""Packet source the transport owns; producers may not publish under it."""

DEFAULT_STOP_TIMEOUT = 3.0
"""Seconds a worker is given to observe cancellation before it is reported."""

MAX_SESSION_RECORDS = 32
"""Retained in-memory session summaries; oldest are dropped first."""

Publish = Callable[[str, float, str, dict[str, Any]], dict[str, Any] | None]
"""``publish(type, session_relative_seconds, source, payload) -> packet or None``."""


class Producer(Protocol):
    """The whole surface the transport requires of a producer.

    A producer that means to be labelled as simulated data must declare
    ``simulated = True``; a producer without the attribute is recorded as a real
    measurement, so scripted or synthetic producers have to say so themselves.
    """

    def run(self, publish: Publish, stop: Event) -> None:
        """Publish results until ``stop`` is set, then return promptly.

        Runs on the session's background thread: no request handler, no event
        loop. ``publish`` may return ``None`` when the session is already
        stopping, and must not be called after ``run`` has returned.
        """
        ...


class Sessions:
    """Own the producer thread and the in-memory session records.

    Args:
        publisher: Publisher every accepted packet goes through.
        producer_factory: Zero-argument callable returning a producer. It is the
            integration boundary (``create_app(producer_factory=...)``); ``None``
            means this server has no producer configured and ``start`` refuses.
        stop_timeout: Seconds a worker gets to observe cancellation.

    Attributes:
        publisher: The publisher passed in.
        records: Session summaries by id, oldest first, capped at
            :data:`MAX_SESSION_RECORDS`.
        current: Id of the most recently started session.
        thread: The producer thread, or ``None`` before the first start.
        closed: Whether the owning server has begun shutting down.
    """

    def __init__(
        self,
        publisher: Publisher,
        producer_factory: Callable[[], Producer] | None = None,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT,
    ) -> None:
        if producer_factory is not None and not callable(producer_factory):
            raise ValueError("producer_factory must be callable or None.")
        if not isinstance(stop_timeout, (int, float)) or stop_timeout <= 0:
            raise ValueError("stop_timeout must be a positive number of seconds.")

        self.publisher = publisher
        self.factory = producer_factory
        self.stop_timeout = float(stop_timeout)
        self.command_lock = Lock()
        self.lock = Lock()
        self.records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.current: str | None = None
        self.thread: Thread | None = None
        self.stop_event = Event()
        self.closed = False

    def list(self) -> list[dict[str, Any]]:
        """Return detached copies of the retained session summaries, oldest first."""

        with self.lock:
            return deepcopy(list(self.records.values()))

    def start(self) -> dict[str, Any]:
        """Start a session, or return the running one unchanged.

        Returns:
            A detached copy of the session record.

        Raises:
            RuntimeError: If the server is shutting down, or if no producer
                factory was configured. The message names which of the two.
        """

        with self.command_lock:
            with self.lock:
                if self.closed:
                    raise RuntimeError("server shutting down")
                if self.thread is not None and self.thread.is_alive():
                    return deepcopy(self.records[self.current])
                if self.factory is None:
                    raise RuntimeError("no producer factory configured")
                producer = self.factory()
                session_id = str(uuid4())
                self.current = session_id
                self.records[session_id] = {
                    "id": session_id,
                    "status": "running",
                    "simulated": bool(getattr(producer, "simulated", False)),
                }
                while len(self.records) > MAX_SESSION_RECORDS:
                    self.records.popitem(last=False)
                self.stop_event = Event()
                self.publisher.begin(session_id)
                self.publisher.publish(
                    "session", 0, RESERVED_SOURCE, session_id, self.records[session_id]
                )
                self.thread = Thread(
                    target=self._run,
                    args=(session_id, producer, self.stop_event),
                    name="nova-transport-producer",
                    daemon=True,
                )
                self.thread.start()
                return deepcopy(self.records[session_id])

    def _run(self, session_id: str, producer: Producer, stop: Event) -> None:
        """Run one producer to completion and publish its terminal session state."""

        began = monotonic()
        failed = False
        # Serializes publication so a producer's packets stay in producer order.
        publication_lock = Lock()

        def publish(
            kind: str, timestamp: float, source: str, payload: dict[str, Any]
        ) -> dict[str, Any] | None:
            """Publish one producer result into the running session."""

            if source == RESERVED_SOURCE:
                raise ValueError(
                    f"packet source {RESERVED_SOURCE!r} is reserved for the transport."
                )
            if stop.is_set():
                return None
            with publication_lock:
                return self.publisher.publish(kind, timestamp, source, session_id, payload)

        try:
            producer.run(publish, stop)
        except Exception:
            # Deliberately not re-raised or logged: the record carries a fixed
            # code, never the exception's text.
            failed = True
        finally:
            with self.lock:
                self.records[session_id]["status"] = "error" if failed else "stopped"
                if failed:
                    self.records[session_id]["error"] = "producer_failed"
                self.publisher.publish(
                    "session",
                    monotonic() - began,
                    RESERVED_SOURCE,
                    session_id,
                    self.records[session_id],
                )
                self.publisher.end(session_id)

    def _stop(self) -> dict[str, Any] | None:
        """Signal the worker, join it, and return the current record."""

        with self.lock:
            self.stop_event.set()
            thread = self.thread
        if thread is not None:
            thread.join(timeout=self.stop_timeout)
            if thread.is_alive():
                raise RuntimeError(
                    f"producer did not stop within the {self.stop_timeout:g}s deadline"
                )
        with self.lock:
            return deepcopy(self.records.get(self.current)) if self.current else None

    def stop(self) -> dict[str, Any] | None:
        """Stop the current session; idempotent and ``None`` before the first start."""

        with self.command_lock:
            return self._stop()

    def close(self) -> dict[str, Any] | None:
        """Refuse further starts, then stop and join the worker."""

        with self.command_lock:
            with self.lock:
                self.closed = True
            return self._stop()


__all__ = [
    "MAX_SESSION_RECORDS",
    "Producer",
    "Publish",
    "RESERVED_SOURCE",
    "Sessions",
]
