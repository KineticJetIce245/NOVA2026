"""The observed media timeline: the browser plays, the transport only transcribes.

By decision D-02 the playback position and its revision are **owned by the
browser**: the frontend validates ``media_id``, ``media_revision`` and
``media_time_s`` field by field and falls back to neutral gain on any
disagreement, so a backend that "improves" a position is worse than one that is
honestly late. This module therefore stores what the controller reported,
acknowledges it with a revision, and refuses commands that contradict the
timeline it has already accepted.

Two rules make the refusals meaningful:

* **Progression must be plausible.** A report may not move the position
  backwards (beyond 20 ms of rounding) and may not jump further ahead than the
  time that actually elapsed plus a 0.5 s allowance per report, so a seek or a
  stalled tab cannot silently re-anchor the timeline.
* **A violation is sticky.** The first contradictory command marks the timeline
  invalid and ``sync_status`` stays ``desynchronized`` until a fresh ``prepare``
  starts a new revision; a later "good" report cannot repair a bad seek.

``sync_status`` is ``observed`` only while a prepared timeline is fresh (a report
within 1.5 s of the clock reading), which is the acknowledgement the attention
path needs before it may attribute a measurement to a playback position.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from .protocol import MAX_SAFE_INTEGER
from .publisher import Publisher

MEDIA_PACKET_TYPE = "media"
"""Packet type the frontend's decoder maps to the media timeline card."""

RESERVED_SOURCE = "server"
"""Source the transport owns; the media timeline is transport state, not a measurement."""

DEFAULT_INTERVAL = 0.25
"""Seconds between media packets: the controller's own tick, so no position is skipped."""

SUPPORTED_SUFFIXES = (".wav", ".mp3", ".m4a", ".mp4", ".webm")
"""Media containers the controller may point at."""

VIDEO_SUFFIXES = (".mp4", ".webm")
"""Suffixes whose descriptor reports ``kind='video'`` instead of ``'audio'``."""

MAX_MEDIA_SECONDS = 864000.0
"""Upper bound for reported positions and durations (10 days)."""

FRESH_SECONDS = 1.5
"""A report older than this leaves the timeline desynchronized."""

PLAYING_ADVANCE_ALLOWANCE = 0.5
"""Extra seconds a report may run ahead of the elapsed time while playing."""

IDLE_ADVANCE_ALLOWANCE = 0.5
"""Forward jump allowed by a command that starts playback from a stopped timeline."""

PAUSED_ADVANCE_ALLOWANCE = 0.1
"""Forward jump allowed while the timeline is not playing and stays not playing."""

BACKWARD_TOLERANCE = 0.02
"""Position regression tolerated as rounding while playing."""

def seconds(value: Any, field: str) -> float:
    """Validate one reported time value and return it as ``float``.

    Args:
        value: Candidate seconds value from a controller command.
        field: Field name used in the error message.

    Returns:
        The value as ``float``.

    Raises:
        ValueError: If ``value`` is not a finite number in
            ``[0, MAX_MEDIA_SECONDS]``. A boolean is not a number here.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= MAX_MEDIA_SECONDS
    ):
        raise ValueError(f"{field} must be a finite number of seconds in [0, {MAX_MEDIA_SECONDS}].")
    return float(value)

class MediaTimeline:
    """One media asset plus the controller timeline observed for it.

    Args:
        path: Existing file to serve. Its resolved path never leaves the process;
            :meth:`descriptor` exposes only ``/api/media/file``.
        title: Display title, trimmed to 128 characters.
        clock: Monotonic clock, injectable so tests can advance time deliberately.

    Raises:
        ValueError: If ``path`` is not an existing file with a supported suffix.
    """

    def __init__(
        self,
        path: str | Path,
        title: str = "Demo Audio",
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError(f"media file does not exist: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"media suffix {self.path.suffix!r} is not one of {SUPPORTED_SUFFIXES}."
            )

        self.title = title.strip()[:128] or "Demo Audio"
        self.media_id = uuid4().hex
        self.kind = "video" if self.path.suffix.lower() in VIDEO_SUFFIXES else "audio"
        self.clock = clock
        self.lock = RLock()
        self.session_id: str | None = None
        self.revision = 0
        self.client_id: str | None = None
        self.request_id = -1
        self.position = 0.0
        self.duration: float | None = None
        self.playback = "stopped"
        self.reference: float | None = None
        self.last_received: float | None = None
        self.valid = False
        self.ever_claimed_at: float | None = None
        self.takeovers = 0
        """How often a new controller reclaimed a slot its owner had left."""

    def descriptor(self) -> dict[str, Any]:
        """Return the asset identity the controller needs, without its path."""

        return {
            "media_id": self.media_id,
            "title": self.title,
            "kind": self.kind,
            "url": "/api/media/file",
        }

    def bind(self, session_id: str) -> None:
        """Attach the timeline to a session, resetting it when the session changes."""

        with self.lock:
            if self.session_id != session_id:
                self.stop()
                self.session_id = session_id
                self.ever_claimed_at = None

    def claimed(self) -> bool:
        """Whether any controller has held this slot since the session was bound.

        This is the question a client that is *deciding whether to compete* needs
        answered, and it is deliberately a different question from "is the slot
        free": :meth:`stop` clears ``client_id`` whether it was the controller
        that stopped or a fresh one taking over, so a second preparer asking
        "is it mine?" cannot tell an idle slot from an abandoned one. Here the
        answer stays ``True`` for the rest of the session once the first
        ``prepare`` has been accepted, which is what makes a stand-in able to
        stand down for good instead of racing for the slot.

        Read-only: it takes the lock, reads one field and changes nothing, so it
        cannot alter any acknowledgement or refusal :meth:`control` produces.

        Returns:
            ``True`` once a ``prepare`` has been accepted since :meth:`bind`
            attached this timeline to its current session.
        """

        with self.lock:
            return self.ever_claimed_at is not None

    def stop(self) -> None:
        """Return to the neutral stopped state and bump the revision.

        Stopping drops the controller identity, so the next command must be a
        ``prepare`` from whichever client takes over.
        """

        with self.lock:
            self.position = 0.0
            self.playback = "stopped"
            self.client_id = None
            self.request_id = -1
            self.valid = False
            self.revision += 1

    def snapshot(self) -> dict[str, Any]:
        """Return the descriptor, position, playback state and sync status.

        ``sync_status`` is ``observed`` only when the timeline is valid *and* a
        report arrived within :data:`FRESH_SECONDS`; otherwise it is
        ``desynchronized`` and the attention path must degrade to neutral.
        """

        with self.lock:
            fresh = (
                self.last_received is not None
                and self.clock() - self.last_received <= FRESH_SECONDS
            )
            return dict(
                **self.descriptor(),
                session_id=self.session_id,
                media_time_s=self.position,
                duration_s=self.duration,
                playback_state=self.playback,
                revision=self.revision,
                server_reference_s=self.reference,
                server_received_s=self.last_received,
                takeovers=self.takeovers,
                sync_status="observed" if self.valid and fresh else "desynchronized",
            )

    def media_reference(self) -> dict[str, Any] | None:
        """The media reference packets may carry, or ``None`` and no invention.

        Decision D-02 gives this process transcription rights over the playback
        position, not authorship: the values returned here are the ones the
        controller last reported, together with the revision this process
        acknowledged. ``None`` means there is nothing honest to copy - no
        controller has prepared the timeline, a contradictory command invalidated
        it, or the last report is older than :data:`FRESH_SECONDS` - and the
        attention path must then publish no media fields at all, which is what
        makes the frontend hold neutral gain instead of trusting a stale number.

        The freshness rule is exactly the one :meth:`snapshot` uses for
        ``sync_status == "observed"``, so a producer that only stamps packets when
        this returns a mapping cannot disagree with the ``media`` stream the
        browser is reading.
        """

        with self.lock:
            if not self.valid or self.last_received is None:
                return None
            if self.clock() - self.last_received > FRESH_SECONDS:
                return None
            return {
                "media_id": self.media_id,
                "media_revision": self.revision,
                "media_time_s": self.position,
            }

    def control(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply one controller command and return the resulting snapshot.

        Args:
            data: Decoded ``POST /api/media/control`` body: ``session_id``,
                ``media_id``, ``client_id``, ``request_id``, ``action``,
                ``media_time_s`` and ``duration_s``.

        Returns:
            The snapshot taken after the command was applied.

        Raises:
            ValueError: If the session or media identity does not match, the
                action is unknown, the controller request is stale or belongs to
                another client, a time value is invalid, or the position moved in
                a way the timeline cannot explain.
        """

        with self.lock:
            if data.get("session_id") != self.session_id or data.get("media_id") != self.media_id:
                raise ValueError("session_id or media_id does not match this timeline.")
            action = data.get("action")
            if action not in ("prepare", "playing", "paused", "stopped", "report"):
                raise ValueError(f"action must be one of the media actions, not {action!r}.")
            client = data.get("client_id")
            request = data.get("request_id")
            if not isinstance(client, str) or not 1 <= len(client) <= 128:
                raise ValueError("client_id must be a nonempty string of at most 128 characters.")
            if type(request) is not int or not 0 <= request <= MAX_SAFE_INTEGER:
                raise ValueError("request_id must be a nonnegative JavaScript-safe integer.")
            # A slot whose owner has stopped reporting is ABANDONED, not owned.
            # Closing the tab, reloading the page, or a sleeping laptop all leave
            # a client that will never send again, and until this existed that
            # client held the slot for the life of the process: `prepare` is
            # refused while playback is not stopped, and `stopped` from a different
            # client is refused by the check below, so neither a new page nor the
            # operator could get it back. Only silence longer than the freshness
            # window releases it, so a live owner still keeps its slot -- which is
            # the protection D-54 is about -- and the release is counted rather
            # than done quietly.
            if (
                self.client_id is not None
                and client != self.client_id
                and (
                    self.last_received is None
                    or self.clock() - self.last_received > FRESH_SECONDS
                )
            ):
                self.takeovers += 1
                self.client_id = None
                self.playback = "stopped"
                self.position = 0.0
                self.valid = False
                self.reference = self.clock()
            if self.client_id is not None and (
                client != self.client_id or request <= self.request_id
            ):
                raise ValueError("request_id must increase within one controller.")

            position = seconds(data.get("media_time_s"), "media_time_s")
            duration = seconds(data.get("duration_s"), "duration_s")
            if duration <= 0 or position > duration:
                raise ValueError("duration_s must be positive and media_time_s must not exceed it.")

            now = self.clock()
            if action == "prepare":
                if self.playback != "stopped" or position != 0:
                    raise ValueError(
                        "media must be stopped at position 0 before it can be prepared."
                    )
                self.revision += 1
                self.client_id = client
                self.reference = now
                self.playback = "paused"
                self.valid = True
                # Sticky for the rest of the session, so a stand-in can see that
                # the slot has an owner even after that owner stops.
                self.ever_claimed_at = now
            elif self.client_id is None:
                if action != "stopped":
                    raise ValueError("prepare must be accepted before any other action.")
            else:
                elapsed = now - self.last_received if self.last_received is not None else 0.0
                if self.playback == "playing":
                    max_advance = elapsed + PLAYING_ADVANCE_ALLOWANCE
                elif action == "playing":
                    max_advance = IDLE_ADVANCE_ALLOWANCE
                else:
                    max_advance = PAUSED_ADVANCE_ALLOWANCE
                if action != "stopped" and (
                    position < self.position - BACKWARD_TOLERANCE
                    or position - self.position > max_advance
                ):
                    # Sticky: only a fresh prepare clears this.
                    self.valid = False
                    raise ValueError(
                        "media position moved further than the elapsed time allows; "
                        "stop and prepare again."
                    )
                if action != "report":
                    self.playback = action

            self.request_id = request
            self.position = 0.0 if action == "stopped" else position
            self.duration = duration
            self.last_received = now
            if action == "stopped":
                self.stop()
            return self.snapshot()

class SessionLookup(Protocol):
    """What the broadcaster needs from the session registry, and nothing more."""

    def list(self) -> list[dict[str, Any]]:
        """Session summaries, oldest first; the last one is the current session."""

        ...

class MediaBroadcaster:
    """Publish the observed media timeline as ``media`` packets while a session runs.

    The controller learns the timeline from the acknowledgement of its own
    ``POST /api/media/control``, but the *dashboard* has no such channel: it reads
    the ``media`` packet that the frontend's decoder turns into ``syncStatus``,
    ``playbackState``, ``revision`` and ``mediaId``, and its gain gate refuses to
    activate - correctly - when that packet is missing. This thread is that
    channel, and it is deliberately a transcriber: every field comes from
    :meth:`MediaTimeline.snapshot` and the position is the one the controller
    reported.

    Two properties keep it from becoming a source of invented time:

    * It publishes only while the newest session is ``running``, so a stopped or
      failed session stops being described rather than freezing on screen.
    * A ``media`` packet never carries a position this process computed. When the
      controller stops reporting, ``sync_status`` falls to ``desynchronized``
      through the timeline's own freshness rule and the frontend's gate closes.

    Args:
        publisher: Publisher the packets go through; the same one the producer and
            the transport's lifecycle packets use, so sequence numbers stay in one
            space.
        media: Timeline to transcribe.
        sessions: Registry consulted for the running session id.
        interval: Seconds between packets. The default matches the controller's
            250 ms tick, which is the rate at which the browser's position
            actually changes.
        clock: Monotonic clock, injectable so a test can drive the cadence.

    Raises:
        ValueError: If ``interval`` is not finite and positive.
    """

    def __init__(
        self,
        publisher: Publisher,
        media: MediaTimeline,
        sessions: SessionLookup,
        *,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("interval must be a finite positive number of seconds.")
        self.publisher = publisher
        self.media = media
        self.sessions = sessions
        self.interval = float(interval)
        self.clock = clock
        self.published = 0
        self.failures = 0
        self._thread: Thread | None = None
        self._stop = Event()
        self._baseline: float | None = None

    def tick(self) -> dict[str, Any] | None:
        """Publish one packet if a session is running. Returns it, or ``None``.

        Returns:
            The published packet, or ``None`` when there is no running session or
            the publisher refused it. A refusal is counted, never raised: a
            transcriber that has lost its session has nothing to say and must not
            be able to end that session by raising on a background thread.
        """

        records = self.sessions.list()
        session_id = (
            records[-1]["id"]
            if records and records[-1]["status"] == "running"
            else None
        )
        if session_id is None:
            return None
        payload = self.media.snapshot()
        try:
            packet = self.publisher.publish(
                MEDIA_PACKET_TYPE,
                self._timestamp(),
                RESERVED_SOURCE,
                session_id,
                payload,
            )
        except ValueError:
            self.failures += 1
            return None
        self.published += 1
        return packet

    def start(self) -> None:
        """Start the loop on a daemon thread; a second call is a no-op."""

        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._loop, name="media-broadcaster", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the loop to finish, join it, and forget the stream's time base."""

        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
        self._baseline = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()

    def _timestamp(self) -> float:
        """Session-relative seconds, nondecreasing within the ``media`` stream."""

        now = self.clock()
        if self._baseline is None or now < self._baseline:
            self._baseline = now
        return max(0.0, now - self._baseline)

__all__ = [
    "BACKWARD_TOLERANCE",
    "DEFAULT_INTERVAL",
    "FRESH_SECONDS",
    "IDLE_ADVANCE_ALLOWANCE",
    "MEDIA_PACKET_TYPE",
    "MAX_MEDIA_SECONDS",
    "PAUSED_ADVANCE_ALLOWANCE",
    "PLAYING_ADVANCE_ALLOWANCE",
    "RESERVED_SOURCE",
    "SUPPORTED_SUFFIXES",
    "VIDEO_SUFFIXES",
    "MediaBroadcaster",
    "MediaTimeline",
    "SessionLookup",
    "seconds",
]
