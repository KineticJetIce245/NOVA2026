"""One attention session, shared by the replay, live and transport paths.

The plan (``final_connection.md`` sections 2, 3.2, 3.4 and 3.5) asks for a single
piece of timing logic, and this module is it. A session

* pulls chunks from an :class:`~nova2026.auditory.sources.AcquisitionSource` and
  feeds the existing streaming chain, so replay and a future LSL path differ only
  in where the chunks come from;
* attaches the two verified reference envelopes through
  :class:`~nova2026.auditory.sources.ReferenceEnvelopes`, which is the only place
  the audio anchor is used;
* offloads scoring to **one** worker with a **two-item** queue and the
  ``drop_oldest`` policy, and turns a dropped item into an explicit
  ``evidence_gap`` estimate instead of silence, so the controller forgets a
  selection that a lost window can no longer support;
* updates the controller with the estimates in evidence order and reports the
  decision, the two raw correlations, the two gains and the reasons;
* degrades explicitly. An invalid window, an uncovered audio range, a dropped
  item, a chain failure or evidence older than the controller's age limit becomes
  ``uncertain`` or ``unavailable`` with a reason - never a confident answer.

Three properties are structural rather than promised:

1. **No label is reachable.** This module never receives a
   :class:`~nova2026.auditory.data.AuditoryTrial`: it receives already-extracted
   chunks, two envelopes and a fitted decoder. There is no code path here that
   could put ground truth into the decoder's input.
2. **No timestamp is invented.** Frames are stamped with source-clock seconds,
   offset to session start; the window grid is the chain's own.
3. **No transport import.** The transport imports this module, never the reverse:
   nothing here knows what a packet is. A :class:`SessionFrame` is what a
   consumer gets, and :mod:`nova2026.auditory.producer` maps one to packets.

Run policy is explicit, not inherited (plan section 3.17 item 1): the chain's own
default (``check_channels=True, max_bad_channels=0``) halts a KU Leuven trial when
faults persist past the recovery budget, so :class:`RunPolicy` has no defaults for
those two fields and every run records the policy it used.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from nova2026.streaming.offload import TaskOffloader

from .config import MIN_MARGIN
from .controller import AttentionController
from .data import AttentionEstimate
from .sources import AcquisitionSource, ReferenceEnvelopes
from .streaming import AuditoryProcessor, stream_config

DECISIONS = ("A", "B", "uncertain", "unavailable")
"""The four decision words of the plan's section 3.5, and the only four emitted."""

ARTIFACT_REASONS = frozenset(
    {"amplitude", "flatline", "bad_channels", "quality", "saturated", "interpolated"}
)
"""Chain reasons that describe the signal rather than the session's own state."""

OFFLOAD_STOP_TIMEOUT = 5.0
"""Seconds the offload worker gets to finish before the session reports it stuck."""

FAILURE_REASON_LIMIT = 240
"""Characters of an exception's text kept in a reason string.

Long enough for the decoder's own refusals (which name the keys that differ) and
short enough that a published frame's reason stays readable. Truncation is marked
rather than silent, so a clipped cause cannot be mistaken for the whole one.
"""


def _failure_reason(error: BaseException) -> str:
    """One line naming what failed: its type and its own message.

    ``str(error)`` alone is empty for some exceptions and ambiguous for the rest
    (a ``ValueError`` and a ``RuntimeError`` can read identically), and the type is
    the part that says which mechanism failed. Newlines are folded so the text
    stays a single reason string in a packet, and the whole line - type included -
    is what :data:`FAILURE_REASON_LIMIT` bounds.
    """

    text = " ".join(str(error).split())
    reason = type(error).__name__ + (": " + text if text else "")
    if len(reason) > FAILURE_REASON_LIMIT:
        reason = reason[: FAILURE_REASON_LIMIT - 3] + "..."
    return reason


@dataclass(frozen=True)
class RunPolicy:
    """The choices a run makes on purpose, so the run record can state them.

    ``check_channels`` and ``max_bad_channels`` deliberately have no defaults.
    Section 3.17 of the plan records that the streaming chain's default policy
    stops a KU Leuven trial mid-run (``S1/trial_003`` raised
    ``EEG quality faults persisted beyond the allowed duration``), and step 5's
    feature cache therefore used ``check_channels=False``. That is a documented
    run policy - the signal is unchanged and only the verdict moves - so a session
    must be handed one instead of inheriting a default that decides for it.

    Args:
        check_channels: Whether the chain may stop a run over dead electrodes.
        max_bad_channels: Bad channels tolerated inside one window.
        exclude_channels: Channels known to be dead, excluded from the census.
        source_unit_exponent: Power of ten of the source unit relative to volts
            (``-6`` for microvolts, ``0`` for the ANT amplifier).
        margin: Correlation-difference threshold the controller commits on. The
            default is :data:`~nova2026.auditory.config.MIN_MARGIN` (0.5), so a
            caller that says nothing gets exactly the value it got before this
            field existed. It is a run-policy choice, not a constant to edit:
            step 9.6 measured (``results/aad_margin_calibration_*.md``) that 0.5
            covers 0.15% of frames at the saved model's 5 s window, and that 0.05
            covers 52.4% at balanced accuracy 0.688 on the decided frames. The
            demo therefore selects
            :data:`~nova2026.auditory.config.CALIBRATED_MARGIN` **explicitly**,
            and the run record states the effective value (decision D-29).
        warmup_seconds: Session seconds published as ``unavailable`` before the
            decision path starts (plan section 3.5: warm up, never ``uncertain``).
        frame_seconds: Cadence of published state, in session seconds.
    """

    check_channels: bool
    max_bad_channels: int
    exclude_channels: tuple[str, ...] = ()
    source_unit_exponent: int = -6
    saturation_limit_uv: float | None = None
    """Repair's absolute-level guard in uV; ``None`` keeps the chain default.

    Run policy, not contract: it decides what counts as an unsafe endpoint in a
    *recording*, and a recording from an amplifier that railed an electrode above
    the chain default would otherwise stop the run at every railed stretch
    (plan section 3.11 measures exactly that rail on the operator's ANT session).
    """
    amplitude_limit_uv: float | None = None
    """The excursion from the run's own anchor level that counts as a fault, in uV.

    ``None`` keeps the chain default (500 uV for both ``Repair``'s endpoint check
    and ``QualityMonitor``'s ``"amplitude"`` fault). One declared value with one
    owner: the run policy states it once and both stages receive it, so the
    quality verdict and the repair verdict cannot disagree about which electrodes
    are still carrying signal. Run policy, not contract -- adding it to
    ``AuditoryProcessor.contract`` would invalidate every trained decoder.

    A recording that genuinely drifts further than the default against its own
    first-level anchor declares its own measured limit here rather than editing
    the default: the operator's ANT session drifts 1.5-3.0 mV from the anchor on
    18 of its 20 electrodes (measured, ``results/antneuro_live_*.json``), and at
    the default every window is an artifact and no decision is ever committed.
    """
    margin: float = MIN_MARGIN
    warmup_seconds: float = 2.0
    frame_seconds: float = 0.25
    display_channel: str | None = None
    """Electrode the ``eeg_display`` tap carries; ``None`` publishes none.

    Run policy, not contract: it decides what a panel draws, never what the
    decoder receives, and adding it to ``AuditoryProcessor.contract`` would
    invalidate every trained model. The default is ``None``, so a caller that
    says nothing gets exactly the run it got before this field existed -- no
    capture, no allocation, no packet.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.check_channels, bool):
            raise TypeError("check_channels must be an explicit bool.")
        if (
            isinstance(self.max_bad_channels, bool)
            or not isinstance(self.max_bad_channels, int)
            or self.max_bad_channels < 0
        ):
            raise ValueError("max_bad_channels must be an explicit non-negative integer.")
        if isinstance(self.exclude_channels, str):
            raise TypeError("exclude_channels must be an iterable of labels.")
        if self.display_channel is not None and (
            not isinstance(self.display_channel, str) or not self.display_channel.strip()
        ):
            raise TypeError(
                "display_channel must be None or a nonempty channel label; a label "
                "that names nothing would make the panel's channel_source text false."
            )
        if isinstance(self.margin, bool) or not isinstance(self.margin, (int, float)):
            raise TypeError("margin must be a number.")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("margin must be finite and positive.")
        object.__setattr__(self, "margin", float(self.margin))
        for name in ("warmup_seconds", "frame_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        object.__setattr__(self, "exclude_channels", tuple(self.exclude_channels))
        if self.display_channel is not None:
            object.__setattr__(self, "display_channel", self.display_channel.strip())

    def to_dict(self) -> dict:
        """A JSON-safe record of the policy, for the run record and the packets."""

        return {
            "check_channels": self.check_channels,
            "max_bad_channels": self.max_bad_channels,
            "exclude_channels": list(self.exclude_channels),
            "source_unit_exponent": self.source_unit_exponent,
            "saturation_limit_uv": self.saturation_limit_uv,
            "amplitude_limit_uv": self.amplitude_limit_uv,
            "margin": self.margin,
            "warmup_seconds": self.warmup_seconds,
            "frame_seconds": self.frame_seconds,
            "display_channel": self.display_channel,
        }

    @classmethod
    def kuleuven_replay(cls, **overrides) -> "RunPolicy":
        """The documented policy for replaying KU Leuven trials.

        ``check_channels=False`` is the recorded choice, for the reason in the
        class docstring: the trials contain excursions past the amplitude limit,
        and the decision that stops a run is not the signal. Every caller that
        wants a different policy has to say so, and the policy travels into the
        run record either way.
        """

        values = {"check_channels": False, "max_bad_channels": 0}
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class SessionFrame:
    """One published state of a session: what a consumer needs to render it.

    ``correlation_a`` and ``correlation_b`` are the raw scores of the newest
    estimate that carried any - never smoothed, never re-scaled - and
    ``window_end`` says how old they are, so a consumer can see staleness rather
    than infer it. ``reasons`` is the newest evidence's reason set, which is what
    makes ``uncertain`` auditable.
    """

    timestamp: float
    decision: str
    correlation_a: float | None
    correlation_b: float | None
    gain_a_db: float
    gain_b_db: float
    quality: float | None
    artifact: bool | None
    reasons: tuple[str, ...]
    window_end: float | None
    evidence_count: int
    media_time_s: float | None
    bad_channels: tuple[str, ...] = ()
    """Channels the chain's census flagged in the newest window. Evidence, not a
    rejection: ``artifact`` is judged from it under a relaxed quality policy,
    where the reason set stays empty (plan section 3.17 item 6)."""

    display: dict = field(default_factory=dict)
    """The newest ``eeg_display`` payload, or ``{}`` when the tap is off.

    The chain built it while the raw signal existed (see
    :meth:`~nova2026.auditory.streaming.AuditoryProcessor._capture_display`); a
    frame carries it so the producer publishes it on the frame cadence without
    reaching back into the chain. Empty is the default and the off state, so the
    field's presence never changes what a run with no display channel publishes.
    """

    def to_dict(self) -> dict:
        """A JSON-safe copy, for a log or a run record."""

        return {
            "timestamp": self.timestamp,
            "decision": self.decision,
            "correlation_a": self.correlation_a,
            "correlation_b": self.correlation_b,
            "gain_a_db": self.gain_a_db,
            "gain_b_db": self.gain_b_db,
            "quality": self.quality,
            "artifact": self.artifact,
            "bad_channels": list(self.bad_channels),
            "reasons": list(self.reasons),
            "window_end": self.window_end,
            "evidence_count": self.evidence_count,
            "media_time_s": self.media_time_s,
            "display": dict(self.display),
        }


class SessionFailed(RuntimeError):
    """The chain refused to continue; the session degrades rather than pretend.

    Args:
        timestamp: Source-clock time of the failure.
        code: Fixed machine-readable reason, e.g. ``processing_failed``.
        detail: The chain's own message. It stays in the local run record; the
            transport's session record deliberately carries only its own fixed
            code, because a transport is not a log sink.
    """

    def __init__(self, timestamp: float, code: str, detail: str = "") -> None:
        super().__init__(f"{code} at {timestamp:g}s" + (f": {detail}" if detail else ""))
        self.timestamp = float(timestamp)
        self.code = str(code)
        self.detail = str(detail)


@dataclass
class SessionSummary:
    """What one run did, in numbers a report can print without re-deriving them."""

    kind: str
    simulated: bool
    policy: dict
    channel_count: int
    window_seconds: float
    step_seconds: float
    started: float
    ended: float
    wall_seconds: float
    windows: int = 0
    submitted: int = 0
    scored: int = 0
    invalid: int = 0
    evidence_gaps: int = 0
    failed: int = 0
    frames: int = 0
    decisions: dict = field(default_factory=dict)
    decision_stream: list = field(default_factory=list)
    """``[(seconds, "A"|"B")]``, one entry per window the controller committed on.

    The seconds are offset to session start and stamped with the window's own
    evidence end, so a report can compare them with a label series without
    re-deriving when each window ended. Entries that the controller later
    abandoned are not removed: this is what was published, in order.
    """

    bad_channel_census: dict = field(default_factory=dict)
    """``{channel: windows}`` for every channel the census flagged at least once.

    Recorded rather than merely counted: case C7 of the plan's perturbation
    matrix asks a bad channel to be *traceable* in the run record, and a single
    "artifact" boolean cannot say which electrode it was.
    """

    recovery_events: list = field(default_factory=list)
    """The streaming chain's own recovery decisions, in order.

    ``{kind, recovery, segment, gap}`` as
    :class:`~nova2026.streaming.recovery.Recovery` recorded them: which fault
    (``irregular_timestamps``, ``large_gap``, ``nonfinite_run``,
    ``unsafe_endpoints``), which restart, and the gap size when there was one.
    Carried into the run record because the plan's perturbation matrix asks for a
    handled fault to be *auditable* (cases C4, C5, C6, E2, E5, E6): a recovery
    that repaired a chunk and restarted the chain leaves no window behind, so
    without this field a fault the chain handled perfectly would look exactly
    like a fault that never happened. It is a record of what the chain did, never
    an input to a decision.
    """

    recovery_segment: int = 0
    """Processing segment the run ended on; one more than the number of recoveries."""

    repaired_samples: int = 0
    """Rows the chain's :class:`~nova2026.streaming.preprocess.repair.Repair`
    reconstructed, the counter behind the ``interpolated`` reason."""

    scoring_failures: list = field(default_factory=list)
    """Named causes of the scoring failures counted by :attr:`failed`.

    ``failed`` alone is a count, and a count cannot say what failed: measured on
    the operator's ANT session, ``scored 0 / failed 116`` was the whole run record
    while the cause - every window refused by the decoder's own contract check -
    lived only inside an exception the session discarded. Each entry is
    ``{"count", "type", "reason"}``, deduplicated by reason so ninety identical
    refusals read as one named cause with a count rather than ninety copies. It is
    the same text the ``scoring_failed`` reason carries, recorded where a report
    can print it after the run.
    """

    failure: dict | None = None

    def to_dict(self) -> dict:
        """A JSON-safe copy of the summary."""

        return {
            "kind": self.kind,
            "simulated": self.simulated,
            "policy": self.policy,
            "channel_count": self.channel_count,
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "started": self.started,
            "ended": self.ended,
            "wall_seconds": self.wall_seconds,
            "windows": self.windows,
            "submitted": self.submitted,
            "scored": self.scored,
            "invalid": self.invalid,
            "evidence_gaps": self.evidence_gaps,
            "failed": self.failed,
            "frames": self.frames,
            "decisions": dict(self.decisions),
            "decision_stream": [[time_s, word] for time_s, word in self.decision_stream],
            "bad_channel_census": dict(self.bad_channel_census),
            "recovery_events": [dict(event) for event in self.recovery_events],
            "recovery_segment": self.recovery_segment,
            "repaired_samples": self.repaired_samples,
            "scoring_failures": [dict(entry) for entry in self.scoring_failures],
            "failure": dict(self.failure) if self.failure else None,
        }


class AttentionSession:
    """Own the acquisition loop, the evidence path and the decision for one run.

    Args:
        source: The acquisition source (see
            :class:`~nova2026.auditory.sources.AcquisitionSource`).
        decoder: A fitted :class:`~nova2026.auditory.decoder.RidgeDecoder`. Its
            contract fixes the window length, the step and the channel order, so
            the chain is configured from the model rather than the other way
            round.
        references: The two verified candidate envelopes.
        policy: The explicit :class:`RunPolicy` for this run.
        controller: Decision controller; a default one is built otherwise, with
            its ``margin`` taken from the policy rather than from the
            constructor's own default (decision D-29). Passing a controller
            keeps winning, and that controller's own margin is then the
            effective one.
        offload_workers: Scoring workers. One is the plan's decision; the
            parameter exists so a test can prove the queue is the bottleneck.
        offload_capacity: Queued windows before the overflow policy applies.
        clock: Monotonic wall-clock source, injectable for tests.

    Raises:
        ValueError: If the decoder records no window/step contract, or if the
            source lacks a channel the decoder needs. Missing channels are never
            truncated to fit - plan section 3.14 case E4.
    """

    def __init__(
        self,
        *,
        source: AcquisitionSource,
        decoder,
        references: ReferenceEnvelopes,
        policy: RunPolicy,
        controller: AttentionController | None = None,
        offload_workers: int = 1,
        offload_capacity: int = 2,
        clock=time.monotonic,
    ) -> None:
        contract = dict(decoder.contract or {})
        history = contract.get("window_seconds")
        step = contract.get("step_seconds")
        if not isinstance(history, (int, float)) or not isinstance(step, (int, float)):
            raise ValueError(
                "The decoder records no window_seconds/step_seconds contract; "
                "a session cannot guess the chain's window length."
            )

        self.source = source
        self.decoder = decoder
        self.references = references
        self.policy = policy
        self.controller = (
            controller
            if controller is not None
            else AttentionController(margin=policy.margin)
        )
        self.offload_workers = int(offload_workers)
        self.offload_capacity = int(offload_capacity)
        self._clock = clock

        settings = stream_config(
            source,
            decoder.config,
            float(history),
            float(step),
            check_channels=policy.check_channels,
            max_bad_channels=policy.max_bad_channels,
            exclude_channels=policy.exclude_channels,
            source_unit_exponent=policy.source_unit_exponent,
            display_channel=policy.display_channel,
        )
        # The endpoint and excursion guards are run policy, so they travel the
        # same way the channel policy and the exponent do: into the chain's
        # settings, where `AuditoryProcessor` hands the saturation limit to
        # `Repair` and the amplitude limit to both `Repair` and `QualityMonitor`.
        # `None` keeps the chain defaults and changes nothing for callers that do
        # not set them.
        settings.saturation_limit_uv = policy.saturation_limit_uv
        settings.amplitude_limit_uv = policy.amplitude_limit_uv

        self.source_channels = tuple(source.channel_names)
        wanted = tuple(contract.get("eeg_channels") or self.source_channels)
        missing = [name for name in wanted if name not in self.source_channels]
        if missing:
            raise ValueError(
                "the decoder needs channels this source does not carry: "
                + ", ".join(missing)
            )
        settings.eeg_channels = wanted
        self.settings = settings
        self.window_seconds = float(history)
        self.step_seconds = float(step)

        # Every controller mutation and read happens under this lock: the
        # acquisition thread applies invalid windows and drops, the offload
        # worker applies scored ones, and the frame emitter only reads.
        self._lock = Lock()
        self._now = float(source.start)
        self._inflight: deque[tuple[int, float, float]] = deque()
        self._next_item = 0
        self._drops_seen = 0
        self._offload: TaskOffloader | None = None
        self.processor = None
        """The chain that ran the last :meth:`run`, or ``None`` before one.

        Public because the chain's own diagnostics are part of the run record:
        :attr:`SessionSummary.recovery_events` and
        :attr:`SessionSummary.repaired_samples` are read from here after the run,
        and a caller that wants the quality monitor's or the repair stage's
        counters can reach them without this class re-exporting every one.
        """
        self._failure: SessionFailed | None = None
        self._census: dict[str, int] = {}

        self.windows = 0
        self.submitted = 0
        self.scored = 0
        self.invalid = 0
        self.evidence_gaps = 0
        self.failed = 0
        # Named causes behind `failed`, deduplicated by reason: a count alone left
        # the operator's ANT run (0 scored, 116 failed) with no cause in its record
        # at all, because the exception that held it was discarded here.
        self.scoring_failures: list[dict] = []
        self.frames = 0
        self.evidence_count = 0
        self.scores: tuple[float, float] | None = None
        self.reasons: tuple[str, ...] = ()
        self.evidence_end: float | None = None
        self.quality: float | None = None
        self.artifact: bool | None = None
        self.bad_channels: tuple[str, ...] = ()
        self.bad_channel_census: dict[str, int] = {}
        self.decisions: list[tuple[float, str]] = []
        self.display: dict = {}
        """Newest display payload, replaced by each window and read by each frame."""

    # ------------------------------------------------------------------ helpers

    @property
    def kind(self) -> str:
        """Short source name recorded by the run policy and the packets."""

        return str(getattr(self.source, "kind", "unknown"))

    @property
    def simulated(self) -> bool:
        """Whether this session's data is a fixture rather than a measurement."""

        return bool(getattr(self.source, "simulated", False))

    def audio_position(self, source_time: float) -> float | None:
        """Media seconds of the audio played at ``source_time``.

        This is the audio *position implied by the EEG clock* through the same
        anchor alignment uses. It is a measurement of the session's own timeline,
        not authority over what a browser is playing: the frontend owns that
        (plan section 2.2, decision D-02).
        """

        anchor = getattr(self.source, "audio_start", None)
        if anchor is None or not math.isfinite(float(anchor)):
            return None
        return float(source_time) - float(anchor)

    def evidence_end_for(self, window) -> float:
        """Evidence time of a window: the last sample with a full decoder lag."""

        index = len(window.timestamps) - self.decoder.config.lag_samples - 1
        if index < 0:
            raise ValueError("Insufficient history for decoder lags.")
        return float(window.timestamps[index])

    def _verdict(self, aligned) -> tuple[float, bool, tuple[str, ...]]:
        """Quality and artifact verdicts for the newest window, from its evidence.

        Two independent sources are read, because either alone is incomplete:

        ``reasons``
            The chain's own rejection reasons. Under the relaxed quality policy
            (``check_channels=False``) the monitor's ``reasons()`` is empty by
            construction, so this source says ``quality == 1.0`` even for a dry
            cap (plan section 3.17 item 6).
        ``bad_channels``
            The channel census, which the policy deliberately does not turn into
            a rejection. A dead electrode is therefore invisible in ``reasons``
            but present here, and judging ``artifact`` from the census is what
            makes the published ``signal_quality`` honest.

        The census never makes a window invalid: that would restore the mid-run
        halt the relaxed policy exists to avoid, and the plan forbids it.
        """

        flagged = bool(ARTIFACT_REASONS.intersection(aligned.reasons))
        bad = tuple(aligned.bad_channels)
        return (0.0 if flagged or bad else 1.0, flagged or bool(bad), bad)

    # --------------------------------------------------------------------- run

    def run(self, emit, stop) -> SessionSummary:
        """Consume the source until it ends or ``stop`` is set.

        Args:
            emit: Called with each :class:`SessionFrame`, on this thread. It must
                not raise for the run to continue; a raising consumer stops the
                session, which is what the transport's ``publish`` does when the
                session is closing.
            stop: A :class:`threading.Event`; set to end the run promptly.

        Returns:
            The :class:`SessionSummary`, including a failure code when the chain
            refused to continue rather than a silent early end.
        """

        processor = AuditoryProcessor(
            self.settings, source_channels=self.source_channels
        )
        # Kept reachable after the run: the chain's recovery decisions and repair
        # counter are part of the run record (see ``SessionSummary``), and a
        # handled fault that left no window behind would otherwise be invisible.
        self.processor = processor
        offload = TaskOffloader(
            self._analyze,
            workers=self.offload_workers,
            capacity=self.offload_capacity,
            overflow="drop_oldest",
            on_result=self._accept,
            on_error=self._note_handler_failure,
        )
        self._offload = offload
        self._drops_seen = 0
        began = self._clock()
        failure: SessionFailed | None = None
        next_frame = float(self.source.start)
        self._now = float(self.source.start)

        try:
            for end_time, samples, timestamps in self.source.chunks(stop):
                self._now = float(end_time)
                try:
                    windows = processor.feed((samples, timestamps))
                except RuntimeError as error:
                    failure = SessionFailed(self._now, "processing_failed", str(error))
                    break
                for window in windows:
                    self._dispatch(window)
                self._collect_drops()
                while end_time - next_frame >= self.policy.frame_seconds - 1e-9:
                    emit(self._frame(next_frame))
                    self.frames += 1
                    next_frame += self.policy.frame_seconds
        finally:
            try:
                offload.close(drain=True, timeout=OFFLOAD_STOP_TIMEOUT)
            except RuntimeError as error:
                failure = failure or SessionFailed(self._now, "offload_stuck", str(error))
            self._offload = None

        self._failure = failure
        # One last frame always: a consumer must learn the final state, including
        # a failure, rather than being left with the previous confident one.
        emit(self._frame(self._now))
        self.frames += 1

        return SessionSummary(
            kind=self.kind,
            simulated=self.simulated,
            policy=self.policy.to_dict(),
            channel_count=len(self.settings.eeg_channels),
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
            started=float(self.source.start),
            ended=float(self._now),
            wall_seconds=self._clock() - began,
            windows=self.windows,
            submitted=self.submitted,
            scored=self.scored,
            invalid=self.invalid,
            evidence_gaps=self.evidence_gaps,
            failed=self.failed,
            frames=self.frames,
            decisions=dict(self._census),
            decision_stream=list(self.decisions),
            bad_channel_census=dict(self.bad_channel_census),
            recovery_events=[
                dict(event) for event in getattr(processor.recovery, "events", [])
            ],
            recovery_segment=int(getattr(processor.recovery, "segment", 0)),
            repaired_samples=int(getattr(processor.repair, "repaired_samples", 0)),
            # Read after the offloader drained, so a failure the worker raised on
            # its way out is in the record too.
            scoring_failures=[dict(entry) for entry in self.scoring_failures],
            failure=(
                {"code": failure.code, "timestamp": failure.timestamp, "detail": failure.detail}
                if failure is not None
                else None
            ),
        )

    # ------------------------------------------------------------- evidence path

    def _dispatch(self, window) -> None:
        """Align one window, then submit it or record its absence explicitly."""

        aligned = self.references.align(window, float(self.source.audio_start))
        self.windows += 1
        # Set before any early return: the display the last window produced is
        # what the frame publishes, and a window the chain rejected still has one.
        self.display = getattr(window, "display", None) or {}
        self.quality, self.artifact, self.bad_channels = self._verdict(aligned)
        for name in self.bad_channels:
            self.bad_channel_census[name] = self.bad_channel_census.get(name, 0) + 1
        if not aligned.valid:
            self.invalid += 1
            # The chain rejected this window: it must not cost the worker time,
            # and the controller has to see that no evidence arrived here.
            self._apply(
                AttentionEstimate(
                    None,
                    self.evidence_end_for(aligned),
                    float(aligned.available_at),
                    False,
                    aligned.reasons,
                )
            )
            return

        item = (self._next_item, aligned)
        self._next_item += 1
        entry = (
            item[0],
            self.evidence_end_for(aligned),
            float(aligned.available_at),
        )
        self._inflight.append(entry)
        self.submitted += 1
        if not offload_accepts(self._offload, item):
            self._inflight.pop()
            self._gap(entry)
        self._collect_drops()

    def _analyze(self, item):
        """Score one aligned window on the worker thread. Labels never reach here."""

        item_id, aligned = item
        estimate = AttentionEstimate(
            self.decoder.score(aligned),
            self.evidence_end_for(aligned),
            float(aligned.available_at),
            True,
            aligned.reasons,
        )
        return item_id, estimate

    def _accept(self, result) -> None:
        """Apply one scored estimate, which is the only path to a decision."""

        item_id, estimate = result
        if self._inflight and self._inflight[0][0] == item_id:
            self._inflight.popleft()
        else:  # pragma: no cover - defensive: a gap already accounted for it
            self._inflight = deque(
                entry for entry in self._inflight if entry[0] != item_id
            )
        self.scored += 1
        self._apply(estimate)

    def _note_handler_failure(self, error: BaseException) -> None:
        """Count a scoring failure, and carry the cause instead of discarding it.

        The exception text is the only thing that says *why* no evidence arrived;
        a count alone hides the next mechanism failure exactly as it hid the
        decoder's contract refusal on the operator's ANT session (measured: 116
        failures, one unnamed cause, ``scored 0``). The text travels both into the
        published ``scoring_failed`` reason and into
        :attr:`SessionSummary.scoring_failures`, so it survives past the frame
        that carried it.
        """

        reason = _failure_reason(error)
        self.failed += 1
        for entry in self.scoring_failures:
            if entry["reason"] == reason:
                entry["count"] += 1
                break
        else:
            self.scoring_failures.append(
                {"type": type(error).__name__, "reason": reason, "count": 1}
            )
        self._apply(
            AttentionEstimate(
                None, self._now, self._now, False, ("scoring_failed", reason)
            )
        )

    def _gap(self, entry) -> None:
        """Turn a dropped window into explicit missing evidence."""

        self.evidence_gaps += 1
        _, evidence_end, emitted_at = entry
        self._apply(
            AttentionEstimate(None, evidence_end, emitted_at, False, ("evidence_gap",))
        )

    def _collect_drops(self) -> None:
        """Account for every item the offloader dropped, oldest first."""

        offload = self._offload
        if offload is None:
            return
        while offload.dropped > self._drops_seen:
            self._drops_seen += 1
            if not self._inflight:  # pragma: no cover - defensive
                return
            self._gap(self._inflight.popleft())

    def _apply(self, estimate: AttentionEstimate) -> None:
        """Advance the controller and the reported state with one estimate."""

        with self._lock:
            self.controller.update(estimate, self._now)
            self.evidence_count += 1
            scores = estimate.scores
            if scores is not None and np.all(np.isfinite(scores)) and scores.shape == (2,):
                self.scores = (float(scores[0]), float(scores[1]))
            self.reasons = tuple(estimate.reasons)
            # What the controller answers *now*, for this estimate's instant. It
            # is the decision the very next frame publishes, so a report can count
            # one decision per window without re-running the controller - the
            # arithmetic that produced it is `AttentionController.update`, and a
            # second implementation of it in a report would be a second truth.
            choice = self.controller.choice(self._now)
            if choice is not None:
                self.decisions.append(
                    (
                        float(estimate.evidence_end) - float(self.source.start),
                        "A" if int(choice) == 0 else "B",
                    )
                )
            # Evidence time never moves backwards. A failure report carries the
            # current instant, which can be later than the evidence it replaces;
            # a stale window arriving from the queue must not rewind the frame's
            # window_end either, because a consumer orders frames by it.
            if math.isfinite(estimate.evidence_end) and (
                self.evidence_end is None or estimate.evidence_end > self.evidence_end
            ):
                self.evidence_end = float(estimate.evidence_end)

    # ------------------------------------------------------------------- frames

    def _frame(self, source_time: float) -> SessionFrame:
        """Build the published state at one source-clock instant."""

        timestamp = float(source_time) - float(self.source.start)
        with self._lock:
            warm = timestamp < self.policy.warmup_seconds
            if self._failure is not None:
                decision, reasons = "unavailable", (self._failure.code,)
            elif warm:
                decision, reasons = "unavailable", ("warmup",)
            else:
                choice = self.controller.choice(source_time)
                if choice is not None:
                    decision = "A" if int(choice) == 0 else "B"
                    reasons = self.reasons
                elif self.evidence_count == 0:
                    decision, reasons = "unavailable", ("no_evidence",)
                elif (
                    self.evidence_end is not None
                    and source_time - self.evidence_end > self.controller.max_age
                ):
                    decision, reasons = "unavailable", ("evidence_stale",)
                else:
                    decision, reasons = "uncertain", (self.reasons or ("no_decision",))
            neutral = self._failure is not None or warm
            gains = (
                np.ones(2)
                if neutral
                else np.asarray(self.controller.gains(source_time), dtype=float)
            )
            frame = SessionFrame(
                timestamp=timestamp,
                decision=decision,
                correlation_a=None if self.scores is None else self.scores[0],
                correlation_b=None if self.scores is None else self.scores[1],
                gain_a_db=attenuation_db(float(gains[0])),
                gain_b_db=attenuation_db(float(gains[1])),
                quality=self.quality,
                artifact=self.artifact,
                bad_channels=tuple(self.bad_channels),
                reasons=tuple(reasons),
                window_end=(
                    None
                    if self.evidence_end is None
                    else self.evidence_end - float(self.source.start)
                ),
                evidence_count=self.evidence_count,
                media_time_s=self.audio_position(source_time),
                display=dict(self.display),
            )
        self._census[decision] = self._census.get(decision, 0) + 1
        return frame


def offload_accepts(offload: TaskOffloader | None, item) -> bool:
    """Submit one item, treating a closed offloader as a refusal rather than an error."""

    if offload is None:
        return False
    return bool(offload.submit(item))


def attenuation_db(gain: float) -> float:
    """Convert a linear gain to dB, never above 0.

    The demo attenuates and never amplifies (plan section 0), so the ceiling is
    part of the conversion rather than a promise a caller has to keep. Six
    decimals is far below any audible or reportable difference and keeps the
    published value readable: an unrounded 6 dB duck renders as
    ``-6.000000000000001 dB`` in the frontend's gain display.
    """

    if not math.isfinite(gain):
        return 0.0
    limited = min(1.0, max(gain, 1e-6))
    decibels = min(0.0, 20.0 * math.log10(limited))
    return 0.0 if decibels == 0.0 else round(decibels, 6)


__all__ = [
    "ARTIFACT_REASONS",
    "DECISIONS",
    "AttentionSession",
    "RunPolicy",
    "SessionFailed",
    "SessionFrame",
    "SessionSummary",
    "attenuation_db",
]
