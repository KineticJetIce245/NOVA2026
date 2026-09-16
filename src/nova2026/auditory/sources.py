"""Acquisition sources and the verified candidate envelopes a session scores against.

A session gets its EEG from an *acquisition source* and its two reference
envelopes from this module. Both are deliberately narrow:

* The source owns its own clock. Chunks carry the trial's or the amplifier's own
  timestamps, and nothing here invents one (plan section 3.17 item 4: alignment
  dominates channel count, so a fabricated timestamp is the most expensive kind
  of convenience).
* The envelopes are loaded from ``datasets/audio/`` and verified before a session
  may start. Section 3.1 rule 6 of the plan: a session refuses to start when the
  envelope it needs is absent, and never computes one at run time. Verification
  re-hashes the source audio through :func:`~nova2026.auditory.envelopes.load_envelope`,
  so a stimulus that changed under a stale envelope is a start-up failure.

Alignment maps a window's *source* timestamps to media seconds through one
measured anchor - where audio sample zero sits on the EEG clock - and then
interpolates the recorded envelope grid. No resampling, no phase guess, and no
envelope is ever re-derived here. A window that reaches outside the recorded
envelope is marked ``audio_unavailable`` and rejected rather than scored on
extrapolated values.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .config import AuditoryConfig
from .data import AuditoryTrial, AuditoryWindow
from .envelopes import EnvelopeRecord, EnvelopeVerificationError, load_envelope

ENVELOPE_SUFFIX = ".npz"
"""Envelopes are stored one per audio file under ``datasets/audio/``."""

AUDIO_UNAVAILABLE = "audio_unavailable"
"""Rejection reason for a window the recorded envelope cannot cover."""


class AcquisitionSource(Protocol):
    """What a session requires of whatever feeds its EEG chain.

    Implemented today by :class:`ReplaySource`; the live path (plan step 11)
    implements the same surface around an eego/LSL stream, which is the whole
    reason this is a protocol rather than a base class.

    Attributes:
        sample_rate: Source sampling rate in Hz.
        channel_names: Channel labels in the order the chunks carry them.
        reference: Recording reference, carried into the chain contract verbatim.
        upstream_processing: Processing already applied upstream, same reason.
        start: Source time of the first sample, in seconds.
        end: Source time after the last sample, in seconds.
        audio_start: Source time at which audio sample zero is played.
        simulated: Whether the data is a fixture rather than a measurement.
        kind: Short source name, recorded in the run policy and the packets.
    """

    sample_rate: float
    channel_names: tuple[str, ...]
    reference: object
    upstream_processing: object
    start: float
    end: float
    audio_start: float
    simulated: bool
    kind: str

    def chunks(self, stop) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """Yield ``(source_time, samples, timestamps)`` until the source ends.

        ``source_time`` is the source clock after the chunk, the samples are
        ``(n, channels)``, and the timestamps are the source times of those rows.
        The iterator must observe ``stop`` within roughly one chunk.
        """

        ...


@dataclass(frozen=True)
class Candidate:
    """One scored candidate: its identity plus the envelope it is scored against."""

    id: str
    label: str
    envelope: EnvelopeRecord

    @property
    def name(self) -> str:
        """Bare file name of the audio this envelope was derived from."""

        return Path(self.envelope.source_path).name


def trial_candidate_names(trial: AuditoryTrial) -> tuple[str, str]:
    """The two stimulus names a converted trial records, in candidate order.

    ``AuditoryTrial.group`` holds ``"|".join(sorted(dry names))`` from the
    conversion, which is also the order the labels index into, so the candidate
    order of a replay is the order the decoder was trained with.

    Raises:
        EnvelopeVerificationError: If the trial does not name exactly two
            candidates, because a session with the wrong number of references
            must not start.
    """

    names = tuple(name for name in str(trial.group).split("|") if name)
    if len(names) != 2:
        raise EnvelopeVerificationError(
            f"a session needs exactly two candidate envelopes; {trial.group!r} names {len(names)}."
        )
    return names  # type: ignore[return-value]


def trial_envelope_paths(
    trial: AuditoryTrial, directory: str | Path
) -> tuple[Path, Path]:
    """Where the two candidate envelopes of ``trial`` live.

    Existence is deliberately not checked here: :meth:`ReferenceEnvelopes.load`
    names the missing file, which is what the start-up refusal has to report.
    """

    return tuple(  # type: ignore[return-value]
        Path(directory) / (Path(name).stem + ENVELOPE_SUFFIX)
        for name in trial_candidate_names(trial)
    )


class ReferenceEnvelopes:
    """Two verified candidate envelopes, aligned to a window by the audio clock.

    Args:
        records: The two :class:`EnvelopeRecord` values, candidate A first.
        labels: Display labels for the candidates; defaults to their file names.

    Raises:
        EnvelopeVerificationError: If there are not exactly two records, if the
            two envelopes were derived at different feature rates or bands, or if
            their recorded configuration differs from the other's.
    """

    def __init__(
        self,
        records: tuple[EnvelopeRecord, EnvelopeRecord],
        labels: tuple[str, str] | None = None,
    ) -> None:
        records = tuple(records)
        if len(records) != 2:
            raise EnvelopeVerificationError(
                f"a session needs exactly two candidate envelopes; got {len(records)}."
            )
        first, second = records[0].metadata, records[1].metadata
        for field in ("sample_rate", "band", "envelope_method", "generator_version"):
            if first.get(field) != second.get(field):
                raise EnvelopeVerificationError(
                    f"candidate envelopes disagree on {field}: "
                    f"{first.get(field)!r} vs {second.get(field)!r}."
                )
        self.records = (records[0], records[1])
        if labels is None:
            labels = (Path(first["source_path"]).name, Path(second["source_path"]).name)
        self.candidates = tuple(
            Candidate(identity, str(label), record)
            for identity, label, record in zip(("A", "B"), labels, self.records)
        )
        self.sample_rate = float(first["sample_rate"])

    @classmethod
    def load(
        cls,
        paths: tuple[str | Path, str | Path],
        config: AuditoryConfig,
        labels: tuple[str, str] | None = None,
    ) -> "ReferenceEnvelopes":
        """Load and verify both envelopes from disk.

        Args:
            paths: The two ``.npz`` envelope files, candidate A first.
            config: The session's feature configuration; a stored envelope whose
                recorded parameters differ is refused rather than resampled.
            labels: Optional display labels for the two candidates.

        Returns:
            A verified :class:`ReferenceEnvelopes`.

        Raises:
            EnvelopeVerificationError: If a file is missing, unreadable, or does
                not match this session or the audio it was derived from. The
                message names the file, so a demo that cannot start says why.
        """

        paths = tuple(paths)
        if len(paths) != 2:
            raise EnvelopeVerificationError(
                f"a session needs exactly two candidate envelopes; got {len(paths)}."
            )
        records = tuple(load_envelope(path, config) for path in paths)
        return cls(records, labels)  # type: ignore[arg-type]

    def align(self, window, audio_start: float) -> AuditoryWindow:
        """Attach both reference envelopes to one processed window.

        Args:
            window: The chain's :class:`~nova2026.streaming.window.EEGWindow`,
                whose timestamps are source-clock seconds.
            audio_start: Source time at which audio sample zero is played. This
                is the only anchor between the two clocks; it is a measurement
                (or, in replay, a construction) and never a guess.

        Returns:
            An :class:`~nova2026.auditory.data.AuditoryWindow` with one envelope
            column per candidate. ``valid`` is false when the window's own chain
            verdict was false or when the recorded envelope cannot cover it, and
            ``reasons`` names which.

        Raises:
            ValueError: If the window has no timestamps or the anchor is not
                finite.
        """

        if not len(window.timestamps):
            raise ValueError("Alignment requires a nonempty window.")
        if not math.isfinite(audio_start):
            raise ValueError("The audio anchor must be finite source-clock seconds.")

        media = np.asarray(window.timestamps, dtype=float) - float(audio_start)
        reasons = list(window.reasons)
        envelopes = np.zeros((len(media), 2))
        covered = True
        for index, record in enumerate(self.records):
            times = np.asarray(record.timestamps, dtype=float)
            values = np.asarray(record.envelope, dtype=float).reshape(-1)
            if len(times) < 2 or media[0] < times[0] - 1e-9 or media[-1] > times[-1] + 1e-9:
                covered = False
                continue
            envelopes[:, index] = np.interp(media, times, values)
        if not covered:
            reasons.append(AUDIO_UNAVAILABLE)
        available = window.available_at
        if available is None:
            available = float(window.timestamps[-1])
        # The reference for a window cannot have been available before the audio
        # it describes was played; the later of the two is the honest bound.
        available = max(float(available), float(audio_start + media[-1]))
        return AuditoryWindow(
            window.data,
            envelopes,
            window.timestamps,
            available,
            window.valid and not reasons,
            reasons,
            window.contract,
            window.segment,
            # The chain's channel census travels with the window: it is evidence
            # the quality policy does not turn into a reason when
            # ``check_channels`` is false, and the session's ``signal_quality``
            # verdict has nowhere else to learn about a dead electrode.
            getattr(window, "bad_channels", ()),
        )

    def coverage(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Media-second range each candidate envelope covers, for the run record."""

        return tuple(  # type: ignore[return-value]
            (
                float(np.asarray(record.timestamps, dtype=float)[0]),
                float(np.asarray(record.timestamps, dtype=float)[-1]),
            )
            for record in self.records
        )


class ReplaySource:
    """Feed one recorded trial to the chain in small chunks, on a virtual clock.

    This mirrors ``scripts/auditory/runner.py::replay_windows`` chunk for chunk:
    the same 32 ms default chunk, the same ``searchsorted`` boundary, and the
    audio side clipped to the EEG length. It differs in what it does *not* do -
    it computes no envelope, so the reference comes from the verified records in
    :class:`ReferenceEnvelopes` instead of from the trial's own audio.

    ``speed`` divides wall-clock time: ``1.0`` is real time and ``50.0`` replays a
    trial fifty times faster, which is how a test reaches a decision without
    waiting minutes for one. ``sleep`` and ``timer`` are injectable so the pacing
    rule is testable without a stopwatch.

    Args:
        trial: A converted recording; its labels are never read by this class.
        chunk_seconds: Source seconds fed per chunk.
        speed: Replay speed multiplier; must be finite and positive.
        audio_offset: Source-time position of audio sample zero.
        simulated: Marks a fixture rather than a measurement, so the packets and
            the run record can say so (plan section 3.6).
        kind: Short source name for the run record and the packets.

    Raises:
        ValueError: If an argument is not finite and positive, or if the trial's
            audio does not reach past the EEG's first sample.
    """

    def __init__(
        self,
        trial: AuditoryTrial,
        *,
        chunk_seconds: float = 0.032,
        speed: float = 1.0,
        audio_offset: float = 0.0,
        simulated: bool = False,
        kind: str | None = None,
        sleep=time.sleep,
        timer=time.perf_counter,
    ) -> None:
        for name, value in (("chunk_seconds", chunk_seconds), ("speed", speed)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(audio_offset):
            raise ValueError("audio_offset must be finite.")
        self.trial = trial
        self.sample_rate = float(trial.sample_rate)
        self.channel_names = tuple(trial.channel_names)
        self.reference = trial.reference
        self.upstream_processing = trial.upstream_processing
        self.start = float(trial.timestamps[0])
        self.end = float(trial.timestamps[-1])
        self.audio_start = self.start + float(audio_offset)
        self.chunk_seconds = float(chunk_seconds)
        self.speed = float(speed)
        self.simulated = bool(simulated)
        self.kind = kind or ("synthetic_replay" if simulated else "replay")
        self._sleep = sleep
        self._timer = timer

    @property
    def duration(self) -> float:
        """Source seconds this source will feed."""

        return self.end - self.start

    def chunks(self, stop) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """Yield chunks of the trial until it ends or ``stop`` is set.

        Pacing happens after each chunk, so the consumer's own work counts
        against the replay budget exactly as it would in real time.
        """

        position = 0
        now = self.start
        began = self._timer()
        while now < self.end and not stop.is_set():
            nxt = min(now + self.chunk_seconds, self.end)
            end_index = int(
                np.searchsorted(self.trial.timestamps, nxt, side="right")
            )
            if end_index > position:
                yield (
                    nxt,
                    self.trial.eeg[position:end_index],
                    self.trial.timestamps[position:end_index],
                )
                position = end_index
            now = nxt
            delay = began + (now - self.start) / self.speed - self._timer()
            if delay > 0:
                self._sleep(delay)


__all__ = [
    "AUDIO_UNAVAILABLE",
    "ENVELOPE_SUFFIX",
    "AcquisitionSource",
    "Candidate",
    "ReferenceEnvelopes",
    "ReplaySource",
    "trial_candidate_names",
    "trial_envelope_paths",
]
