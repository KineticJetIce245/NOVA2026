"""Drive the auditory live path from an LSL outlet carrying recorded ANT EEG.

:class:`~nova2026.auditory.sources.AcquisitionSource` is the seam the live route
was left open at: ``ReplaySource`` implements it from a converted trial file,
and this module implements it around a real LSL inlet, so
:class:`~nova2026.auditory.session.AttentionSession` runs unchanged against
either. Everything below the seam - pre-flight, ``Acquire``, the time base, the
channel contract, the chain - is the same code the amplifier path uses; the only
thing missing compared with the rig is the amplifier.

The seam has one hard requirement: the chain's timestamps must be small numbers
on the recording's own axis, because the reference envelopes and the audio
anchor live there. An LSL stamp is a large absolute clock reading, so the
adapter **rebases**: the first placed sample is time zero and every later sample
keeps its own offset from it. The raw stamps are not thrown away - they are what
``Acquire.max_lag``/``gaps`` and the grid diagnostics are measured on, and they
are the only honest evidence that a transport, rather than a file, produced
these samples.

    python -B -m scripts.getlive.ant_live --session datasets/AAD-ANT/session_19-34-06.npz

Nothing here reads the recording's labels: they are carried by the publisher for
scoring only (plan rule 3, labels never enter the decode path).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from mne_lsl.lsl import local_clock

from nova2026.streaming.acquire import Acquire
from nova2026.streaming.preflight import ChannelContract, validate_source
from nova2026.streaming.timebase import GridPolicy, TimeBase

# How the run treats the inlet's own stamps. "grid" places every received sample
# on a regular grid at the declared rate from a counted index and reports how far
# the source's anchors drifted from it; "stamps" passes them through. Both are
# the live hardware script's own modes (``scripts/getlive/live.py``), kept here
# for the same reason: a rig whose stamps are untrustworthy must be runnable
# without editing the consumer.
TIME_BASE_MODES = ("grid", "stamps")

DEFAULT_MICROVOLT_EXPONENT = -6
"""The chain's own unit convention: microvolts (plan section 3.11 item 6)."""


@dataclass
class TransportDiagnostics:
    """What the LSL transport itself did, as opposed to what the chain decided.

    Every number here is measured on the stamps the inlet delivered, before the
    adapter rebased them. They are the acquisition diagnostics the plan asks the
    live path to expose (``max_lag``, ``gaps``, recovery events, the bad-channel
    census) plus the two grid numbers that say whether the transport stayed on
    rate.
    """

    first_stamp: float | None = None
    last_stamp: float | None = None
    blocks: int = 0
    samples: int = 0
    max_lag_seconds: float = 0.0
    gaps: int = 0
    max_gap_seconds: float = 0.0
    pending_samples_at_end: int = 0
    discarded_backlog_samples: int = 0
    ended: str = "not started"
    raw_rate_hz: float | None = None
    grid_deviation_peak_seconds: float = 0.0
    timebase_relocks: int = 0
    timebase_relocked_samples: float = 0.0
    timebase_large_steps: int = 0
    timebase_anchor_rate_hz: float | None = None
    recovery_events: list = field(default_factory=list)
    pre_resampled_to_hz: float | None = None
    pre_resampled_samples: int = 0
    # What the emitted count should be, and the shortfall against it. A
    # conversion that is sample-exact leaves the deficit at zero; a non-zero
    # value means samples the chain never saw, so it is recorded rather than
    # left for a reader to recompute from `samples` and the two rates.
    pre_resampled_expected_samples: int | None = None
    pre_resampled_deficit_samples: int | None = None

    def to_dict(self) -> dict:
        """A JSON-safe copy, shaped for the run record."""

        return {
            "first_stamp": self.first_stamp,
            "last_stamp": self.last_stamp,
            "transport_seconds": (
                None
                if self.first_stamp is None or self.last_stamp is None
                else round(float(self.last_stamp - self.first_stamp), 6)
            ),
            "blocks": self.blocks,
            "samples": self.samples,
            "max_lag_seconds": round(float(self.max_lag_seconds), 6),
            "gaps": int(self.gaps),
            "max_gap_seconds": round(float(self.max_gap_seconds), 6),
            "pending_samples_at_end": int(self.pending_samples_at_end),
            "discarded_backlog_samples": int(self.discarded_backlog_samples),
            "ended": self.ended,
            "raw_rate_hz": self.raw_rate_hz,
            "grid_deviation_peak_seconds": round(
                float(self.grid_deviation_peak_seconds), 9
            ),
            "timebase_relocks": int(self.timebase_relocks),
            "timebase_relocked_samples": round(float(self.timebase_relocked_samples), 6),
            "timebase_large_steps": int(self.timebase_large_steps),
            "timebase_anchor_rate_hz": self.timebase_anchor_rate_hz,
            "pre_resampled_to_hz": self.pre_resampled_to_hz,
            "pre_resampled_samples": int(self.pre_resampled_samples),
            "pre_resampled_expected_samples": self.pre_resampled_expected_samples,
            "pre_resampled_deficit_samples": self.pre_resampled_deficit_samples,
            "recovery_events": [dict(event) for event in self.recovery_events],
        }


def grid_deviation_seconds(
    placed: np.ndarray, ideal: np.ndarray
) -> float:
    """Largest disagreement between the placed grid and an ideal one, in seconds.

    This is not a decoder-relevant number on its own - the chain's own timeline
    is the grid - but it is the transport's verdict: a loopback publisher that
    cannot keep a 500 Hz grid shows up here, and a run at, say, 50 ms of peak
    deviation is not a run whose timing can be trusted.
    """

    if placed.shape != ideal.shape:
        raise ValueError("placed and ideal must have the same shape.")
    if placed.size == 0:
        return 0.0
    return float(np.max(np.abs(placed - ideal)))


class AntStreamSource:
    """An :class:`AcquisitionSource` fed by a live LSL inlet.

    Construction is also the start-up check: the inlet's metadata is validated
    against the requested contract by the package's own pre-flight, so a wrong
    rate, a wrong declared unit or a missing electrode is refused before a sample
    is used - exactly as it would be on the rig.

    Args:
        stream: A connected ``mne_lsl.stream.StreamLSL`` opened with manual
            acquisition, whose labels and declared units are read back.
        channels: The expected channel labels, in chain order.
        sfreq: The rate the run asserts, in Hz.
        reference: Reference the recording already carries, recorded as
            provenance (the ANT recording is CPz-referenced, the model was
            trained on Cz-referenced data - a difference this adapter reports
            rather than hides).
        upstream_processing: Provenance string.
        kind: Short source name for the run record and the packets.
        source_unit_exponent: Power of ten of the unit the inlet carries.
        block_samples: Samples per acquired block.
        timebase: ``"grid"`` or ``"stamps"``.
        pre_resample_sfreq: Rate to resample the incoming blocks to **before**
            the chain, or ``None``. The decoded model's contract records the rate
            its training trials were at (128 Hz for the KU Leuven models), and
            ``RidgeDecoder.validate`` compares the chain's contract with the
            model's verbatim - so a 500 Hz recording has to reach the chain at
            the model's own input rate. This is an explicit, reported conversion
            in front of the chain, never a change to the contract: the recording
            is untouched and the effective value is printed and recorded.
        max_lag_seconds: Age limit for a delivered block; ``None`` disables it.
        no_data_timeout: Seconds without progress before the run ends.

    Raises:
        RuntimeError: If the source fails pre-flight.
        ValueError: On an invalid argument.
    """

    def __init__(
        self,
        stream,
        *,
        channels: tuple[str, ...],
        sfreq: float,
        reference: str = "not asserted",
        upstream_processing: str = "not asserted",
        kind: str = "ant_lsl_replay",
        source_unit_exponent: int = DEFAULT_MICROVOLT_EXPONENT,
        block_samples: int = 25,
        timebase: str = "grid",
        pre_resample_sfreq: float | None = None,
        max_lag_seconds: float | None = 3.0,
        no_data_timeout: float = 5.0,
    ) -> None:
        if timebase not in TIME_BASE_MODES:
            raise ValueError(f"timebase must be one of {TIME_BASE_MODES}.")
        if block_samples < 1:
            raise ValueError("block_samples must be positive.")
        channels = tuple(str(name) for name in channels)
        if not channels:
            raise ValueError("channels must not be empty.")

        # The package's own pre-flight: identity, rate, units, labels, types and
        # the untouched-inlet rule. Anything it refuses is refused here too.
        validate_source(
            stream,
            sfreq=sfreq,
            channels=channels,
            source_unit_exponent=source_unit_exponent,
        )
        self.contract: ChannelContract = ChannelContract(tuple(stream.ch_names), channels)
        self.stream = stream
        self.sample_rate = float(sfreq)
        # The source's own rate, pinned. ``sample_rate`` is deliberately
        # reassigned after construction by callers whose session reads its
        # settings off the source (``scripts/auditory_ui/live.py``: the session
        # is told the rate the adapter *delivers*), so arithmetic that means
        # "what the transport delivered" must not read it back.
        self._source_rate = float(sfreq)
        self.channel_names = channels
        self.reference = str(reference)
        self.upstream_processing = str(upstream_processing)
        self.kind = str(kind)
        self.simulated = False
        self.source_unit_exponent = int(source_unit_exponent)
        self.timebase_mode = str(timebase)

        self.policy = GridPolicy.for_rate(self.sample_rate)
        self._timebase = TimeBase(self.policy) if timebase == "grid" else None
        # Filled in when consumption actually begins; see ``chunks``.
        self.discarded_backlog = 0
        self.acquire = Acquire(
            stream,
            int(block_samples),
            sfreq=self.sample_rate,
            max_lag_seconds=max_lag_seconds,
            no_data_timeout=no_data_timeout,
        )
        # An inlet that attaches behind a publisher which is already streaming
        # finds its buffer full of samples from before the connection - several
        # seconds old, and refused by ``Acquire``'s age guard as if the source
        # had stalled. Connecting immediately after the publisher starts
        # (``lead_seconds`` in ``ant_publish``) keeps that backlog short, but the
        # lag is a measurement, not a thing to hide: it stays in
        # ``diagnostics.max_lag_seconds``.
        self.diagnostics = TransportDiagnostics()

        # Optional pre-chain rate conversion. It exists because a model's
        # contract names the rate its training trials were at, and the live path
        # must arrive at that rate; the conversion is reported (not silent), and
        # nothing about the contract or the recording is changed by it.
        self.pre_resample_sfreq = (
            None if pre_resample_sfreq is None else float(pre_resample_sfreq)
        )
        if self.pre_resample_sfreq is not None and (
            not float(self.pre_resample_sfreq) > 0
            or math.isclose(self.pre_resample_sfreq, self.sample_rate)
        ):
            raise ValueError(
                "pre_resample_sfreq must be positive and different from the "
                "source rate; leave it unset to pass the source rate through."
            )
        self._converter = None
        self._emitted = 0
        if self.pre_resample_sfreq is not None:
            from nova2026.streaming.preprocess import Resampler

            self._converter = Resampler(
                self.sample_rate,
                self.pre_resample_sfreq,
                len(channels),
                quality="HQ",
            )

        # Filled in once the first block has been placed: the chain's time axis
        # is the recording's own, so it starts at zero, not at an LSL clock
        # reading (an epoch-sized float would cost the alignment its precision).
        self.start = 0.0
        self.end = 0.0
        self.audio_start = 0.0
        self._origin: float | None = None
        self._ideal_first: float | None = None
        self._previous_block_end: float | None = None
        self._rate_span: tuple[float, float] | None = None
        self._closed = False

    def _discard_initial_backlog(self, grace_seconds: float = 0.25,
                                 budget_seconds: float = 10.0) -> int:
        """Discard inlet samples that predate this attachment; return how many.

        Reads the inlet's own buffer until its newest sample is younger than
        ``grace_seconds``, or until nothing more is waiting, or until the budget
        runs out. A source that has published nothing has no backlog and this
        returns 0 immediately -- it does not wait for data to arrive, because an
        amplifier that is not publishing is a different failure with its own
        message (``no_data_timeout``), and hiding it behind a drain would be worse
        than useless.
        """

        if not self.stream.connected:
            return 0
        # Only a real LSL inlet has a buffer to drop. A source that does not expose
        # the acquisition API -- a test double, or any deterministic stand-in --
        # has no backlog, and reaching for methods it does not have would be a
        # requirement invented by this drain rather than by the source.
        for needed in ("acquire", "get_data", "n_new_samples"):
            if not hasattr(self.stream, needed):
                return 0
        discarded = 0
        deadline = time.monotonic() + budget_seconds
        while time.monotonic() < deadline:
            self.stream.acquire()
            if not self.stream.n_new_samples:
                break
            data, stamps = self.stream.get_data(winsize=None, exclude=())
            if stamps is None or len(stamps) == 0:
                break
            discarded += int(len(stamps))
            if local_clock() - float(stamps[-1]) <= grace_seconds:
                break
        return discarded

    # ------------------------------------------------------------------ surface

    def chunks(self, stop) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """Yield ``(source_time, samples, timestamps)`` straight off the inlet.

        Ends when the outlet stops, when ``stop`` is set, or when the transport
        reports a fault the run cannot continue through. Every ending is named in
        :attr:`diagnostics` ``ended`` so a short run is never mistaken for a
        finished one.
        """

        diagnostics = self.diagnostics
        diagnostics.ended = "running"
        # Drop everything that arrived before consumption actually began.
        #
        # This is deliberately here and not at construction. On the rig the
        # amplifier is streaming before the demo is started -- the documented
        # procedure -- so the inlet's first read is a backlog; and the gap between
        # attaching and consuming is seconds long (the media is rendered, the
        # contract is checked, the URL is printed), during which the backlog grows
        # rather than drains. ``Acquire._take_block`` cuts from the FRONT of the
        # buffer, so that backlog makes every block handed back look old and the
        # age guard ends the run: measured, "Source samples are 3.467 seconds old"
        # with zero blocks acquired, after an earlier attempt had already dropped
        # 15 000 samples at construction time.
        #
        # Dropping it here is what makes "connect the EEG, then start the demo"
        # the ordinary case instead of a fault. What was dropped is reported
        # (``diagnostics.discarded_backlog_samples``), never silently swallowed.
        self.discarded_backlog += self._discard_initial_backlog()
        flush = getattr(self.acquire, "flush", None)
        if callable(flush):
            flush()
        diagnostics.discarded_backlog_samples = self.discarded_backlog
        try:
            while not stop.is_set():
                try:
                    data, stamps = self.acquire.read(timeout=0.5)
                except TimeoutError:
                    diagnostics.ended = "no-more-data"
                    return
                except RuntimeError as error:
                    diagnostics.ended = f"transport-error: {error}"
                    return
                placed = self._place(stamps)
                data = np.asarray(data, dtype=np.float64)
                # Contract order, by name: extra columns are dropped, never
                # truncated by position (plan case E4).
                data = self.contract.reorder(data)
                self._record(data, stamps, placed)
                if self._converter is not None:
                    data, placed = self._convert(data, placed)
                    if data.shape[0] == 0:
                        # A resampler holds samples back until it has enough to
                        # emit a whole output frame; an empty block is progress
                        # withheld, not a source that ended.
                        continue
                self.end = float(placed[-1] + 1.0 / self.sample_rate)
                yield self.end, data, placed
        finally:
            if diagnostics.ended == "running":
                diagnostics.ended = "stopped"
            # The converter holds output back until it has enough to emit a whole
            # frame (see `_convert`), so the samples still inside it at the end of
            # the source are *real samples*, not a rounding residue: they are the
            # preset's filter delay, which at 500 -> 128 Hz is about 7.4 s for HQ.
            # Asking for them here is what makes the emitted count match the rate
            # ratio; without it they were dropped when the stream was torn down.
            tail = self._drain()
            if tail is not None:
                yield tail
            self._close()

    def close(self) -> None:
        """Release the acquisition handle; the caller owns disconnecting."""

        self._close()

    @property
    def transport(self) -> TransportDiagnostics:
        """The transport's own counters, for the run record."""

        return self.diagnostics

    # ------------------------------------------------------------------ internals

    def _place(self, stamps: np.ndarray) -> np.ndarray:
        """Turn source stamps into the chain's rebased timeline."""

        stamps = np.asarray(stamps, dtype=np.float64)
        if stamps.size == 0:
            return stamps
        if self._timebase is not None:
            placed, _ = self._timebase.place(stamps)
            placed = np.asarray(placed, dtype=np.float64)
        else:
            placed = stamps
        if self._origin is None:
            # Time zero is the *published window's* first sample, and the grid is
            # anchored on that same sample, so sample n of the window sits at
            # exactly n / rate on the chain's axis - the recording's own axis,
            # rebased. That is what lets the reference envelopes and the audio
            # anchor be reused verbatim from the replay path.
            self._origin = float(placed[0])
            self._ideal_first = float(stamps[0])
        return placed - float(self._origin)

    def _convert(
        self, data: np.ndarray, placed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Put one block on the model's own input rate, and say so on the clock.

        The converted block keeps the invariant the rest of the path relies on -
        sample *n* of the stream sits at ``n / rate`` - so the reference
        envelopes and the audio anchor stay on the same axis, and the chain's own
        rate becomes the one the decoder's contract names.
        """

        rate = float(self.pre_resample_sfreq)
        data, _ = self._converter(data, placed)
        data = np.asarray(data, dtype=np.float64)
        count = int(data.shape[0])
        placed = (self._emitted + np.arange(count, dtype=np.float64)) / rate
        self._emitted += count
        self.diagnostics.pre_resampled_to_hz = rate
        self.diagnostics.pre_resampled_samples += count
        return data, placed

    def _drain(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        """Deliver the tail the converter is still holding, and report the count.

        Called once, at the end of the source. The samples come out on the same
        grid as the rest of the stream - sample *n* sits at ``n / rate`` - so the
        audio anchor and the reference envelopes stay on one axis, and the run
        record states the accounting rather than leaving it to be inferred.
        """

        if self._converter is None:
            return None
        data, _ = self._converter.drain()
        data = np.asarray(data, dtype=np.float64)
        count = int(data.shape[0])
        rate = float(self.pre_resample_sfreq)
        placed = (self._emitted + np.arange(count, dtype=np.float64)) / rate
        self._emitted += count
        self.diagnostics.pre_resampled_samples += count
        # What the emitted count should be, given what the transport delivered:
        # the source's own sample count, mapped through the true rate ratio. A
        # non-zero residual is a fault in the path, not a rounding note - the
        # conversion is sample-exact, so the two must agree to the sample.
        expected = int(
            round(self.diagnostics.samples * rate / self._source_rate)
        )
        self.diagnostics.pre_resampled_expected_samples = expected
        self.diagnostics.pre_resampled_deficit_samples = expected - self._emitted
        if count == 0:
            return None
        self.end = float(placed[-1] + 1.0 / rate)
        return self.end, data, placed

    def _record(
        self, data: np.ndarray, stamps: np.ndarray, placed: np.ndarray
    ) -> None:
        """Copy one block's transport facts into the diagnostics."""

        diagnostics = self.diagnostics
        diagnostics.blocks += 1
        diagnostics.samples += int(data.shape[0])
        if diagnostics.first_stamp is None:
            diagnostics.first_stamp = float(stamps[0])
        diagnostics.last_stamp = float(stamps[-1])
        if self._rate_span is None:
            self._rate_span = (float(stamps[0]), 0.0)
            diagnostics.raw_rate_hz = None
        else:
            span = float(stamps[-1]) - self._rate_span[0]
            if span > 0:
                # Counted samples over the stamps' own span: the source's clock
                # as the transport delivered it, before the grid corrected it.
                diagnostics.raw_rate_hz = round(
                    (diagnostics.samples - 1) / span, 6
                )
        if self._ideal_first is not None:
            # Sample index of this block's first row: what came before it.
            index = self.diagnostics.samples - int(data.shape[0])
            ideal = self._ideal_first + (
                index + np.arange(placed.size, dtype=np.float64)
            ) / self.sample_rate
            deviation = grid_deviation_seconds(placed + float(self._origin), ideal)
            diagnostics.grid_deviation_peak_seconds = max(
                diagnostics.grid_deviation_peak_seconds, deviation
            )
        if self._timebase is not None:
            state = self._timebase.state
            diagnostics.timebase_relocks = len(state.relocks)
            diagnostics.timebase_relocked_samples = state.relocked_samples
            diagnostics.timebase_large_steps = len(state.large_steps)
            diagnostics.timebase_anchor_rate_hz = (
                None if not np.isfinite(state.anchor_rate) else round(state.anchor_rate, 6)
            )
        self._previous_block_end = float(placed[-1])

    def _close(self) -> None:
        """Close the acquisition handle once."""

        if self._closed:
            return
        self._closed = True
        self.diagnostics.pending_samples_at_end = int(self.acquire.pending_samples)
        self.diagnostics.max_lag_seconds = float(self.acquire.max_lag)
        self.diagnostics.gaps = int(self.acquire.gaps)
        self.diagnostics.max_gap_seconds = float(self.acquire.max_gap)
        self.acquire.close()

    def clock_check(self, expected_seconds: float = 0.0) -> dict:
        """Compare the transport's own clock with the wall clock since it started.

        Reports two independent things: how far the LSL stamps moved, and how
        much wall time the process actually spent. It is a loopback measurement
        of *this machine's* delivery, **not** the audio-to-EEG loopback the plan
        requires before a human study (``residual_offset_seconds``, section
        3.11 / 3.17-4) - no audio device is involved anywhere in this path.
        """

        stamps = self.diagnostics
        return {
            "stamp_seconds": (
                None
                if stamps.first_stamp is None or stamps.last_stamp is None
                else round(float(stamps.last_stamp - stamps.first_stamp), 6)
            ),
            "expected_seconds": expected_seconds,
            "note": (
                "transport-only measurement; NOT the audio-to-EEG loopback the "
                "plan requires before a human study"
            ),
        }


def sleep_until_ready(name: str, timeout: float = 15.0, poll: float = 0.1) -> None:
    """Block until an outlet with this name publishes, or raise.

    Thin wrapper over the package's resolver so a runner does not spin on a
    ``connect()`` that is waiting for a publisher which is not there yet.
    """

    from nova2026.streaming.preflight import resolve_outlet

    resolve_outlet(name=name, timeout=timeout, poll_interval=poll)


__all__ = [
    "DEFAULT_MICROVOLT_EXPONENT",
    "TIME_BASE_MODES",
    "AntStreamSource",
    "TransportDiagnostics",
    "grid_deviation_seconds",
    "sleep_until_ready",
]
