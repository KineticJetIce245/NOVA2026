"""One causal EEG chain for auditory training, replay, and live acquisition."""

import math
from types import SimpleNamespace

import numpy as np

from nova2026.streaming import Recovery, UnrepairableError
from nova2026.streaming.circular_buffer import CircularBuffer
from nova2026.streaming.judges import collect_verdict
from nova2026.streaming.preflight import ChannelContract
from nova2026.streaming.preprocess import (
    QualityMonitor, Repair, Resampler, SosFilter, design_bandpass,
)
from nova2026.streaming.window import EEGWindow

# The unit this chain's callers feed: the KU Leuven trials are microvolts, and
# the live path scales its source to microvolts before this chain. For
# microvolts, ``Repair``'s endpoint check (``10 ** (exponent + 6)``) is an
# identity on the values, which is the behaviour this chain has always had. A
# source whose own unit is volts -- the ANT amplifier -- is described by passing
# 0 at the call site instead.
DEFAULT_SOURCE_UNIT_EXPONENT = -6
SUPPORTED_SOURCE_UNIT_EXPONENTS = (0, -3, -6, -9)


class _PassThroughResampler:
    """A chain stage for the case ``input_sfreq == output_sfreq``.

    ``Resampler`` refuses equal rates (a resampler that resamples to the same
    rate is a bug), but a caller that has already converted the source - the live
    ANT route's pre-chain adapter, which brings a 500 Hz outlet to the 128 Hz the
    decoder contract records - feeds this chain at its output rate. The chain
    still needs a stage here: the recovery guard resets it, ``feed`` calls it,
    and ``AuditoryProcessor.contract`` reads its ``quality``. It carries no
    state, invents no sample, and its startup delay is exactly zero, which is
    true in a way a 1:1 ``Resampler`` could not be.
    """

    quality = "pass-through"

    def __init__(self) -> None:
        """Nothing to configure: the rate is already the output rate."""

        self.startup_delay_seconds = 0.0
        self.max_delay_seconds = 0.0

    def reset(self) -> None:
        """A stateless stage has nothing to forget."""

    def __call__(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the block unchanged."""

        return data, timestamps


DISPLAY_TARGET_POINTS = 80.0
"""Trace points a display window is decimated to.

About 80 points per trace for the 5 s window the saved models record, i.e. the
"roughly 16 Hz" the panel draws at. It is a count rather than a rate because the
decimation factor has to stay an integer: points and rate cannot both be exact,
and a trace whose length varies by a rounding rule is what makes a client's
``sample_rate`` field a lie.
"""

DISPLAY_UNIT_LABELS = {-6: "uV", -3: "mV", -9: "nV", 0: "V"}
"""Unit text for the source exponent the chain was configured with.

The exponent is the only thing this chain is told about its input unit
(``nova2026.streaming.preprocess.units.unit_scaler``'s convention), so the label
is derived from it rather than assumed. An exponent outside the supported set
gets a fallback naming the exponent instead of inventing a unit.
"""


class DisplayTap:
    """One display window: the newest raw trace beside the filtered window.

    Both traces are this chain's own signal, and they are carried separately on
    purpose:

    ``samples``
        The channel as it arrived, **before** the 1-9 Hz band-pass. The panel
        shows what the electrode saw, including the drift and line noise the
        band-pass removes -- a display that could only ever show the decoder's own
        input could not tell "no signal" from "signal the filter removed".
    ``filtered_samples``
        The same channel after the causal band-pass and resample, which is what
        the decoder actually consumes. Both are drawn on one panel, so the
        difference between them is visible rather than asserted.

    ``filtered_samples`` is the window's own array, which :class:`EEGWindow`
    already copies on construction and no later stage writes to. ``samples`` is
    not: it is captured inside the chain, before the band-pass, and copied there,
    because the array in flight is storage the resampler and the ring buffer hand
    out. See :meth:`AuditoryProcessor._capture_display`.

    ``original_scale_hint_uv`` and ``filtered_scale_hint_uv`` are per-window span
    hints -- the half-range that window happens to span -- not a calibrated
    full-scale value.
    """

    __slots__ = (
        "sample_rate", "filtered_sample_rate", "channels", "samples",
        "filtered_samples", "original_units", "filtered_units",
        "original_scale_hint_uv", "filtered_scale_hint_uv", "window_seconds",
        "lag_seconds", "channel_source",
    )

    def __init__(
        self, *, sample_rate, filtered_sample_rate, channels, samples,
        filtered_samples, original_units, filtered_units, original_scale_hint_uv,
        filtered_scale_hint_uv, window_seconds, lag_seconds, channel_source,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.filtered_sample_rate = float(filtered_sample_rate)
        self.channels = tuple(str(name) for name in channels)
        self.samples = samples
        self.filtered_samples = filtered_samples
        self.original_units = str(original_units)
        self.filtered_units = str(filtered_units)
        self.original_scale_hint_uv = float(original_scale_hint_uv)
        self.filtered_scale_hint_uv = float(filtered_scale_hint_uv)
        self.window_seconds = float(window_seconds)
        self.lag_seconds = float(lag_seconds)
        self.channel_source = str(channel_source)

    def to_payload(self) -> dict:
        """The frozen ``eeg_display`` field set, as plain JSON types."""

        return {
            "sample_rate": self.sample_rate,
            "channels": list(self.channels),
            "samples": [list(row) for row in self.samples],
            "filtered_samples": [list(row) for row in self.filtered_samples],
            "filtered_sample_rate": self.filtered_sample_rate,
            "original_units": self.original_units,
            "filtered_units": self.filtered_units,
            "original_scale_hint_uv": self.original_scale_hint_uv,
            "filtered_scale_hint_uv": self.filtered_scale_hint_uv,
            "window_seconds": self.window_seconds,
            "lag_seconds": self.lag_seconds,
            "channel_source": self.channel_source,
        }


def decimation_factor(sample_count: int, target_points: float = DISPLAY_TARGET_POINTS) -> int:
    """Largest integer decimation factor that divides ``sample_count`` exactly.

    Chosen from the *actual* window length, not a configured rate that may not be
    the rate being fed (the live ANT route feeds this chain at its output rate
    through :class:`_PassThroughResampler`). The factor divides the window
    exactly, so the decimated length is computed rather than discovered, and no
    zero is ever padded in to make a length fit: a padded sample is a sample the
    electrode never produced.

    Raises:
        ValueError: If ``sample_count`` is not a positive integer, or if no factor
            gets within a factor of two of the target. That second case means the
            window is far shorter than the target asks for; returning a trace that
            aliases would be worse than saying so.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer.")
    target = max(1.0, float(target_points))
    best = max(
        (factor for factor in range(1, sample_count + 1) if sample_count % factor == 0),
        key=lambda factor: (
            sample_count / factor >= target / 2, -abs(sample_count / factor - target)
        ),
    )
    if sample_count / best < target / 2:
        raise ValueError(
            f"a {sample_count}-sample window cannot be decimated to about "
            f"{target:g} points without aliasing."
        )
    return best


def decimate_trace(values: np.ndarray, factor: int) -> np.ndarray:
    """Group-and-average ``values`` by ``factor``: exactly ``len(values)//factor`` rows.

    Anti-aliased on purpose. A bare ``[::factor]`` slice of a 1-9 Hz trace drawn
    at 16 Hz keeps the power above the drawn Nyquist folded down into the drawn
    band, so the panel would show a slow wave that is not in the signal; at
    128 -> 16 Hz the alias of a 30 Hz component lands at 2 Hz, inside the band the
    decoder uses. Averaging each group is a box low-pass followed by the
    decimation, which is what the drawn rate can honestly carry. The input's
    trailing dimensions are preserved, so one trace or many take the same path.

    Raises:
        ValueError: If ``factor`` is not a positive integer or does not divide the
            trace length. Neither is padded nor truncated to fit.
    """

    array = np.asarray(values)
    if array.ndim not in (1, 2):
        raise ValueError("a trace must be one or two dimensional.")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer.")
    if array.shape[0] % factor:
        raise ValueError(
            f"factor {factor} does not divide a {array.shape[0]}-sample trace; "
            "the decimated length would not be the computed one."
        )
    grouped = array.reshape(array.shape[0] // factor, factor, *array.shape[1:])
    return grouped.mean(axis=1)


def decimate_for_display(values, factor: int) -> list[list[float]]:
    """Decimate one trace to JSON rows at the display's own precision.

    Four decimals of a microvolt is 0.1 nV, far below anything an amplifier or a
    panel resolves; the rounding exists because float64 noise in the JSON would
    multiply the packet size for digits no one can read.

    One trace becomes one row of the result. ``np.atleast_2d`` would put the
    *samples* on the column axis -- a ``(1, N)`` array of one row -- and the
    decimation would then be asked to divide a single row, so the reshape is
    explicit here rather than left to ``atleast_2d``.
    """

    rows = np.asarray(values, dtype=float).reshape(-1, 1)
    grouped = decimate_trace(rows, factor)
    return [[round(float(value), 4) for value in row] for row in grouped]


def unit_scale_to_uv(source_unit_exponent: int) -> float:
    """Factor converting the source unit to microvolts: ``10 ** (6 + exponent)``."""

    return float(10.0 ** (6 + int(source_unit_exponent)))


def unit_label(source_unit_exponent: int) -> str:
    """Unit text derived from the exponent the chain declares, never assumed."""

    exponent = int(source_unit_exponent)
    known = DISPLAY_UNIT_LABELS.get(exponent)
    if known is not None:
        return known
    return f"source unit 1e{exponent:+d} V (unlabelled)"


def scale_hint_uv(values: np.ndarray, source_unit_exponent: int) -> float:
    """Half-range the trace actually spans, in uV, rounded up.

    A per-window span hint, not a calibrated full-scale value: it is the largest
    excursion this window contains, converted from the source unit by the same
    declared exponent the unit label comes from. ``1.0`` is the floor, so a flat
    or empty window yields a drawable range instead of ``0``.
    """

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 1.0
    span = float(np.max(np.abs(finite))) * unit_scale_to_uv(source_unit_exponent)
    return float(max(1.0, math.ceil(span)))


def shared_decimation(
    raw_size: int, filtered_size: int, target_points: float = DISPLAY_TARGET_POINTS
) -> tuple[int, int, int]:
    """One point count for two traces of different rates.

    The raw trace and the window it is drawn beside have different lengths -- one
    is the interval at ``input_sfreq``, the other at ``output_sfreq`` -- so the
    two cannot simply share a factor: at 128 Hz in and 64 Hz out a 5 s window is
    640 raw samples and 320 resampled ones, and one factor of 8 would draw 80
    points against 40. The point count is therefore chosen **first** and each trace
    gets the factor that reaches it.

    Returns:
        ``(raw_factor, filtered_factor, points)``. The largest point count at or
        below ``target_points`` that both traces can hit exactly wins; the count
        closest to the target is the caller's own display rate, so this never
        invents a length the format did not ask for.

    Raises:
        ValueError: If either length is not a positive integer, or if the two
            lengths share no whole point count. Nothing is padded or truncated to
            make one fit: a fabricated sample is worse than no drawing.
    """

    for length in (raw_size, filtered_size):
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("both trace lengths must be positive integers.")
    ceiling = int(max(1.0, float(target_points)))
    for points in range(ceiling, 0, -1):
        if raw_size % points or filtered_size % points:
            continue
        return raw_size // points, filtered_size // points, points
    raise ValueError(
        f"a {raw_size}-sample raw trace and a {filtered_size}-sample window share "
        "no whole point count; they could not be drawn on one axis without "
        "fabricating samples."
    )


def capture_display(
    raw_channel, filtered_channel, *, channel, channel_source, source_rate,
    filtered_rate, window_seconds, window_end, available_at, source_unit_exponent,
    target_points=DISPLAY_TARGET_POINTS,
) -> DisplayTap:
    """Build one display window from the two traces over the same interval.

    The two traces must describe the **same seconds**: ``raw_channel`` is the raw
    ring's tail over ``window_seconds`` ending at the newest raw sample, and
    ``filtered_channel`` is the window the ring buffer emitted. A panel drawing
    them on one axis is claiming they are comparable, so both are decimated by
    one factor and must come out the same length; anything else is refused rather
    than drawn.

    Args:
        raw_channel: The channel from **before** the band-pass, copied out of the
            chain's own storage by the caller, covering ``window_seconds`` at
            ``source_rate``.
        filtered_channel: The window's own trace after band-pass and resample.
        channel: Displayed channel label.
        channel_source: Free text naming how the channel was chosen.
        source_rate: Rate of ``raw_channel``, i.e. the rate this chain is fed.
        filtered_rate: Rate of ``filtered_channel``; never assumed equal to
            ``source_rate`` -- the two are separate fields for that reason.
        window_seconds: The interval both traces cover, in seconds.
        window_end: Source time of the newest filtered sample, i.e. where the
            displayed window ends.
        available_at: Source time of the newest raw sample, which is what the
            ``eeg_display`` packet's own timestamp is stamped with.
        source_unit_exponent: Exponent of the fed unit (``-6`` = microvolts).
        target_points: Trace points to decimate to.

    Raises:
        ValueError: If the rates are not a whole-number ratio of each other, if
            the two trace lengths disagree on the interval, or if the filtered
            trace is empty. Each is a window whose two traces could not be drawn
            on one axis, and a drawing that silently compares different intervals
            would make the panel's central claim false.
    """

    raw = np.asarray(raw_channel, dtype=float).reshape(-1)
    filtered = np.asarray(filtered_channel, dtype=float).reshape(-1)
    if filtered.size == 0:
        raise ValueError("a display window needs at least one filtered sample.")
    ratio = float(source_rate) / float(filtered_rate)
    k = int(round(ratio))
    if k < 1 or not math.isclose(ratio, k, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"a {source_rate:g} Hz source cannot be displayed against a "
            f"{filtered_rate:g} Hz window: the rates must be a whole-number ratio, "
            "or the two traces would not share a time axis."
        )
    if filtered.size % k:
        raise ValueError(
            f"a {filtered.size}-sample window at {filtered_rate:g} Hz is not a "
            f"whole number of {source_rate:g} Hz samples; the raw trace could not "
            "cover the same interval."
        )
    expected_raw = filtered.size * k
    if raw.size != expected_raw:
        raise ValueError(
            f"the raw trace covers {raw.size} samples but the window covers "
            f"{expected_raw} at {source_rate:g} Hz; the two would be drawn on "
            "different intervals."
        )
    # The two traces are decimated to **one** point count. Both lengths must be
    # whole windows: the raw one covers the interval, so any factor that divides
    # it exactly divides the filtered window too (its length is the raw interval
    # measured in resampled rows), and one count is what makes the two drawn
    # traces share a time axis rather than merely a starting time.
    raw_factor, filtered_factor, points = shared_decimation(
        raw.size, filtered.size, target_points
    )
    newest = None if available_at is None else float(available_at)
    if newest is None or not math.isfinite(newest):
        newest = 0.0 if window_end is None else float(window_end)
    label = unit_label(source_unit_exponent)
    return DisplayTap(
        sample_rate=float(source_rate),
        filtered_sample_rate=float(filtered_rate),
        channels=(str(channel),),
        samples=decimate_for_display(raw, raw_factor),
        filtered_samples=decimate_for_display(filtered, filtered_factor),
        original_units=label,
        filtered_units=label,
        original_scale_hint_uv=scale_hint_uv(raw, source_unit_exponent),
        filtered_scale_hint_uv=scale_hint_uv(filtered, source_unit_exponent),
        window_seconds=float(window_seconds),
        lag_seconds=max(0.0, newest - float(window_end)),
        channel_source=str(channel_source),
    )


class DisplayWindowUnavailable(ValueError):
    """The chain emitted a window the display cannot describe yet.

    Raised when the produced window is not yet a whole ``window_seconds`` long --
    the first window of a segment, and the first after a recovery, are partial.
    It is a ValueError so a caller that only knows the old contract still sees a
    refusal rather than a trace of the wrong length.
    """


class RawDisplayRing:
    """The displayed channel's own raw history, on its own logical clock.

    The raw EEG is not available when a window is emitted. ``feed`` receives it in
    ~0.03 s chunks, ``bandpass`` and ``resampler`` consume the same call, and the
    ring buffer then hands out a window covering the last ``window_seconds`` --
    minutes of raw signal that no longer exists anywhere once the chunk has been
    filtered. Keeping a reference to the last chunk would give a display trace of
    four points over a five-second axis, drawn beside a filtered trace covering
    five seconds: two different intervals on one shared axis, which is exactly
    what a reader of the panel would take as a comparison.

    So the chain keeps this small ring instead: one channel, ``dtype`` float64,
    ``capacity_samples`` rows, holding the newest raw samples keyed by the same
    logical order they arrived in. ``tail`` returns a contiguous copy of the last
    ``window_samples`` rows, which is the raw interval the emitted window covers.
    It is copied because the next chunk overwrites the cells it came from -- the
    same reason the capture itself is a copy.

    Registered with :class:`~nova2026.streaming.recovery.Recovery` as resettable:
    after a gap, samples from before it must not be drawn as if they were
    adjacent to samples from after it.
    """

    def __init__(self, capacity_samples: int, dtype=np.float64) -> None:
        if isinstance(capacity_samples, bool) or not isinstance(capacity_samples, int):
            raise TypeError("capacity_samples must be an integer.")
        if capacity_samples < 1:
            raise ValueError("capacity_samples must be positive.")
        self._capacity = int(capacity_samples)
        self._ring = np.empty(self._capacity, dtype=dtype)
        self.total_written = 0

    @property
    def capacity_samples(self) -> int:
        """Rows the ring holds."""

        return self._capacity

    def reset(self) -> None:
        """Forget every sample: a recovery means the old ones are not adjacent."""

        self.total_written = 0

    def push(self, values) -> None:
        """Append one chunk's worth of the displayed channel, wrapping as needed."""

        block = np.asarray(values, dtype=self._ring.dtype).reshape(-1)
        if block.size == 0:
            return
        if block.size >= self._capacity:
            self._ring[:] = block[-self._capacity :]
            self.total_written += block.size
            return
        start = self.total_written % self._capacity
        first = min(block.size, self._capacity - start)
        self._ring[start : start + first] = block[:first]
        if first < block.size:
            self._ring[: block.size - first] = block[first:]
        self.total_written += block.size

    def tail(self, window_samples: int):
        """The newest ``window_samples`` rows in order, or ``None`` if too few.

        ``None`` means the ring holds less than a whole window -- the first window
        after the chain starts, or after a recovery -- and a trace shorter than
        the window it would be drawn against is refused rather than padded, since a
        padded sample is a sample the electrode never produced.

        Returns:
            A copy, so no later ``push`` can rewrite what the caller holds.
        """

        if isinstance(window_samples, bool) or not isinstance(window_samples, int):
            raise TypeError("window_samples must be an integer.")
        if window_samples < 1:
            raise ValueError("window_samples must be positive.")
        available = min(self.total_written, self._capacity)
        if available < window_samples:
            return None
        offset = (self.total_written - window_samples) % self._capacity
        if offset + window_samples <= self._capacity:
            return self._ring[offset : offset + window_samples].copy()
        split = self._capacity - offset
        return np.concatenate(
            (self._ring[offset:], self._ring[: window_samples - split])
        )


def stream_config(
    trial,
    config,
    history=5.0,
    step=1.0,
    *,
    check_channels=True,
    max_bad_channels=0,
    exclude_channels=(),
    source_unit_exponent=DEFAULT_SOURCE_UNIT_EXPONENT,
    saturation_limit_uv=None,
    amplitude_limit_uv=None,
    display_channel=None,
):
    """Build the chain settings for one trial and decoder configuration.

    The channel options are run policy, not part of the processing contract:
    ``AuditoryProcessor.contract`` is compared against a trained model's
    contract, so adding keys there would invalidate every existing decoder.

    ``source_unit_exponent`` is the power of ten of the *source* unit relative to
    volts (0 = volts, -6 = microvolts), the convention of
    ``nova2026.streaming.preprocess.units.unit_scaler``. It reaches only
    ``Repair``'s endpoint safety check, which converts endpoints to microvolts to
    compare them against its saturation and amplitude limits; no stage rescales
    samples with it. The default is -6 because this chain's callers feed
    microvolts (the KU Leuven trials are microvolts, and the live path scales its
    source to microvolts first), which makes the check an identity on the values.
    A caller feeding a source whose own unit is volts -- the ANT amplifier -- must
    pass 0, or the check would compare 0.0833 V against a microvolt saturation
    limit and pass a saturated signal through. Like the channel options, it is run
    policy and must not be added to ``contract``.

    ``amplitude_limit_uv`` and ``saturation_limit_uv`` are the two limits
    ``Repair`` judges an endpoint with; ``None`` keeps its own (500 / 75 000 uV).
    ``amplitude_limit_uv`` additionally reaches ``QualityMonitor``, which calls an
    electrode faulted when it moves further than that from the run's own anchor
    level: one declared number, one owner, so a window the monitor publishes as
    artifact-free cannot be one the repairer refuses to bridge. A recording that
    drifts further than 500 uV from its anchor (the operator's ANT session does,
    by more than 3 mV) declares this value for the whole run; nothing here widens
    a limit silently, because a value set here is printed and recorded as the
    effective run policy.

    ``display_channel`` names the electrode the display tap carries. ``None`` --
    the default -- turns the tap off entirely: no capture, no allocation, no
    field, which is what keeps a run without a display byte-for-byte the run it
    was before this option existed. Like the channel and limit options it is run
    policy and must not reach ``contract``.
    """

    if source_unit_exponent not in SUPPORTED_SOURCE_UNIT_EXPONENTS:
        raise ValueError("Supported source exponents are 0, -3, -6, -9.")
    for name, value in (("history", history), ("step", step)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if history * config.sample_rate <= config.lag_samples + 2:
        raise ValueError("History must include the decoder lags and usable samples.")
    if not isinstance(check_channels, bool):
        raise TypeError("check_channels must be a bool.")
    if isinstance(exclude_channels, str):
        raise TypeError("exclude_channels must be an iterable of labels, not a string.")
    return SimpleNamespace(
        input_sfreq=round(trial.sample_rate, 6), output_sfreq=config.sample_rate,
        eeg_channels=tuple(trial.channel_names), bandpass=config.band,
        input_reference=trial.reference, upstream_processing=trial.upstream_processing,
        window_seconds=history, step_seconds=step, warmup_seconds=2.0,
        persistent_fault_seconds=max(15.0, 2 * history + 2),
        check_channels=check_channels, max_bad_channels=max_bad_channels,
        exclude_channels=tuple(exclude_channels),
        source_unit_exponent=int(source_unit_exponent),
        saturation_limit_uv=(
            None if saturation_limit_uv is None else float(saturation_limit_uv)
        ),
        amplitude_limit_uv=(
            None if amplitude_limit_uv is None else float(amplitude_limit_uv)
        ),
        display_channel=(
            None if display_channel is None else str(display_channel).strip() or None
        ),
    )


class AuditoryProcessor:
    """Use the current streaming primitives; retain every hop even in large chunks.

    Inputs are explicitly in microvolts. Live callers scale source units before
    feeding this chain. Channel selection occurs once, by name, at construction.
    """

    def __init__(self, settings, source_channels=None, judges=None):
        self.settings = settings
        self._raw_window = None
        """Newest raw display trace, or ``None`` while the tap is off or the ring is filling."""
        names = settings.eeg_channels
        self.channels = ChannelContract(source_channels or names, names)
        n = len(names)
        # Dead-electrode policy. Both stages that can stop a run for one
        # channel take the same list, so no path is left unguarded.
        check_channels = bool(getattr(settings, "check_channels", True))
        max_bad_channels = int(getattr(settings, "max_bad_channels", 0))
        exclude_channels = tuple(getattr(settings, "exclude_channels", ()) or ())
        self.repair = Repair(
            settings.input_sfreq,
            **self._endpoint_limits(settings),
            source_unit_exponent=int(getattr(
                settings, "source_unit_exponent", DEFAULT_SOURCE_UNIT_EXPONENT
            )),
            channel_names=names, exclude_channels=exclude_channels,
        )
        self.quality = QualityMonitor(
            n_eeg=n, sfreq=settings.input_sfreq, channel_names=names,
            check_channels=check_channels, max_bad_channels=max_bad_channels,
            exclude_channels=exclude_channels, **self._quality_limits(settings),
        )
        self.bandpass = SosFilter(design_bandpass(*settings.bandpass, 3, settings.input_sfreq), n)
        # A source that already arrives at the output rate (the live ANT route's
        # pre-chain 500 -> 128 adapter) must not be resampled again: equal rates
        # are refused by `Resampler`, and the chain's rate declaration has to be
        # the rate it is actually fed or `Repair` reads every step as a gap.
        self.resampler = (
            _PassThroughResampler()
            if math.isclose(settings.input_sfreq, settings.output_sfreq)
            else Resampler(
                settings.input_sfreq, settings.output_sfreq, n, quality="auto",
                max_age_seconds=3.0, reserve_seconds=1.0, allow_qq=True, strict=True,
            )
        )
        self.buffer = CircularBuffer(
            round(settings.window_seconds * settings.output_sfreq),
            round(settings.step_seconds * settings.output_sfreq),
            round((settings.window_seconds + 4) * settings.output_sfreq),
            settings.output_sfreq, n,
        )
        # The display tap is off unless a channel is named. It is read once here
        # rather than per chunk, so a disabled tap costs one attribute lookup at
        # construction and nothing in `feed`.
        self._display_channel = getattr(settings, "display_channel", None) or None
        self._display_ring = (
            None
            if self._display_channel is None
            else self._display_ring_for(settings, self.resampler)
        )
        self.recovery = Recovery(
            resettable=(
                self.repair, self.quality, self.bandpass, self.resampler, self.buffer,
                # The raw display ring forgets with the rest: after a recovery the
                # samples from before the gap are not adjacent to the ones after it.
                *([self._display_ring] if self._display_ring is not None else []),
            ),
            persistent_fault_seconds=settings.persistent_fault_seconds,
        )
        # Judges are injected, so a script can add its own census rule (see
        # BadChannelJudge) without this class knowing about it.
        self.judges = self._resolve_judges(judges)
        self.contract = {
            "eeg_channels": list(names), "output_sfreq": settings.output_sfreq,
            "input_sfreq": settings.input_sfreq, "units": "uV",
            "input_reference": settings.input_reference,
            "upstream_processing": settings.upstream_processing,
            "bandpass": list(settings.bandpass), "filter_order": 3,
            "resample_quality": self.resampler.quality,
            "stage": "auditory-current-streaming-v2",
            "window_seconds": settings.window_seconds, "step_seconds": settings.step_seconds,
        }

    @staticmethod
    def _display_ring_for(settings, resampler) -> "RawDisplayRing":
        """The raw history ring: one window, plus the resampler's own delay.

        A window needs ``window_seconds * input_sfreq`` rows, and the resampler's
        startup and maximum delay are the only reason the raw stream can run ahead
        of the filtered one by more than a chunk. Sizing it here rather than from a
        constant keeps it correct for a source fed at its own output rate, and for
        a window length the decoder's contract chooses.
        """

        seconds = (
            float(settings.window_seconds)
            + 2.0
            + float(getattr(resampler, "startup_delay_seconds", 0.0))
            + float(getattr(resampler, "max_delay_seconds", 0.0))
        )
        return RawDisplayRing(round(seconds * float(settings.input_sfreq)))

    @staticmethod
    def _endpoint_limits(settings):
        """Repair's endpoint safety limits, when the run policy sets them.

        Defaults stay `Repair`'s own (500 / 75 000 uV) for every caller that
        does not set them, so no existing behaviour moves. A recording whose
        amplifier railed an electrode past 75 000 uV (plan section 3.11 documents
        exactly that on the operator's ANT session) needs a declared saturation
        limit above its own rail, and a recording that drifts further than 500 uV
        from its own anchor level needs a declared amplitude limit above its own
        drift; both values are run policy, printed and recorded, never a silent
        widening of the guard.
        """

        limits = {}
        saturation = getattr(settings, "saturation_limit_uv", None)
        amplitude = getattr(settings, "amplitude_limit_uv", None)
        if saturation is not None:
            limits["saturation_limit_uv"] = float(saturation)
        if amplitude is not None:
            limits["amplitude_limit_uv"] = float(amplitude)
        return limits

    @staticmethod
    def _quality_limits(settings):
        """The declared limits `QualityMonitor` shares with `Repair`.

        One declared value has one owner: the run policy states the amplitude
        limit once and both stages that judge an electrode with it receive it, so
        a window the monitor calls artifact-free cannot be one the repairer calls
        unsafe to bridge. `saturation_limit_uv` is deliberately **not** handed to
        the monitor: it is an absolute rail, and the monitor's own saturation
        fault is what names a railed electrode for the census. A caller that sets
        nothing gets the monitor's own 500 uV, exactly as before.
        """

        amplitude = getattr(settings, "amplitude_limit_uv", None)
        return {} if amplitude is None else {"amplitude_limit_uv": float(amplitude)}

    def _resolve_judges(self, judges):
        """Validate the injected judges; default to the monitor and the repairer."""

        if judges is None:
            return (self.quality, self.repair)
        if isinstance(judges, (str, bytes)):
            raise TypeError("judges must be an iterable of window judges.")
        resolved = tuple(judges)
        for judge in resolved:
            if not callable(getattr(judge, "reasons", None)):
                raise TypeError(
                    "Every judge must implement reasons(start, end); "
                    f"{judge!r} does not."
                )
        return resolved

    def _capture_display(self, data):
        """Copy the displayed channel out of the chunk **before** the band-pass.

        The raw EEG exists only in this call: ``bandpass`` and ``resampler`` are
        handed the same array, and the ring buffer hands out views of its own
        storage, so a reference kept here would be overwritten before the frame
        that needs it is built. The copy is what makes the trace raw; a reference
        is what would silently make it band-passed.

        Only the displayed channel is copied -- a few samples per 0.03 s chunk
        instead of the window's ~300 KB every time -- because this runs on the
        acquisition thread for every chunk of a 64-channel, 128 Hz stream.

        Returns:
            ``(channel, raw_window_trace)``: the label and the newest
            ``window_seconds`` of that channel, copied out of the chain's
            :class:`RawDisplayRing`; or ``(None, None)`` when the tap is off, the
            channel is not here, or the ring has not filled yet. A missing channel
            surfaces through the producer's ``channel_source``, never as a guess.
        """

        if self._display_channel is None or not len(data):
            return None, None
        try:
            index = self.settings.eeg_channels.index(self._display_channel)
        except ValueError:
            return None, None
        # The raw interval a window covers is `window_seconds` at the rate this
        # chain is fed -- `input_sfreq`, not the buffer's own `output_sfreq`.
        self._display_ring.push(data[:, index])
        wanted = round(float(self.settings.window_seconds) * float(self.settings.input_sfreq))
        return self._display_channel, self._display_ring.tail(wanted)

    def _display_payload(self, channel, raw, rows, times, available):
        """The display window for one emitted window.

        The raw trace is cut from the ring to the length this window's own
        interval needs -- ``len(rows)`` resampled rows times the rate ratio -- so
        the two traces cover the same seconds by construction rather than by
        coincidence. A window that is not yet a whole one cannot get that trace at
        all: see :class:`DisplayWindowUnavailable`.
        """

        ratio = float(self.settings.input_sfreq) / float(self.settings.output_sfreq)
        k = int(round(ratio))
        needed = len(rows) * k if k >= 1 and math.isclose(
            ratio, k, rel_tol=0.0, abs_tol=1e-6
        ) else 0
        held = min(self._display_ring.total_written, self._display_ring.capacity_samples)
        raw = self._display_ring.tail(needed) if needed else None
        if raw is None:
            raise DisplayWindowUnavailable(
                f"a {len(rows)}-row window needs {needed} raw samples but the ring "
                f"holds {held}; the interval is not covered yet and a shorter trace "
                "must not be drawn against it."
            )
        return capture_display(
            raw,
            rows[:, self.settings.eeg_channels.index(channel)],
            channel=channel,
            channel_source=f"display_channel={channel!r} named by the run policy",
            source_rate=float(self.settings.input_sfreq),
            filtered_rate=float(self.settings.output_sfreq),
            # The interval this window really covers, not the configured window
            # length: a partial window that reaches here must report its own span.
            window_seconds=float(len(rows)) / float(self.settings.output_sfreq),
            window_end=float(times[-1]),
            available_at=available,
            source_unit_exponent=int(
                getattr(
                    self.settings, "source_unit_exponent", DEFAULT_SOURCE_UNIT_EXPONENT
                )
            ),
        ).to_payload()

    def feed(self, chunk):
        data, timestamps = chunk
        data = self.channels.reorder(data)
        if not len(timestamps):
            return []
        available = float(timestamps[-1])
        try:
            data, timestamps = self.repair(data, timestamps)
            self.quality.feed(data, timestamps)
            raw_channel, raw_window = self._capture_display(data)
            if raw_channel is not None:
                # `tail` returns None until the ring holds a whole window, so the
                # previous chunk's trace must not survive into this one's window.
                self._raw_window = raw_window
            data, timestamps = self.bandpass(data, timestamps)
            data, timestamps = self.resampler(data, timestamps)
        except UnrepairableError as error:
            self.recovery.handle(error)
            return []
        result = []
        for rows, times, start in self.buffer.push(data, timestamps):
            reasons, bad_channels = collect_verdict(self.judges, times[0], times[-1])
            warm = start < round(self.settings.warmup_seconds * self.settings.output_sfreq)
            window = EEGWindow(
                rows, rows[:, :0], times, not reasons and not warm, reasons, start,
                segment=self.recovery.segment, channel_names=self.settings.eeg_channels,
                contract=self.contract, available_at=available,
                bad_channels=bad_channels,
            )
            if raw_channel is not None and self._raw_window is not None:
                # Set on the window rather than threaded through its constructor:
                # `EEGWindow` is the streaming package's, and this tap is one
                # consumer's display, not a property every window must carry. The
                # raw trace is the same interval as `data`, so the two drawn
                # traces cover the same seconds.
                window.raw = self._raw_window
                try:
                    window.display = self._display_payload(
                        raw_channel, self._raw_window, rows, times, available
                    )
                except DisplayWindowUnavailable:
                    # The segment's first window is partial, so it has no honest
                    # raw trace to draw. Absence is the truthful display there; the
                    # next window covers a whole one.
                    del window.raw
            self.recovery.watch(window)
            result.append(window)
        return result
