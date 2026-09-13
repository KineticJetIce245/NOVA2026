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
    )


class AuditoryProcessor:
    """Use the current streaming primitives; retain every hop even in large chunks.

    Inputs are explicitly in microvolts. Live callers scale source units before
    feeding this chain. Channel selection occurs once, by name, at construction.
    """

    def __init__(self, settings, source_channels=None, judges=None):
        self.settings = settings
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
        self.recovery = Recovery(
            resettable=(self.repair, self.quality, self.bandpass, self.resampler, self.buffer),
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

    def feed(self, chunk):
        data, timestamps = chunk
        data = self.channels.reorder(data)
        if not len(timestamps):
            return []
        available = float(timestamps[-1])
        try:
            data, timestamps = self.repair(data, timestamps)
            self.quality.feed(data, timestamps)
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
            self.recovery.watch(window)
            result.append(window)
        return result
