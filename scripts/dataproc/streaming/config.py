"""Configuration shared by acquisition, replay, and future live-model training."""

import math

# CA-208 datasheet, page 3: 63 EEG channels and one EOG channel.
# CPz is the reference; AFz is the ground, not a classifier channel.
CA208_CHANNELS = (
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "O2",
    "EOG",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CP4",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO5",
    "PO3",
    "PO4",
    "PO6",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "Oz",
)

# Shared CA-208 / COG-BCI EEG channels, in CA-208 order. The offline code is
# unchanged; future training must explicitly use this contract.
EEG_CHANNELS = tuple(
    name
    for name in CA208_CHANNELS
    if name not in ("Fpz", "M1", "Cz", "M2", "EOG", "PO5", "PO6")
)


class StreamConfig:
    """Define one run's acquisition and preprocessing contract.

    Args:
        input_sfreq: Confirmed source sampling rate in Hz.
        source_unit_exponent: Source multiplier, e.g. -6 for microvolts.
        input_reference: Reference already applied by the source. This runner
            does not apply another reference.
        upstream_processing: Operator-supplied description of source processing.
        stream_name: LSL outlet name. Name or source ID must identify the outlet.
        source_id: LSL outlet source ID.
        eeg_channels: Classifier channels in canonical output order.
        eog_channels: Auxiliary channels, kept separate from classifier data.

    Notes:
        Filter and quality limits are prototype settings, not measured hardware
        specifications. Output samples are microvolts. The saturation limit must
        be set from the selected amplifier gain before hardware use.
    """

    def __init__(
        self,
        input_sfreq: float,
        source_unit_exponent: int,
        input_reference: str,
        upstream_processing: str,
        stream_name: str | None = None,
        source_id: str | None = None,
        stream_type: str | None = None,
        eeg_channels: tuple[str, ...] = EEG_CHANNELS,
        eog_channels: tuple[str, ...] = ("EOG",),
        ground: str = "AFz",
        output_sfreq: float = 128.0,
        window_seconds: float = 2.0,
        step_seconds: float = 0.5,
        warmup_seconds: float = 2.0,
        bandpass: tuple[float, float] = (1.0, 45.0),
        filter_order: int = 3,
        notch_frequency: float | None = 60.0,
        notch_q: float = 30.0,
        resample_quality: str = "LQ",
        buffer_seconds: float = 4.0,
        queue_size: int = 128,
        max_ready_windows: int = 32,
        max_lag_seconds: float = 3.0,
        max_future_seconds: float = 0.1,
        no_data_timeout: float = 3.0,
        connection_timeout: float = 5.0,
        poll_interval: float = 0.005,
        timestamp_tolerance: float = 0.0002,
        amplitude_limit_uv: float = 500.0,
        saturation_limit_uv: float = 75000.0,
        flatline_seconds: float = 0.5,
        flatline_tolerance_uv: float = 0.001,
        recovery_max_events: int = 5,
        recovery_max_gap: float = 0.5,
        persistent_fault_seconds: float = 5.0,
        interpolation_max_seconds: float = 0.02,
        interpolation_max_fraction: float = 0.05,
        expected_channels: tuple[str, ...] | None = None,
    ) -> None:
        """Store explicit run settings and validate them before acquisition."""

        self.input_sfreq = input_sfreq
        self.source_unit_exponent = source_unit_exponent
        self.input_reference = input_reference
        self.upstream_processing = upstream_processing
        self.stream_name = stream_name
        self.source_id = source_id
        self.stream_type = stream_type
        if expected_channels is not None:
            if eeg_channels != EEG_CHANNELS and eeg_channels != expected_channels:
                raise ValueError("Specify one consistent EEG channel order.")
            eeg_channels = expected_channels

        self.eeg_channels = tuple(eeg_channels)
        self.eog_channels = eog_channels
        self.ground = ground
        self.output_sfreq = output_sfreq
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.warmup_seconds = warmup_seconds
        self.bandpass = bandpass
        self.filter_order = filter_order
        self.notch_frequency = notch_frequency
        self.notch_q = notch_q
        self.resample_quality = resample_quality
        self.buffer_seconds = buffer_seconds
        self.queue_size = queue_size
        self.max_ready_windows = max_ready_windows
        self.max_lag_seconds = max_lag_seconds
        self.max_future_seconds = max_future_seconds
        self.no_data_timeout = no_data_timeout
        self.connection_timeout = connection_timeout
        self.poll_interval = poll_interval
        self.timestamp_tolerance = timestamp_tolerance
        self.amplitude_limit_uv = amplitude_limit_uv
        self.saturation_limit_uv = saturation_limit_uv
        self.flatline_seconds = flatline_seconds
        self.flatline_tolerance_uv = flatline_tolerance_uv

        self.recovery_max_events = recovery_max_events
        self.recovery_max_gap = recovery_max_gap
        self.persistent_fault_seconds = persistent_fault_seconds

        self.interpolation_max_seconds = interpolation_max_seconds
        self.interpolation_max_fraction = interpolation_max_fraction

        self.validate()

    def validate(self) -> None:
        """Reject inconsistent settings before connecting to an outlet."""

        if (
            not math.isfinite(self.interpolation_max_seconds)
            or not 0 <= self.interpolation_max_seconds <= 0.1
        ):
            raise ValueError(
                "Interpolation must be disabled (0) or limited to at most 0.1 seconds."
            )
        if (
            not math.isfinite(self.interpolation_max_fraction)
            or not 0 <= self.interpolation_max_fraction <= 0.1
        ):
            raise ValueError("Interpolation fraction must be between 0 and 0.1.")

        positive = (
            "input_sfreq",
            "output_sfreq",
            "window_seconds",
            "step_seconds",
            "buffer_seconds",
            "max_lag_seconds",
            "max_future_seconds",
            "no_data_timeout",
            "connection_timeout",
            "poll_interval",
            "notch_q",
            "amplitude_limit_uv",
            "saturation_limit_uv",
            "flatline_seconds",
            "recovery_max_gap",
            "persistent_fault_seconds",
        )

        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        for name in ("warmup_seconds", "timestamp_tolerance", "flatline_tolerance_uv"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

        for name in (
            "queue_size",
            "max_ready_windows",
            "filter_order",
            "recovery_max_events",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if not self.stream_name and not self.source_id:
            raise ValueError("Specify a stream name or source ID.")

        for name in ("input_reference", "upstream_processing"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be explicitly described.")

        if self.source_unit_exponent not in (0, -3, -6, -9):
            raise ValueError("Supported source units are V, mV, uV, and nV.")

        if self.resample_quality not in ("LQ", "MQ", "HQ", "VHQ"):
            raise ValueError("Choose an anti-aliased SoXR quality mode.")
        channels = self.channels

        if not self.eeg_channels or len(channels) != len(set(channels)):
            raise ValueError("EEG channels must be non-empty and all names unique.")

        if any(not isinstance(name, str) or not name.strip() for name in channels):
            raise ValueError("Channel names must be non-empty strings.")

        if self.step_seconds > self.window_seconds:
            raise ValueError("The step must not exceed the window length.")

        if self.buffer_seconds < self.window_seconds:
            raise ValueError("The buffer must hold at least one window.")

        for seconds in (self.window_seconds, self.step_seconds, self.buffer_seconds):
            samples = seconds * self.output_sfreq
            if round(samples) < 1 or not math.isclose(
                samples, round(samples), abs_tol=1e-8
            ):
                raise ValueError(
                    "Window, step, and buffer must use whole output samples."
                )
        low, high = self.bandpass

        if not 0 < low < high < min(self.input_sfreq, self.output_sfreq) / 2:
            raise ValueError(
                "Band-pass cutoffs must lie below both Nyquist frequencies."
            )

        if (
            self.notch_frequency is not None
            and not 0 < self.notch_frequency < self.input_sfreq / 2
        ):
            raise ValueError("Notch frequency must lie below input Nyquist.")

        if self.timestamp_tolerance >= 0.5 / self.input_sfreq:
            raise ValueError("Timestamp tolerance must be less than half a sample.")

        if self.saturation_limit_uv < self.amplitude_limit_uv:
            raise ValueError("Saturation limit must not be below the amplitude limit.")

    @property
    def channels(self) -> tuple[str, ...]:
        """Return EEG followed by auxiliary channels in canonical queue order."""

        return self.eeg_channels + self.eog_channels

    def sample_count(self, seconds: float) -> int:
        """Convert a configured duration to a whole number of output samples."""

        return round(seconds * self.output_sfreq)

    def to_dict(self) -> dict:
        """Return serializable settings without sharing mutable instance state."""

        return dict(vars(self))

    @classmethod
    def from_dict(cls, values: dict) -> "StreamConfig":
        """Reconstruct a recorded configuration, restoring ordered tuples."""

        values = dict(values)
        # Historical recordings used rejection only. Preserve their replay behavior.
        values.setdefault("interpolation_max_seconds", 0.0)

        for name in ("eeg_channels", "eog_channels", "bandpass"):
            if name in values:
                values[name] = tuple(values[name])

        return cls(**values)

    def updated(self, **changes) -> "StreamConfig":
        """Build a validated copy with explicitly replaced settings."""

        values = self.to_dict()
        if "expected_channels" in changes:
            values["eeg_channels"] = tuple(changes.pop("expected_channels"))
        return self.from_dict({**values, **changes})

    def processing_contract(self) -> dict:
        """Return settings that must match a saved artifact operator.

        Transport identifiers and quality thresholds are excluded. The operator
        is applied after causal filtering and resampling, in microvolts, with no
        additional reference operation.
        """

        names = (
            "eeg_channels",
            "eog_channels",
            "input_sfreq",
            "output_sfreq",
            "input_reference",
            "upstream_processing",
            "bandpass",
            "filter_order",
            "notch_frequency",
            "notch_q",
            "resample_quality",
        )
        result = {name: getattr(self, name) for name in names}
        if self.interpolation_max_seconds > 0:
            result["interpolation_max_seconds"] = self.interpolation_max_seconds
            result["interpolation_max_fraction"] = self.interpolation_max_fraction
            result["interpolation_endpoint_limits"] = (
                self.amplitude_limit_uv,
                self.saturation_limit_uv,
            )
        result["units"] = "uV"
        result["stage"] = "after-causal-filter-and-resample-v1"

        return result

    @property
    def expected_channels(self) -> tuple[str, ...]:
        """Return the EEG order already enforced by ChannelSelectionContract."""

        return self.eeg_channels

    def window_contract(self) -> dict:
        """Return the serializable preprocessing and window contract for inference."""

        import json

        result = self.processing_contract()
        for name in (
            "window_seconds",
            "step_seconds",
            "warmup_seconds",
            "amplitude_limit_uv",
            "saturation_limit_uv",
            "flatline_seconds",
            "flatline_tolerance_uv",
            "timestamp_tolerance",
        ):
            result[name] = getattr(self, name)
        return json.loads(json.dumps(result))
