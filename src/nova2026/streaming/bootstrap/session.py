"""Build every run-once component of a live session around a stream.

``StreamSession`` assembles the boilerplate start-up (channel contract,
optional recorder, acquire handle, window buffer) while the preprocessing
stages stay where the script defines them: the caller constructs
``scale``/``quality``/``filters``/``resample`` explicitly and hands them in.
The session only guarantees the execution order and the gating semantics.
"""

from datetime import datetime

import numpy as np

from ..acquire import Acquire
from ..channels import ChannelContract
from ..circular_buffer import CircularBuffer
from ..preprocess import QualityMonitor, unit_scaler
from ..recording import RunRecorder, RunSpec


class StreamSession:
    """Assemble one live session's components around a connected stream.

    Args:
        stream: Connected ``StreamLSL`` (manual acquisition). Channel labels
            are read from it to build the contract.
        args: Parsed namespace (geometry + recording identity + consumer
            knobs). See :func:`~.args.parse_args` for the expected fields.
        channels: Canonical channel names, EEG first then auxiliary.
        scale: Explicit V->uV stage (``(data, ts) -> (data, ts)``). Defaults
            to ``unit_scaler(source_unit_exponent)``.
        quality: Explicit :class:`~..preprocess.QualityMonitor`. Defaults to a
            monitor over all ``channels`` at the input rate.
        filters: Explicit stateful causal filters applied after quality
            observation, in order (e.g. ``(notch, bandpass)``).
        resample: Explicit :class:`~..preprocess.Resampler` (or None). Its
            output rate drives the window geometry.
        source_unit_exponent: Power of ten of the source unit (0 = volts).
        role: Run role recorded in the metadata.
        ch_types: Optional per-channel MNE types; defaults to all EEG.

    Notes:
        The execution order of preprocessing is fixed and owned here:
        ``scale -> quality.feed -> each filter -> resample``. The session does
        NOT connect the stream, start threads or consume windows; it only
        builds the run-once pieces and exposes the loop helpers.

    Attributes:
        contract, recorder, acquire, buffer: The built components.
        scale, quality, filters, resample: The preprocessing stages as passed.
        n_channels, out_sfreq, warmup_samples: Geometry used by the loop.
    """

    def __init__(
        self,
        stream,
        args,
        channels: tuple[str, ...],
        *,
        scale=None,
        quality=None,
        filters: tuple = (),
        resample=None,
        source_unit_exponent: int = 0,
        role: str = "run",
        ch_types: tuple[str, ...] | None = None,
    ) -> None:
        """Build the contract, recorder, acquire, buffer and chain defaults."""

        self.args = args
        self.channels = tuple(str(name) for name in channels)
        self.n_channels = len(self.channels)
        if ch_types is None:
            ch_types = ("eeg",) * self.n_channels
        self.ch_types = tuple(ch_types)

        sfreq = float(getattr(args, "sfreq", 500.0))

        # Channel contract: fail here if a required label is missing.
        self.contract = ChannelContract(stream.ch_names, self.channels)

        # Optional per-run recorder (raw volts, before any transformation).
        self.recorder = None
        record_root = getattr(args, "record", None)
        if record_root is not None:
            run_name = getattr(args, "run", None) or datetime.now().strftime(
                "run-%Y%m%d-%H%M%S"
            )
            self.recorder = RunRecorder(
                record_root,
                RunSpec(
                    getattr(args, "subject", "demo"),
                    getattr(args, "session", "synthetic"),
                    run_name,
                    role=role,
                ),
                self.channels,
                sfreq,
                ch_types=self.ch_types,
                unit_exponent=source_unit_exponent,
            )

        # Preprocessing stages: caller-provided wins, sensible defaults remain
        # so a bare session still converts units and monitors quality.
        self.scale = scale if scale is not None else unit_scaler(
            source_unit_exponent, desired_exponent=-6
        )
        self.quality = (
            quality
            if quality is not None
            else QualityMonitor(n_eeg=self.n_channels, sfreq=sfreq, warmup_seconds=0.0)
        )
        self.filters = tuple(filters)
        self.resample = resample

        # The window geometry lives at the OUTPUT rate: the resampler's rate
        # when present, otherwise the source rate (or --out-sfreq fallback).
        if resample is not None:
            self.out_sfreq = float(resample.out_sfreq)
        else:
            self.out_sfreq = float(getattr(args, "out_sfreq", sfreq))
        self.window_samples = round(
            float(getattr(args, "window", 2.0)) * self.out_sfreq
        )
        self.hop_samples = round(float(getattr(args, "hop", 0.5)) * self.out_sfreq)
        self.capacity_samples = round(
            float(getattr(args, "capacity", 6.0)) * self.out_sfreq
        )
        self.warmup_samples = round(
            float(getattr(args, "warmup", 2.0)) * self.out_sfreq
        )
        if self.capacity_samples < self.window_samples:
            raise ValueError("capacity must hold at least one window.")

        # Acquisition handle and window storage.
        self.acquire = Acquire(
            stream,
            block_samples=int(getattr(args, "block", 50)),
            sfreq=sfreq,
            no_data_timeout=5.0,
            max_lag_seconds=3.0,
        )
        self.buffer = CircularBuffer(
            self.window_samples,
            self.hop_samples,
            self.capacity_samples,
            sfreq=self.out_sfreq,
            n_channels=self.n_channels,
        )

    def ingest(self, data: np.ndarray, timestamps: np.ndarray):
        """Reorder to canonical columns and save the raw block if recording.

        Returns the reordered block with its timestamps, ready for the chain.
        """

        data = self.contract.reorder(data)
        if self.recorder is not None:
            self.recorder.write(data, timestamps)
        return data, timestamps

    def process(self, data: np.ndarray, timestamps: np.ndarray):
        """Run the caller-defined preprocessing in the fixed order."""

        # 1) Units: source -> uV.
        data, timestamps = self.scale(data, timestamps)
        # 2) Quality observes the RAW uV, before filters can hide faults.
        self.quality.feed(data, timestamps)
        # 3) Each caller-defined causal filter, in order.
        for streaming_filter in self.filters:
            data, timestamps = streaming_filter(data, timestamps)
        # 4) Optional rate conversion (also regenerates timestamps).
        if self.resample is not None:
            data, timestamps = self.resample(data, timestamps)
        return data, timestamps

    def gate(self, start_sample: int, window_times: np.ndarray):
        """Decide whether one finished window may reach the consumer.

        Returns:
            ``(accepted, reasons)`` where ``accepted`` is False for warm-up
            windows or windows overlapped by a quality fault, and ``reasons``
            names the faults (empty when accepted).
        """

        reasons = self.quality.reasons(
            float(window_times[0]), float(window_times[-1])
        )
        accepted = start_sample >= self.warmup_samples and not reasons
        return accepted, reasons

    def close(self, status: str = "completed", error: str | None = None) -> None:
        """Finalize the recorder (locks the run and exports the FIF)."""

        if self.recorder is not None:
            self.recorder.close(status=status, error=error)
