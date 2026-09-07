"""CircularBuffer: independent realtime component."""

from collections.abc import Sequence

import numpy as np

from .acquisition_queue import StreamDiscontinuityError
from .config import StreamConfig
from .quality import SignalQuality
from .streaming_filter import StreamingFilter
from .streaming_resampler import StreamingResampler
from .voltage_converter import VoltageConverter
from .window import EEGWindow


class CircularBuffer:
    """Keep the existing ring-buffer flow and emit timestamped windows.

    Args:
        config: Rates, channel order, window dimensions, and validity settings.
        voltage_converter: Converter matching canonical queue order.
        streaming_resampler: Stateful resampler for all selected channels.
        filters: Causal filters applied in list order at the input sample rate.

    Notes:
        This class belongs to the processing thread only. It can also be used
        directly for offline chunk replay. It never starts or manages a thread.
    """

    def __init__(
        self,
        config: StreamConfig,
        voltage_converter: VoltageConverter,
        streaming_resampler: StreamingResampler,
        filters: Sequence[StreamingFilter],
    ) -> None:
        """Allocate ring storage and fresh filter, quality, and timing state."""

        self.config = config
        self.voltage_converter = voltage_converter
        self.streaming_resampler = streaming_resampler
        self.filters = filters
        self.n_channels = len(config.channels)
        self.window_size = config.sample_count(config.window_seconds)
        self.step_size = config.sample_count(config.step_seconds)
        self.buffer_size = config.sample_count(config.buffer_seconds)
        self.buffer = np.empty((self.buffer_size, self.n_channels), dtype=np.float64)
        self.quality = SignalQuality(config)
        self.reset()

    def reset(self) -> None:
        """Reset signal history, timing, and pending windows between runs."""

        self.write_index = 0
        self.total_written = 0
        self.total_input = 0
        self.next_window_end = self.window_size
        self.first_timestamp: float | None = None
        self.ready_windows: list[EEGWindow] = []
        self.quality.reset()
        self.streaming_resampler.reset()

        for streaming_filter in self.filters:
            streaming_filter.reset()

    def feed(self, queue_item: tuple[np.ndarray, np.ndarray]) -> None:
        """Process one canonical acquisition chunk and prepare complete windows.

        Args:
            queue_item: Samples by channels and their original LSL timestamps.

        Raises:
            StreamDiscontinuityError: If input timing no longer fits the run grid.
            RuntimeError: If windows are not consumed before the ready limit.
        """

        data, timestamps = queue_item

        if data.ndim != 2 or data.shape[1] != self.n_channels:
            raise ValueError("Expected data shaped (samples, configured channels).")

        if timestamps.ndim != 1 or len(timestamps) != len(data) or not len(data):
            raise ValueError("Each non-empty input sample must have a timestamp.")

        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("Data and timestamps must be finite.")

        if self.first_timestamp is None:
            self.first_timestamp = float(timestamps[0])
        expected = (
            self.first_timestamp
            + (self.total_input + np.arange(len(data))) / self.config.input_sfreq
        )

        if np.any(
            np.abs(timestamps - expected) > self.config.timestamp_tolerance + 1e-8
        ):
            raise StreamDiscontinuityError(
                "Input timing left the run grid. Stop and initialize a new run."
            )
        self.total_input += len(data)
        data = self.voltage_converter.convert(data)
        self.quality.feed(data, timestamps)

        for streaming_filter in self.filters:
            data = streaming_filter(data)
        data = self.streaming_resampler(data)

        chunk_position = 0
        while chunk_position < data.shape[0]:
            samples_until_window = self.next_window_end - self.total_written
            samples_to_write = min(samples_until_window, data.shape[0] - chunk_position)
            self._append(data[chunk_position : chunk_position + samples_to_write])
            chunk_position += samples_to_write
            if self.total_written == self.next_window_end:
                if len(self.ready_windows) >= self.config.max_ready_windows:
                    raise RuntimeError(
                        "Pending windows are full; call spit() after feed()."
                    )
                self.ready_windows.append(self._make_window())
                self.next_window_end += self.step_size

    def _append(self, data: np.ndarray) -> None:
        """Append samples, wrapping at the end of the existing ring buffer."""

        n_samples = data.shape[0]
        first_part = min(n_samples, self.buffer_size - self.write_index)
        self.buffer[self.write_index : self.write_index + first_part] = data[
            :first_part
        ]
        remaining = n_samples - first_part

        if remaining > 0:
            self.buffer[:remaining] = data[first_part:]
        self.write_index = (self.write_index + n_samples) % self.buffer_size
        self.total_written += n_samples

    def _get_latest_window(self) -> np.ndarray:
        """Copy the latest complete window, including a wrapped window."""

        start = (self.write_index - self.window_size) % self.buffer_size
        end = start + self.window_size

        if end <= self.buffer_size:
            return self.buffer[start:end].copy()

        return np.concatenate(
            (self.buffer[start:], self.buffer[: end - self.buffer_size]), axis=0
        )

    def _make_window(self) -> EEGWindow:
        """Attach source timing and reject warm-up or contaminated windows."""

        if self.first_timestamp is None:
            raise RuntimeError("Cannot emit a window before receiving samples.")

        start_sample = self.total_written - self.window_size
        timestamps = (
            self.first_timestamp
            + (start_sample + np.arange(self.window_size)) / self.config.output_sfreq
        )
        reasons = list(
            self.quality.reasons(float(timestamps[0]), float(timestamps[-1]))
        )

        if start_sample < self.config.warmup_seconds * self.config.output_sfreq:
            reasons.append("warmup")
        data = self._get_latest_window()

        if not np.all(np.isfinite(data)):
            reasons.append("nonfinite")
        eeg_count = len(self.config.eeg_channels)

        return EEGWindow(
            data=data[:, :eeg_count],
            eog=data[:, eeg_count:],
            timestamps=timestamps,
            valid=not reasons,
            reasons=tuple(reasons),
            start_sample=start_sample,
            channel_names=self.config.eeg_channels,
            contract=self.config.window_contract(),
        )

    def spit(self) -> list[EEGWindow]:
        """Return pending windows and clear the list; the runner gates validity."""

        windows = self.ready_windows.copy()
        self.ready_windows.clear()

        return windows
