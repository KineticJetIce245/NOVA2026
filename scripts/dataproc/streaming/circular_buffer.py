import numpy as np
from channel_selection_contract import ChannelSelectionContract
from filters import StreamingFilter
from mne_lsl.stream import StreamLSL

import soxr
from abc import ABC, abstractmethod
from scripts.dataproc.streaming import channel_selection_contract


DESIRED_EXPONENT = -6


class UnitConverter(ABC):
    stream: StreamLSL
    channel_selection_contract: ChannelSelectionContract

    @abstractmethod
    def create_converter(self) -> np.ndarray: ...

    @abstractmethod
    def convert(self, data: np.ndarray) -> np.ndarray: ...


class VoltageConverter(UnitConverter):
    def __init__(
        self,
        stream: StreamLSL,
        channel_selection_contract: ChannelSelectionContract,
        desired_exponent: int = DESIRED_EXPONENT,
    ) -> None:
        self.stream: StreamLSL = stream
        self.channel_selection_contract: ChannelSelectionContract = (
            channel_selection_contract
        )
        self.desired_exponent: int = desired_exponent
        self.converter: np.ndarray | None = None

    def create_converter(self) -> np.ndarray:
        # Returns an np.ndarray of exponents that convert to the desired units
        # in the order of the selected channels.
        sinfo = self.stream.sinfo

        if sinfo is None:
            raise RuntimeError("The stream is not connected.")

        raw_units = sinfo.get_channel_units()

        if raw_units is None or any(unit is None for unit in raw_units):
            raise RuntimeError("The LSL stream does not declare every channel's unit.")

        expected_channels = self.channel_selection_contract.expected_channel_names

        unit_info = self.stream.get_channel_units(
            picks=expected_channels,
        )

        source_exponents = np.asarray(
            [int(unit_multiplier) for _, unit_multiplier in unit_info],
            dtype=int,
        )

        ordered_source_exponents = self.channel_selection_contract.apply_contract(
            source_exponents
        )

        conversion_values = np.power(
            10.0, ordered_source_exponents - self.desired_exponent
        )

        self.converter = conversion_values

        return conversion_values

    def convert(self, data: np.ndarray) -> np.ndarray:
        # Formula: x_converted = x_original * 10^(original_exponent - desired_exponent)
        if self.converter is None:
            raise RuntimeError("The voltage converter has not been created.")
        return data * self.converter[None, :]


class StreamingResampler:
    def __init__(
        self,
        sfreq: float,
        rfreq: float,
        n_channels: int,
    ) -> None:
        self.sfreq = sfreq
        self.rfreq = rfreq
        self.n_channels = n_channels

        self.resampler = soxr.ResampleStream(
            in_rate=sfreq,
            out_rate=rfreq,
            num_channels=n_channels,
            dtype="float64",
            quality="HQ",
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")

        if data.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[1]}."
            )

        data = np.ascontiguousarray(data, dtype=np.float64)

        return self.resampler.resample_chunk(
            data,
            last=False,
        )

    def reset(self) -> None:
        self.resampler.clear()


class CircularBuffer:
    def __init__(
        self,
        voltage_converter: VoltageConverter,
        streaming_resampler: StreamingResampler,
        filters: list[StreamingFilter],
        n_channels: int,
        window_size: int,
        step_size: int,
        buffer_size: int,
    ) -> None:
        if buffer_size < window_size:
            raise ValueError("Buffer size must be at least the window size.")

        if not 0 < step_size <= window_size:
            raise ValueError("Step size must be between 1 and window size.")

        self.voltage_converter = voltage_converter
        self.streaming_resampler = streaming_resampler
        self.filters = filters
        self.n_channels = n_channels
        self.window_size = window_size
        self.step_size = step_size
        self.buffer_size = buffer_size

        self.buffer = np.empty(
            (buffer_size, n_channels),
            dtype=np.float64,
        )

        self.write_index = 0
        self.total_written = 0
        self.next_window_end = window_size
        self.ready_windows: list[np.ndarray] = []

    def feed(
        self,
        queue_item: tuple[np.ndarray, np.ndarray],
    ) -> None:
        data, _timestamps = queue_item

        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")

        if data.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[1]}."
            )

        data = self.voltage_converter.convert(data)

        for streaming_filter in self.filters:
            data = streaming_filter(data)

        data = self.streaming_resampler(data)

        chunk_position = 0

        while chunk_position < data.shape[0]:
            samples_until_window = self.next_window_end - self.total_written
            samples_to_write = min(
                samples_until_window,
                data.shape[0] - chunk_position,
            )

            self._append(data[chunk_position : chunk_position + samples_to_write])

            chunk_position += samples_to_write

            if self.total_written == self.next_window_end:
                self.ready_windows.append(self._get_latest_window())
                self.next_window_end += self.step_size

    def _append(self, data: np.ndarray) -> None:
        n_samples = data.shape[0]
        first_part = min(
            n_samples,
            self.buffer_size - self.write_index,
        )

        self.buffer[self.write_index : self.write_index + first_part] = data[
            :first_part
        ]

        remaining = n_samples - first_part

        if remaining > 0:
            self.buffer[:remaining] = data[first_part:]

        self.write_index = (self.write_index + n_samples) % self.buffer_size
        self.total_written += n_samples

    def _get_latest_window(self) -> np.ndarray:
        start = (self.write_index - self.window_size) % self.buffer_size
        end = start + self.window_size

        if end <= self.buffer_size:
            return self.buffer[start:end].copy()

        return np.concatenate(
            (
                self.buffer[start:],
                self.buffer[: end - self.buffer_size],
            ),
            axis=0,
        )

    def spit(self) -> list[np.ndarray]:
        windows = self.ready_windows.copy()
        self.ready_windows.clear()

        return windows
