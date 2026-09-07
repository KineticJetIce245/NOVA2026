"""Explicit array shapes for already-ordered EEG windows."""

import numpy as np

from scripts.dataproc.streaming.window import EEGWindow


class WindowAdapter:
    """Validate EEG order and expose explicit batch layouts without reordering.

    Args:
        expected_channels: EEG order already selected at acquisition.
        window_samples: Required samples per window.
    """

    def __init__(self, expected_channels: tuple[str, ...], window_samples: int) -> None:
        """Store an explicit channel and sample contract."""

        self.expected_channels = tuple(expected_channels)
        self.window_samples = window_samples
        if not self.expected_channels or len(set(self.expected_channels)) != len(
            self.expected_channels
        ):
            raise ValueError("Expected channels must be non-empty and unique.")
        if window_samples < 2:
            raise ValueError("A covariance window needs at least two samples.")

    def transform(self, window: EEGWindow) -> np.ndarray:
        """Return one valid EEGWindow as (1, samples, channels), in microvolts."""

        if not window.valid:
            raise ValueError("Rejected windows cannot enter feature extraction.")
        if window.channel_names != self.expected_channels:
            raise ValueError("Window channels differ from the expected order.")
        return self.batch(window.data)

    def batch(self, data: np.ndarray) -> np.ndarray:
        """Validate caller-supplied (samples, channels) or (windows, samples, channels).

        Arrays carry no labels: the caller is responsible for their channel order
        and for applying the same live preprocessing before this method.
        """

        values = np.asarray(data, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        expected = (self.window_samples, len(self.expected_channels))
        if values.ndim != 3 or values.shape[1:] != expected or not len(values):
            raise ValueError(f"Expected (windows, {expected[0]}, {expected[1]}).")
        if not np.all(np.isfinite(values)):
            raise ValueError("EEG arrays must be finite.")
        return np.ascontiguousarray(values)

    def channels_first(
        self, data: np.ndarray, convolution_axis: bool = False
    ) -> np.ndarray:
        """Return (windows, channels, samples), optionally (windows, 1, channels, samples)."""

        values = self.batch(data).transpose(0, 2, 1)
        if convolution_axis:
            values = values[:, None, :, :]
        return np.ascontiguousarray(values)
