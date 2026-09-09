"""Stateful causal filtering, separate from coefficient design."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi


class StreamingFilter(ABC):
    """Define the chunk-processing and reset interface used by the buffer."""

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Filter one data chunk."""

        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter's streaming state."""

        ...


class StreamingSOSFilter(StreamingFilter):
    """Apply one SOS filter continuously across incoming data chunks.

    Args:
        sos: Second-order-section filter coefficients.
        n_channels: Expected number of data channels.
    """

    def __init__(
        self,
        sos: np.ndarray,
        n_channels: int,
    ) -> None:
        """Initialize the streaming filter."""

        self.sos = sos
        self.n_channels = n_channels
        self.zi: np.ndarray | None = None

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Filter one data chunk while preserving state across calls.

        Args:
            data: Data shaped as samples by channels.

        Returns:
            The filtered data with the same shape as the input.
        """

        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")

        if data.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[1]}."
            )

        if data.shape[0] == 0:
            return data.copy()

        if self.zi is None:
            initial_state = sosfilt_zi(self.sos)[:, :, None]
            self.zi = initial_state * data[0][None, None, :]

        filtered_data, self.zi = sosfilt(
            self.sos,
            data,
            axis=0,
            zi=self.zi,
        )

        return filtered_data

    def reset(self) -> None:
        """Clear the saved filter state."""

        self.zi = None
