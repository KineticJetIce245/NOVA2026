"""StreamingResampler: independent realtime component."""

import numpy as np
import soxr


class StreamingResampler:
    """Resample consecutive chunks while retaining the resampler's history.

    Args:
        sfreq: Input sample rate in Hz.
        rfreq: Output sample rate in Hz.
        n_channels: Number of canonical EEG and auxiliary channels.
    """

    def __init__(
        self, sfreq: float, rfreq: float, n_channels: int, quality: str = "LQ"
    ) -> None:
        """Create a stateful SoXR stream for the requested rates and channel count."""

        self.sfreq = sfreq
        self.rfreq = rfreq
        self.n_channels = n_channels
        self.resampler = soxr.ResampleStream(
            in_rate=sfreq,
            out_rate=rfreq,
            num_channels=n_channels,
            # HQ buffers several seconds at EEG rates; LQ retains anti-aliasing
            # while keeping the default 500 -> 128 Hz path within its lag limit.
            dtype="float64",
            quality=quality,
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Return available output samples; some calls may return no samples."""

        if data.ndim != 2 or data.shape[1] != self.n_channels:
            raise ValueError("Expected data shaped (samples, channels).")

        if not len(data):
            return np.empty((0, self.n_channels), dtype=np.float64)

        return self.resampler.resample_chunk(
            np.ascontiguousarray(data, dtype=np.float64),
            last=False,
        )

    def reset(self) -> None:
        """Clear history without flushing an artificial end-of-stream tail."""

        self.resampler.clear()
