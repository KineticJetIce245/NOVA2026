"""Streaming rate conversion stage backed by SoXR.

The module imports cleanly without ``soxr``; only constructing the stage
requires it. If it is missing, construction raises ``ImportError`` with the
install command.
"""

import math

import numpy as np

# Quality/speed trade-off presets understood by python-soxr.
SOXR_QUALITIES = ("LQ", "MQ", "HQ", "VHQ")


class Resampler:
    """Resample consecutive chunks from ``in_sfreq`` to ``out_sfreq``.

    Args:
        in_sfreq: Source sample rate in Hz.
        out_sfreq: Target sample rate in Hz.
        n_channels: Expected number of data columns.
        quality: SoXR quality mode: ``"LQ"``, ``"MQ"``, ``"HQ"`` or ``"VHQ"``.

    Notes:
        The output row count per call varies (SoXR buffers internally), which
        is normal for a streaming resampler. Output timestamps form a uniform
        grid at ``out_sfreq`` anchored to the first finite input timestamp;
        SoXR's internal processing latency is not compensated, matching the
        dataproc prototype. Call ``reset()`` before a new segment.

    Raises:
        ImportError: If ``soxr`` is not installed.
    """

    def __init__(
        self,
        in_sfreq: float,
        out_sfreq: float,
        n_channels: int,
        quality: str = "LQ",
    ) -> None:
        """Validate geometry and create the stateful SoXR stream."""

        if not math.isfinite(in_sfreq) or in_sfreq <= 0:
            raise ValueError("in_sfreq must be finite and positive.")
        if not math.isfinite(out_sfreq) or out_sfreq <= 0:
            raise ValueError("out_sfreq must be finite and positive.")
        if math.isclose(in_sfreq, out_sfreq):
            raise ValueError("in_sfreq and out_sfreq must differ.")
        if (
            isinstance(n_channels, bool)
            or not isinstance(n_channels, int)
            or n_channels < 1
        ):
            raise ValueError("n_channels must be a positive integer.")
        if quality not in SOXR_QUALITIES:
            raise ValueError(f"quality must be one of {SOXR_QUALITIES}.")

        # Optional dependency: import lazily so the rest of the package works
        # without soxr installed.
        try:
            import soxr
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "The Resampler requires 'soxr'. Install it with: "
                "uv add soxr   (or: pip install soxr)"
            ) from error

        self._in_sfreq = float(in_sfreq)
        self._out_sfreq = float(out_sfreq)
        self._channels = n_channels
        # Public rates so a session can derive its output geometry.
        self.in_sfreq = self._in_sfreq
        self.out_sfreq = self._out_sfreq
        # ResampleStream keeps its own history between resample_chunk calls.
        self._resampler = soxr.ResampleStream(
            in_rate=self._in_sfreq,
            out_rate=self._out_sfreq,
            num_channels=n_channels,
            dtype="float64",
            quality=quality,
        )
        self.reset()

    def __call__(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample one chunk and return it with a new output-rate time grid."""

        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self._channels:
            raise ValueError(
                f"Expected samples by {self._channels} channels, "
                f"received shape {data.shape}."
            )
        if timestamps.ndim != 1 or len(timestamps) != len(data):
            raise ValueError("Each sample must have one timestamp.")
        if data.shape[0] == 0:
            return (
                np.empty((0, self._channels), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        # Remember the first real time so output samples get a time grid.
        if self._anchor is None:
            finite = timestamps[np.isfinite(timestamps)]
            if finite.size:
                self._anchor = float(finite[0])

        # SoXR needs contiguous float64 input; may return 0 rows on early calls.
        output = self._resampler.resample_chunk(
            np.ascontiguousarray(data, dtype=np.float64),
            last=False,
        )
        output = np.asarray(output, dtype=np.float64)

        # Timestamps follow a uniform grid at the output rate.
        if self._anchor is None:
            output_times = np.full(len(output), np.nan)
        else:
            output_times = self._anchor + (
                self._output_index + np.arange(len(output))
            ) / self._out_sfreq
        self._output_index += len(output)
        self.output_samples += len(output)

        return output, output_times

    def reset(self) -> None:
        """Clear internal history without flushing an end-of-stream tail."""

        self._resampler.clear()
        self._anchor: float | None = None
        self._output_index = 0
        self.output_samples = 0
