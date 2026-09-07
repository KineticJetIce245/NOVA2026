"""Basic raw-signal checks, independent of threading and filtering."""

from collections import deque

import numpy as np

from .config import StreamConfig


class SignalQuality:
    """Track raw EEG faults and their filter recovery intervals.

    Args:
        config: EEG channel count, raw voltage limits, and recovery duration.

    Notes:
        Checks run before filtering so a high-pass filter cannot hide a stuck
        electrode or a large DC input. EOG is excluded from classifier checks.
    """

    def __init__(self, config: StreamConfig) -> None:
        """Initialize empty raw-quality history for the configured EEG channels."""

        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear intervals and per-channel flatline history at a run boundary."""

        self.intervals: deque[tuple[float, float, str]] = deque()
        self.previous: np.ndarray | None = None
        self.initial_level: np.ndarray | None = None
        self.unchanged = np.zeros(len(self.config.eeg_channels), dtype=int)

    def feed(self, data_uv: np.ndarray, timestamps: np.ndarray) -> None:
        """Inspect one canonical raw chunk and remember invalid time intervals.

        Args:
            data_uv: Samples by channels, with EEG columns first.
            timestamps: Original LSL timestamps for the same samples.
        """

        config = self.config
        eeg = data_uv[:, : len(config.eeg_channels)]

        if self.initial_level is None:
            self.initial_level = eeg[0].copy()
        # DC-coupled amplifiers can have large electrode offsets. Check signal
        # excursions from the run's initial level separately from absolute rails.
        excursions = np.abs(eeg - self.initial_level)
        faults = {
            "amplitude": np.any(excursions > config.amplitude_limit_uv, axis=1),
            "saturation": np.any(np.abs(eeg) >= config.saturation_limit_uv, axis=1),
        }
        # Carry the unchanged-sample count across acquisition chunk boundaries.
        previous = eeg[0] if self.previous is None else self.previous
        differences = np.abs(np.diff(eeg, axis=0, prepend=previous[None, :]))
        changed = differences > config.flatline_tolerance_uv
        indices = np.arange(1, len(eeg) + 1)[:, None]
        last_change = np.where(changed, indices, -self.unchanged[None, :])
        last_change = np.maximum.accumulate(last_change, axis=0)
        run_lengths = indices - last_change
        self.unchanged = run_lengths[-1].copy()
        self.previous = eeg[-1].copy()
        faults["flatline"] = np.any(
            run_lengths >= config.flatline_seconds * config.input_sfreq, axis=1
        )

        for reason, mask in faults.items():
            # Keep separate fault intervals instead of marking an entire chunk
            # between two spikes. Quality must not depend on chunk boundaries.
            edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            for left, right in zip(starts, stops):
                start = float(timestamps[left])
                if reason == "flatline":
                    start -= config.flatline_seconds
                end = float(timestamps[right - 1]) + config.warmup_seconds
                self.intervals.append((start, end, reason))

    def reasons(self, start: float, end: float) -> tuple[str, ...]:
        """Return faults overlapping a window and discard expired history."""

        # Intervals can have different starts (flatlines include a lookback).
        self.intervals = deque(item for item in self.intervals if item[1] >= start)

        return tuple(
            sorted(
                {
                    reason
                    for left, right, reason in self.intervals
                    if left <= end and right >= start
                }
            )
        )
