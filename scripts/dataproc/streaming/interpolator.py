"""Bounded linear interpolation before causal signal processing."""

import numpy as np

from .config import StreamConfig


class SampleInterpolator:
    """Repair short runs of missing values using two finite endpoints.

    Args:
        config: Source rate, timing tolerance, and maximum repair duration.

    Notes:
        Trailing damage waits for a right endpoint, at most the configured
        number of source samples. Unrepairable rows are returned unchanged for
        RunProcessor to reject. No extrapolation or timestamp guessing is used.
    """

    def __init__(self, config: StreamConfig) -> None:
        """Store limits and start with no retained endpoint or pending rows."""

        self.config = config.updated()
        self.limit = int(
            np.floor(config.interpolation_max_seconds * config.input_sfreq)
        )
        self.reset()

    def reset(self) -> None:
        """Discard pending rows and history without inventing a closing endpoint."""

        self.left: np.ndarray | None = None
        self.left_time: float | None = None
        self.previous_time: float | None = None
        self.pending: list[tuple[np.ndarray, float]] = []
        self.blocked = False
        self.events: list[dict] = []

    def feed(
        self, item: tuple[np.ndarray, np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return settled good/bad chunks in order; retain only a short bad tail.

        Args:
            item: Canonical samples in source units and original timestamps.

        Returns:
            Chunks split at good/bad boundaries. events describes each repair.
            Empty output means a short trailing fault is awaiting its endpoint.
        """

        self.events = []
        data, timestamps = item
        if data.ndim != 2 or data.shape[1] != len(self.config.channels):
            raise ValueError("Interpolation needs canonical samples by channels.")
        if timestamps.ndim != 1 or len(timestamps) != len(data) or not len(data):
            raise ValueError("Each non-empty sample needs one timestamp.")
        if not self.limit:
            return [item]

        emitted: list[tuple[np.ndarray, float]] = []
        interval = 1 / self.config.input_sfreq
        tolerance = self.config.timestamp_tolerance + 1e-8

        continuous = (
            self.previous_time is None
            or abs(float(timestamps[0]) - self.previous_time - interval) <= tolerance
        )
        if (
            not self.pending
            and continuous
            and np.all(np.isfinite(data))
            and np.all(np.isfinite(timestamps))
            and np.all(np.abs(np.diff(timestamps) - interval) <= tolerance)
        ):
            self.left = data[-1].copy()
            self.left_time = float(timestamps[-1])
            self.previous_time = self.left_time
            self.blocked = False
            return [item]

        for row, timestamp in zip(data, timestamps):
            timestamp = float(timestamp)
            if not np.isfinite(timestamp):
                emitted.extend(self.pending)
                repairs = self.events
                self.reset()
                self.events = repairs
                emitted.append((row.copy(), timestamp))
                continue

            if self.previous_time is not None:
                steps = (timestamp - self.previous_time) / interval
                missing = round(steps) - 1
                on_grid = abs(steps - round(steps)) * interval <= tolerance

                if on_grid and 0 < missing <= self.limit and self.left is not None:
                    for index in range(missing):
                        gap_time = self.previous_time + (index + 1) * interval
                        self.pending.append((np.full_like(row, np.nan), gap_time))
                elif abs(steps - 1) * interval > tolerance:
                    emitted.extend(self.pending)
                    self.pending = []
                    self.left = None
                    self.blocked = False

            self.previous_time = timestamp
            finite = bool(np.all(np.isfinite(row)))

            if not finite:
                if self.left is None or self.blocked:
                    emitted.append((row.copy(), timestamp))
                else:
                    self.pending.append((row.copy(), timestamp))

                if len(self.pending) > self.limit:
                    emitted.extend(self.pending)
                    self.pending = []
                    self.left = None
                    self.blocked = True
                continue

            if self.pending:
                if len(self.pending) <= self.limit and self._safe_endpoints(row):
                    assert self.left is not None and self.left_time is not None
                    for damaged, time in self.pending:
                        weight = (time - self.left_time) / (timestamp - self.left_time)
                        estimate = self.left + weight * (row - self.left)
                        emitted.append(
                            (np.where(np.isfinite(damaged), damaged, estimate), time)
                        )

                    self.events.append(
                        {
                            "kind": "interpolation",
                            "timestamp": self.pending[0][1],
                            "end": self.pending[-1][1],
                            "samples": len(self.pending),
                            "available_at": timestamp,
                        }
                    )
                else:
                    emitted.extend(self.pending)
                self.pending = []

            emitted.append((row.copy(), timestamp))
            self.left = row.copy()
            self.left_time = timestamp
            self.blocked = False

        if not emitted:
            return []

        values = np.stack([row for row, _ in emitted])
        times = np.asarray([time for _, time in emitted])
        good = np.all(np.isfinite(values), axis=1) & np.isfinite(times)
        boundaries = np.flatnonzero(good[1:] != good[:-1]) + 1
        return list(zip(np.split(values, boundaries), np.split(times, boundaries)))

    def _safe_endpoints(self, right: np.ndarray) -> bool:
        """Reject interpolation between saturated or widely separated EEG endpoints."""

        if self.left is None or self.left_time is None:
            return False
        count = len(self.config.eeg_channels)
        scale = 10.0 ** (self.config.source_unit_exponent + 6)
        endpoints = np.stack((self.left[:count], right[:count])) * scale
        return bool(
            np.all(np.abs(endpoints) < self.config.saturation_limit_uv)
            and np.all(
                np.abs(endpoints[1] - endpoints[0]) <= self.config.amplitude_limit_uv
            )
        )
