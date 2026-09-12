"""Reusable processing and bounded recovery, independent of threads and LSL."""

import numpy as np

from .acquisition_queue import StreamDiscontinuityError
from .buffer_factory import create_buffer
from .config import StreamConfig
from .interpolator import SampleInterpolator
from .spatial_operator import SpatialOperator
from .window import EEGWindow


class RunProcessor:
    """Process canonical chunks, reset on brief faults, and apply a fixed operator.

    Args:
        config: Rates, processing, and bounded recovery policy.
        operator: Optional operator shared by baseline, training, and trial runs.

    Notes:
        Interpolation repairs short bounded faults before processing. Recovery
        discards damage that cannot be repaired. Shape
        errors, long gaps, repeated faults, and persistent bad EEG remain fatal.
        Each restart gets a new segment number and repeats warm-up.
    """

    def __init__(
        self, config: StreamConfig, operator: SpatialOperator | None = None
    ) -> None:
        """Build fresh processing state and validate an optional saved operator."""

        self.config = config.updated()
        config = self.config
        self.operator = operator

        if operator is not None:
            operator.validate_config(config)

        self.buffer = create_buffer(
            config, (config.source_unit_exponent,) * len(config.channels)
        )
        self.segment = 0
        self.recoveries = 0
        self.discarded_chunks = 0
        self.output_samples = 0
        self.last_timestamp: float | None = None
        self.events: list[dict] = []
        self.bad_since: float | None = None
        self.interpolator = SampleInterpolator(config)
        self.repair_history: list[dict] = []
        self.interpolated_samples = 0

    def feed(self, item: tuple[np.ndarray, np.ndarray]) -> list[EEGWindow]:
        """Repair bounded faults, process settled chunks, and annotate output windows.

        Args:
            item: Canonical source-unit samples and original timestamps.

        Returns:
            Available windows, including rejected windows. The caller must gate
            on valid. A damaged tail may wait for its finite right endpoint.
        """

        self.events.clear()
        chunks = self.interpolator.feed(item)
        self.events.extend(self.interpolator.events)
        self.repair_history.extend(self.interpolator.events)
        self.interpolated_samples += sum(
            event["samples"] for event in self.interpolator.events
        )
        windows = []

        for chunk in chunks:
            windows.extend(self._feed_chunk(chunk))

        for window in windows:
            start = float(window.timestamps[0])
            end = float(window.timestamps[-1])
            # Mark the settling interval too: repaired input affects filter history.
            affected = [
                event
                for event in self.repair_history
                if event["timestamp"] <= end
                and event["end"] + self.config.warmup_seconds >= start
            ]
            window.interpolated = bool(affected)
            window.interpolated_samples = sum(
                event["samples"] for event in affected if event["end"] >= start
            )
            window.available_at = max(
                [end, window.available_at or end]
                + [event["available_at"] for event in affected]
            )
            allowed = (
                self.config.interpolation_max_fraction
                * self.config.window_seconds
                * self.config.input_sfreq
            )
            if window.interpolated_samples > allowed:
                window.valid = False
                window.reasons = (*window.reasons, "interpolation_limit")

            faults = set(window.reasons) - {"warmup"}
            if faults:
                if self.bad_since is None:
                    self.bad_since = start
                if end - self.bad_since > self.config.persistent_fault_seconds:
                    raise RuntimeError(
                        "EEG quality faults persisted beyond the allowed duration."
                    )
            else:
                self.bad_since = None

        if windows:
            oldest = float(windows[-1].timestamps[0]) - self.config.warmup_seconds
            self.repair_history = [
                event for event in self.repair_history if event["end"] >= oldest
            ]
        return windows

    def _feed_chunk(self, item: tuple[np.ndarray, np.ndarray]) -> list[EEGWindow]:
        """Return windows from a chunk, or no windows after discarding damage."""

        data, timestamps = item

        if data.ndim != 2 or data.shape[1] != len(self.config.channels):
            raise ValueError("Input channel dimensions changed during the run.")

        if timestamps.ndim != 1 or len(data) != len(timestamps) or not len(data):
            raise ValueError("Each non-empty input sample must have a timestamp.")

        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(timestamps)):
            self._recover("nonfinite_chunk", timestamps, discard=True)
            return []

        tolerance = self.config.timestamp_tolerance + 1e-8
        interval = 1 / self.config.input_sfreq
        intervals = np.diff(timestamps)

        if np.any(np.abs(intervals - interval) > tolerance):
            if np.any(intervals > self.config.recovery_max_gap):
                raise StreamDiscontinuityError(
                    "An internal gap exceeds the recovery limit."
                )
            self._recover("irregular_chunk", timestamps, discard=True)
            return []

        if self.last_timestamp is not None:
            gap = float(timestamps[0]) - self.last_timestamp - interval
            if gap < -tolerance:
                self._recover("overlapping_chunk", timestamps, discard=True)
                return []
            if gap > self.config.recovery_max_gap:
                raise StreamDiscontinuityError(
                    "The source gap exceeds the recovery limit."
                )
            if gap > tolerance:
                self._recover("missing_samples", timestamps, discard=False)

        before = self.buffer.total_written
        try:
            self.buffer.feed(item)
        except StreamDiscontinuityError:
            self._recover("clock_drift", timestamps, discard=False)
            before = 0
            self.buffer.feed(item)

        self.output_samples += self.buffer.total_written - before
        self.last_timestamp = float(timestamps[-1])
        windows = self.buffer.spit()

        for window in windows:
            window.segment = self.segment
            # Conservative source availability includes the resampler input batch.
            window.available_at = max(
                float(timestamps[-1]), float(window.timestamps[-1])
            )

        if self.operator is not None:
            windows = [self.operator.apply_window(window) for window in windows]

        return windows

    def _recover(self, reason: str, timestamps: np.ndarray, discard: bool) -> None:
        """Reset all processing history and record one bounded recovery decision."""

        self.recoveries += 1

        if self.recoveries > self.config.recovery_max_events:
            raise RuntimeError(
                "Too many data faults; the run requires operator attention."
            )

        self.segment += 1
        self.discarded_chunks += int(discard)
        self.buffer.reset()
        self.bad_since = None
        finite = timestamps[np.isfinite(timestamps)]
        timestamp = float(finite[0]) if len(finite) else self.last_timestamp
        self.events.append(
            {"reason": reason, "segment": self.segment, "timestamp": timestamp}
        )

        if (
            discard
            and len(finite) == len(timestamps)
            and np.all(np.diff(timestamps) > 0)
        ):
            last = float(timestamps[-1])
            if self.last_timestamp is None or last > self.last_timestamp:
                self.last_timestamp = last
