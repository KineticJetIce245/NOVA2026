"""Bounded run counters, persisted with the recording at close time.

The demo and scripts keep lightweight counters while the loop runs and hand
them to ``RunRecorder.close(stats=...)`` so every run leaves a structured
summary (windows delivered, faults recovered, samples repaired, transport
health) instead of only printing to a terminal.
"""


class StreamStats:
    """One serializable counter snapshot per run.

    Attributes:
        blocks: Acquired source blocks.
        samples: Source samples acquired.
        windows: Windows produced by the ring buffer.
        valid: Windows that passed the gate.
        rejected: Windows rejected (warm-up or judge reasons).
        recoveries: Bounded-recovery restarts performed.
        repairs: Source rows repaired by the Repair stage.
        dropped: Damaged source rows dropped before the first finite sample.
        gaps: Source timestamp gaps counted by the acquire handle.
        max_lag: Oldest consumed block age observed (seconds).
        offload_dropped: Windows discarded by the offloader's overflow policy.
        offload_failed: Analysis calls that raised on a worker thread.
        timebase_relocked_samples: Drift the time base absorbed by re-locking,
            in samples. This is the number to score: the instantaneous residual
            is bounded by policy, so it cannot show how far the source's clock
            and the grid disagreed over a session.
        timebase_anchor_rate: Rate the source's own anchors imply, in Hz.
        timebase_relocks: Re-locks performed.
        timebase_large_steps: Suspicious single timestamp steps reported.
    """

    def __init__(self) -> None:
        """Start every counter at zero."""

        self.blocks = 0
        self.samples = 0
        self.windows = 0
        self.valid = 0
        self.rejected = 0
        self.recoveries = 0
        self.repairs = 0
        self.dropped = 0
        self.gaps = 0
        self.max_lag = 0.0
        self.offload_dropped = 0
        self.offload_failed = 0
        self.timebase_relocked_samples = 0.0
        self.timebase_anchor_rate = float("nan")
        self.timebase_relocks = 0
        self.timebase_large_steps = 0

    def to_dict(self) -> dict:
        """Return a serializable snapshot of these counters."""

        return dict(vars(self))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StreamStats(blocks={self.blocks}, samples={self.samples}, "
            f"windows={self.windows}, valid={self.valid}, "
            f"rejected={self.rejected}, recoveries={self.recoveries})"
        )
