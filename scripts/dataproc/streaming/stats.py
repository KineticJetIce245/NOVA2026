"""Streaming run counters."""


class StreamStats:
    """Bounded counters for a run; sample arrays are not retained."""

    def __init__(self) -> None:
        """Start all counters at zero for a new run."""

        self.chunks: int = 0
        self.input_samples: int = 0
        self.output_samples: int = 0
        self.windows: int = 0
        self.valid_windows: int = 0
        self.invalid_windows: int = 0
        self.max_window_lag: float = 0.0
        self.max_queue_lag: float = 0.0
        self.max_queue_size: int = 0
        self.recoveries = 0
        self.discarded_chunks = 0
        self.interpolated_samples = 0
        self.pending_interpolation_samples = 0

    def to_dict(self) -> dict:
        """Return a serializable snapshot of these counters."""

        return dict(vars(self))
