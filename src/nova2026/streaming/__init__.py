"""Real-time EEG streaming: decoupled acquisition, preprocessing and output."""

# Public entry points, re-exported so callers use one import line.
from .acquire import Acquire             # fixed-size blocks from the LSL inlet
from .circular_buffer import CircularBuffer   # ring storage -> overlapping windows
from .offload import (                    # per-window analysis on worker threads
    TaskOffloader,
    dummy_offloader,
)
from .preflight import (
    ChannelContract,  # validate + reorder source channels
    prepare,  # pre-flight check -> the run's channel contract
    resolve_outlet,  # confirm an outlet exists before connecting (B1)
    validate_source,  # check a connected inlet's metadata
)
from .preprocess.repair import (  # damage Repair cannot fix, and its grid rule
    UnrepairableError,
    grid_tolerance_samples,
    grid_tolerance_seconds,
)
from .preprocess.resample import (  # stateful 500->128 Hz (SoXR)
    Resampler,
    ResamplerQualityWarning,
    select_quality,
)
from .recording import RunRecorder, RunSpec  # per-run SQLite recording + identity
from .recovery import Recovery          # bounded recovery: reset or stop (A2)
from .spatial import (
    SpatialOperator,
    cut_epochs,
    fit_ssp,
    processing_contract,
)
from .stats import StreamStats          # run counters, persisted at close (E)
from .timebase import (                  # one owner for the LSL timeline
    GridPolicy,
    TimeBase,
    TimeBaseEvent,
    TimeBaseState,
)
from .window import EEGWindow            # one window + verdict, for consumers

__all__ = [
    "Acquire",
    "ChannelContract",
    "CircularBuffer",
    "TaskOffloader",
    "dummy_offloader",
    "Resampler",
    "ResamplerQualityWarning",
    "RunRecorder",
    "RunSpec",
    "SpatialOperator",
    "Recovery",
    "StreamStats",
    "UnrepairableError",
    "cut_epochs",
    "EEGWindow",
    "GridPolicy",
    "TimeBase",
    "TimeBaseEvent",
    "TimeBaseState",
    "fit_ssp",
    "prepare",
    "processing_contract",
    "grid_tolerance_seconds",
    "grid_tolerance_samples",
    "resolve_outlet",
    "select_quality",
    "validate_source",
]
