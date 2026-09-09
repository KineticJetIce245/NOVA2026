"""Real-time EEG streaming: decoupled acquisition, preprocessing and output."""

# Public entry points, re-exported so callers use one import line.
from .acquire import Acquire             # fixed-size blocks from the LSL inlet
from .channels import ChannelContract    # validate + reorder source channels
from .circular_buffer import CircularBuffer   # ring storage -> overlapping windows
from .offload import TaskOffloader       # run per-window analysis on workers
from .preprocess.resample import Resampler   # stateful 500->128 Hz (SoXR)
from .recording import RunRecorder, RunSpec  # per-run SQLite recording + identity
from .source import validate_source      # check a connected inlet's metadata
from .spatial import SpatialOperator, fit_ssp, processing_contract
from .window import EEGWindow            # one window + verdict, for consumers

__all__ = [
    "Acquire",
    "ChannelContract",
    "CircularBuffer",
    "TaskOffloader",
    "Resampler",
    "RunRecorder",
    "RunSpec",
    "SpatialOperator",
    "EEGWindow",
    "fit_ssp",
    "processing_contract",
    "validate_source",
]
