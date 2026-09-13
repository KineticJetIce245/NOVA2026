"""Independent result and algorithm contracts; no source or model implementations."""
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


@dataclass(frozen=True)
class AttentionResult:
    timestamp: float
    source: str
    decision: str = 'unavailable'
    correlation_a: Optional[float] = None
    correlation_b: Optional[float] = None
    simulated: bool = False


@dataclass(frozen=True)
class VigilanceResult:
    timestamp: float
    source: str
    value: Optional[float] = None
    metric: str = 'vigilance'
    simulated: bool = False


@dataclass(frozen=True)
class SyncResult:
    timestamp: float
    source: str
    status: str = 'unknown'
    offset_ms: Optional[float] = None
    drift_warning: Optional[bool] = None
    simulated: bool = False
    clock_domain: Optional[str] = None
    max_block_timing_error_seconds: Optional[float] = None
    estimated_audio_drift_ppm: Optional[float] = None


@dataclass(frozen=True)
class EEGDisplayResult:
    timestamp: float
    source: str
    channels: Optional[Sequence[str]] = None
    samples: Optional[Sequence[Sequence[float]]] = None
    sample_rate: Optional[float] = None
    simulated: bool = False


@dataclass(frozen=True)
class SignalQualityResult:
    timestamp: float
    source: str
    artifact: Optional[bool] = None
    bad_channels: Optional[Sequence[str]] = None
    simulated: bool = False


@dataclass(frozen=True)
class GainResult:
    timestamp: float
    source: str
    a_db: Optional[float] = None
    b_db: Optional[float] = None
    simulated: bool = False


class AttentionAlgorithm(Protocol):
    def evaluate(self, eeg: Any, audio: Any) -> AttentionResult: ...


class VigilanceAlgorithm(Protocol):
    def evaluate(self, eeg: Any) -> VigilanceResult: ...


class Synchronization(Protocol):
    def metadata(self, timestamp: float) -> SyncResult: ...


class PublishCallback(Protocol):
    def __call__(self, kind: str, timestamp: float, source: str, payload: dict) -> Any: ...
