"""Physical source contracts only; implementations must honor bounded reads/stops."""
from typing import Protocol, Any


class Source(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout: float) -> Any: ...


class EEGSource(Source, Protocol):
    pass


class AudioSource(Source, Protocol):
    pass
