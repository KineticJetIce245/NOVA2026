"""Application metadata for already-prepared research inputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreparedInput:
    provider_id: str
    task: str
    timestamp: float | None = None
    window_id: str | None = None
    valid: bool = True
    reasons: tuple[str, ...] = ()
    native: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)