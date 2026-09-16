"""Result contracts — a test double for the vendored frontend tests.

These dataclasses mirror the field names the teammate's adapters use so the
language-boundary test exercises realistic objects. ``src/nova2026/transport``
owns the real contracts and replaces this fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttentionResult:
    """One attention decision with its two candidate correlations."""

    index: int
    provider: str = "aad"
    decision: str = "A"

    def payload(self) -> dict:
        if self.decision not in ("A", "B", "uncertain", "unavailable"):
            raise ValueError(f"Unknown attention decision: {self.decision!r}.")
        attended = self.decision if self.decision in ("A", "B") else None
        return {"decision": self.decision, "attended": attended, "simulated": True}


@dataclass(frozen=True)
class SyncResult:
    """Timeline status; ``status`` is a word, never a fabricated offset."""

    timestamp: float = 0.0
    status: str = "unknown"

    def payload(self) -> dict:
        return {
            "status": self.status,
            "offset_ms": None,
            "drift_warning": None,
            "timeline": "session_relative",
            "fixed_latency_ms": None,
            "simulated": True,
        }


@dataclass(frozen=True)
class VigilanceResult:
    """A vigilance illustration, declared as simulated."""

    score: float = 0.0
    timestamp: float = 0.0

    def payload(self) -> dict:
        return {"score": float(self.score), "metric": "vigilance", "simulated": True}
