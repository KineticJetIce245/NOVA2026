"""Legacy-shaped helper results — a test double for the vendored frontend tests.

The teammate's ``adapters/legacy.py`` bridges the older flat payload shape into
the packet publisher. Only the one helper the frontend test uses is reproduced
here; ``src/nova2026/transport`` owns the real bridging code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyLapseResult:
    """A lapse-risk illustration under the legacy field name ``lapse_score``."""

    lapse_score: float
    timestamp: float = 0.0
    metric: str = "lapse_probability"

    def payload(self) -> dict:
        if not 0.0 <= float(self.lapse_score) <= 1.0:
            raise ValueError("A lapse score is a probability between zero and one.")
        return {
            "metric": self.metric,
            "lapse_score": float(self.lapse_score),
            "simulated": True,
        }


def lapse_result(value, *, timestamp=0.0):
    """Build the legacy lapse result the fixture publisher publishes."""

    return LegacyLapseResult(float(value), float(timestamp))
