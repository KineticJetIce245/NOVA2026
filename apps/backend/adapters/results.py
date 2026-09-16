"""Result adapters — a test double for the vendored frontend tests.

``ResultAdapter`` turns a typed result object into a published packet by asking
the object for its payload. ``src/nova2026/transport`` owns the real adapters;
this exists so the cross-language fixture test drives the same call shape.
"""

from __future__ import annotations

from typing import Callable

from ..app.protocol import make_packet


class ResultAdapter:
    """Publish typed results through a caller-supplied publication callback."""

    def __init__(self, publish: Callable[[str, float, str, dict], dict], source: str = "adapter") -> None:
        if not callable(publish):
            raise ValueError("ResultAdapter needs a callable publisher.")
        if not isinstance(source, str) or not source:
            raise ValueError("ResultAdapter needs a nonempty source name.")
        self._publish = publish
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    def publish(self, result, *, packet_type: str | None = None, timestamp: float | None = None) -> dict:
        """Publish one result whose ``payload()`` returns its packet payload."""

        payload_method = getattr(result, "payload", None)
        if not callable(payload_method):
            raise TypeError("A result must expose payload().")
        stamp = getattr(result, "timestamp", 0.0) if timestamp is None else timestamp
        kind = packet_type or _TYPE_BY_CLASS.get(type(result).__name__, "result")
        return self._publish(kind, float(stamp), self._source, payload_method())


_TYPE_BY_CLASS = {
    "AttentionResult": "attention",
    "SyncResult": "sync",
    "VigilanceResult": "vigilance",
    "LegacyLapseResult": "vigilance",
}

__all__ = ["ResultAdapter", "make_packet"]
