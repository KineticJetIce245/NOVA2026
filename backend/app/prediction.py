"""Provider-neutral, JSON-serializable application prediction results."""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
import json
import math
from typing import Any

STATUSES = frozenset({'ok', 'invalid', 'unavailable', 'error'})
SEMANTIC_TYPES = frozenset({'probability', 'score', 'correlation', 'distance', 'class',
                            'boolean', 'ratio', 'measurement', 'development'})


@dataclass(frozen=True)
class PredictionOutput:
    name: str
    value: Any
    semantic_type: str
    label: str | None = None

    def to_dict(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('output name must be nonempty')
        if self.semantic_type not in SEMANTIC_TYPES:
            raise ValueError('unsupported output semantic type')
        json.dumps(self.value, allow_nan=False)
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError('output label must be a string or None')
        return {'name': self.name, 'value': self.value,
                'semantic_type': self.semantic_type, 'label': self.label}


@dataclass(frozen=True)
class PredictionResult:
    status: str
    provider_id: str
    provider_name: str
    task: str
    outputs: tuple[PredictionOutput, ...] = ()
    timestamp: float | None = None
    window_id: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_version: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError('unsupported prediction status')
        for value in (self.provider_id, self.provider_name, self.task):
            if not isinstance(value, str) or not value:
                raise ValueError('provider and task identifiers are required')
        if any(not isinstance(output, PredictionOutput) for output in self.outputs):
            raise ValueError('outputs must contain PredictionOutput values')
        for value in (self.timestamp, self.window_start, self.window_end):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not math.isfinite(value) or value < 0):
                raise ValueError('times must be finite nonnegative numbers')
        if self.window_start is not None and self.window_end is not None and self.window_end < self.window_start:
            raise ValueError('window_end precedes window_start')
        if any(not isinstance(reason, str) or not reason for reason in self.reasons):
            raise ValueError('reasons must be nonempty strings')
        json.dumps(self.metadata, allow_nan=False)

    def to_dict(self):
        return deepcopy({'status': self.status, 'provider_id': self.provider_id,
            'provider_name': self.provider_name, 'provider_version': self.provider_version,
            'task': self.task, 'timestamp': self.timestamp, 'window_id': self.window_id,
            'window_start': self.window_start, 'window_end': self.window_end,
            'outputs': [output.to_dict() for output in self.outputs],
            'reasons': list(self.reasons), 'metadata': self.metadata})

    @classmethod
    def unavailable(cls, provider_id, provider_name, task, reason):
        return cls('unavailable', provider_id, provider_name, task, reasons=(reason,))