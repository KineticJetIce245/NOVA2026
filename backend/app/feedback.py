"""Development-only policy. Numeric thresholds have NO scientific interpretation."""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
import math
from typing import Protocol
from .prediction import PredictionResult, PredictionOutput


@dataclass(frozen=True)
class FeedbackAction:
    status: str
    action_type: str
    message: str
    severity: str
    source_provider: str
    source_task: str
    timestamp: float
    reason: str
    metadata: dict = field(default_factory=dict)
    simulated: bool = True

    def __post_init__(self):
        if self.status not in ('none', 'active', 'suppressed'):
            raise ValueError('invalid feedback status')
        if self.action_type not in ('visual', 'audio', 'haptic', 'control', 'none'):
            raise ValueError('invalid action type')
        if self.severity not in ('info', 'warning', 'critical'):
            raise ValueError('invalid severity')
        if type(self.timestamp) not in (int, float) or not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('invalid feedback timestamp')
        for value in (self.message, self.source_provider, self.source_task, self.reason):
            if not isinstance(value, str) or not value:
                raise ValueError('feedback text fields required')
        if type(self.simulated) is not bool or not isinstance(self.metadata, dict):
            raise ValueError('invalid feedback provenance')
        json.dumps(self.metadata, allow_nan=False)

    def to_dict(self):
        return deepcopy(asdict(self))


class FeedbackPolicy(Protocol):
    def evaluate(self, result: PredictionResult, *, timestamp: float) -> FeedbackAction: ...


def prediction_from_payload(payload):
    """Rehydrate only the existing PredictionResult contract, not arbitrary fields."""
    keys = ('status', 'provider_id', 'provider_name', 'task', 'timestamp', 'window_id',
            'window_start', 'window_end', 'reasons', 'metadata', 'provider_version')
    values = {key: payload[key] for key in keys if key in payload}
    values['outputs'] = tuple(PredictionOutput(**output) for output in payload.get('outputs', []))
    result = PredictionResult(**values)
    result.to_dict()
    return result


class DevelopmentFeedbackPolicy:
    """One policy instance per session; caller serializes evaluations.

    Uses deterministic session-relative event time. Never executes an actuator.
    """
    lower_threshold = 8
    upper_threshold = 24
    cooldown_seconds = 5.0

    def __init__(self):
        self.last_active = None
        self.last_seen = None

    def evaluate(self, result, *, timestamp):
        stale = self.last_seen is not None and timestamp <= self.last_seen
        timestamp = max(timestamp, self.last_seen) if self.last_seen is not None else timestamp
        provider = getattr(result, 'provider_id', 'unknown')
        task = getattr(result, 'task', 'unknown')
        remaining = max(0.0, self.cooldown_seconds - (timestamp - self.last_active)) if self.last_active is not None else 0.0

        def action(status, reason, message, severity='info'):
            return FeedbackAction(status, 'visual' if status == 'active' else 'none', message,
                severity, provider, task, timestamp, reason,
                {'development_only': True, 'policy': 'mock-numeric-v1',
                 'cooldown_seconds': self.cooldown_seconds, 'cooldown_remaining_seconds': remaining,
                 'last_active_timestamp': self.last_active, 'scientific_interpretation': False}, True)

        if stale:
            return action('suppressed', 'stale_result', 'Development feedback suppressed: stale result.')
        self.last_seen = timestamp
        if not isinstance(result, PredictionResult):
            return action('suppressed', 'malformed_result', 'Development feedback suppressed: malformed result.')
        if result.status != 'ok':
            return action('suppressed', 'prediction_' + result.status, 'Development feedback suppressed: prediction ' + result.status + '.')
        if (result.provider_id != 'mock' or result.task != 'transport_demo' or
                not isinstance(result.metadata, dict) or result.metadata.get('mock') is not True or
                result.metadata.get('scientific_interpretation') is not False):
            return action('suppressed', 'not_development_mock', 'Feedback is enabled only for the development mock.')
        if result.timestamp is None or result.timestamp != timestamp:
            return action('suppressed', 'invalid_result_time', 'Development feedback suppressed: invalid result time.')
        outputs = [o for o in result.outputs if o.name == 'development_value' and o.semantic_type == 'development']
        try:
            numeric = len(outputs) == 1 and type(outputs[0].value) in (int, float) and math.isfinite(outputs[0].value)
        except OverflowError:
            numeric = False
        if not numeric:
            return action('suppressed', 'unavailable_value', 'Development feedback suppressed: numeric output unavailable.')
        value = outputs[0].value
        if value < self.lower_threshold:
            return action('none', 'below_development_threshold', 'No development feedback.')
        if remaining > 0:
            return action('suppressed', 'cooldown', 'Development feedback suppressed during cooldown.')
        self.last_active = timestamp
        remaining = self.cooldown_seconds
        if value < self.upper_threshold:
            return action('active', 'development_middle_range', 'Development test: warning feedback.', 'warning')
        return action('active', 'development_upper_range', 'Development test: stronger feedback.', 'critical')
