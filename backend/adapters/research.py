"""Thin, non-scientific boundaries for external research repositories."""
from collections.abc import Mapping
from backend.app.prediction import PredictionOutput, PredictionResult
from backend.app.providers import UnavailableProvider


def result_from_native(*, provider_id, provider_name, task, outputs, timestamp=None,
                       window_id=None, metadata=None):
    """Map reviewed native fields without renaming their semantic types."""
    typed = tuple(PredictionOutput(**output) if isinstance(output, Mapping) else output for output in outputs)
    return PredictionResult('ok', provider_id, provider_name, task, typed, timestamp,
                            window_id=window_id, metadata=metadata or {})


class NOVA2026Provider(UnavailableProvider):
    def __init__(self):
        super().__init__('nova2026', 'NOVA2026 adapter', 'research_prediction',
                         'NOVA2026 native coordinator is not connected')


class NovaAADProvider(UnavailableProvider):
    def __init__(self):
        super().__init__('nova_aad', 'novaAAD adapter', 'auditory_attention',
                         'novaAAD native live callable is not reviewed')