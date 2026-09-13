"""Common provider protocol and explicit provider selection."""
from typing import Any, Protocol
from .prediction import PredictionOutput, PredictionResult


class PredictionProvider(Protocol):
    provider_id: str
    provider_name: str
    task: str

    def predict(self, prepared_input: Any) -> PredictionResult: ...


class MockPredictionProvider:
    provider_id = 'mock'
    provider_name = 'MOCK development provider'
    task = 'transport_demo'

    def predict(self, prepared_input):
        timestamp = getattr(prepared_input, 'timestamp', None)
        return PredictionResult('ok', self.provider_id, self.provider_name, self.task,
            outputs=(
                PredictionOutput('development_value', 1, 'development', 'mock illustration'),
            ), timestamp=timestamp, metadata={'mock': True, 'scientific_interpretation': False})


class UnavailableProvider:
    def __init__(self, provider_id, provider_name, task, reason):
        self.provider_id, self.provider_name, self.task, self.reason = provider_id, provider_name, task, reason

    def predict(self, prepared_input):
        return PredictionResult.unavailable(self.provider_id, self.provider_name, self.task, self.reason)


def provider_registry():
    return {
        'mock': MockPredictionProvider(),
        'nova2026': UnavailableProvider('nova2026', 'NOVA2026 adapter', 'research_prediction',
                                        'NOVA2026 adapter is not connected in this workspace'),
        'nova_aad': UnavailableProvider('nova_aad', 'novaAAD adapter', 'auditory_attention',
                                        'novaAAD adapter has no reviewed live callable boundary'),
    }


def select_provider(provider_id):
    registry = provider_registry()
    if provider_id not in registry:
        raise ValueError(f'unknown provider: {provider_id}')
    return registry[provider_id]