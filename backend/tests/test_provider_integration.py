import unittest
from dataclasses import dataclass

from backend.adapters.research import result_from_native, NOVA2026Provider, NovaAADProvider
from backend.adapters.results import ResultAdapter, encode_result
from backend.app.prediction import PredictionOutput, PredictionResult
from backend.app.prepared_input import PreparedInput
from backend.app.providers import MockPredictionProvider, provider_registry, select_provider
from backend.app.publisher import Publisher


@dataclass
class Window:
    timestamp: float = 2.0


class ProviderIntegrationTests(unittest.TestCase):
    def test_result_supports_multiple_typed_outputs(self):
        result = result_from_native(provider_id='nova_aad', provider_name='novaAAD',
            task='auditory_attention', timestamp=2, outputs=[
                {'name': 'speaker_a_correlation', 'value': .18, 'semantic_type': 'correlation'},
                {'name': 'attended_speaker', 'value': 'B', 'semantic_type': 'class'},
            ])
        payload = result.to_dict()
        self.assertEqual([output['semantic_type'] for output in payload['outputs']], ['correlation', 'class'])

    def test_probability_output_keeps_probability_semantics(self):
        output = PredictionOutput('lapse_probability', .72, 'probability', 'lapse')
        self.assertEqual(output.to_dict()['semantic_type'], 'probability')

    def test_invalid_unavailable_and_error_statuses_serialize(self):
        for status in ('invalid', 'unavailable', 'error'):
            result = PredictionResult(status, 'p', 'P', 'task', reasons=('not_ready',))
            self.assertEqual(result.to_dict()['status'], status)

    def test_mock_is_deterministic_and_marked(self):
        provider = MockPredictionProvider()
        first = provider.predict(PreparedInput('mock', 'transport_demo', timestamp=1))
        second = provider.predict(PreparedInput('mock', 'transport_demo', timestamp=1))
        self.assertEqual(first, second)
        self.assertTrue(first.metadata['mock'])

    def test_registry_selects_providers_without_fallback(self):
        self.assertEqual(set(provider_registry()), {'mock', 'nova2026', 'nova_aad'})
        self.assertEqual(select_provider('mock').provider_id, 'mock')
        self.assertEqual(select_provider('nova2026').predict(Window()).status, 'unavailable')
        self.assertEqual(select_provider('nova_aad').predict(Window()).status, 'unavailable')
        with self.assertRaises(ValueError):
            select_provider('missing')

    def test_prediction_transport_preserves_semantic_type(self):
        publisher = Publisher()
        publisher.begin('s')
        adapter = ResultAdapter(lambda kind, timestamp, source, payload: publisher.publish(
            kind, timestamp, source, 's', payload))
        adapter.publish(PredictionResult('ok', 'mock', 'MOCK', 'task',
            outputs=(PredictionOutput('correlation', .2, 'correlation'),), timestamp=1))
        packet = publisher.snapshot()['packets'][0]
        self.assertEqual(packet['payload']['outputs'][0]['semantic_type'], 'correlation')


if __name__ == '__main__':
    unittest.main()