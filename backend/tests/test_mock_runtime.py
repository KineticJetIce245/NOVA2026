"""Exercise the actual default session -> provider -> publisher -> WebSocket path."""
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.server import create_app
from backend.app.providers import MockPredictionProvider
from backend.app.prepared_input import PreparedInput
from backend.app.prediction import PredictionResult, PredictionOutput


class MockRuntimeTests(unittest.TestCase):
    def test_default_session_invokes_provider_and_publishes_same_result_and_feedback(self):
        calls = []
        original = MockPredictionProvider.predict
        def predict(provider, prepared):
            self.assertIsInstance(prepared, PreparedInput)
            output = original(provider, prepared)
            calls.append((prepared, output))
            return output
        with patch.object(MockPredictionProvider, 'predict', predict):
            with TestClient(create_app()) as client:
                client.post('/api/session/start')
                packets = []
                with client.websocket_connect('/ws/live') as socket:
                    while len([p for p in packets if p['type'] == 'feedback']) < 2:
                        packets.append(socket.receive_json())
                    self.assertTrue({'session','attention','vigilance','signal_quality','sync','eeg_display','gain','prediction','feedback'} <= {p['type'] for p in packets})
                    predictions = [p for p in packets if p['type'] == 'prediction']
                    self.assertGreaterEqual(len(predictions), 2)
                    self.assertEqual(predictions[1]['timestamp'] - predictions[0]['timestamp'], .25)
                    for packet in predictions:
                        payload = packet['payload']
                        expected = next(r.to_dict() for prepared, r in calls if prepared.timestamp == packet['timestamp'])
                        self.assertEqual(payload, {**expected, 'simulated': True})
                        self.assertEqual(payload['provider_id'], 'mock')
                        self.assertEqual(payload['outputs'][0]['semantic_type'], 'development')
                        feedback = next(p for p in packets if p['type'] == 'feedback' and p['timestamp'] == packet['timestamp'])
                        self.assertEqual(feedback['payload']['status'], 'none')
                        self.assertEqual(feedback['payload']['source_provider'], payload['provider_id'])
                        self.assertEqual(feedback['sequence'], packet['sequence'] + 1)
                    self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')
                    stopped = client.get('/api/state').json()
                    count = len(calls)
                    time.sleep(.35)
                    self.assertEqual(len(calls), count)
                    self.assertEqual(client.get('/api/state').json(), stopped)

    def test_feedback_uses_provider_output_not_independently_generated_value(self):
        def predict(provider, prepared):
            return PredictionResult('ok', 'mock', provider.provider_name, provider.task,
                timestamp=prepared.timestamp,
                outputs=(PredictionOutput('development_value', 12, 'development'),),
                metadata={'mock': True, 'scientific_interpretation': False})
        with patch.object(MockPredictionProvider, 'predict', predict), TestClient(create_app()) as client:
            client.post('/api/session/start')
            with client.websocket_connect('/ws/live') as socket:
                packets = []
                while not any(p['type'] == 'feedback' for p in packets):
                    packets.append(socket.receive_json())
                prediction = next(p for p in packets if p['type'] == 'prediction')
                feedback = next(p for p in packets if p['type'] == 'feedback')
                self.assertEqual(prediction['payload']['outputs'][0]['value'], 12)
                self.assertEqual(feedback['payload']['status'], 'active')
                self.assertEqual(feedback['payload']['severity'], 'warning')
            client.post('/api/session/stop')
