import unittest
from threading import Event
from fastapi.testclient import TestClient
from backend.app.prediction import PredictionOutput, PredictionResult
from backend.app.feedback import DevelopmentFeedbackPolicy, FeedbackAction
from backend.app.server import create_app


def result(value=12, timestamp=0, status='ok', provider='mock'):
    return PredictionResult(status, provider, 'Development fixture', 'transport_demo',
        outputs=(PredictionOutput('development_value', value, 'development'),), timestamp=timestamp,
        metadata={'mock': True, 'scientific_interpretation': False})


class FeedbackTests(unittest.TestCase):
    def test_middle_and_upper_mock_trigger(self):
        for value, severity in [(8, 'warning'), (23, 'warning'), (24, 'critical')]:
            action = DevelopmentFeedbackPolicy().evaluate(result(value), timestamp=0)
            self.assertEqual(action.status, 'active')
            self.assertEqual(action.severity, severity)
            self.assertEqual(action.action_type, 'visual')
            self.assertTrue(action.simulated)

    def test_low_mock_none(self):
        self.assertEqual(DevelopmentFeedbackPolicy().evaluate(result(7), timestamp=0).status, 'none')

    def test_invalid_unavailable_error_never_active(self):
        for status in ('invalid', 'unavailable', 'error'):
            action = DevelopmentFeedbackPolicy().evaluate(result(99, status=status), timestamp=0)
            self.assertEqual(action.status, 'suppressed')
            self.assertEqual(action.action_type, 'none')

    def test_nonmock_and_missing_numeric_never_active(self):
        for value in (None, True, '99', float('nan'), float('inf'), 10**500):
            self.assertEqual(DevelopmentFeedbackPolicy().evaluate(result(value), timestamp=0).status, 'suppressed')
        self.assertEqual(DevelopmentFeedbackPolicy().evaluate(result(provider='nova2026'), timestamp=0).status, 'suppressed')
        self.assertEqual(DevelopmentFeedbackPolicy().evaluate(None, timestamp=0).status, 'suppressed')

    def test_five_second_cooldown(self):
        policy = DevelopmentFeedbackPolicy()
        self.assertEqual(policy.evaluate(result(), timestamp=0).status, 'active')
        action = policy.evaluate(result(timestamp=4.99), timestamp=4.99)
        self.assertEqual(action.status, 'suppressed')
        self.assertEqual(action.reason, 'cooldown')
        self.assertAlmostEqual(action.metadata['cooldown_remaining_seconds'], .01)
        self.assertEqual(policy.evaluate(result(timestamp=5), timestamp=5).status, 'active')

    def test_low_and_invalid_do_not_reset_cooldown(self):
        policy = DevelopmentFeedbackPolicy()
        policy.evaluate(result(), timestamp=0)
        policy.evaluate(result(0, timestamp=1), timestamp=1)
        policy.evaluate(result(timestamp=4, status='error'), timestamp=4)
        self.assertEqual(policy.evaluate(result(timestamp=5), timestamp=5).status, 'active')

    def test_stale_time_suppressed_and_emission_time_does_not_regress(self):
        policy = DevelopmentFeedbackPolicy()
        policy.evaluate(result(timestamp=5), timestamp=5)
        for time in (5, 1):
            action = policy.evaluate(result(timestamp=time), timestamp=time)
            self.assertEqual(action.reason, 'stale_result')
            self.assertEqual(action.status, 'suppressed')
            self.assertEqual(action.timestamp, 5)

    def test_action_contract_validation_and_defensive_copy(self):
        action = DevelopmentFeedbackPolicy().evaluate(result(), timestamp=0)
        payload = action.to_dict()
        for key in ('status','action_type','message','severity','source_provider','source_task','timestamp','reason','metadata','simulated'):
            self.assertIn(key, payload)
        payload['metadata']['policy'] = 'changed'
        self.assertEqual(action.metadata['policy'], 'mock-numeric-v1')
        with self.assertRaises(ValueError):
            FeedbackAction('bad','none','message','info','mock','task',0,'reason')

    def test_prediction_to_feedback_websocket_and_session_reset(self):
        ready = Event()
        class Producer:
            simulated = True
            def run(self, publish, stop):
                publish('prediction', 0, 'mock', result().to_dict())
                ready.set()
                stop.wait()
        with TestClient(create_app(producer_factory=Producer)) as client:
            for _ in range(2):
                ready.clear()
                client.post('/api/session/start')
                self.assertTrue(ready.wait(2))
                packets = client.get('/api/state').json()['packets']
                self.assertEqual([p['type'] for p in packets], ['session','prediction','feedback'])
                self.assertEqual(packets[-1]['payload']['status'], 'active')
                self.assertTrue(packets[-1]['payload']['simulated'])
                with client.websocket_connect('/ws/live') as socket:
                    self.assertEqual([socket.receive_json() for _ in packets], packets)
                client.post('/api/session/stop')

    def test_malformed_prediction_emits_suppression_without_worker_failure(self):
        ready = Event()
        class Producer:
            def run(self, publish, stop):
                publish('prediction', 0, 'bad', {'outputs': [None]})
                ready.set(); stop.wait()
        with TestClient(create_app(producer_factory=Producer)) as client:
            client.post('/api/session/start'); self.assertTrue(ready.wait(2))
            packets = client.get('/api/state').json()['packets']
            self.assertEqual(packets[-1]['payload']['reason'], 'malformed_result')
            self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')
