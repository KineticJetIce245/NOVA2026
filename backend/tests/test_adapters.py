"""Adapter boundary tests use result fixtures, never scientific imports/hardware."""
from dataclasses import dataclass
from threading import Event
import unittest
from fastapi.testclient import TestClient
from backend.adapters.contracts import AttentionResult, VigilanceResult, SyncResult, EEGDisplayResult
from backend.adapters.results import ResultAdapter, ResultProducer, encode_result
from backend.adapters.legacy import aad_result, lapse_result, combined_results
from backend.app.publisher import Publisher
from backend.app.server import create_app


class AdapterTests(unittest.TestCase):
    def test_four_attention_states(self):
        for decision in ('A', 'B', 'uncertain', 'unavailable'):
            payload = encode_result(AttentionResult(0, 'aad', decision))[3]
            self.assertEqual(payload['decision'], decision)
            self.assertEqual(payload['attended'], decision if decision in ('A', 'B') else None)
        with self.assertRaises(ValueError):
            encode_result(AttentionResult(0, 'aad', 'guess'))

    def test_aad_mapping_does_not_recompute_decision(self):
        for index, expected in [(0, 'A'), (1, 'B'), (None, 'unavailable'), ('uncertain', 'uncertain')]:
            result = aad_result(index, (.1, .9), timestamp=3)
            self.assertEqual(result.decision, expected)
            self.assertEqual(encode_result(result)[3]['correlation_a'], .1)
        for invalid in (True, -1, 2, 'A', 0.0):
            with self.assertRaises(ValueError):
                aad_result(invalid, None, timestamp=0)

    def test_combined_object_and_dictionary_explicit_timestamp(self):
        @dataclass
        class Decision:
            end_time_s: float = 10000
            attended_talker: int = 1
            correlations: tuple = (.2, .4)
            lapse_score: float = .8
        for raw in (Decision(), vars(Decision())):
            a, v = combined_results(raw, timestamp=5)
            self.assertEqual(a.timestamp, 5)
            self.assertEqual(a.decision, 'B')
            self.assertEqual(v.value, .8)
            self.assertEqual(v.metric, 'lapse_probability')
        with self.assertRaises(TypeError):
            combined_results(Decision())

    def test_lapse_and_vigilance_semantics_and_unavailable(self):
        p = encode_result(lapse_result(.8, timestamp=1))[3]
        self.assertIsNone(p['score'])
        self.assertEqual(p['lapse_score'], .8)
        self.assertEqual(p['metric'], 'lapse_probability')
        self.assertEqual(encode_result(VigilanceResult(1, 'v', 0))[3]['score'], 0)
        self.assertIsNone(encode_result(lapse_result(None, timestamp=1))[3]['lapse_score'])

    def test_sync_has_no_fabricated_measurement(self):
        p = encode_result(SyncResult(0, 'sync'))[3]
        self.assertEqual(p['status'], 'unknown')
        self.assertIsNone(p['offset_ms'])
        self.assertIsNone(p['drift_warning'])
        p = encode_result(SyncResult(0, 'sync', 'reported', -12.5, False))[3]
        self.assertEqual(p['offset_ms'], -12.5)
        self.assertIs(p['drift_warning'], False)

    def test_invalid_results_do_not_reach_publisher(self):
        calls = []
        adapter = ResultAdapter(lambda *args: calls.append(args))
        bad = [AttentionResult(-1, 'x'), AttentionResult(0, 'server'), AttentionResult(0, ''),
               AttentionResult(0, 'x', correlation_a=float('nan')), AttentionResult(0, 'x', correlation_b=2),
               VigilanceResult(0, 'v', True), VigilanceResult(0, 'v', 2), VigilanceResult(0, 'v', .5, 'invented'),
               SyncResult(0, 's', drift_warning=0), SyncResult(0, 's', offset_ms=float('inf')),
               SyncResult(0, 's', status=''), AttentionResult(None, 'x'), AttentionResult(0, 'x', simulated=1)]
        for result in bad:
            with self.subTest(result=result), self.assertRaises(ValueError):
                adapter.publish(result)
        self.assertEqual(calls, [])

    def test_display_samples_are_detached_and_not_processed(self):
        samples = [[0, .4, -1]]
        result = EEGDisplayResult(0, 'display', ['one'], samples, 32)
        payload = encode_result(result)[3]
        self.assertEqual(payload['samples'], samples)
        samples[0][0] = 4
        self.assertEqual(payload['samples'][0][0], 0)
        for result in [EEGDisplayResult(0, 'd', ['one'], [[0], [1]], 32),
                       EEGDisplayResult(0, 'd', ['a', 'b'], [[1], [2, 3]], 32),
                       EEGDisplayResult(0, 'd', ['a'], [[None]], 32), EEGDisplayResult(0, 'd', sample_rate=0)]:
            with self.assertRaises(ValueError):
                encode_result(result)

    def test_publisher_assigns_sequence_and_session(self):
        publisher = Publisher()
        publisher.begin('actual-session')
        adapter = ResultAdapter(lambda kind, t, source, payload: publisher.publish(kind, t, source, 'actual-session', payload))
        for i, result in enumerate((AttentionResult(0, 'a'), SyncResult(0, 's')), 1):
            packet = adapter.publish(result)
            self.assertEqual(packet['sequence'], i)
            self.assertEqual(packet['session_id'], 'actual-session')
        publisher.end('actual-session')
        with self.assertRaises(ValueError):
            adapter.publish(AttentionResult(1, 'a'))

    def test_result_producer_cancellation_and_error_propagation(self):
        calls = []
        stop = Event()
        stop.set()
        ResultProducer(lambda *_: calls.append('ran')).run(lambda *_: None, stop)
        self.assertEqual(calls, [])
        def failing(adapter, stop):
            raise RuntimeError('coordinator failure')
        with self.assertRaisesRegex(RuntimeError, 'coordinator failure'):
            ResultProducer(failing).run(lambda *_: None, Event())
        with self.assertRaises(ValueError):
            ResultProducer(None)

    def test_independent_producer_to_rest_websocket_lifecycle(self):
        ready = Event()
        def run_results(adapter, stop):
            adapter.publish(AttentionResult(0, 'external-aad', 'uncertain', simulated=True))
            adapter.publish(lapse_result(.7, timestamp=0, simulated=True))
            adapter.publish(SyncResult(0, 'separate-sync', simulated=True))
            ready.set()
            stop.wait()
        with TestClient(create_app(producer_factory=lambda: ResultProducer(run_results, simulated=True))) as client:
            sid = client.post('/api/session/start').json()['id']
            self.assertTrue(ready.wait(2))
            packets = client.get('/api/state').json()['packets']
            self.assertEqual([p['type'] for p in packets], ['session', 'attention', 'vigilance', 'sync'])
            with client.websocket_connect('/ws/live') as socket:
                received = [socket.receive_json() for _ in packets]
                self.assertEqual(received, packets)
                self.assertEqual(received[1]['payload']['decision'], 'uncertain')
                self.assertIsNone(received[3]['payload']['offset_ms'])
                self.assertTrue(all(p['session_id'] == sid for p in received))
                self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
