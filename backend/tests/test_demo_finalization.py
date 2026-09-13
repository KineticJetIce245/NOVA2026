import unittest
from fastapi.testclient import TestClient
from backend.adapters.mock import results, MockProducer
from backend.app.server import create_app


class DemoFinalizationTests(unittest.TestCase):
    def test_required_packets_and_deterministic_focus(self):
        for index, decision in [(0, 'A'), (15, 'A'), (16, 'B'), (31, 'B'), (32, 'A')]:
            packets = list(results(index))
            self.assertEqual(packets, list(results(index)))
            by_type = {kind: payload for kind, _, _, payload in packets}
            self.assertEqual(set(by_type), {'audio_sources', 'attention', 'vigilance', 'sync',
                                          'eeg_display', 'signal_quality', 'gain', 'prediction'})
            self.assertTrue(all(payload['simulated'] is True for payload in by_type.values()))
            self.assertTrue(all(timestamp == index * .25 for _, timestamp, _, _ in packets))
            self.assertEqual(by_type['attention']['decision'], decision)
            self.assertEqual(by_type['attention']['attended'], decision)
            self.assertEqual([s['id'] for s in by_type['audio_sources']['sources']], ['A', 'B'])
            self.assertIsNone(by_type['sync']['offset_ms'])
            self.assertIsNone(by_type['signal_quality']['quality'])

    def test_custom_metadata_through_existing_rest_and_websocket(self):
        sources = [dict(id='A', label='Recorded Interview', input_type='recording', reference='clip-a.wav'),
                   dict(id='B', label='YouTube Speech', input_type='youtube', reference='video-example')]
        producer = MockProducer(sources)
        sources[0]['label'] = 'changed after construction'
        with TestClient(create_app(producer_factory=lambda: producer)) as client:
            sid = client.post('/api/session/start').json()['id']
            with client.websocket_connect('/ws/live') as socket:
                while True:
                    packet = socket.receive_json()
                    if packet['type'] == 'audio_sources':
                        break
            self.assertEqual(packet['session_id'], sid)
            self.assertEqual(packet['payload']['sources'][0]['label'], 'Recorded Interview')
            self.assertEqual(packet['payload']['sources'][1], sources[1])
            snapshot = client.get('/api/state').json()
            self.assertTrue(any(p['type'] == 'audio_sources' for p in snapshot['packets']))
            self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')

    def test_invalid_source_metadata_rejected(self):
        valid = MockProducer().sources
        for sources in [[], valid[:1], valid + valid[:1], [valid[0], valid[0]], [None, valid[1]]]:
            with self.assertRaises(ValueError):
                MockProducer(sources)
        for reference in ['https://example.invalid/a?token=example', '/absolute/path', '../file', 0]:
            with self.assertRaises(ValueError):
                MockProducer([{**valid[0], 'reference': reference}, valid[1]])
