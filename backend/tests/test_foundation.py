"""Phase 1 regression tests: no hardware, datasets, models or external network."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import unittest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from backend.app.protocol import make_packet, validate_packet, MAX_PACKET_BYTES
from backend.app.publisher import Publisher, LaggedSubscriber
from backend.app.sessions import Sessions
from backend.app.server import create_app
from backend.adapters.mock import results


class IdleProducer:
    def run(self, publish, stop):
        stop.wait()


class ProtocolTests(unittest.TestCase):
    def test_packet_creation_and_unknown_payload(self):
        payload = {'future': [None, {'x': 1}]}
        packet = make_packet('future', 1.5, 7, 'source', 'session', payload)
        self.assertEqual(packet['version'], 1)
        self.assertEqual(packet['payload'], payload)
        payload['future'].append('changed')
        self.assertEqual(len(packet['payload']['future']), 2)
        packet['extension'] = {'kept': True}
        self.assertTrue(validate_packet(packet)['extension']['kept'])

    def test_malformed_envelopes(self):
        packet = make_packet('attention', 0, 1, 'source', 'session', {})
        for key, value in [('version', 2), ('version', True), ('sequence', -1),
                           ('sequence', True), ('sequence', 2**53), ('timestamp', float('nan')),
                           ('timestamp', float('inf')), ('timestamp', -1), ('timestamp', True),
                           ('source', ''), ('session_id', None), ('type', 4), ('payload', [])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_packet({**packet, key: value})
        for raw in (None, [], 'not json', {}):
            with self.assertRaises(ValueError):
                validate_packet(raw)

    def test_payload_limits_and_nonfinite(self):
        for payload in ({'x': float('nan')}, {'x': object()}, {1: 'not a JSON key'},
                        {'x': (1, 2)}, {'x': 'a'*MAX_PACKET_BYTES}):
            with self.assertRaises(ValueError):
                make_packet('future', 0, 1, 's', 'id', payload)

    def test_deep_payload_and_oversized_timestamp_rejected(self):
        payload = {}
        for _ in range(34):
            payload = {'nested': payload}
        with self.assertRaises(ValueError):
            make_packet('x', 0, 1, 's', 'id', payload)
        with self.assertRaises(ValueError):
            make_packet('x', 10**400, 1, 's', 'id', {})


class PublisherTests(unittest.TestCase):
    def test_events_preserved_and_snapshot_latest(self):
        p = Publisher()
        p.begin('one')
        for i in range(5):
            p.publish('attention', i, 'aad', 'one', {'value': i})
        self.assertEqual([x['sequence'] for x in p.events_after(0)], list(range(1, 6)))
        snapshot = p.snapshot()
        self.assertEqual(len(snapshot['packets']), 1)
        snapshot['packets'][0]['payload']['value'] = -1
        self.assertEqual(p.snapshot()['packets'][0]['payload']['value'], 4)

    def test_bounds_and_lag_detection(self):
        p = Publisher(event_limit=3, state_limit=2)
        p.begin('one')
        for i in range(5):
            p.publish(str(i), i, 'source', 'one', {})
        self.assertEqual(len(p.snapshot()['packets']), 2)
        self.assertEqual(len(p.events_after(2)), 3)
        with self.assertRaises(LaggedSubscriber):
            p.events_after(1)

    def test_sessions_timestamp_validation_and_rejection_atomicity(self):
        p = Publisher()
        p.begin('one')
        p.publish('x', 2, 's', 'one', {})
        for args in [('x', 1, 's', 'one', {}), ('x', 3, 's', 'old', {})]:
            with self.assertRaises(ValueError):
                p.publish(*args)
        self.assertEqual(p.sequence, 1)
        p.end('one')
        with self.assertRaises(ValueError):
            p.publish('x', 3, 's', 'one', {})
        p.begin('two')
        self.assertEqual(p.snapshot()['packets'], [])
        self.assertEqual(p.publish('x', 0, 's', 'two', {})['sequence'], 2)

    def test_concurrent_publication_sequences(self):
        p = Publisher()
        p.begin('one')
        with ThreadPoolExecutor(max_workers=4) as pool:
            packets = list(pool.map(lambda i: p.publish('x', 0, str(i%4), 'one', {'i': i}), range(100)))
        self.assertEqual(sorted(x['sequence'] for x in packets), list(range(1, 101)))


class WorkerTests(unittest.TestCase):
    def test_mock_deterministic_and_multiple_types(self):
        for index in (0, 16, 32):
            self.assertEqual(list(results(index)), list(results(index)))
        initial = {kind: payload for kind, _, _, payload in results(0)}
        later = {kind: payload for kind, _, _, payload in results(16)}
        self.assertTrue({'attention', 'vigilance', 'sync', 'eeg_display'} <= initial.keys())
        self.assertEqual(initial['attention']['attended'], 'A')
        self.assertEqual(later['attention']['attended'], 'B')
        self.assertEqual(initial['sync']['status'], 'unknown')
        self.assertIsNone(initial['sync']['offset_ms'])
        self.assertTrue(all(value['simulated'] for value in initial.values()))
        self.assertEqual(len(initial['eeg_display']['samples'][0]), 32)

    def test_idempotent_commands_restart_and_close(self):
        sessions = Sessions(Publisher(), IdleProducer)
        try:
            self.assertIsNone(sessions.stop())
            first = sessions.start()
            self.assertEqual(sessions.start()['id'], first['id'])
            self.assertEqual(sessions.stop()['status'], 'stopped')
            self.assertFalse(sessions.thread.is_alive())
            self.assertEqual(sessions.stop()['status'], 'stopped')
            self.assertNotEqual(sessions.start()['id'], first['id'])
        finally:
            sessions.close()
        self.assertFalse(sessions.thread.is_alive())
        with self.assertRaises(RuntimeError):
            sessions.start()

    def test_worker_failure_is_reported(self):
        class Failing:
            def run(self, publish, stop):
                raise ValueError('private diagnostic details')
        sessions = Sessions(Publisher(), Failing)
        sessions.start()
        sessions.thread.join(1)
        self.assertFalse(sessions.thread.is_alive())
        result = sessions.list()[0]
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'producer_failed')
        self.assertNotIn('private', str(result))
        sessions.close()

    def test_stop_deadline_is_explicit(self):
        release = Event()
        entered = Event()
        class Blocked:
            def run(self, publish, stop):
                entered.set()
                release.wait(2)
        sessions = Sessions(Publisher(), Blocked, stop_timeout=.01)
        sessions.start()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                sessions.stop()
        finally:
            release.set()
            sessions.thread.join(1)
            sessions.close()

    def test_concurrent_start_is_single_worker(self):
        sessions = Sessions(Publisher(), IdleProducer)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                records = list(pool.map(lambda _: sessions.start(), range(8)))
            self.assertEqual(len({record['id'] for record in records}), 1)
        finally:
            sessions.close()


class TransportTests(unittest.TestCase):
    def test_rest_health_state_commands_and_shutdown(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/health').json()['status'], 'ok')
            self.assertEqual(client.get('/api/state').json()['packets'], [])
            self.assertIsNone(client.post('/api/session/stop').json())
            record = client.post('/api/session/start').json()
            self.assertEqual(client.post('/api/session/start').json()['id'], record['id'])
            self.assertEqual(client.get('/api/sessions/'+record['id']).status_code, 200)
            self.assertEqual(client.get('/api/sessions/missing').status_code, 404)
            self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')
            client.post('/api/session/start')
        self.assertTrue(app.state.sessions.closed)
        self.assertFalse(app.state.sessions.thread.is_alive())

    def test_websocket_mock_snapshot_events_and_reconnect(self):
        app = create_app()
        with TestClient(app) as client:
            client.post('/api/session/start')
            with client.websocket_connect('/ws/live') as ws:
                seen = set()
                sequences = []
                for _ in range(20):
                    packet = ws.receive_json()
                    validate_packet(packet)
                    seen.add(packet['type'])
                    sequences.append(packet['sequence'])
                    if {'attention', 'vigilance', 'sync', 'eeg_display'} <= seen:
                        break
                self.assertTrue({'attention', 'vigilance', 'sync', 'eeg_display'} <= seen)
                self.assertEqual(sequences, sorted(set(sequences)))
                ws.send_text('not a command')
                client.post('/api/session/stop')
                for _ in range(20):
                    packet = ws.receive_json()
                    if packet['type'] == 'session' and packet['payload']['status'] == 'stopped':
                        break
                else:
                    self.fail('missing terminal session event')
            expected = client.get('/api/state').json()['packets']
            with client.websocket_connect('/ws/live') as ws:
                self.assertEqual([ws.receive_json() for _ in expected], expected)

    def test_unknown_partial_payload_delivered_unchanged(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            sid = client.post('/api/session/start').json()['id']
            packet = app.state.publisher.publish('future', 0, 'extension', sid, {'unknown': None})
            with client.websocket_connect('/ws/live') as ws:
                self.assertEqual(ws.receive_json()['type'], 'session')
                self.assertEqual(ws.receive_json(), packet)

    def test_factory_failure_is_clean_http_error(self):
        def failure():
            raise ValueError('private path')
        with TestClient(create_app(failure)) as client:
            response = client.post('/api/session/start')
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private', response.text)
            self.assertEqual(client.get('/api/health').status_code, 200)

    def test_lagged_websocket_explicitly_requests_reconnect(self):
        class Overrun(Publisher):
            def events_after(self, cursor):
                raise LaggedSubscriber('forced overflow')
        with TestClient(create_app(IdleProducer, Overrun())) as client:
            client.post('/api/session/start')
            with client.websocket_connect('/ws/live') as ws:
                self.assertEqual(ws.receive_json()['type'], 'session')
                with self.assertRaises(WebSocketDisconnect) as caught:
                    ws.receive_json()
                self.assertEqual(caught.exception.code, 1013)

    def test_idle_socket_disconnect_does_not_leave_subscriber(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            with client.websocket_connect('/ws/live') as ws:
                ws.send_bytes(b'ignored')
            # Context exit waits for the handler and its child tasks to finish.
            self.assertEqual(len(app.state.sockets), 0)


if __name__ == '__main__':
    unittest.main()
