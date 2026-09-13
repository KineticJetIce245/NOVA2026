import json
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.media import MediaTimeline
from backend.app.server import create_app
from backend.adapters.mock import results
from backend.app.protocol import make_packet


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'sample.wav'
        self.path.write_bytes(b'RIFF' + bytes(64))
        self.now = 100.0
        self.media = MediaTimeline(self.path, 'Conversation', clock=lambda: self.now)
        self.media.bind('session')
        self.request = 0

    def command(self, action, position=0, **extra):
        self.request += 1
        return self.media.control(dict(session_id='session', media_id=self.media.media_id,
            client_id='browser', request_id=self.request, action=action,
            media_time_s=position, duration_s=90, **extra))

    def test_handshake_pause_resume_and_media_time_predictions(self):
        self.assertEqual(self.command('prepare')['server_reference_s'], 100)
        self.command('playing')
        for position, expected in [(0, 'A'), (3.99, 'A'), (4, 'B'), (6.2, 'B'), (8, 'A')]:
            self.now = 100 + position
            self.command('report', position)
            packets = {k:p for k,t,s,p in results(200, media=self.media.snapshot())}
            self.assertEqual(packets['attention']['decision'], expected)
            self.assertTrue(all(p['simulated'] for p in packets.values()))
        self.command('paused', 8)
        self.now += 30
        self.command('report', 8)
        self.assertEqual(self.media.snapshot()['media_time_s'], 8)
        self.assertEqual(self.media.snapshot()['playback_state'], 'paused')
        self.command('playing', 8)
        self.now += .25
        self.command('report', 8.25)
        self.assertEqual(self.media.snapshot()['media_time_s'], 8.25)

    def test_paused_at_six_point_two_stays_b_then_stop_neutral(self):
        self.command('prepare'); self.command('playing')
        self.now += 6.2; self.command('paused', 6.2)
        self.now += 30; self.command('report', 6.2)
        by_type = {k:p for k,t,s,p in results(300, media=self.media.snapshot())}
        self.assertEqual(by_type['attention']['attended'], 'B')
        self.command('stopped', 0)
        by_type = {k:p for k,t,s,p in results(301, media=self.media.snapshot())}
        self.assertEqual(by_type['attention']['decision'], 'unavailable')
        self.assertEqual((by_type['gain']['a_db'], by_type['gain']['b_db']), (0, 0))
        self.assertEqual(self.media.snapshot()['media_time_s'], 0)

    def test_expired_reports_and_invalid_seek_fail_neutral(self):
        self.command('prepare'); self.command('playing')
        self.now += 2
        self.assertEqual(self.media.snapshot()['sync_status'], 'desynchronized')
        with self.assertRaises(ValueError): self.command('report', 40)
        self.assertEqual(self.media.snapshot()['sync_status'], 'desynchronized')
        self.command('stopped'); self.command('prepare')
        self.assertEqual(self.media.snapshot()['sync_status'], 'observed')

    def test_old_controller_session_and_invalid_values_rejected(self):
        self.command('prepare')
        original = self.media.snapshot()
        data = dict(session_id='session', media_id=self.media.media_id, client_id='browser',
                    request_id=2, action='report', media_time_s=0, duration_s=90)
        for extra in [dict(session_id='old'), dict(client_id='other'), dict(request_id=1),
                      dict(media_time_s=float('nan')), dict(media_time_s=-1),
                      dict(duration_s=0), dict(action='seek')]:
            with self.assertRaises(ValueError): self.media.control({**data, **extra})
        self.assertEqual(self.media.snapshot(), original)

    def test_media_http_shared_asset_range_and_transport(self):
        with TestClient(create_app(media=self.media)) as client:
            descriptor = client.get('/api/media').json()
            self.assertNotIn(str(self.path), json.dumps(descriptor))
            response = client.get(descriptor['url'], headers={'Range':'bytes=0-3'})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b'RIFF')
            sid = client.post('/api/session/start').json()['id']
            data = dict(session_id=sid, media_id=descriptor['media_id'], client_id='browser',
                        request_id=1, action='prepare', media_time_s=0, duration_s=90)
            self.assertEqual(client.post('/api/media/control', json=data).status_code, 200)
            with client.websocket_connect('/ws/live') as socket:
                found = {}
                while not {'media','attention','gain'} <= found.keys():
                    packet = socket.receive_json()
                    if packet['type'] in ('media','attention','gain'):
                        found[packet['type']] = packet
                for packet in found.values():
                    make_packet(packet['type'], packet['timestamp'], packet['sequence'], packet['source'], sid, packet['payload'])
                    self.assertTrue(packet['payload']['simulated'])
                self.assertEqual(found['attention']['payload']['media_id'], descriptor['media_id'])
                sources = next(p['payload']['sources'] for p in client.get('/api/state').json()['packets'] if p['type'] == 'audio_sources')
                self.assertEqual([item['label'] for item in sources], ['Source A', 'Source B'])
                self.assertEqual(client.post('/api/session/stop').json()['status'], 'stopped')
            self.assertEqual(self.media.snapshot()['playback_state'], 'stopped')
            self.assertEqual(client.post('/api/media/control', json={**data,'request_id':2}).status_code, 409)

    def test_unconfigured_media_and_private_path_boundary(self):
        with TestClient(create_app()) as client:
            self.assertIsNone(client.get('/api/media').json())
            self.assertEqual(client.get('/api/media/file').status_code, 404)
        with self.assertRaises(ValueError): MediaTimeline(self.path.parent / 'missing.wav')
