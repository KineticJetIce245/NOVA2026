"""Shared recording reference and observed browser timeline; no clock inference."""
import math
from pathlib import Path
from threading import RLock
from time import monotonic
from uuid import uuid4


def seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 864000:
        raise ValueError('invalid media time')
    return float(value)


class MediaTimeline:
    def __init__(self, path, title='Demo Audio', clock=monotonic):
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.suffix.lower() not in ('.wav', '.mp3', '.m4a', '.mp4', '.webm'):
            raise ValueError('configure an existing supported media file')
        self.title = title.strip()[:128] or 'Demo Audio'
        self.media_id = uuid4().hex
        self.kind = 'video' if self.path.suffix.lower() in ('.mp4', '.webm') else 'audio'
        self.clock = clock
        self.lock = RLock()
        self.session_id = None
        self.revision = 0
        self.client_id = None
        self.request_id = -1
        self.position = 0.0
        self.duration = None
        self.playback = 'stopped'
        self.reference = None
        self.last_received = None
        self.valid = False

    def descriptor(self):
        return dict(media_id=self.media_id, title=self.title, kind=self.kind, url='/api/media/file')

    def bind(self, session_id):
        with self.lock:
            if self.session_id != session_id:
                self.stop()
                self.session_id = session_id

    def stop(self):
        with self.lock:
            self.position = 0.0
            self.playback = 'stopped'
            self.client_id = None
            self.request_id = -1
            self.valid = False
            self.revision += 1

    def snapshot(self):
        with self.lock:
            fresh = self.last_received is not None and self.clock() - self.last_received <= 1.5
            return dict(**self.descriptor(), session_id=self.session_id,
                        media_time_s=self.position, duration_s=self.duration,
                        playback_state=self.playback, revision=self.revision,
                        server_reference_s=self.reference, server_received_s=self.last_received,
                        sync_status='observed' if self.valid and fresh else 'desynchronized')

    def control(self, data):
        with self.lock:
            if data.get('session_id') != self.session_id or data.get('media_id') != self.media_id:
                raise ValueError('media/session mismatch')
            action = data.get('action')
            if action not in ('prepare', 'playing', 'paused', 'stopped', 'report'):
                raise ValueError('invalid media action')
            client = data.get('client_id')
            request = data.get('request_id')
            if not isinstance(client, str) or not 1 <= len(client) <= 128 or type(request) is not int or not 0 <= request <= 9007199254740991:
                raise ValueError('invalid controller request')
            if self.client_id is not None and (client != self.client_id or request <= self.request_id):
                raise ValueError('media controller conflict or old request')
            position = seconds(data.get('media_time_s'))
            duration = seconds(data.get('duration_s'))
            if duration <= 0 or position > duration:
                raise ValueError('invalid media duration')
            now = self.clock()
            if action == 'prepare':
                if self.playback != 'stopped' or position != 0:
                    raise ValueError('stop media before preparing')
                self.revision += 1
                self.client_id = client
                self.reference = now
                self.playback = 'paused'
                self.valid = True
            elif self.client_id is None:
                if action != 'stopped':
                    raise ValueError('prepare media first')
            else:
                elapsed = now - self.last_received if self.last_received is not None else 0
                max_advance = elapsed + .5 if self.playback == 'playing' else .5 if action == 'playing' else .1
                if action != 'stopped' and (position < self.position - .02 or position - self.position > max_advance):
                    self.valid = False
                    raise ValueError('media progression mismatch; stop and prepare again')
                if action != 'report':
                    self.playback = action
                # Fresh reports cannot silently repair an invalid seek.
            self.request_id = request
            self.position = 0.0 if action == 'stopped' else position
            self.duration = duration
            self.last_received = now
            if action == 'stopped':
                self.stop()
            return self.snapshot()
