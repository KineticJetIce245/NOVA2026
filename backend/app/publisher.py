"""Thread-safe, bounded event log and latest state; no scientific computation."""
from collections import OrderedDict, deque
from copy import deepcopy
from threading import RLock
from .protocol import make_packet


class LaggedSubscriber(Exception):
    """Reconnect to obtain a fresh snapshot; event history was overrun."""


class Publisher:
    def __init__(self, event_limit=256, state_limit=128):
        if event_limit < 1 or state_limit < 1:
            raise ValueError('limits must be positive')
        self.lock = RLock()
        self.sequence = 0
        self.latest = OrderedDict()
        self.events = deque(maxlen=event_limit)
        self.state_limit = state_limit
        self.session_id = None
        self.accepting = False

    def begin(self, session_id):
        with self.lock:
            self.session_id = session_id
            self.accepting = True
            self.latest.clear()
            # Retain bounded events so existing clients see session boundaries.

    def end(self, session_id):
        with self.lock:
            if session_id == self.session_id:
                self.accepting = False

    def publish(self, kind, timestamp, source, session_id, payload):
        with self.lock:
            if session_id != self.session_id or not self.accepting:
                raise ValueError('inactive producer session')
            packet = make_packet(kind, timestamp, self.sequence + 1, source, session_id, payload)
            key = (source, kind)
            previous = self.latest.get(key)
            if previous and timestamp < previous['timestamp']:
                raise ValueError('timestamp regressed within source/type')
            self.sequence += 1
            self.latest[key] = packet
            self.latest.move_to_end(key)
            while len(self.latest) > self.state_limit:
                self.latest.popitem(last=False)
            self.events.append(packet)
            return deepcopy(packet)

    def snapshot(self):
        with self.lock:
            return dict(sequence=self.sequence, session_id=self.session_id,
                        packets=deepcopy(sorted(self.latest.values(), key=lambda p: p['sequence'])))

    def events_after(self, cursor):
        with self.lock:
            if self.events and cursor < self.events[0]['sequence'] - 1:
                raise LaggedSubscriber('event buffer overrun')
            return deepcopy([packet for packet in self.events if packet['sequence'] > cursor])
