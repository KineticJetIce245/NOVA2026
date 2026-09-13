"""Command serialization and cooperative worker lifecycle, independent of HTTP."""
from collections import OrderedDict
from copy import deepcopy
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4
from backend.adapters.mock import MockProducer
from .feedback import DevelopmentFeedbackPolicy, prediction_from_payload


class Sessions:
    def __init__(self, publisher, producer_factory=MockProducer, stop_timeout=3):
        self.publisher = publisher
        self.factory = producer_factory
        self.stop_timeout = stop_timeout
        self.command_lock = Lock()
        self.lock = Lock()
        self.records = OrderedDict()
        self.current = None
        self.thread = None
        self.stop_event = Event()
        self.closed = False

    def list(self):
        with self.lock:
            return deepcopy(list(self.records.values()))

    def start(self):
        with self.command_lock:
            with self.lock:
                if self.closed:
                    raise RuntimeError('server shutting down')
                if self.thread and self.thread.is_alive():
                    return deepcopy(self.records[self.current])
                producer = self.factory()
                sid = str(uuid4())
                self.current = sid
                self.records[sid] = dict(id=sid, status='running',
                                         simulated=bool(getattr(producer, 'simulated', False)))
                while len(self.records) > 32:
                    self.records.popitem(last=False)
                self.stop_event = Event()
                self.publisher.begin(sid)
                self.publisher.publish('session', 0, 'server', sid, self.records[sid])
                self.thread = Thread(target=self._run, args=(sid, producer, self.stop_event),
                                     name='attune-producer', daemon=True)
                self.thread.start()
                return deepcopy(self.records[sid])

    def _run(self, sid, producer, stop):
        began = monotonic()
        error = False
        feedback = DevelopmentFeedbackPolicy()
        publication_lock = Lock()
        try:
            def publish(kind, timestamp, source, payload):
                if source == 'server':
                    raise ValueError('server source is reserved')
                if stop.is_set():
                    return None
                with publication_lock:
                    packet = self.publisher.publish(kind, timestamp, source, sid, payload)
                    if kind == 'prediction':
                        try:
                            result = prediction_from_payload(packet['payload'])
                        except (ValueError, TypeError, KeyError, AttributeError):
                            result = None
                        action = feedback.evaluate(result, timestamp=timestamp)
                        self.publisher.publish('feedback', action.timestamp, 'development-feedback', sid, action.to_dict())
                    return packet
            producer.run(publish, stop)
        except Exception:
            error = True
        finally:
            with self.lock:
                self.records[sid]['status'] = 'error' if error else 'stopped'
                if error:
                    # Do not leak exception contents, paths or credentials to transport.
                    self.records[sid]['error'] = 'producer_failed'
                self.publisher.publish('session', monotonic()-began, 'server', sid, self.records[sid])
                self.publisher.end(sid)

    def _stop(self):
        with self.lock:
            self.stop_event.set()
            thread = self.thread
        if thread:
            thread.join(timeout=self.stop_timeout)
            if thread.is_alive():
                raise RuntimeError('producer did not stop within deadline')
        with self.lock:
            return deepcopy(self.records.get(self.current))

    def stop(self):
        with self.command_lock:
            return self._stop()

    def close(self):
        with self.command_lock:
            with self.lock:
                self.closed = True
            return self._stop()
