"""Deterministic Python illustration; not connected to participant measurements."""
import math
from backend.adapters.audio_sources import validate_sources
from backend.app.providers import MockPredictionProvider
from backend.app.prepared_input import PreparedInput


def results(index, provider=None, sources=None, media=None):
    if type(index) is not int or index < 0:
        raise ValueError('index must be a nonnegative integer')
    t = index * .25
    media_time = media['media_time_s'] if media else t
    a = int(media_time // 4) % 2 == 0
    available = media is None or (media['playback_state'] != 'stopped' and media['sync_status'] == 'observed')
    if media is not None:
        yield 'media', t, 'mock-media', dict(**media, simulated=True)
    yield 'audio_sources', t, 'mock-audio', dict(sources=validate_sources(sources), simulated=True)
    yield 'attention', t, 'mock-aad', dict(decision=('A' if a else 'B') if available else 'unavailable', attended=('A' if a else 'B') if available else None,
                                        correlation_a=.42 if a else .19,
                                        correlation_b=.19 if a else .42, simulated=True)
    yield 'vigilance', t, 'mock-vigilance', dict(score=.5-.4*math.cos(t/4), simulated=True)
    yield 'signal_quality', t, 'mock-quality', dict(quality=None, artifact=None, simulated=True)
    yield 'sync', t, 'mock-sync', dict(status='unknown', offset_ms=None, drift_warning=False,
                                     timeline='session_seconds', fixed_latency_ms=None, simulated=True)
    yield 'eeg_display', t, 'mock-display', dict(sample_rate=32, channels=['illustration'],
        samples=[[math.sin(t + i / 8) for i in range(32)]], simulated=True)
    yield 'gain', t, 'mock-gain', dict(a_db=0 if a or not available else -6, b_db=0 if not a or not available else -6, simulated=True)
    provider = provider if provider is not None else MockPredictionProvider()
    prepared = PreparedInput(provider.provider_id, provider.task, timestamp=t,
                             window_id=f'mock-{index}', metadata={'mock': True})
    prediction = provider.predict(prepared).to_dict()
    prediction['simulated'] = True
    yield 'prediction', t, provider.provider_id, prediction


class MockProducer:
    simulated = True

    def __init__(self, sources=None, timeline=None):
        self.timeline = timeline
        if timeline is not None and sources is None:
            sources = [dict(id=key, label=f'Source {key}', input_type='media', reference=None) for key in ('A', 'B')]
        self.sources = validate_sources(sources)

    def run(self, publish, stop):
        index = 0
        provider = MockPredictionProvider()
        while not stop.is_set():
            media = self.timeline.snapshot() if self.timeline else None
            for kind, timestamp, source, payload in results(index, provider, self.sources, media):
                if media is not None and kind in ('attention', 'gain'):
                    payload.update(media_id=media['media_id'], media_revision=media['revision'], media_time_s=media['media_time_s'])
                if stop.is_set():
                    return
                publish(kind, timestamp, source, payload)
            index += 1
            stop.wait(.25)
