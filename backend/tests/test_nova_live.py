"""Real producer contract, lifecycle and malformed status regressions."""
import unittest
from dataclasses import replace
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from backend.adapters.contracts import GainResult, SignalQualityResult, SyncResult
from backend.adapters.results import encode_result, ResultAdapter
from backend.adapters.nova_live import make_coordinator
from nova2026.auditory.data import AttentionEstimate


class NovaLiveTests(unittest.TestCase):
    def test_gain_and_quality_ranges(self):
        self.assertEqual(encode_result(GainResult(1, 'nova_aad', -6, 0))[3]['a_db'], -6)
        self.assertIsNone(encode_result(SignalQualityResult(1, 'nova_aad', True, ['F8']))[3]['quality'])
        for value in [GainResult(1, 'nova_aad', 1, 0), GainResult(1, 'nova_aad', float('nan'), 0),
                      SignalQualityResult(1, 'nova_aad', 1), SignalQualityResult(1, 'nova_aad', True, 'F8')]:
            with self.assertRaises(ValueError):
                encode_result(value)

    def test_sync_software_diagnostics_do_not_invent_offset(self):
        payload = encode_result(SyncResult(1, 'nova_aad', clock_domain='local_lsl',
                      max_block_timing_error_seconds=.002, estimated_audio_drift_ppm=30))[3]
        self.assertIsNone(payload['offset_ms'])
        self.assertIsNone(payload['drift_warning'])
        self.assertEqual(payload['estimated_audio_drift_ppm'], 30)

    def setup_coordinator(self):
        trial = SimpleNamespace(audio_rate=200)
        model = SimpleNamespace(training_info={'decision_gate': dict(center=0., scale=.1, threshold=2.,
                               false_fire_rate=.01, null_windows=100)},
                                config=SimpleNamespace(sample_rate=64), contract={'eeg_channels': ['F3', 'F4']})
        with patch('nova2026.auditory.data.load_trial', return_value=trial), patch(
                'nova2026.auditory.decoder.RidgeDecoder.load', return_value=model):
            return make_coordinator('trial', 'model', 'source')

    def test_push_publishes_valid_invalid_and_uncertain_records(self):
        coordinator = self.setup_coordinator()
        packets = []
        stream = SimpleNamespace(connected=True, connect=lambda **kw: None, disconnect=lambda: None)
        def run(trial, model, source, **options):
            from mne_lsl.lsl import local_clock
            now = local_clock()
            raw = SimpleNamespace(bad_channels=('F8',))
            aligned = SimpleNamespace(timestamps=np.linspace(now, now+1, 64))
            for i, (scores, valid) in enumerate([([.8, -.8], True), ([.01, 0], True), (None, False)]):
                options['on_estimate'](AttentionEstimate(scores, now+i+1, now+i+1, valid), aligned, raw)
            return np.empty((0, 2)), {'consumer_errors': 0}
        with patch('mne_lsl.stream.StreamLSL', return_value=stream), patch('scripts.auditory.live.run', side_effect=run):
            coordinator(ResultAdapter(lambda *p: packets.append(p)), Event())
        attention = [p[3] for p in packets if p[0] == 'attention']
        self.assertEqual([p['decision'] for p in attention], ['A', 'uncertain', 'unavailable'])
        for kind, timestamp, source, payload in packets:
            self.assertGreaterEqual(timestamp, 0)
            self.assertIs(payload['simulated'], False)
        self.assertEqual(len([p for p in packets if p[0] == 'prediction']), 3)

    def test_stop_during_source_search_is_bounded(self):
        coordinator = self.setup_coordinator()
        def connect(**kwargs):
            sleep(.05)
            raise RuntimeError('no stream')
        stream = SimpleNamespace(connected=False, connect=connect)
        stop = Event()
        with patch('mne_lsl.stream.StreamLSL', return_value=stream):
            worker = Thread(target=coordinator, args=(ResultAdapter(lambda *p: None), stop))
            worker.start()
            sleep(.1)
            tick = monotonic()
            stop.set()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertLess(monotonic()-tick, 1)
