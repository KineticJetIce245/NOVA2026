"""NOVA push-loop coordinator; transport is independent of scientific imports."""
import logging
from pathlib import Path
from threading import Event
from time import monotonic
import json
import math
from uuid import uuid4
from backend.adapters.contracts import (AttentionResult, EEGDisplayResult, SyncResult,
                                        SignalQualityResult, GainResult)
from backend.app.prediction import PredictionOutput, PredictionResult

log = logging.getLogger(__name__)


def make_coordinator(trial_path, model_path, stream_name, *, source_unit_exponent=0,
                     presentation='dichotic', output='wav', timing_profile=None,
                     check_channels=True, report_dir=None, timebase='grid', ledger=None,
                     player_raw_path=None, marker_name=None):
    # Preload before session workers start so a large file load cannot consume
    # the session's cooperative stop deadline. Paths come only from local config.
    from nova2026.auditory.data import load_trial
    from nova2026.auditory.decoder import RidgeDecoder
    from nova2026.auditory.gate import DecisionGate
    trial = load_trial(trial_path)
    model = RidgeDecoder.load(model_path)
    gate_data = model.training_info.get('decision_gate')
    if gate_data is None:
        raise ValueError('ATTUNE requires a null-calibrated decoder; retrain the model.')
    gate = DecisionGate(**gate_data)
    if presentation not in ('dichotic', 'diotic'):
        raise ValueError('Invalid presentation.')
    mapping = {'mode': presentation, 'left': 'A' if presentation == 'dichotic' else 'A+B',
               'right': 'B' if presentation == 'dichotic' else 'A+B', 'simulated': False,
               'source_mode': 'recorded EEG via PlayerLSL' if player_raw_path else 'external LSL source'}

    def run_results(adapter, stop: Event):
        from mne_lsl.stream import StreamLSL
        from mne_lsl.lsl import local_clock
        from scripts.auditory import live
        origin = local_clock()

        def on_estimate(estimate, aligned, raw):
            if stop.is_set():
                return
            t = max(0., estimate.evidence_end-origin)
            valid = bool(estimate.valid and estimate.scores is not None)
            scores = estimate.scores if valid else None
            choice = gate.choice(scores)
            decision = ('A' if choice == 0 else 'B') if choice is not None else ('uncertain' if valid else 'unavailable')
            a, b = (float(scores[0]), float(scores[1])) if valid else (None, None)
            adapter.publish(AttentionResult(t, 'nova_aad', decision, a, b, False))
            adapter.publish(PredictionResult(
                'ok' if valid else 'invalid', 'nova_aad', 'NOVA auditory attention', 'auditory_attention',
                outputs=(PredictionOutput('correlation_a', a, 'correlation', 'A / left' if presentation == 'dichotic' else 'A'),
                         PredictionOutput('correlation_b', b, 'correlation', 'B / right' if presentation == 'dichotic' else 'B')),
                timestamp=t, window_start=max(0., float(aligned.timestamps[0])-origin),
                window_end=t, reasons=tuple(estimate.reasons),
                metadata={'simulated': False, 'scientific_interpretation': True,
                          'presentation': mapping, 'decision_gate': gate_data,
                          'bad_channels': list(raw.bad_channels),
                          'acoustic_latency_verified': timing_profile is not None,
                          'diotic_accuracy': 'unmeasured'}))

        def on_status(now, aligned, status):
            if stop.is_set():
                return
            t = max(0., now-origin)
            adapter.publish(SyncResult(t, 'nova_aad', status='software_clock_observed',
                offset_ms=None, drift_warning=None, clock_domain=status['clock_domain'],
                max_block_timing_error_seconds=status['max_block_timing_error_seconds'],
                estimated_audio_drift_ppm=status['estimated_audio_drift_ppm']))
            gains = status['gains']
            adapter.publish(GainResult(t, 'nova_aad', min(0., 20*math.log10(gains[0])),
                                       min(0., 20*math.log10(gains[1]))))
            bad = list(status['quality'].get('current_bad_channels', status['quality']['bad_channel_windows']))
            adapter.publish(SignalQualityResult(t, 'nova_aad',
                artifact=None if aligned is None else bool(aligned.reasons or bad), bad_channels=bad))
            if aligned is not None:
                step = max(1, round(model.config.sample_rate/32))
                # At 2 Hz send only a short recent display, never a raw source block.
                values = aligned.eeg[-round(model.config.sample_rate): :step, :4].T.tolist()
                adapter.publish(EEGDisplayResult(t, 'nova_aad', model.contract['eeg_channels'][:4],
                                                 values, model.config.sample_rate/step, False))

        stream = StreamLSL(bufsize=20., name=stream_name)
        deadline = monotonic()+20
        player = None
        marker_receiver = None
        try:
            if player_raw_path:
                import subprocess
                import sys
                # One finite source per session. No GIL contention and no wrap.
                player = subprocess.Popen([sys.executable, '-m', 'scripts.attune.player',
                    '--raw', str(player_raw_path), '--name', stream_name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if marker_name:
                from scripts.attune.events import MarkerReceiver
                marker_receiver = MarkerReceiver(marker_name, ledger=ledger)
            while not stop.is_set():
                try:
                    stream.connect(acquisition_delay=None, processing_flags=['clocksync'], timeout=1.)
                    break
                except RuntimeError:
                    if monotonic() >= deadline:
                        raise
                    stop.wait(.05)
            if stop.is_set():
                return
            adapter._publish('presentation', 0., 'nova_aad', mapping)
            audio, report = live.run(trial, model, stream,
                source_unit_exponent=source_unit_exponent, output=output,
                timing_profile=timing_profile, presentation=presentation,
                check_channels=check_channels, timebase=timebase, ledger=ledger,
                stop_event=stop, on_estimate=on_estimate, on_status=on_status, marker_receiver=marker_receiver)
            if report_dir:
                from scipy.io import wavfile
                folder = Path(report_dir)/uuid4().hex
                folder.mkdir(parents=True, exist_ok=False)
                wavfile.write(folder/'mixed.wav', round(trial.audio_rate), audio)
                (folder/'timing.json').write_text(json.dumps(report, indent=2))
            if report.get('consumer_errors', 0):
                raise RuntimeError(f"Dashboard consumer failed {report['consumer_errors']} times")
        except Exception:
            log.exception('NOVA session failed')
            raise
        finally:
            if marker_receiver is not None:
                marker_receiver.close()
            if player is not None:
                player.terminate()
                try:
                    player.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    player.kill()
                    player.wait(timeout=.5)
            if stream.connected:
                stream.disconnect()

    return run_results
