"""ATTUNE wiring regressions independent of EEG or audio hardware."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import subprocess
import sys
import numpy as np
import pytest
from scipy.io import wavfile
from nova2026.auditory.audio import AudioMixer, DichoticMixer, DioticMixer
from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import AttentionEstimate, AuditoryWindow
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.gate import DecisionGate
from nova2026.auditory.timing import OnlineTimestampedAudio, TimestampedAudio
from scripts.attune.events import EpochLedger
from scripts.attune.stimuli import load_candidates, check_candidates, make_live_trial
from scripts.getlive.cap import EEGO24_CHANNELS, select_profile


@pytest.mark.parametrize('mixer', [AudioMixer, DichoticMixer, DioticMixer])
@pytest.mark.parametrize('samples,target', [([[2., 0.]], [1., 1.]),
    ([[0., np.nan]], [1., 1.]), ([[0., 0.]], [-1., 1.]), ([[0., 0.]], [1.]),
    ([0., 0.], [1., 1.])])
def test_mixers_reject_unsafe_input(mixer, samples, target):
    with pytest.raises(ValueError):
        mixer(100).process(samples, target)


def test_stereo_routing_ducking_and_mono_equivalence():
    rng = np.random.default_rng(72)
    audio = rng.uniform(-1, 1, (4000, 2))
    mono, diotic, dichotic = AudioMixer(100), DioticMixer(100), DichoticMixer(100)
    np.testing.assert_array_equal(dichotic.process(audio[:100], [1, 1]), (.49*audio[:100]).astype('float32'))
    for target in ([1, 1], [1, 10**(-6/20)], [10**(-6/20), 1]):
        m = mono.process(audio, target)
        d = diotic.process(audio, target)
        np.testing.assert_array_equal(d[:, 0], m)
        np.testing.assert_array_equal(d[:, 0], d[:, 1])
    steady = dichotic.process(np.ones((4000, 2)), [1, 10**(-6/20)])[-1]
    np.testing.assert_allclose(steady, [.49, .49*10**(-6/20)], rtol=1e-6)
    assert dichotic.process(np.empty((0, 2)), [1, 1]).shape == (0, 2)


def test_online_equivalence_causality_and_bounded_ring():
    rng = np.random.default_rng(13)
    audio = rng.uniform(-.9, .9, (16000, 2))
    config = AuditoryConfig()
    offline = TimestampedAudio(audio, 200, config)
    online = OnlineTimestampedAudio(200, config, retain_seconds=10)
    checked = 0
    for i in range(0, len(audio), 8):
        offline.record(i, 8, 100+i/200)
        online.record(i, 8, 100+i/200, audio[i:i+8])
        if i > 1200 and i % 200 == 0:
            times = 100+i/200-4+np.arange(96)/32
            w = SimpleNamespace(timestamps=times, data=np.zeros((96, 2)), reasons=(),
                                valid=True, available_at=times[-1], contract=None, segment=0)
            a, b = offline.align(w), online.align(w)
            assert a.reasons == b.reasons
            np.testing.assert_array_equal(a.envelopes, b.envelopes)
            checked += 1
    assert checked > 60 and len(online.values) <= 10*config.sample_rate and len(online.blocks) < 310
    w.timestamps += 1000
    assert 'audio_unavailable' in online.align(w).reasons
    with pytest.raises(ValueError):
        online.record(len(audio)+1, 8, 200, audio[:8])
    with pytest.raises(ValueError):
        online.record(len(audio), 8, 200, audio[:7])


def test_gate_null_quantile_centering_and_dwell_expiry():
    null = np.random.default_rng(51).normal(.1, .02, 1000)
    gate = DecisionGate.fit(null)
    assert sum(gate.choice([d, 0]) is not None for d in null)/len(null) <= .01
    assert gate.choice([gate.center, 0]) is None
    assert gate.choice([.9, -.9]) == 0
    assert gate.choice([-.9, .9]) == 1
    c = AttentionController(gate=gate)
    for i in range(3):
        c.update(AttentionEstimate([.9, -.9], i, i), i)
        assert c.choice(i) == (0 if i == 2 else None)
    assert c.choice(6) is None
    c.update(AttentionEstimate(None, 7, 7, False), 7)
    np.testing.assert_array_equal(c.gains(7), [1, 1])
    assert DecisionGate(**json.loads(json.dumps(gate.to_dict()))) == gate


@pytest.mark.parametrize('values', [[], [0]*100, [np.nan]*100, [0, 1]*5])
def test_gate_rejects_inadequate_null(values):
    with pytest.raises(ValueError):
        DecisionGate.fit(values)


def test_gate_precision_is_unavailable_without_fires():
    gate = DecisionGate(0, 1, 3)
    assert gate.metrics([[0, 0]], [0]) == {'windows': 1, 'fires': 0, 'fire_rate': 0., 'precision': None}


def test_candidate_pcm_layout_and_unknown_truth(tmp_path):
    a = np.random.default_rng(3).integers(-30000, 30000, (1000, 2), dtype=np.int16)
    wavfile.write(tmp_path/'stereo.wav', 200, a)
    wavfile.write(tmp_path/'a.wav', 200, a[:, 0])
    wavfile.write(tmp_path/'b.wav', 200, a[:, 1])
    stereo, rate = load_candidates('dichotic', tmp_path/'stereo.wav', level_match=False)
    pair, _ = load_candidates('pair', tmp_path/'a.wav', tmp_path/'b.wav', level_match=False)
    np.testing.assert_array_equal(stereo, pair)
    np.testing.assert_array_equal(stereo, a/32768.)
    trial = make_live_trial('dichotic', [tmp_path/'stereo.wav'], 500, EEGO24_CHANNELS, 'CPz', 'test')
    assert np.all(trial.labels == -1) and trial.channel_names == EEGO24_CHANNELS
    with pytest.raises(ValueError):
        load_candidates('dichotic', tmp_path/'a.wav')
    with pytest.raises(ValueError):
        check_candidates(np.repeat(stereo[:, :1], 2, axis=1), rate, AuditoryConfig())


def test_ledger_masks_boundaries_and_refuses_overlap():
    ledger = EpochLedger()
    first = ledger.add(100, 110, 't', 'A', 'dichotic', '1')
    ledger.add(110, 125, 't', 'B', 'diotic', '2')
    assert ledger.for_window(101, 106) == first
    assert ledger.for_window(100.5, 105) is None
    assert ledger.for_window(108, 113) is None
    with pytest.raises(ValueError):
        ledger.add(120, 130, 't', 'A', 'diotic', '3')


def test_montage_profile_operator_declaration():
    profile = select_profile('eego24', EEGO24_CHANNELS)
    assert (profile.reference, profile.ground) == ('CPz', 'Fpz')
    assert len(profile.channels) == 24
    with pytest.raises(ValueError):
        select_profile('eego24', EEGO24_CHANNELS[:-1])


def test_data_package_does_not_import_torch():
    subprocess.run([sys.executable, '-c',
        "import nova2026.data; import sys; assert 'torch' not in sys.modules; assert nova2026.data.Pipeline"], check=True)


def test_flat_channel_requires_explicit_policy_and_survives_save(tmp_path):
    rng = np.random.default_rng(2)
    eeg = rng.normal(size=(160, 2))
    eeg[:, 1] = 0
    contract = {'eeg_channels': ['A', 'B']}
    w = AuditoryWindow(eeg, rng.normal(size=(160, 2)), np.arange(160)/AuditoryConfig().sample_rate, 5, contract=contract)
    examples = [(w, np.zeros(160, dtype=int))]
    with pytest.raises(ValueError, match='flat'):
        RidgeDecoder().fit(examples)
    model = RidgeDecoder().fit(examples, allow_flat_channels=('B',))
    scores = model.score(w)
    model.save(tmp_path/'model.npz')
    loaded = RidgeDecoder.load(tmp_path/'model.npz')
    w.eeg[:, 1] = 1e9
    np.testing.assert_array_equal(loaded.score(w), scores)


def test_replay_policy_and_offset_forwarding():
    from scripts.auditory.train import prepare
    with patch('scripts.auditory.train.replay_windows', return_value=[]) as replay:
        prepare([object()], AuditoryConfig(), 5, audio_offset=.123,
                check_channels=False, max_bad_channels=2, exclude_channels=('F8',))
    assert replay.call_args.kwargs == dict(audio_offset=.123, check_channels=False,
                                           max_bad_channels=2, exclude_channels=('F8',))
