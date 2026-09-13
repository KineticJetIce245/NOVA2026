from types import SimpleNamespace
import numpy as np
import pytest
from scripts.attune.calibration_session import calibration_callback
from scripts.attune.xdf_to_trial import convert_streams
from scripts.getlive.cap import EEGO24_CHANNELS
from nova2026.auditory.data import save_trial, load_trial


def test_onset_comes_from_first_dac_callback_and_both_arms():
    for arm in ('dichotic', 'diotic'):
        events = []
        markers = SimpleNamespace(push=lambda event, stamp: events.append((event, stamp)))
        callback = calibration_callback(np.ones((4000, 2))*.1, 200, arm, 10., markers, lambda: 100.)
        assert events == []
        out = np.empty((8, 2))
        callback(out, 8, SimpleNamespace(outputBufferDacTime=20.07, currentTime=20.), False)
        assert events[0][0]['event'] == 'audio:onset'
        assert events[0][1] == pytest.approx(100.07)
        assert events[1][0].condition == arm
        assert np.all(out > 0)


def streams(labels=True, onset=True):
    eeg = {'info': {'type': ['EEG'], 'name': ['eego'], 'nominal_srate': ['500']},
           'time_stamps': 100.001+np.arange(1000)/500,
           'time_series': np.random.default_rng(8).normal(0, 1e-6, (1000, 24))}
    if labels:
        eeg['info']['desc'] = [{'channels': [{'channel': [{'label': [n]} for n in EEGO24_CHANNELS]}]}]
    event = 'audio:onset' if onset else 'session:start'
    marker = {'info': {'type': ['Markers'], 'name': ['AAD_Markers']},
              'time_stamps': [100.], 'time_series': [['{"event": "'+event+'"}']]}
    return [eeg, marker]


def test_xdf_requires_labels_and_audio_anchor():
    for labels, onset, match in [(False, True, 'labels'), (True, False, 'audio:onset')]:
        with pytest.raises(ValueError, match=match):
            convert_streams(streams(labels, onset), np.ones((1000, 2))*.1, 200,
                            subject='s', trial_id='t', group='g')


def test_xdf_preserves_samples_unknown_truth_and_offset_roundtrip(tmp_path):
    trial = convert_streams(streams(), np.ones((1000, 2))*.1, 200,
                            subject='s', trial_id='t', group='g')
    assert len(trial.eeg) == 1000 and np.all(trial.labels == -1)
    assert trial.audio_offset == pytest.approx(-.001)
    assert trial.reference == 'CPz'
    save_trial(trial, tmp_path/'t.npz')
    restored = load_trial(tmp_path/'t.npz')
    assert restored.audio_offset == trial.audio_offset
    np.testing.assert_array_equal(restored.timestamps, trial.timestamps)
