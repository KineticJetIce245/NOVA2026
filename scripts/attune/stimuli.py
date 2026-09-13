"""Candidate A/left and B/right input, with measured envelope independence."""
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from nova2026.auditory.audio import read_audio
from nova2026.auditory.data import AuditoryTrial
from nova2026.auditory.envelopes import EnvelopeExtractor


def load_candidates(form, *paths, level_match=True):
    if form == 'pair' and len(paths) == 2:
        a, rate = read_audio(paths[0])
        b, other = read_audio(paths[1])
        if rate != other:
            raise ValueError('Candidate rates differ; resample first.')
        audio = np.column_stack([a[:min(len(a), len(b))], b[:min(len(a), len(b))]])
    elif form == 'dichotic' and len(paths) == 1:
        rate, samples = wavfile.read(paths[0])
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError('A dichotic stimulus needs exactly 2 channels.')
        dtype = samples.dtype
        audio = samples.astype(float)
        if np.issubdtype(dtype, np.unsignedinteger):
            midpoint = (np.iinfo(dtype).max + 1)/2
            audio = (audio-midpoint)/midpoint
        elif np.issubdtype(dtype, np.signedinteger):
            audio /= np.iinfo(dtype).max + 1
    else:
        raise ValueError('Use pair with two paths or dichotic with one path.')
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError('Candidate audio must be nonempty and finite.')
    rms = np.sqrt(np.mean(audio**2, axis=0))
    if np.min(rms) <= 0:
        raise ValueError('A candidate track is silent.')
    if level_match:
        audio = audio * (np.sqrt(rms[0]*rms[1])/rms)
    audio /= max(1., float(np.max(np.abs(audio))))
    return audio, float(rate)


def check_candidates(audio, rate, config, *, limit=.30):
    if not np.isfinite(limit) or not 0 < limit < 1:
        raise ValueError('Candidate correlation limit must be inside (0,1).')
    values = EnvelopeExtractor(rate, config).feed(audio[:int(min(300., len(audio)/rate)*rate)])
    values = values[int(config.sample_rate*2):]
    if len(values) < config.sample_rate or np.any(np.std(values, axis=0) < 1e-12):
        raise ValueError('Candidate audio too short or constant to validate.')
    r = float(np.corrcoef(values.T)[0, 1])
    if not np.isfinite(r) or abs(r) > limit:
        raise ValueError(f'Candidate envelopes correlate at r={r:+.3f} (limit {limit}); '
                         'check for duplicate, collapsed stereo, or HRTF tracks.')
    return r


def make_live_trial(form, paths, eeg_sfreq, channel_names, reference, upstream_processing,
                    *, subject='demo', trial_id='live'):
    audio, rate = load_candidates(form, *paths)
    # EEG is supplied by LSL. Two rows suffice to declare the rate; no invented truth.
    return AuditoryTrial(np.zeros((2, len(channel_names))), np.arange(2)/eeg_sfreq,
                         audio, rate, [-1, -1], subject, trial_id, channel_names,
                         reference, upstream_processing,
                         group='|'.join(Path(p).name for p in paths))
