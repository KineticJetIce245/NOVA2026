"""Convert the two known ANT recordings to aligned, stimulus-disjoint trials.

Operator-supplied onset anchors are used. Acoustic residual remains unverified.
The shared stimulus region [267,306) is removed from BOTH partitions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from nova2026.auditory.data import AuditoryTrial, save_trial
from scripts.getlive.cap import EEGO24_CHANNELS, EEGO24_REFERENCE, EEGO24_GROUND
from .stimuli import load_candidates, check_candidates
from nova2026.auditory.config import AuditoryConfig

SESSIONS = {
    'Lacroix_Flo2_2026-09-12_19-34-06': (5.006, 0., 0., 267.),
    'Lacroix_Flo2_2026-09-12_19-41-11': (12.838, 267., 306., 582.),
}


def convert(path, stimulus):
    import mne
    path = Path(path)
    if path.stem not in SESSIONS:
        raise ValueError('Unknown session: declare its anchors and stimulus split explicitly.')
    anchor, offset, low, high = SESSIONS[path.stem]
    raw = mne.io.read_raw_ant(path, preload=True, verbose='ERROR')
    if tuple(raw.ch_names) != EEGO24_CHANNELS or raw.info['sfreq'] != 500.:
        raise ValueError('CNT montage/rate differs from the measured eego24 recording.')
    eeg = raw.get_data().T * 1e6
    # Whole-session centering is used only for this offline recording/replay.
    # A physical source must supply its declared upstream DC removal independently.
    eeg -= eeg.mean(axis=0)
    times = np.arange(len(eeg))/500.
    cues, starts = [], []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        code = str(description).split('/')[0]
        # Operator-confirmed accidental presses in session 2, not heuristics.
        if offset == 267. and any(abs(onset-t) < .002 for t in (148.354, 148.402, 326.692)):
            continue
        if code == '1004':
            starts.append(float(onset))
        if code in ('1001', '1006'):
            cues.append((float(onset), 0 if code == '1001' else 1))
    if len(starts) != 1 or abs(starts[0]-anchor) > .002:
        raise ValueError('Start marker differs from operator anchor.')
    cue_counts = [sum(c == side for _, c in cues) for side in (0, 1)]
    # Actual supplied session 2 has 12/11 after the mandated exclusions,
    # contradicting the handoff's 12/12 claim. Do not invent the missing cue.
    if cue_counts != ([12, 11] if offset == 267. else [12, 12]):
        raise ValueError('Unexpected cue census after explicit exclusions.')
    labels = np.full(len(eeg), -1)
    for onset, side in cues:
        labels[times >= onset] = side
    for onset, _ in cues:
        labels[np.abs(times-onset) <= .5] = -1
    labels[times < anchor] = -1
    if offset == 267.:
        labels[(times >= 148.354-.5) & (times <= 161.224+.5)] = -1
    stimulus_times = offset + times-anchor
    keep = (stimulus_times >= low) & (stimulus_times < high)
    audio, rate = load_candidates('dichotic', stimulus, level_match=False)
    # Start both arrays at exactly the same stimulus sample, absorbing the
    # operator offset here rather than relying on a per-file training CLI flag.
    first = float(stimulus_times[keep][0])
    count = int(keep.sum())
    begin = round(first*rate)
    audio = audio[begin:begin+round(count/500.*rate)]
    corr = check_candidates(audio, rate, AuditoryConfig())
    trial = AuditoryTrial(eeg[keep], np.arange(count)/500., audio, rate, labels[keep],
                          'Lacroix_Flo2', path.stem, EEGO24_CHANNELS, EEGO24_REFERENCE,
                          'ANT recording; volts to uV; per-session DC mean removed',
                          group=f'dichotic_15min[{low:g},{high:g})')
    trial.presentation = 'dichotic'
    report = {'source': path.name, 'presentation': 'dichotic', 'reference': EEGO24_REFERENCE,
              'ground': EEGO24_GROUND, 'reference_ground_source': 'operator declaration 2026-09-13',
              'stimulus_interval': [first, first+count/500.], 'dropped_overlap': [267., 306.],
              'anchor_seconds': anchor, 'stimulus_offset_seconds': offset,
              'audio_offset_seconds': 0., 'acoustic_residual_seconds': None,
              'cue_counts': cue_counts, 'candidate_correlation': corr,
              'eeg_abs_p95_uv': float(np.quantile(np.abs(trial.eeg), .95)),
              'eeg_abs_max_uv': float(np.max(np.abs(trial.eeg))),
              'limitation': 'Same participant and source pair; disjoint segments, not independent-pair generalization.'}
    return trial, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=Path('datasets/experiment'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SESSIONS:
        trial, report = convert(args.recordings/(name+'.cnt'),
                                args.recordings/'audio_files/dichotic_15min.wav')
        destination = args.out/(name+'.npz')
        save_trial(trial, destination)
        destination.with_suffix('.json').write_text(json.dumps(report, indent=2))
        print(destination, report['eeg_abs_p95_uv'])


if __name__ == '__main__':
    main()
