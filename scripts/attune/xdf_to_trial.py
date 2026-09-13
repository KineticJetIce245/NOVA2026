"""Convert explicitly labelled eego24 EEG using audio:onset and cue epoch markers."""
import argparse
import json
from pathlib import Path
import numpy as np
from nova2026.auditory.data import AuditoryTrial, save_trial
from nova2026.streaming.timebase import TimeBase, GridPolicy
from scripts.getlive.cap import EEGO24_CHANNELS, EEGO24_REFERENCE
from .stimuli import load_candidates


def convert_streams(streams, audio, rate, *, subject, trial_id, group,
                    source_unit_exponent=0, timebase='grid'):
    eegs = [s for s in streams if s['info']['type'][0].lower() == 'eeg']
    markers = [s for s in streams if s['info']['name'][0] == 'AAD_Markers']
    if len(eegs) != 1 or len(markers) != 1:
        raise ValueError('Select exactly one EEG and AAD_Markers stream.')
    eeg = eegs[0]
    try:
        names = tuple(c['label'][0] for c in eeg['info']['desc'][0]['channels'][0]['channel'])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Real EEG requires channel labels; Ch1..ChN fallback is forbidden.') from error
    if names != EEGO24_CHANNELS:
        raise ValueError('XDF montage must match the declared eego24 order.')
    sfreq = float(eeg['info']['nominal_srate'][0])
    times = np.asarray(eeg['time_stamps'], float)
    if timebase == 'grid':
        times, _ = TimeBase(GridPolicy.for_rate(sfreq)).place(times)
    elif timebase != 'off' or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid timebase or non-increasing raw timestamps.')
    events = [(float(t), json.loads(s[0])) for t, s in zip(markers[0]['time_stamps'], markers[0]['time_series'])]
    onsets = [t for t, e in events if e.get('event') == 'audio:onset']
    if len(onsets) != 1:
        raise ValueError('One first-DAC-block audio:onset marker is required; session:start is not an audio anchor.')
    onset = onsets[0]
    keep = (times >= onset) & (times < onset+len(audio)/rate)
    samples = np.asarray(eeg['time_series'], float)[keep]*10.**(source_unit_exponent+6)
    samples -= samples.mean(axis=0)
    selected = times[keep]
    labels = np.full(len(selected), -1)
    for _, e in events:
        if 'cued_side' in e:
            side = e['cued_side']
            if side not in ('A', 'B'):
                raise ValueError('Invalid cue side.')
            labels[(selected > e['t_start']+.5) & (selected < e['t_end']-.5)] = int(side == 'B')
    trial = AuditoryTrial(samples, selected-onset, audio, rate, labels, subject, trial_id,
                          names, EEGO24_REFERENCE, 'XDF; declared units to uV; per-session DC mean removed', group)
    trial.audio_offset = -float(trial.timestamps[0])
    trial.presentation = next(e.get('presentation') for _, e in events if e.get('event') == 'audio:onset')
    return trial


def main():
    import pyxdf
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xdf', required=True)
    p.add_argument('--form', choices=('pair', 'dichotic'), required=True)
    p.add_argument('--audio', nargs='+', required=True)
    p.add_argument('--subject', required=True)
    p.add_argument('--trial-id', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--source-unit-exponent', type=int, default=0)
    p.add_argument('--timebase', choices=('grid', 'off'), default='grid')
    args = p.parse_args()
    audio, rate = load_candidates(args.form, *args.audio)
    streams, _ = pyxdf.load_xdf(args.xdf, dejitter_timestamps=False)
    trial = convert_streams(streams, audio, rate, subject=args.subject, trial_id=args.trial_id,
                            group='|'.join(Path(a).name for a in args.audio),
                            source_unit_exponent=args.source_unit_exponent, timebase=args.timebase)
    save_trial(trial, args.out)


if __name__ == '__main__':
    main()
