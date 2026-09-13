"""Launch ATTUNE with a fresh finite PlayerLSL recording source per UI session.

Run the frontend in a second terminal with npm run dev --prefix ../attune-ui/frontend.
This serves recorded data with real inference, paced WAV output, and no hardware claim.
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from nova2026.auditory.data import load_trial, save_trial


def main():
    import mne
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ui', type=Path, default=Path('../attune-ui').resolve())
    p.add_argument('--presentation', choices=('dichotic', 'diotic'), default='dichotic')
    p.add_argument('--seconds', type=float, default=240.)
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--out', type=Path, default=Path('results/attune-server'))
    args = p.parse_args()
    trial = load_trial(args.trial)
    if not np.isfinite(args.seconds) or not 10 <= args.seconds < len(trial.eeg)/trial.sample_rate-15:
        raise ValueError('Demo needs >=10 seconds plus at least 15 seconds of real source tail.')
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = (args.out/'source_raw.fif').resolve()
    raw = mne.io.RawArray(trial.eeg.T/1e6, mne.create_info(list(trial.channel_names),
                           round(trial.sample_rate), 'eeg'), verbose='ERROR')
    raw.save(raw_path, overwrite=True, verbose='ERROR')
    trial.audio = trial.audio[:round(args.seconds*trial.audio_rate)]
    trial_path = (args.out/'live_trial.npz').resolve()
    save_trial(trial, trial_path)
    root = Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH'] = os.pathsep.join([str(root/'src'), str(root), str(args.ui.resolve())])
    sys.path.insert(0, str(args.ui.resolve()))
    os.environ.update(ATTUNE_NOVA_TRIAL=str(trial_path), ATTUNE_NOVA_MODEL=str(args.model.resolve()),
        ATTUNE_NOVA_STREAM='attune-recording-player', ATTUNE_NOVA_PLAYER_RAW=str(raw_path),
        ATTUNE_NOVA_OUTPUT='wav', ATTUNE_NOVA_CHANNEL_POLICY='record-only',
        ATTUNE_NOVA_PRESENTATION=args.presentation, ATTUNE_NOVA_REPORT_DIR=str(args.out.resolve()/'runs'))
    uvicorn.run('backend.app.server:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
