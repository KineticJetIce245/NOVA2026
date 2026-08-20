from pathlib import Path

import mne


def load(set_file: str | Path) -> mne.io.eeglab.eeglab.RawEEGLAB:
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
