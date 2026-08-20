from pathlib import Path

import mne


def load(set_file: str | Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
