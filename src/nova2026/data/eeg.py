import mne
from pathlib import Path


def load(set_file: str | Path):
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
