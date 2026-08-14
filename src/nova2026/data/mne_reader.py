import mne
from nova2026.config import DATASET

file_path = DATASET
raw = mne.io.read_raw_eeglab(file_path, preload=True)

print(raw.info)
raw.plot(n_channels=30, duration=10, block=True)
print(raw.annotations)
