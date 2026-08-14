import mne

file_path = "datasets/COG-BCI/sub-01/sub-01/ses-S1/eeg/zeroBACK.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)

print(raw.info)
raw.plot(n_channels=30, duration=10, block=True)
print(raw.annotations)
