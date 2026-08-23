"""Rewritten script for loading PVT EEG data from the COG-BCI dataset."""

import mne
import numpy as np
import torch
from pipelines import AttUPipeline
from save_pt import save_chkpt

from nova2026.config import DATA_DIR, SAMPLE_RATE, SAMPLE_SIZE
from nova2026.data.eeg import Loader

OUTPUT_DIR = DATA_DIR / "COG-BCI" / "outputs"
DIR_MASK = (DATA_DIR / "COG-BCI").parts
EEG_CHANNELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "FCz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
]


def tagging_cogbci(datarf: Loader.DataFileRef) -> list[str]:
    tag_list = [t for t in datarf.path.parts[:-2] if t not in DIR_MASK]
    tag_list[0] = tag_list[0].replace("sub-", "")
    tag_list[1] = tag_list[1].replace("ses-S", "")
    return tag_list


def data_integrity_check(raw: mne.io.BaseRaw) -> None:
    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
        raise ValueError(
            f"Pipeline {type(pipeline).__name__} produced "
            f"{raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
        )

    missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Recording is missing EEG channels: {missing_channels}.")


# create a new loader
loader = Loader("COG-BCI", root=DATA_DIR)
data_type = "RS_Beg_EC"
# search for files named as RS_Beg/End_EC/EO
loader.search(data_type, "name")
# refine for files with .set extension
loader.refine(lambda datarf: datarf.path.suffix == ".set")
# tag the files based on "sub" and "ses"
loader.tag(tagging_cogbci)
# sort the quary cache
loader.query_cache.sort(key=lambda q: (int(q.tags[0]), int(q.tags[1])))  # pyright: ignore
# load the data into TaggedData
tagged_data_list: list[Loader.TaggedData] = loader.load(mode="eeglab")

pipeline = AttUPipeline()
# 25 participants, 3 sessions
data_list = []


for tagged_data in tagged_data_list:
    raw: mne.io.BaseRaw = tagged_data.raw
    raw.load_data()
    assert raw is not None

    pipeline.rundown(raw)
    data_integrity_check(raw)
    raw.pick(EEG_CHANNELS)

    data_raw = raw.get_data(picks=EEG_CHANNELS) * 1e6  # pyright: ignore
    # Discard the first and last second of data to avoid filter edge effects
    data_raw = data_raw[:, 128:-128]
    windows_list = []
    for w in range(data_raw.shape[1] // 256):
        windows_list.append(data_raw[:, w * 256 : w * 256 + 256])
    windows_list_np = np.stack(windows_list, axis=0)
    data_list.append(windows_list_np)

data_list = np.asarray(data_list)
print(data_list.shape)
data_list = data_list.reshape(25, 3, *(data_list.shape[-3:]))
print(data_list.shape)

checkpoint = {
    "data": torch.from_numpy(data_list).float(),
    "channel_names": list(EEG_CHANNELS),
    "pipeline": type(pipeline).__name__,
    "sample_rate_hz": SAMPLE_RATE,
    "window_length_ms": SAMPLE_SIZE,
    "data_type": data_type,
}
save_chkpt(checkpoint, OUTPUT_DIR)
