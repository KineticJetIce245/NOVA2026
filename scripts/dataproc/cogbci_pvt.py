import mne
import numpy as np
import torch
from pipelines import AttUPipeline

from nova2026.config import DATA_DIR, SAMPLE_RATE, SAMPLE_SIZE, WINDOW_SIZE
from nova2026.data.eeg import Loader, save_chkpt

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
    return [t for t in datarf.path.parts[:-2] if t not in DIR_MASK]


def get_trials(raw: mne.io.BaseRaw) -> np.ndarray:
    """Extract reaction-time and timing fields from PVT annotations."""
    trials: list[np.ndarray] = []
    stimulus_timestamp = 0
    response_timestamp = 0
    error_timestamp = 0

    for annotation in raw.annotations:
        timestamp = int(np.int32(annotation["onset"] * 1000))  # pyright: ignore
        description = str(annotation["description"])

        if description == "13":
            stimulus_timestamp = timestamp

        elif description == "14":
            trials.append(
                np.array(
                    [
                        timestamp - stimulus_timestamp,
                        min(
                            stimulus_timestamp - response_timestamp,
                            stimulus_timestamp - error_timestamp,
                        ),
                        stimulus_timestamp,
                        timestamp,
                    ],
                    dtype=np.int64,
                )
            )
            response_timestamp = timestamp

        elif description == "12":
            trials.append(np.array([-1, -1, -1, timestamp], dtype=np.int64))
            error_timestamp = timestamp

    if not trials:
        return np.empty((0, 4), dtype=np.int64)

    return np.stack(trials, axis=0)


def data_integrity_check(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:

    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
        raise ValueError(
            f"Pipeline produced {raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
        )

    missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Recording is missing EEG channels: {missing_channels}.")

    return raw


def pick_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.pick(EEG_CHANNELS)
    return raw


# create a new loader
loader = Loader("COG-BCI", root=DATA_DIR)
# search for files named as PVT
loader.search("PVT", "name")
# refine for files with .set extension
loader.refine(lambda datarf: datarf.path.suffix == ".set")
# tag the files based on "sub" and "ses"
loader.tag(tagging_cogbci)
# load the data into TaggedData
tagged_data_list: list[Loader.TaggedData] = loader.load(mode="eeglab")
pipeline = AttUPipeline()
print("Runing pipeline ... ")
loader.run_pipe(pipeline)
print("Performing integrity check ...")
loader.run(data_integrity_check)
loader.run(pick_channels)

cuts_list = []
rt_list = []
print("Generating cuts ...")
for tagged_data in tagged_data_list:
    raw: mne.io.BaseRaw = tagged_data.raw
    trials = get_trials(raw)
    trials = trials[trials[:, 1] > WINDOW_SIZE + 200]
    rt_list.extend(trials[:, 0])
    cuts_list.append(raw.time_as_index((trials[:, 2] - 100) / 1000))

print("Cuting the data ...")
meta_list, data_list = loader.slice(
    cuts_list,
    eeg_channels=EEG_CHANNELS,
    sample_size=SAMPLE_SIZE,
    cut_at_start=False,
    permutate=lambda array: array * 1e6,
)

if not data_list:
    raise ValueError("No qualified PVT trials were found.")

meta_list = [(tags[0], tags[1]) for tags in meta_list]
data_array = np.stack(data_list, axis=0)  # (n trials, 62, 256)
rt_array = np.asarray(rt_list)
# meta_array = np.asarray(meta_list, dtype=object)

checkpoint = {
    "data": data_array,
    "rt": rt_array,
    "metadata": meta_list,
    "channel_names": list(EEG_CHANNELS),
    "pipeline": type(pipeline).__name__,
    "sample_rate_hz": SAMPLE_RATE,
    "window_length_ms": WINDOW_SIZE,
    "data_type": "PVT",
}
save_chkpt(checkpoint, OUTPUT_DIR)
