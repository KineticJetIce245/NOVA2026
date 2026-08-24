"""Build the PVT trial checkpoint: ``outputs/PVT_128Hz_AttUPipeline.pt``.

Preprocessing
-------------
Every recording passes through :class:`AttUPipeline` (``pipelines.py``), i.e.
the operator P = K o B o R o B o N of the engagement-z-scoring design doc
(``documents/engagement_zscoring.pdf``, Eq. 1): 60 Hz notch (10 Hz width) ->
4-20 Hz zero-phase Butterworth band-pass -> resample 500 -> 128 Hz ->
4-20 Hz band-pass again -> per-channel centre-scale-clip
((v - mean)/8 clipped to [-4, 4] uV).

Trial windows
-------------
For each PVT trial, the window is the 256 samples (2000 ms @ 128 Hz) ending
100 ms before stimulus onset (guard delta = 100 ms).  A trial is admitted
only if the preceding response *and* the preceding error event both cleared
by more than 2200 ms (T/fs + 200 ms), so no post-response activity enters
the window.  The slowest 10 % of admitted trials by RT (within-session
decile) are labelled 1, the remaining 90 % are labelled 0.

Checkpoint schema (dict returned by ``torch.load``)
---------------------------------------------------
data            (N, 62, 256) float32   EEG in microvolts; channel order is
                    EEG_CHANNELS (62 channels); 256 samples @ 128 Hz =
                    2000 ms, window ends 100 ms before stimulus onset
labels          (N,) int64             0/1, 1 = slowest-RT decile of the session
metadata        (N, 3) object array    columns [subject, session, RT_ms];
                    subject like "sub-01", session like "ses-S1"
channel_names   list[str] of 62 channel names (EEG_CHANNELS)
pipeline        str                    "AttUPipeline"
sample_rate_hz  int                    128
window_length_ms int                   2000
data_type       str                    "PVT"

Row ``i`` of ``data``/``labels``/``metadata`` refer to the same trial.
"""

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


def select_labeled_trials(trials: np.ndarray) -> list[tuple[np.ndarray, int]]:
    # Apply the existing qualification and slowest-10-percent labeling rule.
    if trials.ndim != 2 or trials.shape[1] != 4:
        raise ValueError("trials must have shape (n_trials, 4).")
    if not len(trials):
        return []

    qualified = trials[trials[:, 1] > SAMPLE_SIZE + 200]
    if not len(qualified):
        return []

    ordered = qualified[np.argsort(qualified[:, 0])]
    positive_count = int(len(ordered) * 0.1)  # 10%

    if positive_count == 0:
        positive = ordered[:0]
        negative = ordered
    else:
        positive = ordered[-positive_count:]
        negative = ordered[:-positive_count]

    return [
        *((trial, 1) for trial in positive),
        *((trial, 0) for trial in negative),
    ]


def data_integrity_check(raw: mne.io.BaseRaw) -> None:

    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
        raise ValueError(
            f"Pipeline produced {raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
        )

    missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Recording is missing EEG channels: {missing_channels}.")


def extract_trial_window(raw: mne.io.BaseRaw, stimulus_time_ms: int) -> np.ndarray:
    # Extract one DNN-aligned EEG window in microvolts.

    # translate the sample size from ms to index
    sample_points_num = int(SAMPLE_SIZE / 1000 * SAMPLE_RATE)
    stop_s = (stimulus_time_ms - 100) / 1000
    # converts s to index
    stop_idx = int(raw.time_as_index(stop_s)[0])
    start_idx = stop_idx - sample_points_num

    if start_idx < 0 or stop_idx > raw.n_times:
        raise ValueError(
            "Trial window falls outside the recording: "
            f"start={start_idx}, stop={stop_idx}, n_times={raw.n_times}."
        )

    window_uv = (
        raw.get_data(
            picks=EEG_CHANNELS,
            start=start_idx,
            stop=stop_idx,
        )
        * 1e6
    )  # pyright: ignore

    expected_shape = (len(EEG_CHANNELS), sample_points_num)
    if window_uv.shape != expected_shape:
        raise ValueError(
            f"Expected trial window shape {expected_shape}, got {window_uv.shape}."
        )
    return window_uv


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
data_list: list[np.ndarray] = []
label_list: list[int] = []
metadata_list: list[tuple[str, str, int]] = []


for tagged_data in tagged_data_list:
    raw: mne.io.BaseRaw = tagged_data.raw
    raw.load_data()
    assert raw is not None

    labeled_trials = select_labeled_trials(get_trials(raw))
    pipeline.rundown(raw)
    data_integrity_check(raw)
    raw.pick(EEG_CHANNELS)

    for trial, label in labeled_trials:
        window_uv = extract_trial_window(raw, int(trial[2]))
        data_list.append(window_uv)
        label_list.append(label)
        metadata_list.append((tagged_data.tags[0], tagged_data.tags[1], int(trial[0])))


if not data_list:
    raise ValueError("No qualified PVT trials were found.")

data_array = np.asarray(data_list)  # (n-trials, 62, 256)
label_array = np.asarray(label_list, dtype=np.int64)
metadata_array = np.asarray(metadata_list, dtype=object)

checkpoint = {
    "data": torch.from_numpy(data_array).float(),
    "labels": torch.from_numpy(label_array).long(),
    "metadata": metadata_array,
    "channel_names": list(EEG_CHANNELS),
    "pipeline": type(pipeline).__name__,
    "sample_rate_hz": SAMPLE_RATE,
    "window_length_ms": SAMPLE_SIZE,
    "data_type": "PVT",
}
save_chkpt(checkpoint, OUTPUT_DIR)
