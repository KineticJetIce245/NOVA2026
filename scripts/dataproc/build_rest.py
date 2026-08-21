"""Build prototype Resting datasets."""

from pathlib import Path

import numpy as np
import torch
from loadnsave import load_run, load_sessions, load_subjects, save_chkpt
from pipelines import AttUPipeline

from nova2026.config import DATA_DIR, SAMPLE_RATE
from nova2026.data.pipeline import DefaultPipe, Pipeline

DATASET_ROOT = DATA_DIR / "COG-BCI"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "outputs"
DATASET = "RS_Beg_EC"

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


def label_data(
    data_type: str = DATASET,
    pipeline: Pipeline | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    # Build one pipeline-specific PVT dataset and return its output path.
    selected_subjects = load_subjects(DATASET_ROOT)
    pipeline = pipeline if pipeline is not None else DefaultPipe()
    data_list: list[np.ndarray] = []
    metadata_list: list[tuple[str, str]] = []

    for subject in selected_subjects:
        subject_path = DATASET_ROOT / subject
        if not subject_path.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        for session in load_sessions(subject, DATASET_ROOT):
            raw = load_run(subject, session, DATASET_ROOT, data_type)
            print(raw.n_times)

            pipeline.rundown(raw)

            if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
                raise ValueError(
                    f"Pipeline {type(pipeline).__name__} produced "
                    f"{raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
                )

            missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
            if missing_channels:
                raise ValueError(
                    f"Recording is missing EEG channels: {missing_channels}."
                )

            raw.pick(EEG_CHANNELS)

            data_raw = raw.get_data(picks=EEG_CHANNELS) * 1e6  # pyright: ignore
            # Discard the first and last second of data to avoid filter edge effects
            data_raw = data_raw[:, 128:-128]
            print(data_raw.shape)
            for w in range(data_raw.shape[1] // 256):
                data_list.append(data_raw[:, w * 256 : w * 256 + 256])
                metadata_list.append((subject, session))

    if not data_list:
        raise ValueError(f"No qualified {DATASET} trials were found.")

    data_array = np.asarray(data_list)  # (n-trials, 62, 256)
    metadata_array = np.asarray(metadata_list, dtype=object)

    print(data_array.shape)
    print(metadata_array.shape)
    print(pipeline.__class__.__name__)
    checkpoint = {
        "data": torch.from_numpy(data_array).float(),
        "metadata": metadata_array,
        "channel_names": list(EEG_CHANNELS),
        "pipeline": type(pipeline).__name__,
        "sample_rate_hz": SAMPLE_RATE,
        "signal_unit": "microvolts",
        "data_type": DATASET,
    }
    save_chkpt(checkpoint, output_dir)


label_data(DATASET, pipeline=AttUPipeline())
