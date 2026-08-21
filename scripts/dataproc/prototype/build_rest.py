"""Build prototype Resting datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from mne.io import BaseRaw
from pipelines import AttUPipeline

from nova2026.config import DATA_DIR, SAMPLE_RATE
from nova2026.data import eeg
from nova2026.data.pipeline import DefaultPipe, Pipeline

DATASET_ROOT = DATA_DIR / "COG-BCI"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "prototype_outputs"
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

PIPELINES: dict[str, Pipeline] = {
    "default": DefaultPipe(),
    "attentive-u": AttUPipeline(),
}


def load_subjects(root: Path = DATASET_ROOT) -> list[str]:
    """Return sorted COG-BCI subject directory names."""
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    )


def load_sessions(subject: str, root: Path = DATASET_ROOT) -> list[str]:
    """Return sorted session directory names for one subject."""
    subject_path = root / subject
    return sorted(
        path.name
        for path in subject_path.iterdir()
        if path.is_dir() and path.name.startswith("ses-")
    )


def load_run(subject: str, session: str, root: Path = DATASET_ROOT) -> BaseRaw:
    """Load one PVT recording."""
    return eeg.load(root / subject / session / "eeg" / f"{DATASET}.set")


def _output_path(pipeline_name: str, output_dir: Path) -> Path:
    return output_dir / f"{DATASET}_data_{pipeline_name}.pt"


def _save_checkpoint_atomically(checkpoint: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def label_data(
    pipeline: Pipeline | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    subjects: Sequence[str] | None = None,
) -> Path:
    # Build one pipeline-specific PVT dataset and return its output path.
    selected_subjects = list(subjects) if subjects is not None else load_subjects()
    pipeline = pipeline if pipeline is not None else DefaultPipe()
    data_list: list[np.ndarray] = []
    metadata_list: list[tuple[str, str]] = []

    for subject in selected_subjects:
        subject_path = DATASET_ROOT / subject
        if not subject_path.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        for session in load_sessions(subject):
            raw = load_run(subject, session)

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

            data_list.append(raw.get_data(picks=EEG_CHANNELS) * 1e6)  # pyright: ignore
            metadata_list.append((subject, session))

    if not data_list:
        raise ValueError(f"No qualified {DATASET} trials were found.")

    min_shape = tuple(np.min([a.shape for a in data_list], axis=0))
    print(f"Target crop shape: {min_shape}")

    # 2. Crop every array to that minimum shape (removes extra data at the end)
    cropped_list = [a[tuple(slice(0, s) for s in min_shape)] for a in data_list]

    # 3. Stack them into a single 3D array
    result = np.stack(cropped_list, axis=0)
    print(f"Final array shape: {result.shape}")  # (3, 62, 190)

    data_array = np.asarray(cropped_list)  # (n-trials, 62, 256)
    metadata_array = np.asarray(metadata_list, dtype=object)

    output_path = _output_path(type(pipeline).__name__, Path(output_dir))
    checkpoint = {
        "data": torch.from_numpy(data_array).float(),
        "metadata": metadata_array,
        "channel_names": list(EEG_CHANNELS),
        "pipeline": type(pipeline).__name__,
        "sample_rate_hz": SAMPLE_RATE,
        "signal_unit": "microvolts",
    }
    _save_checkpoint_atomically(checkpoint, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype PVT datasets.")
    parser.add_argument(
        "--pipeline",
        choices=[*sorted(PIPELINES)],
        default="default",
        help="Preprocessing pipeline to build. Defaults to both variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Torch files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional subject names, for example: --subjects sub-01 sub-02.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = PIPELINES.get(args.pipeline)
    output_path = label_data(
        pipeline=pipeline,
        output_dir=args.output_dir,
        subjects=args.subjects,
    )
    print(output_path)


if __name__ == "__main__":
    main()
