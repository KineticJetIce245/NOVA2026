"""Build prototype PVT datasets with trial-level engagement features.

Run as a module from the repository root, for example::

    python -m scripts.dataproc.prototype.build_pvt --pipeline all
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from mne.io import BaseRaw

from nova2026.config import DATA_DIR, SAMPLE_LENGTH, SAMPLE_RATE, WINDOW_LENGTH
from nova2026.data import eeg

from .pipelines import PIPELINES, apply_pipeline
from .spectral import compute_trial_spectral_features


DATASET_ROOT = DATA_DIR / "COG-BCI"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "prototype_outputs"

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
    return eeg.load(root / subject / session / "eeg" / "PVT.set")


def get_trials(raw: BaseRaw) -> np.ndarray:
    """Extract reaction-time and timing fields from PVT annotations."""
    trials: list[np.ndarray] = []
    stimulus_timestamp = 0
    response_timestamp = 0
    error_timestamp = 0

    for annotation in raw.annotations:
        timestamp = int(np.int32(annotation["onset"] * 1000))
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
            trials.append(
                np.array([-1, -1, -1, timestamp], dtype=np.int64)
            )
            error_timestamp = timestamp

    if not trials:
        return np.empty((0, 4), dtype=np.int64)
    return np.stack(trials, axis=0)


def select_labeled_trials(trials: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Apply the existing qualification and slowest-10-percent labeling rule."""
    if trials.ndim != 2 or trials.shape[1] != 4:
        raise ValueError("trials must have shape (n_trials, 4).")
    if not len(trials):
        return []

    qualified = trials[trials[:, 1] > SAMPLE_LENGTH + 200]
    if not len(qualified):
        return []

    ordered = qualified[np.argsort(qualified[:, 0])]
    positive_count = int(len(ordered) * 0.1)

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


def extract_trial_window(raw: BaseRaw, stimulus_time_ms: int) -> np.ndarray:
    """Extract one DNN-aligned EEG window in microvolts."""
    window_samples = int(WINDOW_LENGTH / 1000 * SAMPLE_RATE)
    end_time_seconds = (stimulus_time_ms - 100) / 1000
    stop = int(raw.time_as_index(end_time_seconds)[0])
    start = stop - window_samples

    if start < 0 or stop > raw.n_times:
        raise ValueError(
            "Trial window falls outside the recording: "
            f"start={start}, stop={stop}, n_times={raw.n_times}."
        )

    window_uv = raw.get_data(
        picks=EEG_CHANNELS,
        start=start,
        stop=stop,
    ) * 1e6

    expected_shape = (len(EEG_CHANNELS), window_samples)
    if window_uv.shape != expected_shape:
        raise ValueError(
            f"Expected trial window shape {expected_shape}, got {window_uv.shape}."
        )
    return window_uv


def _output_path(pipeline_name: str, output_dir: Path) -> Path:
    return output_dir / f"PVT_data_{WINDOW_LENGTH}ms__{pipeline_name}.pt"


def _save_checkpoint_atomically(checkpoint: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def label_data(
    pipeline_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    subjects: Sequence[str] | None = None,
) -> Path:
    """Build one pipeline-specific PVT dataset and return its output path."""
    if pipeline_name not in PIPELINES:
        available = ", ".join(sorted(PIPELINES))
        raise ValueError(
            f"Unknown preprocessing pipeline {pipeline_name!r}. "
            f"Available: {available}."
        )

    selected_subjects = list(subjects) if subjects is not None else load_subjects()
    data_list: list[np.ndarray] = []
    label_list: list[int] = []
    metadata_list: list[tuple[str, str, int]] = []

    for subject in selected_subjects:
        subject_path = DATASET_ROOT / subject
        if not subject_path.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        for session in load_sessions(subject):
            raw = load_run(subject, session)
            labeled_trials = select_labeled_trials(get_trials(raw))

            apply_pipeline(pipeline_name, raw)
            if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
                raise ValueError(
                    f"Pipeline {pipeline_name!r} produced "
                    f"{raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
                )

            missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
            if missing_channels:
                raise ValueError(
                    f"Recording is missing EEG channels: {missing_channels}."
                )
            raw.pick(EEG_CHANNELS)

            for trial, label in labeled_trials:
                window_uv = extract_trial_window(raw, int(trial[2]))
                data_list.append(window_uv[np.newaxis, ...])
                label_list.append(label)
                metadata_list.append((subject, session, int(trial[0])))

    if not data_list:
        raise ValueError("No qualified PVT trials were found.")

    data_array = np.stack(data_list, axis=0)
    label_array = np.asarray(label_list, dtype=np.int64)
    metadata_array = np.asarray(metadata_list, dtype=object)

    trial_windows = np.squeeze(data_array, axis=1)
    spectral = compute_trial_spectral_features(
        trial_windows,
        sfreq=float(SAMPLE_RATE),
    )

    trial_count = data_array.shape[0]
    expected_feature_shape = (trial_count, len(EEG_CHANNELS))
    for feature_name in ("theta", "alpha", "beta", "engagement"):
        if spectral[feature_name].shape != expected_feature_shape:
            raise ValueError(
                f"{feature_name} has shape {spectral[feature_name].shape}; "
                f"expected {expected_feature_shape}."
            )

    output_path = _output_path(pipeline_name, Path(output_dir))
    checkpoint = {
        "data": torch.from_numpy(data_array).float(),
        "labels": torch.from_numpy(label_array).long(),
        "metadata": metadata_array,
        "channel_names": list(EEG_CHANNELS),
        "psd_frequencies": torch.from_numpy(spectral["frequencies"]).float(),
        "theta_power": torch.from_numpy(spectral["theta"]).float(),
        "alpha_power": torch.from_numpy(spectral["alpha"]).float(),
        "beta_power": torch.from_numpy(spectral["beta"]).float(),
        "engagement": torch.from_numpy(spectral["engagement"]).float(),
        "pipeline": pipeline_name,
        "sample_rate_hz": SAMPLE_RATE,
        "window_length_ms": WINDOW_LENGTH,
        "window_end_offset_ms": -100,
        "signal_unit": "microvolts",
    }
    _save_checkpoint_atomically(checkpoint, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prototype PVT datasets with trial-level spectral features."
    )
    parser.add_argument(
        "--pipeline",
        choices=[*sorted(PIPELINES), "all"],
        default="all",
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
    pipeline_names = sorted(PIPELINES) if args.pipeline == "all" else [args.pipeline]
    for pipeline_name in pipeline_names:
        output_path = label_data(
            pipeline_name=pipeline_name,
            output_dir=args.output_dir,
            subjects=args.subjects,
        )
        print(output_path)


if __name__ == "__main__":
    main()
