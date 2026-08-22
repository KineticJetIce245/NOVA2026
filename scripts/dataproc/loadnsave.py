from pathlib import Path

import torch
from mne.io import BaseRaw

from nova2026.data import eeg


def load_subjects(root: Path) -> list[str]:
    """Return sorted COG-BCI subject directory names."""
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    )


def load_sessions(subject: str, root: Path) -> list[str]:
    """Return sorted session directory names for one subject."""
    subject_path = root / subject
    return sorted(
        path.name
        for path in subject_path.iterdir()
        if path.is_dir() and path.name.startswith("ses-")
    )


def load_run(subject: str, session: str, root: Path, data_type: str) -> BaseRaw:
    """Load one PVT recording."""
    return eeg.load(root / subject / session / "eeg" / f"{data_type}.set")


def _gen_output_path(
    data_type: str, pipeline_name: str, sample_rate: float, output_dir: Path
) -> Path:
    return output_dir / f"{data_type}_{sample_rate}Hz_{pipeline_name}.pt"


def save_chkpt(chkpt, output_dir) -> None:
    output_path: Path = _gen_output_path(
        chkpt["data_type"], chkpt["pipeline"], chkpt["sample_rate_hz"], output_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(chkpt, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
