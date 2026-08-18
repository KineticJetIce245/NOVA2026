from pathlib import Path
import os


def locate_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return current.parent.parent


PROJECT_ROOT = locate_project_root()
DATA_DIR = PROJECT_ROOT / "datasets"
# modify this path to point to the dataset
DATASET = DATA_DIR / "COG-BCI/sub-01/ses-S1/eeg/PVT.set"
# DATASET = DATA_DIR / "CAP-POS/DDE-OP-3345rev02 electrode positions for CA-208.elc"
