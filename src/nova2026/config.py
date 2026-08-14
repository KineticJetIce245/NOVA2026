from pathlib import Path
import os


def locate_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return current.parent.parent


PROJECT_ROOT = locate_project_root()
print(f"[config.py]: Current project root: {PROJECT_ROOT}.")
DATA_DIR = PROJECT_ROOT / "datasets"
print(f"[config.py]: Current data directory: {DATA_DIR}.")
# modify this path to point to your dataset
DATASET = DATA_DIR / "COG-BCI/sub-01/sub-01/ses-S1/eeg/PVT.set"
print(f"[config.py]: Current dataset path: {DATASET}.")
