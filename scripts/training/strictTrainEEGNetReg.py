from enum import unique
from pathlib import Path

import numpy as np
import torch

from nova2026.architecture.cnn import EEGNet
from nova2026.architecture.lossfun import FocalLoss
from nova2026.config import DATA_DIR
from nova2026.trainer import Trainer

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "outputs/PVT_128Hz_AttUPipeline.pt"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
VAL_FRAC = 0.2  # fraction of TRAINING subjects held out for epoch selection
NEG_CAP = 90  # max random negatives kept per training subject (None = keep all)
SEED = 0


# model hyperparameters
criterion = FocalLoss(gamma=3.0, alpha=[1, 3.5], reduction="mean")


def regularizer(model, max_value: float = 1.0, eps: float = 1e-8) -> None:
    """Apply the max-norm constraint to a parameter."""
    param_dept = model.depthwise_conv.weight
    param_sep = model.sep_pointwise_conv.weight
    with torch.no_grad():
        norm_dept = param_dept.norm(2)
        norm_sep = param_sep.norm(2)
        if norm_dept > max_value:
            param_dept.mul_(max_value / (norm_dept + eps))
        if norm_sep > max_value:
            param_sep.mul_(max_value / (norm_sep + eps))


def label_trials(data_chkpt: dict) -> np.ndarray:
    return np.array([0 if t < 500 else 1 for t in data_chkpt["rt"]])


def organizer(data_chkpt: dict, test_sub: str):
    chkpt = {
        "data": {},
        "labels": {},
    }
    labels = label_trials(data_chkpt)
    train_mask: np.ndarray = np.array(
        [tags[0] != test_sub for tags in data_chkpt["metadata"]]
    )
    data = np.asarray(data_chkpt["data"])
    chkpt["data"]["train"] = data[train_mask]
    chkpt["data"]["test"] = data[~train_mask]
    chkpt["labels"]["train"] = labels[train_mask]
    chkpt["labels"]["test"] = labels[~train_mask]
    return chkpt


def main():
    trainer = Trainer()
    trainer.fetch(Path(DATASET))

    meta = trainer.chkpt["metadata"]
    subs = list({tags[0] for tags in meta})
    for sub in subs:  # main LOSO loop
        # Prepare Model
        model = EEGNet(chn=62)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # load model and hyperparameters
        trainer.uses(model, optimizer, criterion, regularizer)
        trainer.organize(lambda x, s=sub: organizer(x, s))
        d = trainer.data
        print(
            f"\n=== Fold test_sub={sub} | "
            f"train {len(d['data']['train'])} | test {len(d['data']['test'])} ==="
        )
        trainer.train(BATCH_SIZE, EPOCHS)


if __name__ == "__main__":
    main()
