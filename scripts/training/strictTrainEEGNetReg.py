from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from nova2026.architecture.cnn import EEGNet
from nova2026.architecture.lossfun import FocalLoss
from nova2026.config import DATA_DIR
from nova2026.trainer import SupervisedTrainer

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "outputs/PVT_128Hz_AttUPipeline.pt"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
VAL_FRAC = 0.2  # fraction of TRAINING subjects held out for epoch selection
NEG_CAP = 90  # max random negatives kept per training subject (None = keep all)
SEED = 0


# model hyperparameters
# criterion = torch.nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=3.0, alpha=[1, 1], reduction="mean")
permutates = {
    "epoch": lambda d: print(
        f"At epoch {d['epoch']}, train_loss={d['train_loss']:.4f}, best_so_far={d['best_so_far']:.4f}"
    ),
    "test_output": lambda logits: torch.argmax(logits, dim=1),
}


def regularizer(model, eps: float = 1e-8) -> None:
    """Apply the max-norm constraint to a parameter."""
    param_dept = model.depthwise_conv.weight
    param_sep = model.sep_pointwise_conv.weight
    with torch.no_grad():
        norm_dept = param_dept.norm(2)
        norm_sep = param_sep.norm(2)
        if norm_dept > 1.0:
            param_dept.mul_(1.0 / (norm_dept + eps))
        if norm_sep > 0.25:
            param_sep.mul_(0.25 / (norm_sep + eps))


def label_trials(data_chkpt: dict, seed: int, neg_cap: NEG_CAP) -> np.ndarray:
    meta_list = data_chkpt["metadata"]
    meta_set = {}
    sub_set = {}
    for i, tags in enumerate(meta_list):
        # dict: (sub, ses) -> [trial.n]
        meta_set.setdefault((tags[0], tags[1]), []).append(i)
        sub_set.setdefault(tags[0], []).append(i)  # dict: sub -> [trial.n]

    labels = [-1] * len(meta_list)  # -1 trials are not going to be used

    # label positive trials
    for idx_list in meta_set.values():
        idx_list.sort(key=lambda i: data_chkpt["rt"][i])
        num = 10 * len(idx_list) // 100
        pos = idx_list[-num:] if num > 0 else []
        for idx in pos:
            labels[idx] = 1

    # select negative trials (underfitting)
    for idx_list in sub_set.values():
        neg_list = [i for i in idx_list if labels[i] != 1]
        rng = np.random.default_rng(seed)
        neg_list = rng.choice(subs_not_test, size=n_vals, replace=False)

    return np.array(labels)


def organizer(
    data_chkpt: dict,
    test_sub: str,
    subs: list,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
):
    print(f"Omitting subject {test_sub} from training data.")
    chkpt = {}
    labels = label_trials(data_chkpt)
    label_mask = labels != -1
    data = np.asarray(data_chkpt["data"])
    subs_meta = np.asarray([tags[0] for tags in data_chkpt["metadata"]])

    # test DataLoader
    test_mask = subs_meta == test_sub
    chkpt["test"] = DataLoader(
        TensorDataset(
            torch.tensor(data[test_mask & label_mask], dtype=torch.float32),
            torch.tensor(labels[test_mask & label_mask], dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # select val subjects
    if val_frac >= 1:
        raise ValueError("Val fraction must be less than 1.")
    n_vals = max(1, int((len(subs) - 1) * val_frac))
    rng = np.random.default_rng(seed)
    subs_not_test = [s for s in subs if s != test_sub]
    val_subs = rng.choice(subs_not_test, size=n_vals, replace=False)
    train_subs = np.setdiff1d(subs_not_test, val_subs)

    train_mask = np.isin(subs_meta, train_subs)
    val_mask = np.isin(subs_meta, val_subs)

    # train DataLoader
    chkpt["train"] = DataLoader(
        TensorDataset(
            torch.tensor(data[train_mask & label_mask], dtype=torch.float32),
            torch.tensor(labels[train_mask & label_mask], dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # val DataLoader
    chkpt["val"] = DataLoader(
        TensorDataset(
            torch.tensor(data[val_mask & label_mask], dtype=torch.float32),
            torch.tensor(labels[val_mask & label_mask], dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return chkpt


trainer = SupervisedTrainer()
trainer.fetch(Path(DATASET))

meta = trainer.chkpt["metadata"]
subs = list({tags[0] for tags in meta})
subs.sort(key=lambda s: int(s[-2:]))  # sort by the last two char as number
for i, sub in enumerate(subs):  # main LOSO loop
    # Prepare Model
    model = EEGNet(chn=62)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # load model and hyperparameters
    trainer.uses(model, optimizer, criterion, regularizer)
    trainer.organize(lambda x, sb=sub, sbs=subs, sd=i: organizer(x, sb, sbs, seed=sd))
    d = trainer.data
    print(
        f"=== Fold test_sub={sub} | "
        f"train {len(d['train'].dataset)} | test {len(d['test'].dataset)} | val {len(d['val'].dataset)} ==="
    )
    result = trainer.train(EPOCHS, permutates)
    print(f"Completed training for subject {sub}.")
    preds = np.asarray(result["test_preds"])
    targets = np.asarray(result["test_targets"])
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="binary", pos_label=1)
    print(f"Test accuracy: {acc:.4f} | F1: {f1:.4f}")
