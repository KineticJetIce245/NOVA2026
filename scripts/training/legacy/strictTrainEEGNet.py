"""Strict LOSO training/evaluation for EEGNet.

Fixes the optimistic biases in ``trainEEGNet.py`` so its F1 can be compared
fairly against the algorithmic arm (``predict.py``):

1. **Epoch selection no longer peeks at the test subject.**  Inside each
   LOSO fold the *training* subjects are split (subject-disjoint) into train
   and validation; the epoch is chosen by the validation macro F1, and the
   held-out test subject's F1 is reported at that fixed epoch.  No
   test-subject label is used for any decision.

2. **No hard-easy training split.**  ``trainEEGNet.py`` trained each subject
   on the slowest-RT positives + fastest-RT negatives (an easy extreme
   contrast).  Here the model trains on all positives plus a *random* sample
   of negatives per subject (no RT-based selection), so it sees the real
   decision boundary rather than two extreme clusters.

3. **Explicit imbalance handling.**  The test set keeps the real ~10 %
   positive prior; both macro and binary F1 are reported, plus a final
   pooled F1 over all held-out test trials concatenated.

Seeds are fixed per fold for reproducibility.
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from nova2026.architecture.cnn import EEGNet
from nova2026.architecture.lossfun import FocalLoss
from nova2026.config import DATA_DIR

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "outputs/old_PVT_chkpt.pt"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
VAL_FRAC = 0.2  # fraction of TRAINING subjects held out for epoch selection
NEG_CAP = 90  # max random negatives kept per training subject (None = keep all)
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def max_norm_(param, max_value: float = 1.0, eps: float = 1e-8) -> None:
    """Apply the max-norm constraint to a parameter."""
    with torch.no_grad():
        norm = param.norm(2)
        if norm > max_value:
            param.mul_(max_value / (norm + eps))


def load_data(path=DATASET):
    """Load the PVT checkpoint; returns (data, labels, subjects, rt)."""
    ckpt = torch.load(path, weights_only=False)
    return (
        ckpt["data"],
        ckpt["labels"],
        ckpt["metadata"][:, 0],
        ckpt["metadata"][:, 2].astype(float),
    )


def select_train_trials(labels, rt, subjects, train_subjects, rng, neg_cap=NEG_CAP):
    """All positives + a *random* sample of negatives per training subject.

    Unlike ``trainEEGNet.py`` this never selects by RT, so it does not build
    an artificially easy extreme-vs-extreme contrast.
    """
    idx = []
    for sub in train_subjects:
        s = np.where(subjects == sub)[0]
        pos = s[labels[s] == 1]
        neg = s[labels[s] == 0]
        if neg_cap is not None:
            neg = rng.choice(neg, size=min(len(neg), neg_cap), replace=False)
        idx.extend(pos.tolist())
        idx.extend(neg.tolist())
    return np.asarray(idx)


def train_model(x_train, y_train, x_val, y_val):
    """Train one EEGNet; return the state dict of the best-validation epoch."""
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = EEGNet(chn=62).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=3.0, alpha=[1, 3.5], reduction="mean")

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                max_norm_(model.depthwise_conv.weight, max_value=1.0)
                max_norm_(model.classifier.weight, max_value=0.25)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds.extend(torch.argmax(model(bx), dim=1).cpu().numpy())
                targets.extend(by.cpu().numpy())
        val_f1 = f1_score(targets, preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    return best_state, best_epoch, best_val_f1


def _build_and_predict(state, x, y):
    """Load ``state`` into a fresh EEGNet; return (preds, probs, targets).

    ``probs`` are the softmax probabilities of the positive (lapse) class,
    used for a threshold-free AUROC comparison against the algorithmic arm.
    """
    model = EEGNet(chn=62).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            targets.extend(by.cpu().numpy())
    return np.asarray(preds), np.asarray(probs), np.asarray(targets)


def evaluate(state, x_test, y_test):
    """Score one held-out test subject at the fixed best-validation epoch."""
    preds, probs, targets = _build_and_predict(state, x_test, y_test)
    return {
        "preds": preds,
        "probs": probs,
        "targets": targets,
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "binary_f1": float(f1_score(targets, preds, average="binary", pos_label=1)),
        "accuracy": float(accuracy_score(targets, preds)),
        "auroc": float(roc_auc_score(targets, probs)),
        "n_pred_pos": int(preds.sum()),
        "n_pos": int(targets.sum()),
    }


def main():
    data, labels, sub_names, rt = load_data(DATASET)
    sub_names = np.asarray(sub_names)
    labels = np.asarray(labels)
    rt = np.asarray(rt)
    print(
        f"Total trials: {len(labels)}, subjects: {len(np.unique(sub_names))}, "
        f"positive rate: {labels.mean():.3f}  (device: {DEVICE})"
    )

    unique_subs = np.unique(sub_names)
    macro_f1s, binary_f1s, aurocs = [], [], []
    all_probs, all_targets = [], []

    for fold, test_sub in enumerate(unique_subs):
        set_seed(SEED + fold)
        print(f"\n=== Fold: test subject = {test_sub} ===")

        test_idx = np.where(sub_names == test_sub)[0]
        train_subjects = np.unique(sub_names[sub_names != test_sub])
        rng = np.random.default_rng(SEED + fold)
        n_val = max(1, int(round(len(train_subjects) * VAL_FRAC)))
        val_subjects = rng.choice(train_subjects, size=n_val, replace=False)
        tr_subjects = np.setdiff1d(train_subjects, val_subjects)

        tr_idx = select_train_trials(labels, rt, sub_names, tr_subjects, rng)
        val_idx = np.where(np.isin(sub_names, val_subjects))[0]

        print(
            f"  train {len(tr_subjects)} subjects / {len(tr_idx)} trials "
            f"({labels[tr_idx].mean():.3f} pos) | val {len(val_subjects)} subjects / "
            f"{len(val_idx)} trials ({labels[val_idx].mean():.3f} pos) | "
            f"test 1 subject / {len(test_idx)} trials ({labels[test_idx].mean():.3f} pos)"
        )

        best_state, best_epoch, best_val_f1 = train_model(
            data[tr_idx], labels[tr_idx], data[val_idx], labels[val_idx]
        )
        res = evaluate(best_state, data[test_idx], labels[test_idx])
        print(
            f"  val macro F1 {best_val_f1:.4f} @ epoch {best_epoch} | "
            f"TEST: macro F1 {res['macro_f1']:.4f}, binary F1 {res['binary_f1']:.4f}, "
            f"AUROC {res['auroc']:.4f}, pred-pos {res['n_pred_pos']}/{res['n_pos']}"
        )

        macro_f1s.append(res["macro_f1"])
        binary_f1s.append(res["binary_f1"])
        aurocs.append(res["auroc"])
        all_probs.extend(res["probs"].tolist())
        all_targets.extend(res["targets"].tolist())

    macro_f1s = np.asarray(macro_f1s)
    binary_f1s = np.asarray(binary_f1s)
    aurocs = np.asarray(aurocs)
    pooled_f1 = float(
        f1_score(
            all_targets, np.asarray(all_probs) >= 0.5, average="binary", pos_label=1
        )
    )
    pooled_auroc = float(roc_auc_score(all_targets, all_probs))

    print("\n================ Strict LOSO Results ================")
    for sub, mf, bf, auc in zip(unique_subs, macro_f1s, binary_f1s, aurocs):
        print(
            f"Subject {sub}: macro F1 {mf:.4f} | binary F1 {bf:.4f} | AUROC {auc:.4f}"
        )
    print(f"Mean macro F1:  {macro_f1s.mean():.4f} ± {macro_f1s.std():.4f}")
    print(f"Mean binary F1: {binary_f1s.mean():.4f} ± {binary_f1s.std():.4f}")
    print(f"Mean AUROC:     {aurocs.mean():.4f} ± {aurocs.std():.4f}")
    print(f"Pooled binary F1 (all held-out trials): {pooled_f1:.4f}")
    print(f"Pooled AUROC (all held-out trials):     {pooled_auroc:.4f}")


if __name__ == "__main__":
    main()
