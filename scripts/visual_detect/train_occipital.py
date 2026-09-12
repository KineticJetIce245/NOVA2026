"""Train EEGNet to read a PVT stimulus out of the occipital EEG.

Input is the checkpoint built by ``build_dataset.py``: a fixed
``[pre_ms, post_ms]`` epoch around each alignment point. The classification
window used here is the one the task asks about -- ``[--win-lo, --win-hi]``
(100-300 ms by default) -- sliced out of that stored context, baseline
corrected on ``[--base-lo, --base-hi]``, and optionally per-trial z-scored.

Two label schemes from the checkpoint, selected with ``--scheme``:

``vr``
    stimulus against silence (the default). Balanced, so the number to read is
    the mean per-subject AUC against the permutation null; most of the
    separability is necessarily the stimulus-locked transient rather than a
    cognitive decision.
``rt``
    fast against slow reaction time, the closer proxy for "seen in time".
    Imbalanced, so the loss is re-weighted and negatives are undersampled.

Every fold is scored with the mean per-subject AUC, which is also the
best-epoch selection metric. The final summary adds pooled AUC, macro F1,
accuracy against the majority-class rate and a label-permutation null.

Protocol
--------
``--protocol loso`` (default) trains on all subjects but one, with a fraction of
the *remaining* subjects held out for best-epoch selection, mirroring
``scripts/training/strictTrainEEGNet.py``. ``--protocol within`` trains on one
subject and holds out one of its sessions, which is the realistic passive-BCI
calibration setting; the split is grouped by session so no two trials from the
same session are on opposite sides.

Either way every epoch is centred and scaled with statistics from the training
trials only (``standardize``); without that the LOSO folds sit at chance because
the ~3 uV evoked response is drowned by between-session offsets in microvolts.

Reported per run: pooled AUC, mean per-subject AUC (also the best-epoch
selection metric), macro F1, accuracy against the majority-class rate and a
label-permutation null.

Usage
-----
    .venv\\Scripts\\python.exe -m scripts.visual_detect.train_occipital
    .venv\\Scripts\\python.exe -m scripts.visual_detect.train_occipital ^
        --scheme rt --neg-cap 20 --epochs 30
    .venv\\Scripts\\python.exe -m scripts.visual_detect.train_occipital ^
        --protocol within --epochs 30
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from nova2026.config import PROJECT_ROOT
from nova2026.training import Metric, SupervisedTrainer

from .device import VIS_DIR
from .models import EEGNetWindow

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
VAL_FRAC = 0.2  # fraction of the TRAINING subjects held out for epoch selection
NEG_CAP = 20  # max random negatives per training subject (imbalance control)
SPARSE_POSITIVES = 10  # a fold with fewer positives has an unstable AUC
SEED = 0

DEFAULT_DATASET = (
    VIS_DIR / "PVT_VIS_occipital_m50_500ms_128.0Hz_VisualPipeline.pt"
)


def windows(
    data: np.ndarray,
    sample_rate: float,
    pre_ms: int,
    lo: int,
    hi: int,
    base: tuple[int, int],
    zscore: bool,
) -> np.ndarray:
    """Slice ``[lo, hi)`` ms out of stored epochs and baseline-correct them.

    ``data`` is ``(n, C, T)`` in microvolts, where sample 0 is ``pre_ms``.
    The baseline mean is removed first, then the slice is taken. With
    ``--zscore`` the slice is additionally scaled by its own standard deviation
    only -- the window *mean is kept*, because the evoked response this task
    looks for lives largely in the window's mean level (the 100-150 ms
    positivity in this dataset is ~+4 uV against ~+0.2 uV of silence) and
    centring the window would delete it.

    Measured on 1340 pooled PVT trials (15 sessions, 4-20 Hz, 128 Hz): a
    stimulus-locked positivity of +3.7 uV at 100-150 ms (t=+13.6) and +1.1 uV
    at 150-200 ms. The same window also contains a pre-motor negativity that
    moves with reaction time (-4.1 uV at 200-250 ms overall, -6.1 uV larger for
    fast than for slow trials), so the late part of the window is not purely
    visual.
    """
    per_ms = sample_rate / 1000.0
    start = int(round((lo - pre_ms) * per_ms))
    stop = int(round((hi - pre_ms) * per_ms))
    if start < 0 or stop > data.shape[-1] or stop <= start:
        raise ValueError(
            f"Window [{lo}, {hi}] ms is outside the stored epoch "
            f"[{pre_ms}, {pre_ms + data.shape[-1] / per_ms:.1f}] ms."
        )
    b0 = int(round((base[0] - pre_ms) * per_ms))
    b1 = int(round((base[1] - pre_ms) * per_ms))
    b0, b1 = max(0, b0), min(data.shape[-1], b1)
    if b1 <= b0:
        raise ValueError(f"Empty baseline window [{base[0]}, {base[1]}] ms.")

    corrected = data.astype(np.float32, copy=True)
    corrected -= data[:, :, b0:b1].mean(axis=-1, keepdims=True, dtype=np.float64)
    out = corrected[:, :, start:stop]
    if zscore:
        scale = out.std(axis=(1, 2), keepdims=True, dtype=np.float64)
        out = out / np.maximum(scale, 1e-8)
    return out


def regularizer(model, eps: float = 1e-8) -> None:
    """EEGNet max-norm constraint on the spatial and pointwise kernels."""
    with torch.no_grad():
        for param, limit in (
            (model.depthwise_conv.weight, 1.0),
            (model.sep_pointwise_conv.weight, 0.25),
        ):
            norm = param.norm(2)
            if norm > limit:
                param.mul_(limit / (norm + eps))


def macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    """Macro F1, defined as 0.0 when only one class is present in ``yt``."""
    yt = np.ravel(yt)
    if np.unique(yt).size < 2:
        return 0.0
    return float(f1_score(yt, np.ravel(yp), average="macro"))


def loso_val_mask(
    subject_of: np.ndarray,
    train_subjects: list[str],
    val_frac: float,
    seed: int,
) -> np.ndarray | None:
    """Hold out a fraction of the *training* subjects for epoch selection."""
    if val_frac <= 0 or not train_subjects:
        return None
    n_val = max(1, int(len(train_subjects) * val_frac))
    rng = np.random.default_rng(seed)
    picked = (
        set(rng.choice(train_subjects, size=n_val, replace=False))
        if n_val < len(train_subjects)
        else set(train_subjects)
    )
    mask = np.isin(subject_of, list(picked))
    return mask if mask.any() else None


def within_splits(
    subject_of: np.ndarray,
    session_of: np.ndarray,
    subject: str,
    n_splits: int = 3,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Per-subject folds grouped by session, with a session held out for val.

    Grouping by session matters because the trials of one session share an
    electrode mount and a slow drift; a random trial split would leak that
    session's offset into the test set.
    """
    mine = np.where(subject_of == subject)[0]
    sessions = np.unique(session_of[mine])
    if sessions.size < 2:
        return [("all", np.isin(np.arange(len(subject_of)), mine), None)]
    n_splits = max(2, min(n_splits, sessions.size))
    splitter = GroupKFold(n_splits=n_splits)
    groups = session_of[mine]
    specs = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(mine, groups=groups)):
        test_mask = np.isin(np.arange(len(subject_of)), mine[test_idx])
        val_sessions = np.unique(groups[train_idx])[:1]
        val_mask = np.isin(np.arange(len(subject_of)), mine[train_idx]) & np.isin(
            subject_of, [subject]
        ) & np.isin(session_of, val_sessions)
        specs.append((f"ses{fold}", test_mask, val_mask))
    return specs


def run_fold(
    data: np.ndarray,
    labels: np.ndarray,
    subject_of: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    val_mask: np.ndarray | None,
    name: str,
    args: argparse.Namespace,
    criterion: torch.nn.Module,
    channels: int,
    sample_rate: float,
    n_samples: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Train and test one fold; returns the trainer result and the test arrays."""
    if np.any(train_mask & test_mask):
        raise ValueError("A trial cannot be both training and test data.")
    # scaling statistics come from the training trials only
    fold_data = standardize(data, train_mask)
    model = EEGNetWindow(
        chn=channels,
        fs=sample_rate,
        t=n_samples,
        kernel_ms=args.kernel_ms,
        f1=args.f1,
        d=args.d,
        p=args.dropout,
        pool1=args.pool1,
        pool2=args.pool2 if args.pool2 > 0 else None,
    )
    train_idx = np.where(train_mask)[0]
    if args.scheme == "rt" and args.neg_cap >= 0:
        train_idx = undersample(train_idx, labels, subject_of, args.neg_cap, seed)
    trainer = SupervisedTrainer(seed=seed)
    trainer.uses(
        model,
        torch.optim.Adam(model.parameters(), lr=args.lr),
        criterion,
        regularizer,
        predict_func=lambda out: (out[:, 1] - out[:, 0]),
        metric=Metric.with_extras(subject_mean_auc, name="mean_subject_auc"),
        unpack=unpack,
        collect=collect,
    )
    split: dict[str, DataLoader | None] = {
        "train": _loader(
            fold_data[train_idx], labels[train_idx], subject_of[train_idx], True
        ),
        "val": _loader(
            fold_data[val_mask], labels[val_mask], subject_of[val_mask], False
        )
        if val_mask is not None and val_mask.any()
        else None,
        "test": _loader(
            fold_data[test_mask], labels[test_mask], subject_of[test_mask], False
        ),
    }
    for key in ("train", "val", "test"):
        loader = split[key]
        n = len(loader.dataset) if loader is not None else 0
        print(f"  {key:5s}: {n} trials")

    def on_epoch(d):
        best = d["best_metric"]
        print(
            f"  epoch {d['epoch']:2d} loss={d['train_loss']:.4f} "
            f"val_auc={float('nan') if best is None else best:.4f}"
        )

    trainer.data = split
    result = trainer.train(args.epochs, {"epoch": on_epoch})
    scores = np.asarray(result["test_preds"], dtype=np.float64)
    truth = np.asarray(result["test_targets"], dtype=np.int64)
    fold_subjects = subject_column(result["test_extras"])
    print(
        f"  {name}: n={truth.size} pos={int(truth.sum())} "
        f"best_epoch={result['best_epoch']}"
    )
    return result, scores, truth, fold_subjects


def undersample(
    index: np.ndarray,
    labels: np.ndarray,
    subject_of: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Keep every positive and at most ``cap`` negatives per training subject."""
    kept: list[np.ndarray] = []
    for i, subject in enumerate(np.unique(subject_of[index])):
        rows = index[subject_of[index] == subject]
        pos = rows[labels[rows] == 1]
        neg = rows[labels[rows] == 0]
        if neg.size > cap:
            neg = np.random.default_rng(seed + i).choice(neg, size=cap, replace=False)
        kept.append(np.concatenate([pos, neg]))
    return np.concatenate(kept) if kept else index


def mean_auc(yt: np.ndarray, yp: np.ndarray, groups: np.ndarray) -> float:
    """Mean per-subject AUC over the folds that contain both classes."""
    yt, yp, groups = np.ravel(yt), np.ravel(yp, order="C"), np.ravel(groups)
    scores = []
    for group in np.unique(groups):
        mask = groups == group
        if np.unique(yt[mask]).size < 2:
            continue
        scores.append(float(roc_auc_score(yt[mask], yp[mask])))
    return float(np.mean(scores)) if scores else float("nan")


def collect(batch):
    """Per-sample ``(n, 1)`` object array of subject tags.

    A list of tag tuples would be flattened by
    ``SupervisedTrainer._concat_extras`` into a 1-D array of strings, losing the
    row structure; an array keeps the tabular layout that ``subject_column``
    reads.
    """
    return np.asarray(batch[2], dtype=object).reshape(-1, 1)


def subject_column(extras) -> np.ndarray:
    """Subject id per trial from a collected extras array.

    Accepts both the 1-D object array of tag tuples that
    ``SupervisedTrainer._concat_extras`` produces today and an already-tabular
    ``(n, >=1)`` array, so a change in that helper fails loudly instead of
    scoring the wrong grouping.
    """
    arr = np.asarray(extras, dtype=object)
    if arr.ndim == 1 and arr.size and isinstance(arr[0], (tuple, list)):
        arr = np.stack([np.asarray(item, dtype=object) for item in arr])
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Expected an (n, >=1) extras array, got {arr.shape}.")
    return arr[:, 0].astype(str)


def subject_mean_auc(yt, yp, extras) -> float:
    """``Metric.with_extras`` entry point: extras carry the subject per trial."""
    return mean_auc(yt, yp, subject_column(extras))


def permutation_null(
    yt: np.ndarray,
    yp: np.ndarray,
    n_perm: int = 200,
    seed: int = SEED,
) -> dict[str, float]:
    """Macro-F1 and accuracy of a label-shuffled null with fixed predictions.

    The decision rule is fixed (``score >= 0``) and the labels are permuted, so
    the null answers "what does this prediction rate score against chance".
    AUC is 0.5 by symmetry and is not simulated.
    """
    rng = np.random.default_rng(seed)
    yt = np.ravel(yt).copy()
    predicted = (np.ravel(yp) >= 0).astype(np.int64)
    f1s = np.empty(n_perm, dtype=np.float64)
    accs = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = rng.permutation(yt)
        f1s[i] = macro_f1(shuffled, predicted)
        accs[i] = accuracy_score(shuffled, predicted)
    return {
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_p95": float(np.percentile(f1s, 95)),
        "accuracy_mean": float(accs.mean()),
        "accuracy_p95": float(np.percentile(accs, 95)),
    }


class EpochDataset(Dataset):
    """``(eeg, label, subject)`` triples.

    ``TensorDataset`` cannot carry the subject tags: it refuses object-dtype
    arrays, and the tags are what makes per-subject scoring possible without
    depending on loader order.
    """

    def __init__(
        self, data: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
        self.subjects = np.asarray(subjects, dtype=object)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index], self.subjects[index]


def _loader(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    return DataLoader(
        EpochDataset(data, labels, subjects),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def standardize(
    data: np.ndarray, train_mask: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Centre and scale every epoch by the *training* trials' mean and SD.

    Without this the LOSO folds are at chance: the evoked response is ~3 uV
    while between-session offsets in the raw microvolt scale are much larger, so
    the network spends its capacity on a per-subject shift it cannot know at
    test time. Statistics come from the training indices only.
    """
    if not train_mask.any():
        raise ValueError("standardize() needs at least one training trial.")
    reference = data[train_mask]
    centre = reference.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    scale = reference.std(axis=(0, 2), keepdims=True, dtype=np.float64)
    return ((data - centre) / np.maximum(scale, eps)).astype(np.float32)


def unpack(batch):
    """``(eeg, label, subject)`` -> model input and target."""
    return batch[0], batch[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--scheme", choices=("vr", "rt"), default="vr")
    parser.add_argument("--win-lo", type=float, default=100.0)
    parser.add_argument("--win-hi", type=float, default=300.0)
    parser.add_argument("--base-lo", type=float, default=-50.0)
    parser.add_argument("--base-hi", type=float, default=0.0)
    parser.add_argument(
        "--zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale each window by its own SD (the window mean is kept)",
    )
    parser.add_argument("--kernel-ms", type=float, default=125.0)
    parser.add_argument("--pool1", type=int, default=2)
    parser.add_argument(
        "--pool2",
        type=int,
        default=4,
        help="second temporal pool; 0 keeps the whole post-convolution time axis",
    )
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument(
        "--neg-cap",
        type=int,
        default=NEG_CAP,
        help="max negatives per training subject for the rt scheme (-1 = all)",
    )
    parser.add_argument(
        "--class-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inverse-frequency loss weight for the imbalanced rt scheme",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also write the best-epoch weights of every fold under models/",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--protocol",
        choices=("loso", "within"),
        default="loso",
        help="leave-one-subject-out, or per-subject folds grouped by session",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="session folds per subject under --protocol within",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="evaluate only the first N subjects (0 = all)",
    )
    parser.add_argument("--tag", default="", help="suffix for the results file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.dataset, map_location="cpu", weights_only=False)
    sample_rate = float(checkpoint["sample_rate_hz"])
    labels = np.asarray(checkpoint["labels"][args.scheme], dtype=np.int64)
    meta = checkpoint["metadata"] if args.scheme == "vr" else checkpoint["rt_metadata"]
    subjects = sorted({tag[0] for tag in meta}, key=lambda s: int(s.split("-")[-1]))
    subject_of = np.asarray([tag[0] for tag in meta], dtype=object)

    print(f"Dataset   : {args.dataset}")
    print(f"Scheme    : {args.scheme} ({labels.sum()} positive / {len(labels)})")
    print(f"Geometry  : {sample_rate:g} Hz, epoch [{checkpoint['pre_ms']}, "
          f"{checkpoint['post_ms']}] ms, {checkpoint['channel_names']}")

    data_key = "data_vr" if args.scheme == "vr" else "data_rt"
    data = windows(
        np.asarray(checkpoint[data_key], dtype=np.float32),
        sample_rate,
        int(checkpoint["pre_ms"]),
        args.win_lo,
        args.win_hi,
        (args.base_lo, args.base_hi),
        args.zscore,
    )
    channels, n_samples = data.shape[1], data.shape[2]
    print(f"Input     : ({channels}, {n_samples}) samples = "
          f"{args.win_hi - args.win_lo:g} ms, zscore={args.zscore}")

    model_probe = EEGNetWindow(
        chn=channels,
        fs=sample_rate,
        t=n_samples,
        kernel_ms=args.kernel_ms,
        f1=args.f1,
        d=args.d,
        p=args.dropout,
        pool1=args.pool1,
        pool2=args.pool2 if args.pool2 > 0 else None,
    )
    print(f"Model     : EEGNetWindow kernel={model_probe.kernel} samples, "
          f"features={model_probe.features}")

    folds = subjects[: args.folds] if args.folds else subjects
    session_of = np.asarray([tag[1] for tag in meta], dtype=object)
    trainer = SupervisedTrainer(seed=args.seed)
    trainer.fetch(args.dataset)
    print(f"Device    : {trainer.device}")
    print(f"Protocol  : {args.protocol}")

    if args.scheme == "rt" and args.class_weight and labels.sum() and (labels == 0).sum():
        class_weight = torch.tensor(
            [1.0, float((labels == 0).sum() / max(1, labels.sum()))],
            dtype=torch.float32,
            device=trainer.device,
        )
    else:
        class_weight = None
    print(
        "Criterion : CrossEntropyLoss(weight="
        f"{'none' if class_weight is None else class_weight.tolist()})"
    )

    predictions, targets, fold_extras = [], [], []
    fold_scores = []
    for i, subject in enumerate(folds):
        if args.protocol == "within":
            specs = within_splits(subject_of, session_of, subject, n_splits=args.inner_folds)
            train_subjects = [subject]
        else:
            specs = [("loso", subject_of == subject, None)]
            train_subjects = [s for s in subjects if s != subject]

        for name, test_mask, val_mask in specs:
            if args.protocol == "loso":
                train_mask = np.isin(subject_of, train_subjects) & ~test_mask
                if val_mask is None:
                    val_mask = loso_val_mask(
                        subject_of, train_subjects, args.val_frac, args.seed + i
                    )
            else:
                train_mask = (subject_of == subject) & ~test_mask
            if val_mask is not None:
                train_mask = train_mask & ~val_mask
            weight = class_weight if args.scheme == "rt" else None
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
            result, scores, truth, fold_subjects = run_fold(
                data,
                labels,
                subject_of,
                train_mask,
                test_mask,
                val_mask,
                f"{subject}/{name}",
                args,
                criterion,
                channels,
                sample_rate,
                n_samples,
                seed=args.seed + i,
            )
            predictions.append(scores)
            targets.append(truth)
            fold_extras.append(fold_subjects)
            fold_auc = mean_auc(truth, scores, fold_subjects)
            fold_scores.append(
                {
                    "subject": subject,
                    "fold": name,
                    "n": int(truth.size),
                    "n_pos": int(truth.sum()),
                    "auc": fold_auc,
                    "accuracy": float(
                        accuracy_score(truth, (scores >= 0).astype(int))
                    ),
                }
            )
            print(
                f"  -> {name} n={truth.size} pos={int(truth.sum())} "
                f"AUC={fold_auc:.4f} acc={fold_scores[-1]['accuracy']:.4f} "
                f"(best epoch {result['best_epoch']})"
            )
            if args.save_models:
                model_dir = MODELS_DIR / f"visual_detect_{args.scheme}_{args.protocol}"
                model_dir.mkdir(parents=True, exist_ok=True)
                out = model_dir / f"{subject}_{name}.pt"
                torch.save(result["model"].state_dict(), out)
                print(f"     saved {out}")

    scores = np.concatenate(predictions)
    truth = np.concatenate(targets)
    fold_subjects = np.concatenate(fold_extras)
    predicted = (scores >= 0).astype(np.int64)

    summary = {
        "scheme": args.scheme,
        "protocol": args.protocol,
        "n_trials": int(truth.size),
        "n_positive": int(truth.sum()),
        "pooled_auc": float(roc_auc_score(truth, scores))
        if np.unique(truth).size > 1
        else float("nan"),
        "pooled_accuracy": float(accuracy_score(truth, predicted)),
        "pooled_macro_f1": macro_f1(truth, predicted),
        "pooled_binary_f1": float(f1_score(truth, predicted, zero_division=0)),
        "mean_subject_auc": mean_auc(truth, scores, fold_subjects),
        "majority_accuracy": float(max(truth.mean(), 1 - truth.mean())),
        "sparse_folds": {
            "threshold_positives": SPARSE_POSITIVES,
            "n_folds": int(sum(1 for row in fold_scores if row["n_pos"] < SPARSE_POSITIVES)),
            "note": "per-subject AUC in these folds rests on too few positives",
        },        "null": permutation_null(truth, scores, seed=args.seed),
        "fold_scores": fold_scores,
        "config": {
            "window_ms": [args.win_lo, args.win_hi],
            "baseline_ms": [args.base_lo, args.base_hi],
            "zscore": args.zscore,
            "channels": checkpoint["channel_names"],
            "n_samples": int(n_samples),
            "kernel_ms": args.kernel_ms,
            "pool1": args.pool1,
            "pool2": args.pool2,
            "f1": args.f1,
            "d": args.d,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "val_frac": args.val_frac,
            "neg_cap": args.neg_cap,
            "class_weight": args.class_weight,
            "band_hz": checkpoint.get("band_hz"),
            "sample_rate_hz": sample_rate,
            "dataset": str(args.dataset),
        },
    }

    print("\n===== summary =====")
    print(f"pooled AUC        : {summary['pooled_auc']:.4f}")
    print(f"mean subject AUC  : {summary['mean_subject_auc']:.4f}")
    print(f"pooled macro F1   : {summary['pooled_macro_f1']:.4f}")
    print(f"pooled accuracy   : {summary['pooled_accuracy']:.4f} "
          f"(majority {summary['majority_accuracy']:.4f})")
    null = summary["null"]
    print(f"permutation null  : macro F1 {null['macro_f1_mean']:.4f} "
          f"(p95 {null['macro_f1_p95']:.4f}) | acc {null['accuracy_mean']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"visual_detect_{args.scheme}{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return summary


if __name__ == "__main__":
    main()
