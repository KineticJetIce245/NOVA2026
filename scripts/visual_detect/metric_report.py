"""What accuracy would hide: score the same predictions with several metrics.

The decoders in ``decode.py`` are ranked by AUC because AUC needs no decision
threshold and does not reward the majority class. This script makes that choice
concrete: it takes one decoder's out-of-fold scores and shows what accuracy,
balanced accuracy and macro F1 would have said at a range of thresholds, plus
the accuracy of a model that always answers "majority".

    .venv\\Scripts\\python.exe -m scripts.visual_detect.metric_report
    .venv\\Scripts\\python.exe -m scripts.visual_detect.metric_report --scheme rt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from nova2026.config import PROJECT_ROOT

from .block_decode import per_trial_scores
from .decode import DATASETS, DECODERS, load_arrays, parse_window
from .device import VIS_DIR

RESULTS_DIR = PROJECT_ROOT / "results"


def threshold_row(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Every headline metric at one decision threshold.

    ``recall_minority`` and ``f1_minority`` always refer to the rarer class, so
    the table reads the same whether the positive class is the rare one
    (``rt``) or not (``vr``).
    """
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    rare = 1 if labels.mean() < 0.5 else 0
    recall = (tp / (tp + fn)) if rare == 1 else (tn / (tn + fp))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, average="macro", zero_division=0)
        ),
        "f1_minority": float(
            f1_score(labels, predicted, pos_label=rare, zero_division=0)
        ),
        "recall_minority": float(recall),
        "predicted_positive_rate": float(predicted.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    path = VIS_DIR / DATASETS[args.channel_set]
    if not path.exists():
        raise SystemExit(f"{path.name} is not built.")
    window = parse_window(args.window)
    data, labels, subjects, sessions = load_arrays(
        str(path), args.scheme, window, (args.base[0], args.base[1])
    )
    decoder = DECODERS[args.decoder]
    scores = per_trial_scores(data, labels, subjects, sessions, decoder, args)
    usable = np.isfinite(scores)
    labels, scores = labels[usable], scores[usable]

    majority = max(labels.mean(), 1 - labels.mean())
    rare = 1 if labels.mean() < 0.5 else 0
    # rank the rare class first so average precision reads the same way
    ranking = scores if rare == 1 else -scores
    print(
        f"=== {args.scheme} | {args.channel_set} | {args.decoder} | "
        f"{len(labels)} trials ==="
    )
    print(
        f"class balance          : {int(labels.sum())} positive / {len(labels)} "
        f"({100 * labels.mean():.1f} %); rare class is {rare}"
    )
    print(
        f"AUC                    : {roc_auc_score(labels, scores):.4f}  "
        "(threshold-free)"
    )
    ap = average_precision_score(labels == rare, ranking)
    base_rate = float((labels == rare).mean())
    print(
        f"average precision      : {ap:.4f}  "
        f"(random ranking would give {base_rate:.4f}, so x{ap / base_rate:.2f})"
    )
    print(f"always-majority accuracy: {majority:.4f}  (recall of rare class = 0.000)")

    rows = []
    thresholds = [0.0]
    thresholds += [float(value) for value in np.quantile(scores, [0.1, 0.5, 0.9])]
    # plus the threshold that maximises balanced accuracy
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best = max(
        (threshold_row(labels, scores, t) for t in candidates),
        key=lambda row: row["balanced_accuracy"],
    )
    thresholds.append(best["threshold"])
    thresholds = sorted({round(value, 6) for value in thresholds})
    for threshold in thresholds:
        rows.append(threshold_row(labels, scores, threshold))

    print("\n  threshold   accuracy  bal_acc  macroF1  rec_rare  pred_pos_rate")
    for row in rows:
        marker = (
            "  <- best balanced acc" if row["threshold"] == best["threshold"] else ""
        )
        print(
            f"  {row['threshold']:+9.4f}   {row['accuracy']:.4f}   "
            f"{row['balanced_accuracy']:.4f}   {row['macro_f1']:.4f}   "
            f"{row['recall_minority']:>8.3f}   {row['predicted_positive_rate']:.3f}"
            f"{marker}"
        )

    accuracies = [row["accuracy"] for row in rows]
    print(
        f"\naccuracy over these {len(rows)} thresholds spans "
        f"{min(accuracies):.4f}-{max(accuracies):.4f} on the SAME predictions; "
        f"AUC never moves."
    )

    summary = {
        "scheme": args.scheme,
        "channel_set": args.channel_set,
        "decoder": args.decoder,
        "window_ms": list(window),
        "n_trials": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(ap),
        "average_precision_random": float(base_rate),
        "always_majority_accuracy": float(majority),
        "thresholds": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"visual_detect_metrics{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", choices=("vr", "rt"), default="vr")
    parser.add_argument("--channel-set", default="all")
    parser.add_argument("--window", default="100-300")
    parser.add_argument("--decoder", default="riemann")
    parser.add_argument("--base", type=float, nargs=2, default=(-50.0, 0.0))
    parser.add_argument("--n-filters", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
