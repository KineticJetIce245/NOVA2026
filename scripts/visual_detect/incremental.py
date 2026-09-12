"""Does the EEG add anything beyond reaction time?

The uncomfortable question this file answers: if a slow reaction time already
tells you the subject was lapsing, is the EEG worth recording at all? It is only
worth it if the EEG carries information reaction time does not -- ideally
*before* reaction time degrades.

Three predictors are compared on the same held-out session:

``rt_only``
    The mean reaction time of a block of K consecutive trials. This is the
    behavioural baseline: it is what you would use if you had no EEG.
``eeg_only``
    The mean decoder score of the same K trials, where the score comes from the
    stimulus-versus-silence decoder trained on the other sessions.
``eeg_plus_rt``
    Both together, fitted on the training sessions.

The target is the *next* block's mean reaction time, so nothing is predicted
from itself. If ``eeg_plus_rt`` does not beat ``rt_only``, the EEG is redundant
with the button press and the honest conclusion is that this pipeline should be
used for something else.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nova2026.config import PROJECT_ROOT
from nova2026.training import load_chkpt

from .block_decode import per_trial_scores
from .decode import DATASETS, DECODERS, parse_window
from .device import VIS_DIR
from .train_occipital import windows

RESULTS_DIR = PROJECT_ROOT / "results"


def block_features(
    order: np.ndarray,
    values: np.ndarray,
    k: int,
    max_gap_ms: float,
    times_ms: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping run means and their first-trial index.

    ``order`` holds the row numbers of one session in time order. Runs are cut at
    any gap longer than ``max_gap_ms``; incomplete runs are dropped. With
    ``valid``, a block containing an unscored trial is dropped rather than
    averaged with a NaN, so every model sees exactly the same blocks.
    """
    means, firsts = [], []
    run: list[int] = []
    for row in order:
        if run and times_ms[row] - times_ms[run[-1]] > max_gap_ms:
            run = []
        run.append(int(row))
        if len(run) == k:
            if valid is None or all(valid[index] for index in run):
                means.append(float(np.mean(values[run])))
                firsts.append(run[0])
            run = []
    return np.asarray(means, dtype=np.float64), np.asarray(firsts, dtype=np.int64)


def build_panel(
    scores: np.ndarray,
    rts: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    times_ms: np.ndarray,
    k: int,
    max_gap_ms: float,
) -> dict:
    """One row per block: current EEG score, current RT, and the *next* RT.

    A block's predictors come from its own K trials; its target is the next
    block's mean RT, so the model is asked to anticipate, not to describe. Both
    predictors are built on the same blocks, so ``rt_only`` and ``eeg_only`` are
    compared on identical rows.
    """
    valid = np.isfinite(scores)
    rows = []
    for session in np.unique(sessions):
        order = np.where(sessions == session)[0]
        order = order[np.argsort(times_ms[order], kind="stable")]
        if len(order) < 2 * k:
            continue
        score_means, _ = block_features(
            order, scores, k, max_gap_ms, times_ms, valid=valid
        )
        rt_means, _ = block_features(order, rts, k, max_gap_ms, times_ms, valid=valid)
        if score_means.size < 2 or rt_means.size != score_means.size:
            continue
        subject = str(subjects[order[0]])
        for i in range(score_means.size - 1):
            rows.append(
                {
                    "subject": subject,
                    "session": str(session),
                    "eeg": score_means[i],
                    "rt": rt_means[i],
                    "next_rt": rt_means[i + 1],
                }
            )
    if not rows:
        return {}
    return {
        "eeg": np.asarray([r["eeg"] for r in rows], dtype=np.float64),
        "rt": np.asarray([r["rt"] for r in rows], dtype=np.float64),
        "next_rt": np.asarray([r["next_rt"] for r in rows], dtype=np.float64),
        "subject": np.asarray([r["subject"] for r in rows], dtype=object),
        "session": np.asarray([r["session"] for r in rows], dtype=object),
    }


def _design(panel: dict, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([panel[name] for name in columns])


def evaluate(
    panel: dict,
    columns: tuple[str, ...],
    seed: int,
    ridge_alphas=np.logspace(-2, 2, 9),
) -> dict:
    """Out-of-sample prediction of the next block's mean RT, session-grouped.

    Standardisation and the ridge penalty are fitted on the training sessions
    only, so the held-out session contributes nothing but its trials.
    """
    x = _design(panel, columns)
    y = panel["next_rt"]
    subjects, sessions = panel["subject"], panel["session"]
    predicted = np.full(len(y), np.nan)
    for subject in sorted(set(subjects), key=lambda s: int(s.split("-")[-1])):
        rows = np.where(subjects == subject)[0]
        fold_sessions = sessions[rows]
        if len(set(fold_sessions)) < 2:
            continue
        n_splits = min(3, len(set(fold_sessions)))
        if min(np.bincount(np.unique(fold_sessions, return_inverse=True)[1])) < 2:
            continue
        labels_for_split = (y[rows] > np.median(y[rows])).astype(int)
        if len(set(labels_for_split)) < 2:
            continue
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for train_idx, test_idx in splitter.split(
            rows, labels_for_split, groups=fold_sessions
        ):
            train_rows, test_rows = rows[train_idx], rows[test_idx]
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ridge_alphas))
            model.fit(x[train_rows], y[train_rows])
            predicted[test_rows] = model.predict(x[test_rows])
    usable = np.isfinite(predicted)
    if usable.sum() < 20 or len(set(subjects[usable])) < 2:
        return {"n": int(usable.sum()), "r2": float("nan"), "spearman": float("nan"),
                "mae_ms": float("nan")}
    truth, guess = y[usable], predicted[usable]
    residual = truth - guess
    r2 = 1.0 - float(np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2))
    rho = float(spearmanr(truth, guess).statistic)
    return {
        "n": int(usable.sum()),
        "r2": r2,
        "spearman": rho,
        "mae_ms": float(np.mean(np.abs(residual))),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channel-set", default="all")
    parser.add_argument("--window", default="100-300")
    parser.add_argument("--decoder", default="riemann")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[5, 8, 12])
    parser.add_argument("--max-gap-s", type=float, default=15.0)
    parser.add_argument("--base", type=float, nargs=2, default=(-50.0, 0.0))
    parser.add_argument("--n-filters", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    path = VIS_DIR / DATASETS[args.channel_set]
    if not path.exists():
        raise SystemExit(f"{path.name} is not built; run build_dataset first.")
    checkpoint = load_chkpt(str(path))
    window = parse_window(args.window)
    sample_rate = float(checkpoint["sample_rate_hz"])
    data = windows(
        np.asarray(checkpoint["data_rt"], dtype=np.float32),
        sample_rate,
        int(checkpoint["pre_ms"]),
        window[0],
        window[1],
        (args.base[0], args.base[1]),
        False,
    )
    labels = np.asarray(checkpoint["labels"]["rt"], dtype=np.int64)
    rts = np.asarray(checkpoint["labels"]["rt_ms"], dtype=np.float64)
    meta = checkpoint["rt_metadata"]
    subjects = np.asarray([tag[0] for tag in meta], dtype=object)
    sessions = np.asarray([f"{tag[0]}/{tag[1]}" for tag in meta], dtype=object)
    times_ms = np.asarray(checkpoint["rt_onset_ms"], dtype=np.float64)
    if not (len(data) == len(labels) == len(rts) == len(times_ms)):
        raise SystemExit(
            "checkpoint arrays disagree in length ("
            f"data {len(data)}, labels {len(labels)}, rt {len(rts)}, "
            f"onsets {len(times_ms)}); rebuild it"
        )

    scores = per_trial_scores(
        data, labels, subjects, sessions, DECODERS[args.decoder], args
    )
    print(
        f"=== incremental value of EEG | {args.channel_set} | {args.decoder} | "
        f"{int(np.isfinite(scores).sum())} scored trials ==="
    )
    print(
        "target: the NEXT block's mean reaction time; "
        "predictors come from the current block only"
    )
    results = []
    for k in args.block_sizes:
        panel = build_panel(
            scores, rts, subjects, sessions, times_ms, k, args.max_gap_s * 1000.0
        )
        if not panel:
            print(f"  K={k}: no complete blocks")
            continue
        print(f"\n  --- K={k} | {len(panel['next_rt'])} block transitions ---")
        for name, columns in (
            ("rt_only", ("rt",)),
            ("eeg_only", ("eeg",)),
            ("eeg_plus_rt", ("rt", "eeg")),
        ):
            outcome = evaluate(panel, columns, args.seed)
            outcome.update({"block_size": k, "predictor": name})
            results.append(outcome)
            print(
                f"    {name:12s} n={outcome['n']:5d}  "
                f"rho={outcome['spearman']:+.4f}  R2={outcome['r2']:+.4f}  "
                f"MAE={outcome['mae_ms']:6.1f} ms"
            )
        base = next(
            r for r in results if r["block_size"] == k and r["predictor"] == "rt_only"
        )
        both = next(
            r
            for r in results
            if r["block_size"] == k and r["predictor"] == "eeg_plus_rt"
        )
        gain = both["spearman"] - base["spearman"]
        verdict = (
            "EEG adds information" if gain > 0.01 else "EEG looks redundant with RT"
        )
        print(f"    -> adding EEG changes rho by {gain:+.4f} ({verdict})")

    summary = {
        "channel_set": args.channel_set,
        "decoder": args.decoder,
        "window_ms": list(window),
        "max_gap_s": args.max_gap_s,
        "seed": args.seed,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"visual_detect_incremental{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return summary


if __name__ == "__main__":
    main()
