"""Does averaging a few trials together rescue the single-trial decoder?

Measurement, not a new task. The single-trial result is ~0.575 mean subject AUC
with an SNR of about 1.3, so the obvious next question is how the score grows
when several consecutive trials are combined. There are two ways to combine, and
they answer slightly different questions:

``--mode block``
    Average the *EEG* of K consecutive trials, then decode the averaged epoch.
    A decoder is fitted on blocks made from the training sessions and scored on
    blocks made from the held-out session. Because averaging is linear, the
    evoked response stays at its amplitude while independent noise falls by
    ``sqrt(K)`` -- this is the experiment that shows whether the signal is there
    at all.
``--mode score``
    Decode each trial on its own, then average the *decoder scores* of K
    consecutive trials. Nothing is refitted, which is what a real-time system
    would do: the per-trial decoder stays as it is and the output is smoothed.

Blocks always stay inside one session, because trials of one recording share an
electrode mount and a slow drift. They are built within one label class as well,
sorted by onset time and non-overlapping, so a "block" is always several stimuli
(or several silence epochs) really presented one after another and never a mix.
A block is dropped when a gap inside it exceeds ``--max-gap-s`` (an unplanned
break) or when fewer than ``--min-per-class`` blocks of a class exist in a
session.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from nova2026.config import PROJECT_ROOT
from nova2026.training import load_chkpt

from .decode import DATASETS, DECODERS, load_arrays, parse_window
from .device import VIS_DIR, channels_for

RESULTS_DIR = PROJECT_ROOT / "results"

BLOCK_SIZES = (1, 2, 3, 5, 8, 12)
MIN_SAMPLES_PER_CLASS = 2


def sessions_of(meta: list) -> np.ndarray:
    return np.asarray([f"{tag[0]}/{tag[1]}" for tag in meta], dtype=object)


def form_blocks(
    index: np.ndarray, times_ms: np.ndarray, k: int, max_gap_ms: float
) -> list[np.ndarray]:
    """Group a time-ordered run of same-class trials into blocks of ``k``.

    A run is cut wherever two consecutive trials are further apart than
    ``max_gap_ms``, so a block never spans an unplanned break. Only complete
    blocks of exactly ``k`` trials are returned; a leftover shorter run at the
    end (or between two breaks) is dropped rather than padded.
    """
    if k <= 0:
        raise ValueError("Block size must be positive.")
    blocks: list[np.ndarray] = []
    run: list[int] = []
    for position, row in enumerate(index):
        if run and times_ms[row] - times_ms[run[-1]] > max_gap_ms:
            run = []  # a break: whatever is pending is abandoned
        run.append(int(row))
        if len(run) == k:
            blocks.append(np.asarray(run, dtype=np.int64))
            run = []
    return blocks


def block_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    times_ms: np.ndarray,
    k: int,
    max_gap_ms: float,
    min_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Averaged epochs, one label per block, and each block's session/subject.

    Trials are grouped by session *and* by class, so a block never averages a
    stimulus with a silence epoch.
    """
    averaged, block_labels, block_sessions, block_subjects = [], [], [], []
    for session in np.unique(sessions):
        for label in (0, 1):
            rows = np.where((sessions == session) & (labels == label))[0]
            if rows.size == 0:
                continue
            rows = rows[np.argsort(times_ms[rows], kind="stable")]
            blocks = form_blocks(rows, times_ms, k, max_gap_ms)
            if min_per_class > 0 and len(blocks) < min_per_class:
                continue
            for block in blocks:
                averaged.append(data[block].mean(axis=0))
                block_labels.append(label)
                block_sessions.append(session)
                block_subjects.append(subjects[block[0]])
    if not averaged:
        return (
            np.empty((0,) + data.shape[1:], dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )
    return (
        np.stack(averaged).astype(np.float32),
        np.asarray(block_labels, dtype=np.int64),
        np.asarray(block_sessions, dtype=object),
        np.asarray(block_subjects, dtype=object),
    )


def cross_session_scores(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    decoder,
    args,
) -> list[dict]:
    """Per-subject AUC with folds grouped by session (train 2, test 1)."""
    rows_out = []
    for subject in sorted(set(subjects), key=lambda s: int(s.split("-")[-1])):
        rows = np.where(subjects == subject)[0]
        session_ids = sessions[rows]
        if len(set(session_ids)) < 2 or len(set(labels[rows])) < 2:
            continue
        if min(int(labels[rows].sum()), int((labels[rows] == 0).sum())) < (
            MIN_SAMPLES_PER_CLASS
        ):
            continue
        scores = np.full(len(rows), np.nan)
        for train_idx, test_idx in StratifiedGroupKFold(
            n_splits=min(3, len(set(session_ids))),
            shuffle=True,
            random_state=args.seed,
        ).split(rows, labels[rows], groups=session_ids):
            train_rows, test_rows = rows[train_idx], rows[test_idx]
            if len(set(labels[train_rows])) < 2:
                continue
            result = decoder(
                data[train_rows], labels[train_rows], data[test_rows], args
            )
            scores[test_idx] = result.values
        usable = np.isfinite(scores)
        if len(set(labels[rows][usable])) < 2:
            continue
        rows_out.append(
            {
                "subject": subject,
                "n": int(usable.sum()),
                "n_pos": int(labels[rows][usable].sum()),
                "auc": float(roc_auc_score(labels[rows][usable], scores[usable])),
            }
        )
    return rows_out


def smooth_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    times_ms: np.ndarray,
    k: int,
    max_gap_ms: float,
    min_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-trial scores over runs of ``k`` same-class consecutive trials.

    Grouping matches :func:`block_dataset` so the two modes are directly
    comparable: same sessions, same class separation, same gap rule. The score
    is the mean of the run's per-trial scores -- exactly what a real-time system
    would average -- and the label is the run's (shared) class.
    """
    out_scores, out_labels = [], []
    for session in np.unique(sessions):
        for label in (0, 1):
            rows = np.where((sessions == session) & (labels == label))[0]
            if rows.size == 0:
                continue
            rows = rows[np.argsort(times_ms[rows], kind="stable")]
            blocks = form_blocks(rows, times_ms, k, max_gap_ms)
            if min_per_class > 0 and len(blocks) < min_per_class:
                continue
            for block in blocks:
                values = scores[block]
                if not np.isfinite(values).all():
                    continue
                out_scores.append(float(values.mean()))
                out_labels.append(label)
    return np.asarray(out_scores), np.asarray(out_labels, dtype=np.int64)


def score_mode_auc(
    per_trial_scores: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    times_ms: np.ndarray,
    k: int,
    max_gap_ms: float,
    min_per_class: int,
) -> float:
    """Mean per-subject AUC after averaging ``k`` consecutive decoder scores."""
    aucs = []
    for subject in sorted(set(subjects), key=lambda s: int(s.split("-")[-1])):
        rows = np.where(subjects == subject)[0]
        if len(set(labels[rows])) < 2:
            continue
        smoothed, smooth_labels = smooth_scores(
            per_trial_scores[rows],
            labels[rows],
            sessions[rows],
            times_ms[rows],
            k,
            max_gap_ms,
            min_per_class,
        )
        if len(set(smooth_labels)) < 2:
            continue
        aucs.append(float(roc_auc_score(smooth_labels, smoothed)))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_trial_scores(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    decoder,
    args,
) -> np.ndarray:
    """Out-of-fold decoder score for every trial, session-grouped folds."""
    scores = np.full(len(data), np.nan)
    for subject in sorted(set(subjects), key=lambda s: int(s.split("-")[-1])):
        rows = np.where(subjects == subject)[0]
        session_ids = sessions[rows]
        if len(set(session_ids)) < 2 or len(set(labels[rows])) < 2:
            continue
        for train_idx, test_idx in StratifiedGroupKFold(
            n_splits=min(3, len(set(session_ids))),
            shuffle=True,
            random_state=args.seed,
        ).split(rows, labels[rows], groups=session_ids):
            train_rows, test_rows = rows[train_idx], rows[test_idx]
            if len(set(labels[train_rows])) < 2:
                continue
            result = decoder(
                data[train_rows], labels[train_rows], data[test_rows], args
            )
            scores[test_rows] = result.values
    return scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", choices=("vr", "rt"), default="vr")
    parser.add_argument("--channel-set", default="all")
    parser.add_argument("--window", default="100-300")
    parser.add_argument("--decoders", nargs="+", default=["riemann", "template"])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--mode", choices=("block", "score", "both"), default="both")
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=4,
        help="a session needs at least this many blocks per class to be used",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=15.0,
        help="a block is not formed across a longer pause",
    )
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
    window = parse_window(args.window)
    data, labels, subjects, sessions = load_arrays(
        str(path), args.scheme, window, (args.base[0], args.base[1])
    )
    checkpoint = load_chkpt(str(path))
    times_ms = np.asarray(checkpoint["onset_ms"], dtype=np.float64)
    if times_ms.size != len(data):
        raise ValueError(
            f"The checkpoint stores {times_ms.size} onsets for {len(data)} epochs; "
            "rebuild it with the current build_dataset."
        )
    n_channels = len(channels_for(args.channel_set))
    max_gap_ms = args.max_gap_s * 1000.0
    print(
        f"=== {args.channel_set} ({n_channels} ch) | {window[0]}-{window[1]} ms | "
        f"{data.shape[0]} trials | max gap {args.max_gap_s:g} s ==="
    )

    rows = []
    for name in args.decoders:
        decoder = DECODERS[name]
        modes = ("block", "score") if args.mode == "both" else (args.mode,)
        for k in args.block_sizes:
            for mode in modes:
                if mode == "block":
                    blocks, block_labels, block_sessions, block_subjects = (
                        block_dataset(
                            data,
                            labels,
                            subjects,
                            sessions,
                            times_ms,
                            k,
                            max_gap_ms,
                            args.min_per_class,
                        )
                    )
                    if len(blocks) < 10 or len(set(block_labels)) < 2:
                        print(f"  {name:10s} block K={k:2d}  too few blocks, skipped")
                        continue
                    per_subject = cross_session_scores(
                        blocks,
                        block_labels,
                        block_subjects,
                        block_sessions,
                        decoder,
                        args,
                    )
                    units = len(blocks)
                else:
                    trial_scores = per_trial_scores(
                        data, labels, subjects, sessions, decoder, args
                    )
                    mean_auc = score_mode_auc(
                        trial_scores,
                        labels,
                        subjects,
                        sessions,
                        times_ms,
                        k,
                        max_gap_ms,
                        args.min_per_class,
                    )
                    per_subject = []
                    units = int(np.isfinite(trial_scores).sum())
                    if np.isfinite(mean_auc):
                        per_subject = [{"auc": mean_auc}]
                aucs = [row["auc"] for row in per_subject]
                mean_auc = float(np.mean(aucs)) if aucs else float("nan")
                rows.append(
                    {
                        "decoder": name,
                        "mode": mode,
                        "block_size": k,
                        "n_units": units,
                        "mean_subject_auc": mean_auc,
                        "n_subjects": len(aucs),
                    }
                )
                print(
                    f"  {name:10s} {mode:5s} K={k:2d}  "
                    f"units={units:6d}  mean subject AUC = {mean_auc:.4f}"
                )

    summary = {
        "channel_set": args.channel_set,
        "window_ms": list(window),
        "min_per_class": args.min_per_class,
        "max_gap_s": args.max_gap_s,
        "seed": args.seed,
        "results": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"visual_detect_averaging{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return summary


if __name__ == "__main__":
    main()
