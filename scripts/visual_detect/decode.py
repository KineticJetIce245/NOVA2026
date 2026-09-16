"""Decoders for the post-stimulus visual-detection task.

Why this file exists
--------------------
``EEGNet`` reduces the time axis by a factor of 8 before it decides anything.
On this task that is fatal: the evidence is the *shape* of a ~3.7 uV evoked
response inside a 200 ms window, and pooling averages that shape away. The
measured symptom is a network stuck at chance while a plain linear model on the
same epochs reaches 0.88-0.91 AUC within a session.

Every decoder here keeps the time axis (except ``window_mean``, which is kept
only as a deliberately blind control). They are fitted on training sessions and
scored on a held-out session of the same subject, so nothing about the test
session -- not even its mean amplitude -- reaches the model.

Decoders
--------
``template``
    Parameter-free matched filter. The mean epoch of each class is the template;
    a trial is scored by the difference between its correlation with the
    stimulus template and with the silence template. This is the ceiling probe:
    if it works, the information is in the waveform.
``xdawnd``
    xDAWN spatial filtering (Rivet et al., 2009) followed by the same template
    correlation. xDAWN learns ``n_filters`` spatial weights per class that
    maximise the evoked response against the background, then keeps the whole
    time course.
``riemann``
    Log-Euclidean tangent-space projection of the per-trial covariance matrix,
    fed to logistic regression. Uses spatial covariance, which carries the
    cross-channel structure of the response, and no temporal averaging at all.
``linear``
    Flattened and standardised samples with logistic regression. The simple
    model that already showed the signal is there.
``peak_to_peak``
    One feature per channel (max minus min inside the window) with logistic
    regression. Interpretable, and a check on whether amplitude alone suffices.
``window_mean``
    One feature per channel (mean inside the window) with logistic regression.
    Deliberately waveform-blind: it can only win through slow level differences,
    so it is the control that says how much of a result is drift rather than
    response.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nova2026.config import PROJECT_ROOT
from nova2026.training import load_chkpt

from .device import VIS_DIR, channels_for
from .train_occipital import windows

RESULTS_DIR = PROJECT_ROOT / "results"

#: Default checkpoint stems per channel set, as written by ``build_dataset``.
DATASETS = {
    "occipital": "PVT_VIS_occipital_m50_500ms_128.0Hz_VisualPipeline.pt",
    "posterior": "PVT_VIS_posterior_m50_500ms_128.0Hz_VisualPipeline.pt",
    "all": "PVT_VIS_all_m50_500ms_128.0Hz_VisualPipeline.pt",
}

WINDOWS = ((100, 300), (50, 500))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unit_norm(flat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norm, 1e-12)


def fit_templates(
    x: np.ndarray, y: np.ndarray, classes=(0, 1)
) -> dict[int, np.ndarray]:
    """Mean epoch per class, flattened and unit-normalised."""
    templates = {}
    for label in classes:
        rows = x[y == label]
        if rows.size == 0:
            raise ValueError(f"No training trials for class {label}.")
        mean = rows.reshape(len(rows), -1).mean(axis=0, keepdims=True)
        templates[label] = _unit_norm(mean)
    return templates


def template_scores(x: np.ndarray, templates: dict[int, np.ndarray]) -> np.ndarray:
    """corr(trial, stimulus template) - corr(trial, silence template)."""
    flat = _unit_norm(x.reshape(len(x), -1))
    return (flat @ templates[1].ravel()) - (flat @ templates[0].ravel())


def xdawn_filters(
    x: np.ndarray, y: np.ndarray, n_filters: int, classes=(0, 1)
) -> np.ndarray:
    """Spatial filters maximising evoked response over background, per class.

    Solves the generalised eigenproblem
    ``C_x w = lambda (C_x + C_n) w`` for each class and keeps the
    ``n_filters`` eigenvectors with the largest eigenvalues, exactly as in the
    original xDAWN formulation. Returns ``(n_classes * n_filters, n_channels)``.
    """
    n_channels = x.shape[1]

    def sample_cov(rows: np.ndarray) -> np.ndarray:
        # (n, C, T) -> (n * T, C): every sample is one observation
        samples = rows.transpose(0, 2, 1).reshape(-1, n_channels)
        return np.cov(samples, rowvar=False)

    cov_all = sample_cov(x)
    filters = []
    for label in classes:
        rows = x[y == label]
        if rows.size == 0:
            raise ValueError(f"No training trials for class {label}.")
        # xDAWN scales only the ratio, so the two covariances need no
        # normalisation before the generalised eigenproblem
        eigenvalues, eigenvectors = eigh(sample_cov(rows), cov_all + sample_cov(rows))
        order = np.argsort(eigenvalues)[::-1][:n_filters]
        filters.append(eigenvectors[:, order].T)
    return np.concatenate(filters, axis=0)


def apply_filters(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Project ``(n, C, T)`` onto spatial filters -> ``(n, K, T)``."""
    return np.einsum("kf,nft->nkt", filters, x)


def _matrix_log(matrix: np.ndarray) -> np.ndarray:
    """Matrix logarithm of a symmetric positive-definite matrix."""
    values, vectors = eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-12))) @ vectors.T


def riemannian_tangent(
    x: np.ndarray, reference: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Log-Euclidean tangent-space features of every per-trial covariance.

    Log-Euclidean rather than affine-invariant: for symmetric positive-definite
    matrices the matrix logarithm of the geometric mean is the mean of the
    matrix logarithms, so both the reference point and the projection are a few
    lines of ``eigh`` and the projection is a genuine vector-space embedding.
    Returns the features and the reference that was used, so a test fold can be
    projected onto the training fold's reference.
    """
    n_trials, n_channels, n_samples = x.shape
    if n_samples < 2:
        raise ValueError(
            f"A covariance needs at least 2 samples per trial, got {n_samples}."
        )
    covs = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)
    eye = np.eye(n_channels)
    for i, trial in enumerate(x):
        cov = np.cov(trial.astype(np.float64))
        covs[i] = cov + 1e-6 * np.trace(cov) / n_channels * eye

    logs = np.empty_like(covs)
    for i, cov in enumerate(covs):
        logs[i] = _matrix_log(cov)
    if reference is None:
        reference = _matrix_log(logs.mean(axis=0))
    log_reference = _matrix_log(np.asarray(reference))

    rows, cols = np.triu_indices(n_channels)
    features = np.empty((n_trials, rows.size), dtype=np.float64)
    for i in range(n_trials):
        centred = logs[i] - log_reference
        features[i] = centred[rows, cols]
    # off-diagonal entries appear twice in the Frobenius inner product
    features[:, rows != cols] *= np.sqrt(2.0)
    return features, reference


# --------------------------------------------------------------------------- #
# scoring interface
# --------------------------------------------------------------------------- #
@dataclass
class Scores:
    """Scores for the test trials plus whatever the decoder wants to report."""

    values: np.ndarray
    info: dict


class Decoder(Protocol):
    """Fit on training epochs, score test epochs."""

    name: str

    def __call__(
        self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, args
    ) -> Scores: ...


def decoder_template(train_x, train_y, test_x, args) -> Scores:
    templates = fit_templates(train_x, train_y)
    return Scores(template_scores(test_x, templates), {})


def decoder_xdawnd(train_x, train_y, test_x, args) -> Scores:
    filters = xdawn_filters(train_x, train_y, args.n_filters)
    filtered_train = apply_filters(train_x, filters)
    filtered_test = apply_filters(test_x, filters)
    templates = fit_templates(filtered_train, train_y)
    scores = template_scores(filtered_test, templates)
    return Scores(scores, {"n_filters": args.n_filters})


def _regression(args) -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=args.c, class_weight="balanced")


def decoder_linear(train_x, train_y, test_x, args) -> Scores:
    model = make_pipeline(StandardScaler(), _regression(args))
    model.fit(train_x.reshape(len(train_x), -1), train_y)
    return Scores(model.decision_function(test_x.reshape(len(test_x), -1)), {})


def decoder_peak_to_peak(train_x, train_y, test_x, args) -> Scores:
    def features(x):
        return (x.max(axis=2) - x.min(axis=2)).astype(np.float64)

    model = make_pipeline(StandardScaler(), _regression(args))
    model.fit(features(train_x), train_y)
    return Scores(model.decision_function(features(test_x)), {})


def decoder_window_mean(train_x, train_y, test_x, args) -> Scores:
    def features(x):
        return x.mean(axis=2, dtype=np.float64)

    model = make_pipeline(StandardScaler(), _regression(args))
    model.fit(features(train_x), train_y)
    return Scores(model.decision_function(features(test_x)), {})


def decoder_riemann(train_x, train_y, test_x, args) -> Scores:
    train_features, reference = riemannian_tangent(train_x)
    test_features, _ = riemannian_tangent(test_x, reference=reference)
    model = make_pipeline(StandardScaler(), _regression(args))
    model.fit(train_features, train_y)
    return Scores(model.decision_function(test_features), {})


DECODERS: dict[str, Decoder] = {
    "template": decoder_template,
    "xdawnd": decoder_xdawnd,
    "riemann": decoder_riemann,
    "linear": decoder_linear,
    "peak_to_peak": decoder_peak_to_peak,
    "window_mean": decoder_window_mean,
}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def load_arrays(
    dataset: str,
    scheme: str,
    window: tuple[int, int],
    base: tuple[int, int],
    zscore: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Epochs inside ``window``, plus labels, subjects and sessions."""
    checkpoint = load_chkpt(dataset)
    data_key = "data_vr" if scheme == "vr" else "data_rt"
    meta_key = "metadata" if scheme == "vr" else "rt_metadata"
    sample_rate = float(checkpoint["sample_rate_hz"])
    data = windows(
        np.asarray(checkpoint[data_key], dtype=np.float32),
        sample_rate,
        int(checkpoint["pre_ms"]),
        window[0],
        window[1],
        base,
        zscore,
    )
    labels = np.asarray(checkpoint["labels"][scheme], dtype=np.int64)
    meta = checkpoint[meta_key]
    subjects = np.asarray([tag[0] for tag in meta], dtype=object)
    sessions = np.asarray(
        [f"{tag[0]}/{tag[1]}" for tag in meta], dtype=object
    )
    return data, labels, subjects, sessions


def within_subject_auc(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    decoder: Decoder,
    args,
) -> tuple[float, list[dict]]:
    """Mean per-subject AUC over session-grouped folds.

    Folds never mix sessions, because the trials of one session share an
    electrode mount and a slow drift; training on two sessions and testing on a
    third is the realistic calibration setting. ``--split same_session`` runs a
    single session with a random trial split instead, which measures how much of
    a score comes from session-specific structure that cannot transfer: that
    contrast is the point of the option.
    """
    per_subject = []
    for subject in sorted(set(subjects), key=lambda s: int(s.split("-")[-1])):
        rows = np.where(subjects == subject)[0]
        session_ids = sessions[rows]
        if len(set(session_ids)) < 2 or len(set(labels[rows])) < 2:
            continue
        scores = np.full(len(rows), np.nan)
        if args.split == "same_session":
            for session in sorted(set(session_ids)):
                fold_rows = rows[session_ids == session]
                if len(set(labels[fold_rows])) < 2 or len(fold_rows) < 20:
                    continue
                splitter = StratifiedKFold(
                    n_splits=max(2, min(3, int(labels[fold_rows].min()))),
                    shuffle=True,
                    random_state=args.seed,
                )
                for train_idx, test_idx in splitter.split(fold_rows, labels[fold_rows]):
                    inside = np.where(session_ids == session)[0]
                    train_rows, test_rows = (
                        fold_rows[train_idx],
                        fold_rows[test_idx],
                    )
                    result = decoder(
                        data[train_rows], labels[train_rows], data[test_rows], args
                    )
                    scores[inside[test_idx]] = result.values
        else:
            n_folds = min(3, len(set(session_ids)))
            splitter = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=args.seed
            )
            for train_idx, test_idx in splitter.split(
                rows, labels[rows], groups=session_ids
            ):
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
        per_subject.append(
            {
                "subject": subject,
                "n": int(usable.sum()),
                "n_pos": int(labels[rows][usable].sum()),
                "auc": float(roc_auc_score(labels[rows][usable], scores[usable])),
            }
        )
    aucs = [row["auc"] for row in per_subject]
    return (float(np.mean(aucs)) if aucs else float("nan")), per_subject


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", choices=("vr", "rt"), default="vr")
    parser.add_argument(
        "--channel-sets",
        nargs="+",
        default=["occipital", "posterior", "all"],
        help="which built checkpoints to compare",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=[f"{lo}-{hi}" for lo, hi in WINDOWS],
        help="analysis windows in ms, e.g. 100-300 50-500",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=list(DECODERS),
        help=f"any of {sorted(DECODERS)}",
    )
    parser.add_argument("--base", type=float, nargs=2, default=(-50.0, 0.0))
    parser.add_argument(
        "--split",
        choices=("cross_session", "same_session"),
        default="cross_session",
        help="cross_session trains on the subject's other sessions; same_session "
        "splits one session at random, which inflates the score",
    )
    parser.add_argument(
        "--n-filters", type=int, default=4, help="xDAWN filters per class"
    )
    parser.add_argument("--c", type=float, default=1.0, help="logistic regression C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


def parse_window(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition("-")
    window = (int(lo), int(hi))
    if window[1] <= window[0]:
        raise ValueError(f"Invalid window {text!r}.")
    return window


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    base = (args.base[0], args.base[1])
    unknown = [name for name in args.decoders if name not in DECODERS]
    if unknown:
        raise SystemExit(f"Unknown decoders {unknown}; available {sorted(DECODERS)}")

    rows = []
    for channel_set in args.channel_sets:
        path = VIS_DIR / DATASETS[channel_set]
        if not path.exists():
            print(f"skip {channel_set}: {path.name} not built")
            continue
        n_channels = len(channels_for(channel_set))
        for window in [parse_window(text) for text in args.windows]:
            data, labels, subjects, sessions = load_arrays(
                str(path), args.scheme, window, base
            )
            if data.shape[1] != n_channels:
                raise ValueError(
                    f"{channel_set} checkpoint has {data.shape[1]} channels, "
                    f"expected {n_channels}."
                )
            print(
                f"\n=== {channel_set} ({n_channels} ch) | {window[0]}-{window[1]} ms | "
                f"{data.shape[0]} trials, {data.shape[2]} samples ==="
            )
            for name in args.decoders:
                auc, per_subject = within_subject_auc(
                    data, labels, subjects, sessions, DECODERS[name], args
                )
                rows.append(
                    {
                        "channel_set": channel_set,
                        "n_channels": n_channels,
                        "window_ms": list(window),
                        "n_samples": int(data.shape[2]),
                        "decoder": name,
                        "mean_subject_auc": auc,
                        "per_subject": per_subject,
                    }
                )
                print(
                    f"  {name:14s} mean subject AUC = {auc:.4f}  "
                    f"({sum(1 for r in per_subject if r['auc'] > 0.5)}/"
                    f"{len(per_subject)} subjects above 0.5)"
                )

    summary = {
        "scheme": args.scheme,
        "baseline_ms": list(base),
        "n_filters": args.n_filters,
        "c": args.c,
        "seed": args.seed,
        "results": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"visual_detect_arm{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    print("\n=== ranking (mean subject AUC) ===")
    for row in sorted(rows, key=lambda r: -r["mean_subject_auc"])[:12]:
        print(
            f"  {row['mean_subject_auc']:.4f}  {row['decoder']:14s} "
            f"{row['channel_set']:10s} {row['window_ms'][0]}-{row['window_ms'][1]} ms"
        )
    return summary


if __name__ == "__main__":
    main()
