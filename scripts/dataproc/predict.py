"""Lapse decision rule and leave-one-subject-out F1 for the engagement arm.

Consumes per-trial composite scores Z_bar (design doc Eq. 13; computed by
``engage_z.normalize_composite``), the RT-decile labels and the metadata of
the PVT checkpoint (schema documented in ``cogbci_pvt.py``).

Decision rules (both parameter-free, matching the design doc's deliberately
parameter-free algorithmic arm):

* ``rank`` (default): within each (subject, session), flag the lowest
  ``frac`` of composite scores as lapses.  The label prior is the same
  (within-session slowest RT decile = 10 % positives), and ranking within
  the session makes the rule immune to session-level offsets (and to the
  composite's mean-vs-median reference, since ranks are shift-invariant).
* ``threshold``: flag every trial whose composite score lies below
  ``threshold``.  Pass the **mean-referenced** composite (see
  ``engage_z.normalize_composite(..., mean_referenced=True)``) so that
  ``threshold = 0`` reads "below the resting *mean* engagement"; the
  design-doc (median-referenced) composite sits ~-0.1 below its mean, so an
  unadjusted 0 threshold would be slightly too conservative.

LOSO (leave-one-subject-out): both rules are parameter-free, so nothing is
fitted on the training folds; each held-out subject's trials are scored
directly with the rule.  Reported are the pooled F1 over all held-out
trials, the macro F1 (mean of per-subject F1), a chance baseline (F1 of an
uninformative rule with the same prediction rate) and a label-permutation
null (labels shuffled within sessions, fixed predictions), so the observed
F1 can be read against its null distribution.

Usage (runs the full evaluation and prints a summary):
    .venv\\Scripts\\python.exe scripts\\dataproc\\predict.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

FRAC = 0.1  # within-session slowest decile (matches the label prior)


def session_keys(metadata: np.ndarray) -> np.ndarray:
    """One "(subject)/(session)" key per trial, e.g. "sub-01/ses-S1"."""
    meta = np.asarray(metadata, dtype=object)
    return np.asarray([f"{s}/{e}" for s, e in meta[:, :2]], dtype=object)


def lapse_decision_rank(
    z_bar: np.ndarray,
    session_ids: np.ndarray,
    frac: float = FRAC,
) -> np.ndarray:
    """Within-session bottom ``frac`` of Z_bar -> 1, rest -> 0 (int array).

    Uses the same count convention as the label build in ``cogbci_pvt.py``
    (``int(n * frac)`` positives per session), so the prediction rate equals
    the label rate exactly.
    """
    z = np.asarray(z_bar, dtype=np.float64)
    sessions = np.asarray(session_ids)
    pred = np.zeros(len(z), dtype=np.int64)
    for ses in np.unique(sessions):
        idx = np.where(sessions == ses)[0]
        k = int(len(idx) * frac)
        if k == 0:
            continue
        order = idx[np.argsort(z[idx], kind="stable")]
        pred[order[:k]] = 1
    return pred


def lapse_decision_threshold(z_bar: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Z_bar < threshold -> 1, else 0 (int array)."""
    return (np.asarray(z_bar, dtype=np.float64) < threshold).astype(np.int64)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def loso_f1(
    z_bar: np.ndarray,
    labels: np.ndarray,
    metadata: np.ndarray,
    rule: str = "rank",
    frac: float = FRAC,
    threshold: float = 0.0,
    n_perm: int = 20,
    seed: int = 0,
) -> dict:
    """Leave-one-subject-out F1 of the chosen decision rule.

    Parameters
    ----------
    z_bar, labels, metadata
        Per-trial composite scores, RT-decile labels and checkpoint metadata.
        For the ``threshold`` rule pass the *mean-referenced* composite so
        the default threshold of 0 means "the resting mean" (see the module
        docstring); the ``rank`` rule is reference-invariant.
    rule
        "rank" or "threshold" (see the module docstring).
    frac, threshold
        Rule parameters (prediction fraction / composite threshold).
    n_perm, seed
        Number of label permutations for the null distribution and the RNG
        seed (labels are permuted within sessions; predictions are fixed).

    Returns a dict with:
      rows       per-subject results (n, n_pos, n_pred, precision, recall, f1)
      pooled     precision/recall/F1 over all held-out trials concatenated
      macro_f1   mean per-subject F1 over subjects with at least one positive
      chance_f1  F1 of an uninformative rule with the same prediction rate
      null       {'mean', 'p95'} of the permutation null F1
    """
    meta = np.asarray(metadata, dtype=object)
    y = np.asarray(labels, dtype=np.int64)
    z = np.asarray(z_bar, dtype=np.float64)
    keys = session_keys(meta)
    subjects = sorted({str(s) for s in meta[:, 0]}, key=lambda s: int(s[4:]))

    if rule == "rank":
        pred_all = lapse_decision_rank(z, keys, frac=frac)
    elif rule == "threshold":
        pred_all = lapse_decision_threshold(z, threshold=threshold)
    else:
        raise ValueError(f"Unknown rule: {rule!r}")

    rows = []
    for sub in subjects:
        m = meta[:, 0].astype(str) == sub
        true, pred = y[m], pred_all[m]
        rows.append(
            {
                "subject": sub,
                "n": int(m.sum()),
                "n_pos": int(true.sum()),
                "n_pred": int(pred.sum()),
                **_scores(true, pred),
            }
        )

    pooled = _scores(y, pred_all)
    f1s = [r["f1"] for r in rows if r["n_pos"] > 0]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # chance: an uninformative rule with the same prediction rate
    fl = float(y.mean())
    fr = float(pred_all.mean())
    chance_f1 = 2 * fl * fr / (fl + fr) if fl + fr > 0 else 0.0

    # permutation null: shuffle labels within sessions, predictions fixed
    rng = np.random.default_rng(seed)
    null_f1s = []
    for _ in range(n_perm):
        y_perm = y.copy()
        for ses in np.unique(keys):
            idx = np.where(keys == ses)[0]
            y_perm[idx] = y_perm[idx][rng.permutation(len(idx))]
        null_f1s.append(f1_score(y_perm, pred_all, zero_division=0))
    null_arr = np.asarray(null_f1s, dtype=np.float64)

    return {
        "rows": rows,
        "pooled": pooled,
        "macro_f1": macro_f1,
        "chance_f1": chance_f1,
        "null": {"mean": float(null_arr.mean()), "p95": float(np.percentile(null_arr, 95))},
    }


if __name__ == "__main__":
    import torch

    from engage_z import fit_baselines, lapse_contrast, normalize_composite
    from nova2026.config import DATA_DIR

    DATASET_ROOT = DATA_DIR / "COG-BCI" / "outputs"
    rest = torch.load(DATASET_ROOT / "RS_Beg_EO_128Hz_AttUPipeline.pt", weights_only=False)
    pvt = torch.load(DATASET_ROOT / "PVT_128Hz_AttUPipeline.pt", weights_only=False)

    baselines = fit_baselines(rest)
    z_bar = normalize_composite(pvt, baselines)  # design-doc (median-referenced)
    z_bar_mean = normalize_composite(pvt, baselines, mean_referenced=True)
    labels = np.asarray(pvt["labels"], dtype=np.int64)
    meta = np.asarray(pvt["metadata"], dtype=object)
    print(f"Scored N={len(z_bar)} trials against {len(baselines)} baselines.")
    print(
        f"  resting composite offset (mean - median) ~ "
        f"{np.mean([b.composite_mean for b in baselines.values()]):+.4f} sigma"
    )

    print("\n== Paired lapse contrast (validation 3) ==")
    c = lapse_contrast(z_bar, labels, meta)
    print(
        f"  mean session diff = {c['mean']:+.4f} sigma (sd {c['sd']:.4f}, "
        f"n={c['n_sessions']} sessions)"
    )
    print(f"  t({c['n_sessions'] - 1}) = {c['t']:.2f}, p = {c['p']:.4f}")

    for rule in ("rank", "threshold"):
        # rank: median-referenced (shift-invariant); threshold: mean-referenced
        z_rule = z_bar if rule == "rank" else z_bar_mean
        out = loso_f1(z_rule, labels, meta, rule=rule)
        print(f"\n== LOSO F1, rule='{rule}' ==")
        p = out["pooled"]
        print(
            f"  pooled: precision={p['precision']:.3f} recall={p['recall']:.3f} "
            f"f1={p['f1']:.3f}"
        )
        print(
            f"  macro F1 = {out['macro_f1']:.3f} | chance F1 = {out['chance_f1']:.3f} "
            f"| null mean/p95 = {out['null']['mean']:.3f}/{out['null']['p95']:.3f}"
        )
        print("  per subject: n, n_pos, n_pred, precision, recall, f1")
        for r in out["rows"]:
            print(
                f"    {r['subject']}: n={r['n']:3d} pos={r['n_pos']:2d} "
                f"pred={r['n_pred']:2d} P={r['precision']:.2f} R={r['recall']:.2f} "
                f"F1={r['f1']:.2f}"
            )
