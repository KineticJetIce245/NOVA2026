import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from nova2026.architecture.cnn import EEGNet
from nova2026.config import DATA_DIR, PROJECT_ROOT
from nova2026.training import Metric, SupervisedTrainer

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "outputs/PVT_128Hz_AttUPipeline.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DROPOUT = 0.5  # EEGNet dropout (regularised setting)
VAL_FRAC = 0.2  # fraction of TRAINING subjects held out for epoch selection
SEED = 0
N_BOOT_FOLD = 1000  # trial-level bootstrap resamples for a per-fold Spearman CI
N_BOOT_MEAN = 10000  # fold-level bootstrap resamples for the mean CIs


# model hyperparameters
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
callbacks = {
    "epoch": lambda d: print(
        f"At epoch {d['epoch']}, train_loss={d['train_loss']:.4f}, "
        f"best_metric={d['best_metric']:.4f}"
    ),
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


def log_rt_targets(data_chkpt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw log RT, validity mask); invalid RTs are NaN and dropped.

    NOTE: this returns the *raw* log RT.  The z-scoring (subtract train mean,
    divide by train std) happens later in ``organizer``, because it needs the
    train/val/test split to know which trials count as "training".
    """
    rt = np.asarray(data_chkpt["rt"], dtype=float)
    valid = np.isfinite(rt) & (rt > 0)
    targets = np.full(rt.shape, np.nan, dtype=np.float32)
    targets[valid] = np.log(rt[valid])
    return targets[:, None], valid


def val_spearman(yt, yp, to_ms) -> float:
    """Pooled Spearman rho over all validation trials; -1.0 when undefined.

    ``spearmanr`` returns NaN if either input is constant, which would break
    the best-epoch comparison, so map that case to a value worse than any rho.
    """
    rho = float(spearmanr(to_ms(yt), to_ms(yp)).statistic)
    return rho if np.isfinite(rho) else -1.0


def grouped_spearman(yt, yp, to_ms, groups) -> float:
    """Mean of per-subject Spearman rho (every subject weighted equally).

    Epoch selection uses this instead of a pooled rho so that between-subject
    differences cannot dominate the score.  ``groups`` must be aligned with the
    prediction order (validation loaders use ``shuffle=False``).
    """
    yt_ms, yp_ms, groups = to_ms(np.ravel(yt)), to_ms(np.ravel(yp)), np.ravel(groups)
    rhos = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 3:
            continue
        rho = float(spearmanr(yt_ms[mask], yp_ms[mask]).statistic)
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else -1.0


def bootstrap_mean_ci(values, n_boot=N_BOOT_MEAN, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean of ``values`` (fold-level)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def spearman_ci(y_true, y_pred, n_boot=N_BOOT_FOLD, seed=SEED, alpha=0.05):
    """Trial-level percentile bootstrap CI for one fold's Spearman rho."""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[b] = spearmanr(y_true[idx], y_pred[idx]).statistic
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def organizer(
    data_chkpt: dict,
    test_sub: str,
    subs: list,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
):
    """Split one LOSO fold and z-score the targets with TRAIN statistics.

    The target is ``log(RT)``; standardising it (``z = (log RT - mu) / sd``)
    puts the mean target at 0, so the randomly initialised output layer already
    sits at the constant baseline instead of having to walk from 0 up to ~5.8
    under Adam's lr-bounded steps.  ``mu``/``sd`` are estimated from the
    training subjects of THIS fold only (never from val/test -- that would be
    label leakage) and are returned in ``chkpt["norm"]`` so predictions can be
    mapped back to milliseconds.
    """
    print(f"Omitting subject {test_sub} from training data.")
    chkpt = {}
    log_rt, valid = log_rt_targets(data_chkpt)  # raw log RT, shape (N, 1)
    data = np.asarray(data_chkpt["data"])
    subs_meta = np.asarray([tags[0] for tags in data_chkpt["metadata"]])
    print(f"Dropped {int((~valid).sum())} trials with invalid RT (<= 0 or NaN).")

    # --- step 1: split train / val / test FIRST ---
    # The split must come before normalisation so the statistics below can
    # only ever see training subjects.
    test_mask = (subs_meta == test_sub) & valid
    if 0 > val_frac or val_frac >= 1:
        raise ValueError("Val fraction must be less than 1.")
    n_vals = max(1, int((len(subs) - 1) * val_frac))
    rng = np.random.default_rng(seed)
    subs_not_test = [s for s in subs if s != test_sub]
    val_subs = rng.choice(subs_not_test, size=n_vals, replace=False)
    val_mask = np.isin(subs_meta, val_subs) & valid
    train_mask = np.where(valid & ~test_mask & ~val_mask)[0]
    # Subject label per validation trial, aligned with the val loader order
    # (shuffle=False), so per-subject metrics can group the predictions.
    chkpt["val_subjects"] = subs_meta[val_mask]

    # --- step 2: z-score the target using TRAIN statistics only ---
    # z = (log RT - mu) / sd gives the train target mean 0 and std 1, so an
    # output layer initialised near 0 already equals the constant baseline and
    # no epochs are wasted teaching Adam the global bias.  Bonus: in z space
    # the constant baseline's MSE is exactly 1.0, an easy reference to beat.
    mu = float(log_rt[train_mask].mean())
    sd = float(log_rt[train_mask].std())
    if sd <= 0:
        raise ValueError("Training log-RT has zero variance; cannot standardise.")
    z = (log_rt - mu) / sd
    chkpt["norm"] = {"mu": mu, "sd": sd}  # used by the main loop to map back to ms
    print(
        f"Train set: {len(train_mask)} trials, "
        f"log RT mu={mu:.3f}, sd={sd:.3f} (targets z-scored)"
    )

    # --- step 3: all three DataLoaders carry the z-scored target ---
    chkpt["test"] = DataLoader(
        TensorDataset(
            torch.tensor(data[test_mask], dtype=torch.float32),
            torch.tensor(z[test_mask], dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    chkpt["val"] = DataLoader(
        TensorDataset(
            torch.tensor(data[val_mask], dtype=torch.float32),
            torch.tensor(z[val_mask], dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    chkpt["train"] = DataLoader(
        TensorDataset(
            torch.tensor(data[train_mask], dtype=torch.float32),
            torch.tensor(z[train_mask], dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return chkpt


trainer = SupervisedTrainer(seed=SEED)
trainer.fetch(Path(DATASET))

meta = trainer.chkpt["metadata"]
subs = list({tags[0] for tags in meta})
subs.sort(key=lambda s: int(s[-2:]))  # sort by the last two char as number
fold_results = []
for i, sub in enumerate(subs):  # main LOSO loop
    # Prepare Model
    model = EEGNet(chn=62, classes=1, p=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Split first: normalisation needs this fold's mu/sd, so organize() must
    # run before uses().
    trainer.organize(lambda x, sb=sub, sbs=subs, sd=i: organizer(x, sb, sbs, seed=sd))
    d = trainer.data

    # z space -> milliseconds: RT = exp(z * sd + mu)
    mu, sd = d["norm"]["mu"], d["norm"]["sd"]
    to_ms = lambda z_: np.exp(z_ * sd + mu)  # noqa: E731
    val_groups = d["val_subjects"]  # subject id per val trial, for per-subject rho

    # Register model and hyperparameters.  The epoch is picked by the mean
    # PER-SUBJECT validation Spearman rho (not a pooled rho, and not MAE):
    # pooling would let between-subject differences dominate, and MAE is
    # dominated by the per-subject RT offset that LOSO can never know.
    trainer.uses(
        model,
        optimizer,
        criterion,
        regularizer,
        predict_func=lambda out: out.squeeze(-1),
        metric=Metric.maximize(
            # yt/yp are z scores; f=to_ms maps them back to ms (rho is
            # scale-invariant, but the inverse transform keeps intent clear).
            # Default arguments f=to_ms / g=val_groups bind THIS fold's mu/sd
            # and subject labels, avoiding late binding on the next fold.
            lambda yt, yp, f=to_ms, g=val_groups: grouped_spearman(yt, yp, f, g),
            name="spearman_subject_mean",
        ),
    )
    print(
        f"=== Fold test_sub={sub} | "
        f"train {len(d['train'].dataset)} | test {len(d['test'].dataset)} | val {len(d['val'].dataset)} ==="
    )
    result = trainer.train(EPOCHS, callbacks)
    print(f"Completed training for subject {sub}.")
    preds = np.asarray(result["test_preds"])
    targets = np.asarray(result["test_targets"])
    # targets/preds are z scores; map back to ms before any error metric
    true_rt = to_ms(np.ravel(targets))
    pred_rt = to_ms(np.ravel(preds))
    mae = float(mean_absolute_error(true_rt, pred_rt))
    rmse = float(np.sqrt(np.mean((true_rt - pred_rt) ** 2)))
    rho = float(spearmanr(true_rt, pred_rt).statistic)
    rho_lo, rho_hi = spearman_ci(true_rt, pred_rt)  # trial-level 95% bootstrap CI

    # Validation diagnostics with the selected (best) weights: the pooled rho
    # vs the per-subject mean actually used for epoch selection.
    val_preds, val_targets = trainer._predict(d["val"], trainer.predict_func)
    val_rho_pooled = val_spearman(val_targets, val_preds, to_ms)
    val_rho_grouped = grouped_spearman(val_targets, val_preds, to_ms, val_groups)

    # Constant baselines: fitted on the TRAINING split only (no test labels)
    train_z = d["train"].dataset.tensors[1].numpy().ravel()
    train_rt = to_ms(train_z)
    const_geo = float(to_ms(0.0))  # z=0 is the train mean -> geometric mean RT
    const_med = float(np.median(train_rt))  # median RT (minimises ms-space MAE)
    mae_geo = float(mean_absolute_error(true_rt, np.full_like(true_rt, const_geo)))
    mae_med = float(mean_absolute_error(true_rt, np.full_like(true_rt, const_med)))
    best_baseline = min(mae_geo, mae_med)
    # oracle: test subject's own mean RT (uses test labels; reference ceiling only)
    mae_oracle = float(
        mean_absolute_error(true_rt, np.full_like(true_rt, true_rt.mean()))
    )
    # Centered MAE: subtract each series' own mean, so the unknowable
    # per-subject offset disappears and only within-subject accuracy remains.
    # Its reference is the oracle above (predicting the subject's own mean),
    # whose MAE equals the subject's mean absolute deviation.
    true_centered = true_rt - true_rt.mean()
    pred_centered = pred_rt - pred_rt.mean()
    mae_centered = float(mean_absolute_error(true_centered, pred_centered))

    fold_results.append(
        {
            "subject": str(sub),
            "mu": mu,
            "sd": sd,
            "n_train": len(d["train"].dataset),
            "n_val": len(d["val"].dataset),
            "n_test": len(d["test"].dataset),
            "best_epoch": int(result["best_epoch"]),
            "val_metric": float(result["best_metric"]),
            "val_metric_name": "spearman_subject_mean",
            "val_rho_pooled": val_rho_pooled,
            "val_rho_subject_mean": val_rho_grouped,
            "test_mae_ms": mae,
            "test_rmse_ms": rmse,
            "test_mae_centered_ms": mae_centered,
            "spearman": rho,
            "spearman_ci95_low": rho_lo,
            "spearman_ci95_high": rho_hi,
            "baseline_geo_mae_ms": mae_geo,
            "baseline_median_mae_ms": mae_med,
            "baseline_oracle_mae_ms": mae_oracle,
            "delta_vs_baseline_ms": best_baseline - mae,
            "delta_centered_ms": mae_oracle - mae_centered,
        }
    )
    print(
        f"Test MAE: {mae:.2f} ms | RMSE: {rmse:.2f} ms | Spearman: {rho:.4f} "
        f"[{rho_lo:+.3f}, {rho_hi:+.3f}] "
        f"|| baseline geo={mae_geo:.2f} median={mae_med:.2f} oracle={mae_oracle:.2f} "
        f"| delta={best_baseline - mae:+.2f} ms "
        f"|| centered MAE={mae_centered:.2f} ms vs oracle {mae_oracle:.2f} "
        f"| delta_c={mae_oracle - mae_centered:+.2f} ms"
    )

# ---------------- summary + persistence ----------------
maes = np.array([r["test_mae_ms"] for r in fold_results])
rmses = np.array([r["test_rmse_ms"] for r in fold_results])
centered = np.array([r["test_mae_centered_ms"] for r in fold_results])
rhos = np.array([r["spearman"] for r in fold_results])
rho_lo = np.array([r["spearman_ci95_low"] for r in fold_results])
rho_hi = np.array([r["spearman_ci95_high"] for r in fold_results])
deltas = np.array([r["delta_vs_baseline_ms"] for r in fold_results])
deltas_c = np.array([r["delta_centered_ms"] for r in fold_results])
baselines = np.array(
    [min(r["baseline_geo_mae_ms"], r["baseline_median_mae_ms"]) for r in fold_results]
)
rho_ci = bootstrap_mean_ci(rhos)
delta_ci = bootstrap_mean_ci(deltas)
delta_c_ci = bootstrap_mean_ci(deltas_c)
summary = {
    "n_folds": len(fold_results),
    "test_mae_ms_mean": float(maes.mean()),
    "test_mae_ms_std": float(maes.std()),
    "test_rmse_ms_mean": float(rmses.mean()),
    "test_rmse_ms_std": float(rmses.std()),
    "test_mae_centered_ms_mean": float(centered.mean()),
    "test_mae_centered_ms_std": float(centered.std()),
    "spearman_mean": float(rhos.mean()),
    "spearman_std": float(rhos.std()),
    "spearman_mean_ci95": list(rho_ci),
    "baseline_mae_ms_mean": float(baselines.mean()),
    "baseline_mae_ms_std": float(baselines.std()),
    "delta_vs_baseline_ms_mean": float(deltas.mean()),
    "delta_vs_baseline_ms_std": float(deltas.std()),
    "delta_vs_baseline_ms_ci95": list(delta_ci),
    "delta_centered_ms_mean": float(deltas_c.mean()),
    "delta_centered_ms_std": float(deltas_c.std()),
    "delta_centered_ms_ci95": list(delta_c_ci),
    "folds_beating_baseline": int((deltas > 0).sum()),
    "folds_beating_oracle_centered": int((deltas_c > 0).sum()),
    "folds_rho_ci_excluding_zero": int(((rho_lo > 0) | (rho_hi < 0)).sum()),
}
print("\n================ LOSO Summary (log RT regression) ================")
print(f"folds: {summary['n_folds']}")
print(
    f"test MAE     : {summary['test_mae_ms_mean']:.2f} "
    f"+/- {summary['test_mae_ms_std']:.2f} ms"
)
print(
    f"constant base: {summary['baseline_mae_ms_mean']:.2f} "
    f"+/- {summary['baseline_mae_ms_std']:.2f} ms"
)
print(
    f"delta (base - model): {summary['delta_vs_baseline_ms_mean']:+.2f} "
    f"+/- {summary['delta_vs_baseline_ms_std']:.2f} ms | "
    f"folds beating baseline: {summary['folds_beating_baseline']}/{summary['n_folds']}"
)
print(
    f"test RMSE    : {summary['test_rmse_ms_mean']:.2f} "
    f"+/- {summary['test_rmse_ms_std']:.2f} ms"
)
print(
    f"centered MAE : {summary['test_mae_centered_ms_mean']:.2f} "
    f"+/- {summary['test_mae_centered_ms_std']:.2f} ms | "
    f"delta_c (oracle - model): {summary['delta_centered_ms_mean']:+.2f} "
    f"+/- {summary['delta_centered_ms_std']:.2f} ms | "
    f"folds beating oracle: {summary['folds_beating_oracle_centered']}/{summary['n_folds']}"
)
print(
    f"Spearman     : {summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f} "
    f"| 95% CI [{summary['spearman_mean_ci95'][0]:+.4f}, "
    f"{summary['spearman_mean_ci95'][1]:+.4f}] "
    f"| folds with CI excluding 0: "
    f"{summary['folds_rho_ci_excluding_zero']}/{summary['n_folds']}"
)
print(
    f"delta 95% CI : [{summary['delta_vs_baseline_ms_ci95'][0]:+.2f}, "
    f"{summary['delta_vs_baseline_ms_ci95'][1]:+.2f}] ms"
)
print(
    f"delta_c 95%CI: [{summary['delta_centered_ms_ci95'][0]:+.2f}, "
    f"{summary['delta_centered_ms_ci95'][1]:+.2f}] ms"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = RESULTS_DIR / f"strictTrainEEGNetReg_logrt_{stamp}.json"
out_path.write_text(
    json.dumps(
        {
            "config": {
                "dataset": str(DATASET),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "val_frac": VAL_FRAC,
                "seed": SEED,
                "criterion": type(criterion).__name__,
                "dropout": DROPOUT,
                "regularizer": "max-norm: depthwise<=1.0, sep_pointwise<=0.25",
                "epoch_selection_metric": (
                    "mean per-subject spearman on validation (higher is better)"
                ),
                "bootstrap": {
                    "per_fold_trial_resamples": N_BOOT_FOLD,
                    "mean_fold_resamples": N_BOOT_MEAN,
                    "alpha": 0.05,
                },
                "target": (
                    "log(rt_ms) z-scored per fold with TRAIN mean/std; "
                    "trials with rt <= 0 or NaN dropped"
                ),
            },
            "summary": summary,
            "folds": fold_results,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"saved -> {out_path}")
