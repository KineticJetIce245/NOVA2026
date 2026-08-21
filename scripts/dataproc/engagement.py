"""Baseline-normalised engagement index (AttentivU pipeline steps 8-12).

The trial engagement index E = Pbeta / (Palpha + Ptheta) is computed per
(trial, channel) from a Welch PSD of the 2 s pre-stimulus window.  It is then
z-scored -- in log space -- against the distribution of E measured over
eyes-open resting-state windows from the *same subject and the same session*.
That per-participant resting distribution plays the role of AttentivU's
calibration phase (paper Eq. 2), which we cannot reproduce literally because
COG-BCI has no explicit high-engagement calibration block.

Two invariants make the normalisation meaningful; both are enforced here:

1.  Rest and trials go through the *identical* spectral estimator.  E is a
    ratio of band powers and is therefore biased by window length, n_fft,
    taper and segment count.  Estimating rest over 60 s and trials over 2 s
    would put the two distributions on different scales and the z-score would
    be meaningless.  ``epoch`` cuts rest into windows of exactly
    ``WINDOW_SECONDS`` and ``welch`` is the single entry point for both.

2.  mu and sigma are estimated per (subject, session, channel).  Absolute band
    power depends on impedance, skull thickness and cap placement, so pooling
    across subjects leaves exactly the nuisance variance the baseline was
    supposed to remove.  ``variance_decomposition`` quantifies how bad that
    would be for this dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import mne.time_frequency

from nova2026.config import DATA_DIR, SAMPLE_RATE, SAMPLE_SIZE

PROTOTYPE_DIR = DATA_DIR / "COG-BCI" / "prototype_outputs"
PIPELINE = "AttUPipeline"  # class name build_pvt.py / build_rest.py stamp into the filename
BASELINE_CONDITION = "RS_Beg_EO"  # eyes-open, pre-task; build_rest.py DATASET must match
PROBE_CONDITION = "RS_Beg_EC"  # eyes-closed, used only for the sanity check

# Half-open bands: the upper edge belongs to the next band only.  Using closed
# intervals on both sides (>= fmin & <= fmax) double-counts 7 Hz in theta and
# alpha and 11 Hz in alpha and beta, which inflates the denominator of E.
THETA_BAND: tuple[float, float] = (4.0, 7.0)
ALPHA_BAND: tuple[float, float] = (7.0, 11.0)
BETA_BAND: tuple[float, float] = (11.0, 20.0)

# Derived from config, never restated: the baseline window must be the same
# length as the trial window build_pvt.py hands the network, or the band-power
# ratio is estimated on two different scales and the z-score is meaningless.
WINDOW_SECONDS = SAMPLE_SIZE / 1000.0
HOP_SECONDS = WINDOW_SECONDS / 2.0  # 50 % overlap
WINDOW_SAMPLES = int(round(WINDOW_SECONDS * SAMPLE_RATE))

# n_per_seg = 1 s gives Welch three averaged segments inside a 2 s window
# instead of the single noisy periodogram the current spectual.py computes.
# n_fft = 2 s zero-pads each segment back onto a 0.5 Hz grid, so the band
# integrals are not evaluated on only three theta bins.
_N_PER_SEG = min(int(1.0 * SAMPLE_RATE), WINDOW_SAMPLES)
_N_FFT = WINDOW_SAMPLES

WELCH_KWARGS = dict(
    sfreq=float(SAMPLE_RATE),
    fmin=THETA_BAND[0],
    fmax=BETA_BAND[1],
    n_fft=_N_FFT,
    n_per_seg=_N_PER_SEG,
    n_overlap=_N_PER_SEG // 2,
    window="hamming",
    average="mean",
    remove_dc=True,
    verbose=False,
)


# --------------------------------------------------------------------------
# spectral estimation -- one entry point for rest and trials
# --------------------------------------------------------------------------
def welch(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of ``(..., n_channels, n_times)`` windows.

    Returns ``(psd, freqs)`` with ``psd`` shaped ``(..., n_channels, n_freqs)``.
    """
    return mne.time_frequency.psd_array_welch(
        np.asarray(windows, dtype=np.float64), **WELCH_KWARGS
    )


def band_power(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integrate ``psd`` over a half-open frequency band."""
    fmin, fmax = band
    mask = (freqs >= fmin) & (freqs < fmax)
    if mask.sum() < 2:
        raise ValueError(f"Band {band} covers {mask.sum()} bins; increase n_fft.")
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def engagement_index(psd: np.ndarray, freqs: np.ndarray, log: bool = True) -> np.ndarray:
    """E = Pbeta / (Palpha + Ptheta), optionally in log space.

    Band power is close to log-normal and E is a ratio, so the raw index is
    strongly right-skewed: a plain z-score on it produces an asymmetric scale
    whose sigma is set by a handful of high outliers.  ``log=True`` returns
    ln E = ln Pbeta - ln(Palpha + Ptheta), which is near-symmetric and is the
    quantity that should be z-scored.  Scale factors cancel in the ratio, so
    the /8 in AttUPipeline does not affect E either way.
    """
    theta = band_power(psd, freqs, THETA_BAND)
    alpha = band_power(psd, freqs, ALPHA_BAND)
    beta = band_power(psd, freqs, BETA_BAND)
    denominator = alpha + theta
    tiny = np.finfo(np.float64).tiny
    if log:
        return np.log(np.maximum(beta, tiny)) - np.log(np.maximum(denominator, tiny))
    return beta / np.maximum(denominator, tiny)


# --------------------------------------------------------------------------
# baseline: window the continuous resting run the same way as a trial
# --------------------------------------------------------------------------
# A zero-phase Butterworth band-pass rings for up to 0.88 s either side of a
# recording edge at 128 Hz, so the first and last window of a continuous run are
# part filter transient.  Trials are unaffected (the first PVT stimulus is at
# ~10.9 s) but a 60 s rest run starts at sample zero.  Trimming 1 s from each
# end leaves 56 windows of 2 s at 50 % overlap, not the untrimmed 59.
EDGE_TRIM_SECONDS = 1.0


def epoch(
    continuous: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    trim_seconds: float = EDGE_TRIM_SECONDS,
) -> np.ndarray:
    """Cut ``(n_channels, n_times)`` into ``(n_windows, n_channels, n_samples)``."""
    n_samples = int(round(window_seconds * SAMPLE_RATE))
    hop = int(round(hop_seconds * SAMPLE_RATE))
    trim = int(round(trim_seconds * SAMPLE_RATE))

    if trim:
        continuous = continuous[..., trim:-trim] if 2 * trim < continuous.shape[-1] else continuous

    n_times = continuous.shape[-1]
    if n_times < n_samples:
        raise ValueError(f"Recording has {n_times} samples, need {n_samples}.")
    starts = range(0, n_times - n_samples + 1, hop)
    return np.stack([continuous[..., s : s + n_samples] for s in starts], axis=0)


@dataclass(frozen=True)
class Baseline:
    """Per-channel location and scale of ln E over one resting recording."""

    center: np.ndarray  # (n_channels,)
    scale: np.ndarray  # (n_channels,)
    composite_scale: float  # spread of the channel-averaged z over baseline windows
    n_windows: int
    robust: bool

    def z(self, values: np.ndarray) -> np.ndarray:
        """Per-channel z-score: ``(..., n_channels)`` in, same shape out."""
        return (values - self.center) / self.scale

    def composite(self, values: np.ndarray) -> np.ndarray:
        """Collapse channels to one score per trial, in baseline sigma units.

        The channel mean of 62 unit-spread z-scores is *not* itself unit
        spread.  Were the channels independent it would be 1/sqrt(62) = 0.13;
        measured on COG-BCI resting runs it is 0.69-1.00, because the
        engagement index is strongly correlated across the scalp (mean
        inter-channel r = 0.47, i.e. roughly 1-2 effective independent
        channels).  Dividing by the measured spread is what makes this
        comparable to a single-channel z.
        """
        return self.z(values).mean(axis=-1) / self.composite_scale


def baseline_from_rest(rest: np.ndarray, robust: bool = True) -> Baseline:
    """Fit a Baseline from one continuous ``(n_channels, n_times)`` rest run.

    ``robust=True`` uses median and 1.4826 * MAD.  Resting EEG contains blinks
    and movement; a plain std is inflated by a few artefact windows, which
    silently shrinks every downstream z-score.  Compare both -- a large gap
    means artefact rejection is doing real work.
    """
    windows = epoch(rest)
    psd, freqs = welch(windows)
    log_e = engagement_index(psd, freqs, log=True)  # (n_windows, n_channels)

    if robust:
        center = np.median(log_e, axis=0)
        scale = 1.4826 * np.median(np.abs(log_e - center), axis=0)
    else:
        center = log_e.mean(axis=0)
        scale = log_e.std(axis=0, ddof=1)

    floor = np.maximum(np.median(scale) * 1e-3, np.finfo(np.float64).eps)
    scale = np.maximum(scale, floor)

    # Spread of the channel-averaged z across the baseline windows themselves.
    composite = ((log_e - center) / scale).mean(axis=-1)
    if robust:
        spread = 1.4826 * np.median(np.abs(composite - np.median(composite)))
    else:
        spread = composite.std(ddof=1)

    return Baseline(center, scale, max(float(spread), 1e-6), len(windows), robust)


def fit_baselines(rest_checkpoint: Mapping, robust: bool = True) -> dict[tuple[str, str], Baseline]:
    """Fit one Baseline per (subject, session) from a build_rest.py checkpoint."""
    data = np.asarray(rest_checkpoint["data"])
    metadata = np.asarray(rest_checkpoint["metadata"], dtype=object)
    baselines: dict[tuple[str, str], Baseline] = {}
    for index, (subject, session) in enumerate(metadata[:, :2]):
        key = (str(subject), str(session))
        if key in baselines:
            raise ValueError(f"Duplicate resting recording for {key}.")
        baselines[key] = baseline_from_rest(data[index], robust=robust)
    return baselines


# --------------------------------------------------------------------------
# trials
# --------------------------------------------------------------------------
def trial_engagement(pvt_checkpoint: Mapping) -> np.ndarray:
    """ln E per trial and channel -> ``(n_trials, n_channels)``."""
    data = np.asarray(pvt_checkpoint["data"], dtype=np.float64)
    if data.ndim == 4:  # legacy PVT_data_window_1.pt is (n, 1, 62, 256)
        data = data[:, 0]
    psd, freqs = welch(data)
    return engagement_index(psd, freqs, log=True)


def normalize(
    pvt_checkpoint: Mapping,
    baselines: Mapping[tuple[str, str], Baseline],
) -> np.ndarray:
    """E_norm: trial ln E z-scored against its own subject-session baseline."""
    log_e = trial_engagement(pvt_checkpoint)
    metadata = np.asarray(pvt_checkpoint["metadata"], dtype=object)

    normalized = np.empty_like(log_e)
    for index, (subject, session) in enumerate(metadata[:, :2]):
        key = (str(subject), str(session))
        baseline = baselines.get(key)
        if baseline is None:
            raise KeyError(f"No resting baseline for {key}; rebuild RS_Beg_EO.")
        normalized[index] = baseline.z(log_e[index])
    return normalized


def normalize_composite(
    pvt_checkpoint: Mapping,
    baselines: Mapping[tuple[str, str], Baseline],
) -> np.ndarray:
    """One engagement score per trial: ``(n_trials,)``.

    Channels are collapsed *after* per-channel z-scoring, never before -- each
    site has its own resting beta/(alpha+theta) offset, so averaging raw ln E
    across channels first would mix incompatible baselines.
    """
    log_e = trial_engagement(pvt_checkpoint)
    metadata = np.asarray(pvt_checkpoint["metadata"], dtype=object)

    out = np.empty(len(log_e), dtype=np.float64)
    for index, (subject, session) in enumerate(metadata[:, :2]):
        key = (str(subject), str(session))
        baseline = baselines.get(key)
        if baseline is None:
            raise KeyError(f"No resting baseline for {key}; rebuild RS_Beg_EO.")
        out[index] = baseline.composite(log_e[index])
    return out


# --------------------------------------------------------------------------
# diagnostics -- run these before trusting any of the above
# --------------------------------------------------------------------------
def variance_decomposition(log_e: np.ndarray, metadata: np.ndarray) -> dict[str, float]:
    """Between- vs within-subject variance of raw ln E, averaged over channels.

    If ``ratio`` is well above 1 the subject identity dominates the index and a
    pooled (cross-subject) sigma is not an acceptable approximation.
    """
    subjects = np.asarray(metadata, dtype=object)[:, 0].astype(str)
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("Need at least two subjects to decompose variance.")
    means = np.stack([log_e[subjects == s].mean(axis=0) for s in unique])
    between = means.var(axis=0, ddof=1)
    within = np.stack([log_e[subjects == s].var(axis=0, ddof=1) for s in unique]).mean(axis=0)
    return {
        "between_subject": float(between.mean()),
        "within_subject": float(within.mean()),
        "ratio": float((between / within).mean()),
        "icc": float((between / (between + within)).mean()),
    }


def baseline_sanity_check(
    baselines: Mapping[tuple[str, str], Baseline],
    probe_checkpoint: Mapping,
) -> dict[str, float]:
    """z-score another resting condition against the RS_Beg_EO baseline.

    Two free checks on whether the index measures anything:
      * RS_Beg_EC should come out clearly *negative* (eyes-closed alpha inflates
        the denominator of E).  If it does not, the bands or the PSD are wrong.
      * RS_End_EO should sit below zero if E tracks time-on-task fatigue.
    """
    data = np.asarray(probe_checkpoint["data"], dtype=np.float64)
    metadata = np.asarray(probe_checkpoint["metadata"], dtype=object)
    scores: list[float] = []
    for index, (subject, session) in enumerate(metadata[:, :2]):
        baseline = baselines.get((str(subject), str(session)))
        if baseline is None:
            continue
        psd, freqs = welch(epoch(data[index]))
        scores.append(float(baseline.z(engagement_index(psd, freqs, log=True)).mean()))
    array = np.asarray(scores)
    if not len(array):
        raise ValueError("No probe recording matched a fitted baseline.")
    spread = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return {"mean_z": float(array.mean()), "sd_z": spread, "n": len(array)}


def lapse_contrast(normalized: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Mean E_norm for lapse vs non-lapse trials, averaged over channels."""
    labels = np.asarray(labels)
    lapse = normalized[labels == 1].mean(axis=0)
    normal = normalized[labels == 0].mean(axis=0)
    return {
        "lapse": float(lapse.mean()),
        "non_lapse": float(normal.mean()),
        "difference": float((lapse - normal).mean()),
    }


def main() -> None:
    # Names must track build_pvt.py / build_rest.py, which template them on
    # SAMPLE_SIZE and the pipeline class name.
    pvt_path = PROTOTYPE_DIR / f"PVT_data_{SAMPLE_SIZE}ms__{PIPELINE}.pt"
    rest_path = PROTOTYPE_DIR / f"{BASELINE_CONDITION}_data_{PIPELINE}.pt"
    for path in (pvt_path, rest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build it first; note build_rest.py has "
                f"DATASET hardcoded and must be set to {BASELINE_CONDITION!r} "
                "for the baseline run."
            )
    pvt = torch.load(pvt_path, weights_only=False)
    rest = torch.load(rest_path, weights_only=False)

    baselines = fit_baselines(rest, robust=True)
    print(f"baselines: {len(baselines)} recordings, "
          f"{next(iter(baselines.values())).n_windows} windows each")

    log_e = trial_engagement(pvt)
    print("raw ln E variance:", variance_decomposition(log_e, pvt["metadata"]))

    normalized = normalize(pvt, baselines)
    print("E_norm shape:", normalized.shape)
    print("lapse contrast:", lapse_contrast(normalized, pvt["labels"]))

    torch.save(
        {
            "engagement_norm": torch.from_numpy(normalized).float(),
            "log_engagement": torch.from_numpy(log_e).float(),
            "labels": pvt["labels"],
            "metadata": pvt["metadata"],
            "channel_names": pvt["channel_names"],
            "baseline": "RS_Beg_EO",
            "baseline_scope": "per subject, session and channel",
            "statistic": "median / 1.4826*MAD of ln E over 2 s windows, 50% overlap",
        },
        PROTOTYPE_DIR / "PVT_engagement_norm.pt",
    )


if __name__ == "__main__":
    main()
