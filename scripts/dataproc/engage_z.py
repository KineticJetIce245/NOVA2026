"""Engagement z-scoring: the baseline-normalised engagement index.

Implements the z-scoring stage of ``documents/engagement_zscoring.pdf``
(Sec. 1.3-2.4): the Welch estimator, the engagement functional
ln E = ln P_beta - ln(P_alpha + P_theta), per-(subject, session, channel)
baseline location/scale from the resting windows, per-trial z and the
composite score.  This module is the single implementation of that stage
and is written from the design doc only -- it does not reuse
``engagement.py`` (an AI-written reference).

Inputs are the checkpoints built by ``cogbci_rest.py`` (RS_Beg_EO) and
``cogbci_pvt.py`` (PVT); their exact schema is documented in those files.
The windowing itself (edge trim + hop tiling) happens in ``cogbci_rest.py``;
this module consumes windowed arrays.

Math summary (notation follows the design doc)
----------------------------------------------
For fixed (s, e, c) with x_w = ln E of the resting windows w = 1..W:

    mu    = median_w x_w                          (Eq. 8)
    sigma = 1.4826 * median_w |x_w - median x|    (Eq. 9)
    guard: sigma <- max(sigma, 1e-3 * median_c sigma)   (Eq. 9)

Trials are z-scored against their own subject/session baseline:

    z_{i,c} = (ln E_{i,c} - mu_{s(i),e(i),c}) / sigma_{s(i),e(i),c}   (Eq. 10)

The channel mean of z is not unit-dispersion (inter-channel correlation is
high), so the composite divides by the resting spread of that mean:

    g_{s,e,w}   = mean_c z_rest                  (Eq. 11)
    sigma_G     = 1.4826 * MAD_w g               (Eq. 12)
    Z_bar_i     = mean_c z_{i,c} / sigma_G       (Eq. 13)

The 1.4826 constant is 1 / Phi^-1(0.75), the MAD-to-sigma consistency
factor under normality.
"""

from typing import NamedTuple

import numpy as np
from mne.time_frequency import psd_array_welch

from nova2026.config import SAMPLE_RATE

THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (7.0, 11.0)
BETA_BAND = (11.0, 20.0)

MAD_SIGMA = 1.4826  # 1 / Phi^-1(0.75): MAD-to-sigma consistency factor
SIGMA_FLOOR_RATIO = 1e-3  # degenerate-channel guard (Eq. 9)


def welch_psd(windows: np.ndarray, sfreq: float = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per the design doc (Sec. 1.3).

    Hamming taper, segment length L = fs (1 s), 50 % overlap, zero-padded to
    the window length (n_fft = T), mean averaging over the K = 3 segments,
    DC removed, evaluated on [4, 20] Hz at df = fs / n_fft = 0.5 Hz.

    ``windows`` has shape ``(..., n_channels, T)``; the returned PSD has
    shape ``(..., n_channels, 33)``.
    """
    n_fft = windows.shape[-1]
    return psd_array_welch(
        windows,
        sfreq=sfreq,
        fmin=THETA_BAND[0],
        fmax=BETA_BAND[1],
        n_fft=n_fft,
        n_per_seg=int(sfreq),  # L = 1 s segments
        n_overlap=int(sfreq) // 2,  # 50 % overlap
        window="hamming",
        average="mean",  # K = 1 + (T - L) / (L - overlap) segments
        remove_dc=True,
        verbose=False,
    )


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Trapezoidal band power over the half-open interval [fmin, fmax) (Eq. 6).

    The 0.5 Hz grid puts 7 Hz and 11 Hz exactly on grid points; a closed
    mask would count those bins in two bands at once (theta inflated by
    ~21 % on a realistic 1/f-plus-alpha spectrum).
    """
    mask = (freqs >= fmin) & (freqs < fmax)
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def engagement_index(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """ln E = ln P_beta - ln(P_alpha + P_theta) (Eq. 7).

    Band power is close to log-normal, so ln E is near-symmetric and admits
    a location-scale description; the raw ratio E does not.  Scale factors
    cancel in the ratio (the /8 of the centre-scale-clip has no effect).
    """
    p_theta = band_power(psd, freqs, *THETA_BAND)
    p_alpha = band_power(psd, freqs, *ALPHA_BAND)
    p_beta = band_power(psd, freqs, *BETA_BAND)
    tiny = np.finfo(np.float64).tiny
    return np.log(np.maximum(p_beta, tiny)) - np.log(np.maximum(p_alpha + p_theta, tiny))


class Baseline(NamedTuple):
    """Per-channel location and scale of ln E over one resting recording.

    Attributes
    ----------
    center: (n_channels,) median of ln E over the resting windows (Eq. 8).
    scale: (n_channels,) 1.4826 * MAD with the degenerate-channel floor
        (Eq. 9 + guard).
    sigma_G: float, resting spread of the channel-averaged z (Eq. 12).
    n_windows: int, number of resting windows the baseline was fitted on.
    """

    center: np.ndarray
    scale: np.ndarray
    sigma_G: float
    n_windows: int

    def z(self, log_e: np.ndarray) -> np.ndarray:
        """Per-channel z-score (Eq. 10); ``(..., n_channels)`` in, same shape out."""
        return (log_e - self.center) / self.scale

    def composite(self, log_e: np.ndarray) -> np.ndarray:
        """Collapse channels to one score per trial, in baseline sigma units.

        The channel mean of unit-spread z-scores is not itself unit-spread
        (inter-channel correlation is high, so the effective dimensionality
        is ~1-2, not 62); dividing by ``sigma_G`` makes the composite
        comparable to a single-channel z (Eq. 13).
        """
        return self.z(log_e).mean(axis=-1) / self.sigma_G


def baseline_from_rest(windows: np.ndarray) -> Baseline:
    """Fit location/scale from one recording's resting windows (Eq. 8-12).

    ``windows`` has shape ``(W, n_channels, T)`` in microvolts and must
    already be trimmed and tiled (see ``cogbci_rest.py``).
    """
    log_e = engagement_index(*welch_psd(np.asarray(windows, dtype=np.float64)))
    center = np.median(log_e, axis=0)  # Eq. 8
    scale = MAD_SIGMA * np.median(np.abs(log_e - center), axis=0)  # Eq. 9
    floor = np.maximum(np.median(scale) * SIGMA_FLOOR_RATIO, np.finfo(np.float64).eps)
    scale = np.maximum(scale, floor)  # Eq. 9 guard

    composite = ((log_e - center) / scale).mean(axis=-1)  # Eq. 11 (g_w)
    sigma_g = MAD_SIGMA * np.median(np.abs(composite - np.median(composite)))  # Eq. 12
    return Baseline(center, scale, max(float(sigma_g), 1e-6), log_e.shape[0])


def fit_baselines(rest_checkpoint: dict) -> dict[tuple[str, str], Baseline]:
    """One Baseline per (subject, session) from the RS_Beg_EO checkpoint.

    ``rest_checkpoint`` is the dict produced by ``cogbci_rest.py``: ``data``
    is a list of R windowed arrays ``(W_r, 62, 256)`` and ``metadata`` is an
    ``(R, 2)`` object array ``[subject, session]`` whose row i corresponds to
    ``data[i]``.
    """
    data = rest_checkpoint["data"]
    metadata = np.asarray(rest_checkpoint["metadata"], dtype=object)
    if len(data) != len(metadata):
        raise ValueError(
            f"Rest checkpoint mismatch: {len(data)} recordings vs "
            f"{len(metadata)} metadata rows."
        )
    channel_names = list(rest_checkpoint["channel_names"])
    n_samples = int(rest_checkpoint["window_length_ms"] / 1000 * rest_checkpoint["sample_rate_hz"])

    baselines: dict[tuple[str, str], Baseline] = {}
    for row, windows in zip(metadata, data):
        key = (str(row[0]), str(row[1]))
        if key in baselines:
            raise ValueError(f"Duplicate resting recording for {key}.")
        windows = np.asarray(windows)
        if windows.ndim != 3 or windows.shape[1:] != (len(channel_names), n_samples):
            raise ValueError(
                f"Rest recording {key}: expected ({len(channel_names)}, {n_samples}) "
                f"windows, got shape {windows.shape}."
            )
        baselines[key] = baseline_from_rest(windows)
    return baselines


def trial_log_engagement(pvt_checkpoint: dict) -> np.ndarray:
    """ln E per trial and channel: ``(N, 62)`` (Welch + Eq. 7)."""
    data = np.asarray(pvt_checkpoint["data"], dtype=np.float64)
    channel_names = list(pvt_checkpoint["channel_names"])
    n_samples = int(pvt_checkpoint["window_length_ms"] / 1000 * pvt_checkpoint["sample_rate_hz"])
    if data.ndim != 3 or data.shape[1:] != (len(channel_names), n_samples):
        raise ValueError(f"PVT data: expected (N, 62, {n_samples}), got {data.shape}.")
    return engagement_index(*welch_psd(data))


def normalize(pvt_checkpoint: dict, baselines: dict[tuple[str, str], Baseline]) -> np.ndarray:
    """Per-channel z of every trial against its own (s, e) baseline (Eq. 10).

    Returns ``(N, 62)``.  Raises KeyError when a trial's (subject, session)
    has no fitted baseline (i.e. the RS_Beg_EO build is missing a session).
    """
    log_e = trial_log_engagement(pvt_checkpoint)
    metadata = np.asarray(pvt_checkpoint["metadata"], dtype=object)
    if len(log_e) != len(metadata):
        raise ValueError(
            f"PVT checkpoint mismatch: {len(log_e)} trials vs {len(metadata)} metadata rows."
        )
    out = np.empty_like(log_e)
    for i, (subject, session) in enumerate(metadata[:, :2]):
        key = (str(subject), str(session))
        baseline = baselines.get(key)
        if baseline is None:
            raise KeyError(f"No resting baseline for {key}; rebuild RS_Beg_EO.")
        out[i] = baseline.z(log_e[i])
    return out


def normalize_composite(
    pvt_checkpoint: dict, baselines: dict[tuple[str, str], Baseline]
) -> np.ndarray:
    """One engagement score per trial: Z_bar (Eq. 13), shape ``(N,)``.

    Channels are collapsed *after* per-channel z-scoring, never before --
    each site has its own resting beta/(alpha+theta) offset, so averaging
    raw ln E across channels first would mix incommensurable offsets.
    """
    log_e = trial_log_engagement(pvt_checkpoint)
    metadata = np.asarray(pvt_checkpoint["metadata"], dtype=object)
    out = np.empty(len(log_e), dtype=np.float64)
    for i, (subject, session) in enumerate(metadata[:, :2]):
        key = (str(subject), str(session))
        baseline = baselines.get(key)
        if baseline is None:
            raise KeyError(f"No resting baseline for {key}; rebuild RS_Beg_EO.")
        out[i] = baseline.composite(log_e[i])
    return out
