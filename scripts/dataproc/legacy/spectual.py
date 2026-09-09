"""Spectral estimates and the raw engagement functional (pre-z-scoring).

Welch parameters follow the design doc (``documents/engagement_zscoring.pdf``,
Sec. 1.3): Hamming taper, 1 s segments (L = fs), 50 % overlap, zero-padded to
the window length (n_fft = T), mean averaging over the K = 3 segments, DC
removed, evaluated on [4, 20] Hz at df = fs / n_fft = 0.5 Hz.

``band_power`` integrates over the *half-open* interval [fmin, fmax)
(Eq. 6).  Half-open matters: the 0.5 Hz grid puts 7 Hz and 11 Hz exactly on
grid points, so closed intervals would count 7 Hz in both theta and alpha
and 11 Hz in both alpha and beta, inflating theta by ~21 % on a realistic
1/f-plus-alpha spectrum.

``compute_engagement_metrics`` returns the raw index E = P_beta /
(P_alpha + P_theta); the log form ln E (Eq. 7) is taken by the caller (see
``script.py`` / ``engage_z.py``).
"""

from pathlib import Path

import mne
import numpy as np
import torch

from nova2026.config import DATA_DIR, SAMPLE_RATE

THETA_BAND: tuple[int, int] = (4, 7)
ALPHA_BAND: tuple[int, int] = (7, 11)
BETA_BAND: tuple[int, int] = (11, 20)


def compute_and_save_psd(data_set_path: Path, output_set: str):
    checkpoint = torch.load(
        data_set_path,
        weights_only=False,
    )
    data = checkpoint["data"]
    n_samples = data.shape[-1]

    psd, freq = mne.time_frequency.psd_array_welch(
        data,
        sfreq=checkpoint["sample_rate_hz"],
        fmin=THETA_BAND[0],
        fmax=BETA_BAND[1],
        n_fft=n_samples,
        n_per_seg=SAMPLE_RATE,  # Taking 1s as the length of each segment
        n_overlap=SAMPLE_RATE // 2,  # 50% overlap
        window="hamming",
        average="mean",
        remove_dc=True,
        verbose=False,
    )

    torch.save(
        {"psd": psd, "freq": freq},
        DATA_DIR / f"COG-BCI/outputs/{output_set}",
    )


def band_power(psd, frequencies, fmin, fmax):
    """Trapezoidal band power over the half-open interval [fmin, fmax).

    The Welch grid is df = 0.5 Hz, so 7 Hz and 11 Hz lie exactly on grid
    points; a closed mask would assign them to two bands at once and bias
    the band powers (see the module docstring).
    """
    freq_mask = (frequencies >= fmin) & (frequencies < fmax)
    power = np.trapezoid(psd[..., freq_mask], frequencies[freq_mask], axis=-1)
    return power


def compute_engagement_metrics(data_set: str):
    ckpt = torch.load(DATA_DIR / f"COG-BCI/outputs/{data_set}", weights_only=False)
    psd = ckpt["psd"]
    freq = ckpt["freq"]

    theta_power = band_power(psd, freq, THETA_BAND[0], THETA_BAND[1])
    alpha_power = band_power(psd, freq, ALPHA_BAND[0], ALPHA_BAND[1])
    beta_power = band_power(psd, freq, BETA_BAND[0], BETA_BAND[1])

    epsilon = np.finfo(psd.dtype).eps
    return np.divide(
        beta_power,
        np.maximum(alpha_power + theta_power, epsilon),
    )
