import mne
import numpy as np
import torch

from nova2026.config import DATA_DIR

DATASET = DATA_DIR / "COG-BCI/prototype_outputs/PVT_data_2000ms__AttUPipeline.pt"

THETA_BAND: tuple[int, int] = (4, 7)
ALPHA_BAND: tuple[int, int] = (7, 11)
BETA_BAND: tuple[int, int] = (11, 20)


def load_data(path):
    checkpoint = torch.load(path, weights_only=False)
    data = checkpoint["data"]
    labels = checkpoint["labels"]
    meta = checkpoint["metadata"]

    subjects = meta[:, 0]
    rt = meta[:, 2].astype(float)
    return checkpoint, data, labels, subjects, rt


def compute_and_save_psd(path):
    checkpoint = torch.load(
        DATA_DIR / f"COG-BCI/prototype_outputs/{path}_data_AttUPipeline.pt",
        weights_only=False,
    )
    data = checkpoint["data"]
    n_samples = data.shape[-1]

    # ---- Suggested Parameters by Claude ----
    #
    # Current problem: n_per_seg = n_samples (256), resulting in only a single
    # segment, so "multi-segment averaging denoising" is not achieved, and the
    # spectral estimate is noisy, especially for short windows.
    #
    # Suggestion: Properly reduce n_per_seg and increase overlap to allow Welch
    # to segment the data and average over multiple segments.
    psd, freq = mne.time_frequency.psd_array_welch(
        data,
        sfreq=checkpoint["sample_rate_hz"],
        fmin=THETA_BAND[0],
        fmax=BETA_BAND[1],
        n_fft=n_samples,
        n_per_seg=n_samples,  # Taking the whole legnth of the window for the FFT
        n_overlap=0,
        window="hamming",
        average="mean",
        remove_dc=True,
        verbose=False,
    )

    torch.save(
        {"psd": psd, "freq": freq},
        DATA_DIR / f"COG-BCI/prototype_outputs/{path}_psd.pt",
    )


def band_power(psd, frequencies, fmin, fmax):
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
    power = np.trapezoid(psd[..., freq_mask], frequencies[freq_mask], axis=-1)
    return power


def compute_engagement_metrics():
    ckpt = torch.load(
        DATA_DIR / "COG-BCI/prototype_outputs/PVT_psd.pt", weights_only=False
    )
    psd = ckpt["psd"]
    freq = ckpt["freq"]

    theta_power = band_power(psd, freq, THETA_BAND[0], THETA_BAND[1])
    alpha_power = band_power(psd, freq, ALPHA_BAND[0], ALPHA_BAND[1])
    beta_power = band_power(psd, freq, BETA_BAND[0], BETA_BAND[1])

    epsilon = np.finfo(psd.dtype).eps
    engagement = np.divide(
        beta_power,
        np.maximum(alpha_power + theta_power, epsilon),
    )
