import mne
import numpy as np
import torch

from nova2026.config import DATA_DIR, SAMPLE_RATE

DATASET = DATA_DIR / "COG-BCI/prototype_outputs/PVT_data_2000ms__AttUPipeline.pt"

THETA_BAND: tuple[int, int] = (4, 7)
ALPHA_BAND: tuple[int, int] = (7, 11)
BETA_BAND: tuple[int, int] = (11, 20)


def compute_and_save_psd(data_set: str, output_set: str):
    checkpoint = torch.load(
        DATA_DIR / f"COG-BCI/outputs/{data_set}",
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
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
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
