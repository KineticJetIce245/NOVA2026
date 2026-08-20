import mne
import numpy as np

from nova2026.data.pipeline import Pipeline

THETA_BAND: tuple[int, int] = (4, 7)
ALPHA_BAND: tuple[int, int] = (7, 11)
BETA_BAND: tuple[int, int] = (11, 20)


class PSDPipeline(Pipeline[np.ndarray]):
    def __init__(self, sfreq: float) -> None:
        super().__init__()

        def input_validation(windows_uv) -> tuple[None, np.ndarray]:
            windows = np.asarray(windows_uv)
            if windows.ndim != 3:
                raise ValueError(
                    "trial_windows_uv must have shape (n_trials, n_channels, n_samples)."
                )
            if not windows.shape[0] or not windows.shape[1] or not windows.shape[2]:
                raise ValueError("trial_windows_uv cannot contain an empty dimension.")
            if sfreq <= 2.0 * BETA_BAND[1]:
                raise ValueError(
                    f"sample freq must be greater than {2.0 * BETA_BAND[1]} Hz for the beta band."
                )
            return None, windows

        # ---- Suggested Parameters by Claude ----
        #
        # Current problem: n_per_seg = n_samples (256), resulting in only a single
        # segment, so "multi-segment averaging denoising" is not achieved, and the
        # spectral estimate is noisy, especially for short windows.
        #
        # Suggestion: Properly reduce n_per_seg and increase overlap to allow Welch
        # to segment the data and average over multiple segments.
        def compute_psd(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_samples = windows.shape[-1]
            psd, frequencies = mne.time_frequency.psd_array_welch(
                windows,
                sfreq=sfreq,
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
            return frequencies, psd

        self.add_tube(input_validation)
        self.add_tube(compute_psd)


def _mean_band_power(
    psd: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
    *,
    include_high: bool = False,
) -> np.ndarray:
    if include_high:
        mask = (frequencies >= low) & (frequencies <= high)
    else:
        mask = (frequencies >= low) & (frequencies < high)

    if not np.any(mask):
        raise ValueError(
            f"PSD frequency grid contains no values in the {low}-{high} Hz band."
        )

    return psd[..., mask].mean(axis=-1)


def compute_trial_spectral_features(
    trial_windows_uv: np.ndarray,
    sfreq: float,
) -> dict[str, np.ndarray]:
    """
    Compute PSD and engagement features per trial and per channel.

    Parameters
    ----------
    trial_windows_uv
        EEG windows in microvolts with shape
        ``(n_trials, n_channels, n_samples)``.
    sfreq
        Sampling frequency in hertz.

    Returns
    -------
    dict
        ``psd`` has shape ``(n_trials, n_channels, n_frequencies)``;
        ``theta``, ``alpha``, ``beta``, and ``engagement`` each have shape
        ``(n_trials, n_channels)``. ``frequencies`` has shape
        ``(n_frequencies,)``.
    """

    psd_pipeline = PSDPipeline(sfreq)
    frequencies, psd = psd_pipeline.rundown(trial_windows_uv)

    assert isinstance(frequencies, np.ndarray)

    theta = _mean_band_power(psd, frequencies, *THETA_BAND)
    alpha = _mean_band_power(psd, frequencies, *ALPHA_BAND)
    beta = _mean_band_power(
        psd,
        frequencies,
        *BETA_BAND,
        include_high=True,
    )

    epsilon = np.finfo(psd.dtype).eps
    engagement = np.divide(
        beta,
        np.maximum(alpha + theta, epsilon),
    )

    return {
        "psd": psd,
        "frequencies": frequencies,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "engagement": engagement,
    }
