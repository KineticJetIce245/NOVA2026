import numpy as np
from mne.time_frequency import psd_array_welch

from nova2026.config import SAMPLE_RATE

THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (7.0, 11.0)
BETA_BAND = (11.0, 20.0)

MAD_SIGMA = 1.4826  # 1 / Phi^-1(0.75): MAD-to-sigma consistency factor


def _welch_psd(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_fft = data.shape[-1]
    return psd_array_welch(
        data,
        sfreq=SAMPLE_RATE,
        fmin=int(THETA_BAND[0]),
        fmax=int(BETA_BAND[1]),
        n_fft=n_fft,
        n_per_seg=SAMPLE_RATE,  # L = 1 s segments
        n_overlap=SAMPLE_RATE // 2,  # 50 % overlap
        window="hamming",
        average="mean",  # K = 1 + (T - L) / (L - overlap) segments
        remove_dc=True,
        verbose=False,
    )


def _band_power(
    psd: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])
    theta_band = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    maks = (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])
    alpha_band = np.trapezoid(psd[..., maks], freqs[maks], axis=-1)
    mask = (freqs >= BETA_BAND[0]) & (freqs < BETA_BAND[1])
    beta_band = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return theta_band, alpha_band, beta_band


def log_engagement(data: np.ndarray) -> np.ndarray:
    # data is np.ndarray(n trials, n channels, n samples)
    psds, freqs = _welch_psd(data)
    # psds and freqs is np.ndarray(n trials, n channels, n freqs)
    theta_band, alpha_band, beta_band = _band_power(psds, freqs)
    # bands are np.ndarray(n trials, n channels)
    # TODO: Zero Division Check
    log_engs = np.log(beta_band / (theta_band + alpha_band))
    # log_engs are np.ndarray(n trials = 75 sessions * (n windows / sessions), n channels)
    return log_engs


def compute_baseline(data: np.ndarray, meta: list[tuple]):
    log_engs = log_engagement(data)

    sessions = set(meta)  # all the unique tuple of (sub, ses) in meta
    meds = []
    sigs = []
    sigGs = []
    z_avgs = []
    result_meta = []
    for session in sessions:
        result_meta.append(session)
        windows = log_engs[[tag == session for tag in meta], :]
        # windows are np.ndarray(n windows (for that session), n channels)
        med = np.median(windows, axis=0)
        # med is np.ndarray(n channels) (median for a session)

        sig = MAD_SIGMA * np.median(np.abs(windows - med), axis=0)
        # sig is np.ndarray(n channels) (median absolute deviation for a session)
        # TODO: ZERO Divison Check
        z_avg = np.mean((windows - med) / sig, axis=1)  # channel aggregation
        # (windows - med) / sig is np.ndarray(n windows, n channels)
        # taking the mean of channel here: axis=1
        # z_avg is np.ndarray(n windows)
        # TODO: Check if sig_G is out of bound
        sigG = MAD_SIGMA * np.median(np.abs(z_avg - np.median(z_avg)), axis=0)
        # sigG is scalar
        meds.append(med)  # (n sessions, n channels)
        sigs.append(sig)  # (n sessions, n channels)
        z_avgs.append(z_avg)  # (n sessions, n windows)
        sigGs.append(sigG)  # (n sessions)

    return result_meta, meds, sigs, z_avgs, sigGs
