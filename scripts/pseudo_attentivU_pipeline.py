eeg = notch_filter(
    eeg,
    center=60,
    bandwidth=10,
    filter_type="butterworth_iir",
)

eeg = bandpass_filter(
    eeg,
    low=4,
    high=20,
    filter_type="butterworth_iir",
)

eeg = resample(eeg, target_sfreq=TARGET_SAMPLE_RATE)

eeg = bandpass_filter(
    eeg,
    low=4,
    high=20,
    filter_type="butterworth_iir",
)

eeg = mean_center(eeg)
eeg = scale(eeg, divisor=8)
eeg = clip(eeg, minimum=-4, maximum=4)

windows = sliding_windows(eeg, duration=5)

psd = compute_psd(windows)

theta = average_band_power(psd, 4, 7)
alpha = average_band_power(psd, 7, 11)
beta = average_band_power(psd, 11, 20)

engagement = beta / (alpha + theta)
engagement = ewma(engagement)

normalized = (
    (engagement - calibration_min)
    / (calibration_max - calibration_min)
    * 100
)