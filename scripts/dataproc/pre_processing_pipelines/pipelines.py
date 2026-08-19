import mne

from nova2026.config import(
    SAMPLE_RATE,
    SAMPLE_LENGTH,
    WINDOW_LENGTH,
    STEP_SIZE
)

def default_pipeline(raw: mne.io.Raw) -> None:
    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    raw.resample(SAMPLE_RATE)