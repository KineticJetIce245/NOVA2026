import mne

from nova2026.config import SAMPLE_RATE


def default_pipeline(raw: mne.io.eeglab.eeglab.RawEEGLAB | mne.io.Raw) -> None:
    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    raw.resample(SAMPLE_RATE)


def attentivU_pipeline(raw: mne.io.Raw) -> None:
    pass
