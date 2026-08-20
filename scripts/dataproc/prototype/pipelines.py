"""Continuous EEG preprocessing pipelines used by the prototype builder."""

import numpy as np
from mne.io import BaseRaw

from nova2026.config import SAMPLE_RATE
from nova2026.data.pipeline import Pipeline

IIR_PARAMS = {
    "order": 4,
    "ftype": "butter",
}


class DefaultPipe(Pipeline[BaseRaw]):
    """Apply the existing default filter and resample ``raw`` in place."""

    def __init__(
        self,
        l_freq=0.5,
        h_freq=45.0,
        picks="eeg",
        sample_rate=SAMPLE_RATE,
        verbose=False,
    ):
        super().__init__()

        def filter_raw_eeg(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                fir_design="firwin",
                verbose=verbose,
            )
            return (None, raw)

        def resample_raw_eeg(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.resample(sample_rate, verbose=verbose)
            return (None, raw)

        self.add_tube(filter_raw_eeg)
        self.add_tube(resample_raw_eeg)


def center_scale_clip_channel(values_v: np.ndarray) -> np.ndarray:
    """Apply AttentivU normalization to one channel while preserving SI units.

    MNE stores EEG values in volts. AttentivU's scale factor and clipping range
    operate on microvolt-valued samples, so this function converts to
    microvolts, normalizes, and converts the result back to volts.
    """
    values_uv = values_v * 1e6
    normalized_uv = (values_uv - values_uv.mean()) / 8.0
    return np.clip(normalized_uv, -4.0, 4.0) / 1e6


def attentivu_pipeline(raw: BaseRaw) -> None:
    """Apply the offline AttentivU-inspired pipeline to ``raw`` in place."""
    raw.load_data()
    raw.notch_filter(
        freqs=60.0,
        notch_widths=10.0,
        method="iir",
        iir_params=dict(IIR_PARAMS),
        picks="eeg",
        phase="zero",
        verbose=False,
    )
    raw.filter(
        l_freq=4.0,
        h_freq=20.0,
        method="iir",
        iir_params=dict(IIR_PARAMS),
        picks="eeg",
        phase="zero",
        verbose=False,
    )
    raw.resample(SAMPLE_RATE, verbose=False)
    raw.filter(
        l_freq=4.0,
        h_freq=20.0,
        method="iir",
        iir_params=dict(IIR_PARAMS),
        picks="eeg",
        phase="zero",
        verbose=False,
    )
    raw.apply_function(
        center_scale_clip_channel,
        picks="eeg",
        channel_wise=True,
        verbose=False,
    )


PIPELINES: dict[str, Pipeline] = {
    "default": default_pipeline,
    "attentivu": attentivu_pipeline,
}


def apply_pipeline(name: str, raw: BaseRaw) -> None:
    """Apply a named preprocessing pipeline to ``raw`` in place."""
    try:
        pipeline = PIPELINES[name]
    except KeyError as error:
        available = ", ".join(sorted(PIPELINES))
        raise ValueError(
            f"Unknown preprocessing pipeline {name!r}. Available: {available}."
        ) from error

    pipeline(raw)
