"""Continuous EEG preprocessing pipelines used by the prototype builder."""

import numpy as np
from mne.io import BaseRaw

from nova2026.config import SAMPLE_RATE
from nova2026.data.pipeline import Pipeline

IIR_PARAMS = {
    "order": 4,
    "ftype": "butter",
}


class AttUPipeline(Pipeline):
    # Apply the offline AttentivU-inspired pipeline to ``raw`` in place.
    def __init__(
        self,
        method="iir",
        iir_params=None,
        picks="eeg",
        phase="zero",
        freqs=60.0,
        l_freq=4.0,
        h_freq=20.0,
        notch_widths=10.0,
        sample_rate=SAMPLE_RATE,
        verbose=False,
    ):
        super().__init__()

        if iir_params is None:
            iir_params = dict(IIR_PARAMS)

        def notch_filter_raw(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.notch_filter(
                freqs=freqs,
                notch_widths=notch_widths,
                method=method,
                iir_params=iir_params,
                picks=picks,
                phase=phase,
                verbose=verbose,
            )
            return None, raw

        def filter_raw(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                method=method,
                iir_params=iir_params,
                picks=picks,
                phase=phase,
                verbose=verbose,
            )
            return None, raw

        def resample_raw(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.resample(sample_rate, verbose=verbose)
            return None, raw

        def center_scale_clip_channel(values_v: np.ndarray) -> np.ndarray:
            # Apply AttentivU normalization to one channel while preserving SI units.
            # MNE stores EEG values in volts. AttentivU's scale factor and clipping range
            # operate on microvolt-valued samples, so this function converts to
            # microvolts, normalizes, and converts the result back to volts.
            values_uv = values_v * 1e6
            normalized_uv = (values_uv - values_uv.mean()) / 8.0
            return np.clip(normalized_uv, -4.0, 4.0) / 1e6

        def center_scale_clip_raw(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.apply_function(
                center_scale_clip_channel,
                picks=picks,
                channel_wise=True,
                verbose=verbose,
            )
            return None, raw

        self.add_tube(notch_filter_raw)
        self.add_tube(filter_raw)
        self.add_tube(resample_raw)
        self.add_tube(filter_raw)
        self.add_tube(center_scale_clip_raw)
