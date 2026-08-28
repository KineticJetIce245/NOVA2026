"""Continuous EEG preprocessing pipelines used by the prototype builder."""

import numpy as np
from mne.io import BaseRaw

from nova2026.config import SAMPLE_RATE, SAMPLE_SIZE
from nova2026.data.pipeline import Pipeline

IIR_PARAMS = {
    "order": 4,
    "ftype": "butter",
}

EEG_CHANNELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "FCz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
]


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


class WinPipe(Pipeline):
    def __init__(
        self,
        trim: tuple[int, int],
        overlap_ratio: float,
        sample_size: int = SAMPLE_SIZE,
        scale: float = 0.0,
    ) -> None:
        super().__init__()

        def get_windowed_samples(raw: BaseRaw) -> tuple[list[np.ndarray], BaseRaw]:
            """Returns list of starting points for each window"""
            if raw is None:
                raise ValueError("BaseRaw is None.")

            increment = round(sample_size * (1 - overlap_ratio))

            if increment < 1:
                raise ValueError(f"Nothing to increment: increment = {increment}")

            if trim[0] < 0:
                raise ValueError(
                    f"start trim is a negative number: trim[0] = {trim[0]}"
                )

            if trim[1] < 0:
                raise ValueError(f"end trim is a negative number: trim[1] = {trim[1]}")

            if trim[0] + trim[1] >= raw.n_times:
                raise ValueError(
                    f"trimming more than the length of the recording: {trim[0] + trim[1]} >= {raw.n_times}"
                )

            windows = []
            end = raw.n_times - trim[1]
            for i in range(trim[0], end - sample_size + 1, increment):
                samples = raw.get_data(
                    picks=EEG_CHANNELS, start=i, stop=i + sample_size
                )
                if not isinstance(samples, np.ndarray):
                    raise TypeError(
                        f"samples is not of type np.ndarray: type(samples) = {type(samples)}"
                    )
                if scale != 0.0:
                    samples *= scale
                windows.append(samples)
            return windows, raw

        self.add_tube(get_windowed_samples)
