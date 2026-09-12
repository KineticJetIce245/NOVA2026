"""Preprocessing pipeline for the post-stimulus visual-detection task.

This pipeline is written for *this* task only and shares no code with the
pre-stimulus attention pipeline in ``scripts/dataproc/``. Two of its defaults
are the opposite of what a slow-wave task would want, because this task reads a
short evoked response rather than a spectral index:

* the pass-band stops at 20 Hz, which keeps the 100-150 ms evoked positivity
  measured at +3.7 uV while attenuating the pre-motor negativity from -13 uV to
  -5 uV;
* the AttentivU-style per-channel ``center/scale/clip`` is available but off by
  default, because dividing by 8 uV shrinks that same positivity to about
  0.5 uV.

Operator order (all optional except the first band-pass):

    notch -> band-pass -> resample -> band-pass -> [center/scale/clip]

The same operator can be reached two ways, which matters because the dataset
never ships a montage: :meth:`VisualPipeline.run` takes a list of recordings and
preprocesses whatever channels each one has, and :func:`missing_channels`
reports which requested channels a recording does not carry (MNE restores only
the channels it had already seen, so callers that want a subset after filtering
must pick it before calling :meth:`run`).
"""

from __future__ import annotations

from .device import EEG_CHANNELS

#: Raw EEG sampling rate of the COG-BCI PVT recordings, in Hz.
RAW_SAMPLE_RATE = 500.0

IIR_PARAMS = {"order": 4, "ftype": "butter", "output": "sos"}


def center_scale_clip(values_v):
    """Per-channel center, scale by 8 uV and clip to +-4 sigma.

    Operates on the whole recording, so the location and scale never come from
    the epoch under test. Returns microvolts, as the operator is defined on
    microvolt-valued samples.
    """
    import numpy as np

    values_uv = values_v * 1e6
    return np.clip((values_uv - values_uv.mean()) / 8.0, -4.0, 4.0) / 1e6


class VisualPipeline:
    """Notch, band-pass, optional resample and optional AttentivU scaling."""

    def __init__(
        self,
        l_freq: float = 4.0,
        h_freq: float = 20.0,
        sample_rate: float = 128.0,
        notch_freq: float = 60.0,
        notch_widths: float = 10.0,
        normalize: bool = False,
        iir_params: dict | None = None,
        verbose: bool = False,
    ) -> None:
        if l_freq <= 0 or h_freq <= l_freq:
            raise ValueError(f"Invalid band ({l_freq}, {h_freq}) Hz.")
        if sample_rate <= 2 * h_freq:
            raise ValueError(
                f"Sample rate {sample_rate} Hz cannot carry a {h_freq} Hz "
                "low-pass; raise --sample-rate or lower --band."
            )
        if h_freq >= RAW_SAMPLE_RATE / 2.0:
            raise ValueError(
                f"High cutoff {h_freq} Hz is at or above the raw Nyquist "
                f"({RAW_SAMPLE_RATE / 2.0} Hz)."
            )
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.sample_rate = float(sample_rate)
        self.notch_freq = float(notch_freq)
        self.notch_widths = float(notch_widths)
        self.normalize = bool(normalize)
        self.iir_params = dict(iir_params or IIR_PARAMS)
        self.verbose = verbose

    def channels(self) -> list[str]:
        """Every channel this pipeline expects to see."""
        return list(EEG_CHANNELS)

    def run(self, recordings: list) -> None:
        """Preprocess every recording in ``recordings`` in place.

        Each element is an ``mne.io.BaseRaw``. MNE keeps only the channels it
        already had when a later ``pick`` happens, so callers that need a
        channel subset must pick it *before* :meth:`run`.
        """
        import mne  # local import: keeps this module importable without MNE
        import numpy as np

        for raw in recordings:
            if not isinstance(raw, mne.io.BaseRaw):
                raise TypeError(f"Expected an MNE Raw, got {type(raw)}")
            nyq = raw.info["sfreq"] / 2.0
            if self.h_freq >= nyq:
                raise ValueError(
                    f"High cutoff {self.h_freq} Hz is at or above Nyquist "
                    f"({nyq} Hz)."
                )
            # one stop-band per call: MNE refuses several with an IIR notch, and
            # a stop-band above the pass-band would do nothing anyway
            for harmonic in np.arange(
                self.notch_freq, self.h_freq, self.notch_freq
            ):
                if harmonic > self.l_freq:
                    raw.notch_filter(
                        freqs=float(harmonic),
                        notch_widths=self.notch_widths,
                        method="iir",
                        iir_params=self.iir_params,
                        verbose=self.verbose,
                    )
            band = dict(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method="iir",
                iir_params=self.iir_params,
                verbose=self.verbose,
            )
            raw.filter(**band)
            if not np.isclose(raw.info["sfreq"], self.sample_rate):
                raw.resample(self.sample_rate, verbose=self.verbose)
                raw.filter(**band)  # the antialiasing filter leaves transients
            if self.normalize:
                raw.apply_function(
                    center_scale_clip, channel_wise=True, verbose=self.verbose
                )


def missing_channels(channels, present) -> list[str]:
    """Requested channels that the recording does not have."""
    return [ch for ch in channels if ch not in set(present)]
