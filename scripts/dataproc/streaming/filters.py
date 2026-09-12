"""Design filter coefficients independently of streaming state."""

from abc import ABC
from typing import Any, cast

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos

EXPECTED_NB_CHANNELS = 64
BANDPASS_ORDER = 3
BANDPASS_LOW_CUTOFF = 1.0
BANDPASS_HIGH_CUTOFF = 45.0

NOTCH_FREQUENCY = 60.0
NOTCH_Q = 30.0  # 2 Hz -3 dB bandwidth around 60 Hz
SUPPORTED_FREQ_MODES = ("sfreq", "pipeline")


class FilterListMismatchError(RuntimeError):
    """Raised when the filter and parameter lists have different lengths."""


class FilterNotFoundError(RuntimeError):
    """Raised when an unsupported filter is requested."""


class FrequencyModeError(RuntimeError):
    """Raised when an unsupported frequency mode is requested."""


class FilterBuilder(ABC):
    """Describe the sampling rates available to coefficient builders."""

    sfreq: float
    pipeline_freq: float
    supported_freq_modes: tuple[str, ...]


class SOSFilterBuilder(FilterBuilder):
    """Design ordered SOS filters for streaming or pipeline data.

    Args:
        sfreq: Sampling frequency of the EEG stream.
        pipeline_freq: Sampling frequency of the resampled pipeline data.
    """

    def __init__(
        self,
        sfreq: float,
        pipeline_freq: float,
        supported_freq_modes: tuple[str, ...] = SUPPORTED_FREQ_MODES,
    ) -> None:
        """Initialize the filter builder."""

        self.sfreq = sfreq
        self.pipeline_freq = pipeline_freq
        self.supported_freq_modes = supported_freq_modes

    def _build_notch_filter(
        self,
        sfreq: float,
        filter_parameters: tuple[float, ...],
    ) -> np.ndarray:
        """Design a notch filter in SOS form.

        Args:
            sfreq: Sampling frequency used to design the filter.
            filter_parameters: Quality factor and notch frequency.

        Returns:
            The notch filter coefficients in SOS form.
        """

        if len(filter_parameters) != 2:
            raise ValueError("Notch filter requires two parameters.")

        quality_factor = filter_parameters[0]
        notch_frequency = filter_parameters[1]

        if not np.isfinite(sfreq) or sfreq <= 2 * notch_frequency:
            raise ValueError(
                f"sfreq must exceed {2 * notch_frequency} Hz, got {sfreq}."
            )

        notch_b, notch_a = iirnotch(
            w0=notch_frequency,
            Q=quality_factor,
            fs=sfreq,
        )
        notch_sos = tf2sos(notch_b, notch_a)

        return notch_sos

    def _build_bandpass_filter(
        self,
        sfreq: float,
        filter_parameters: tuple[float, float, int],
    ) -> np.ndarray:
        """Design a Butterworth band-pass filter in SOS form.

        Args:
            sfreq: Sampling frequency used to design the filter.
            filter_parameters: Low cutoff, high cutoff, and filter order.

        Returns:
            The band-pass filter coefficients in SOS form.
        """

        if len(filter_parameters) != 3:
            raise ValueError("Band-pass filter requires three parameters.")

        low_cutoff = filter_parameters[0]
        high_cutoff = filter_parameters[1]
        bandpass_order = filter_parameters[2]

        if not np.isfinite(sfreq) or sfreq <= 2 * high_cutoff:
            raise ValueError(f"sfreq must exceed {2 * high_cutoff} Hz, got {sfreq}.")

        bandpass_sos = butter(
            N=bandpass_order,
            Wn=(low_cutoff, high_cutoff),
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )

        return cast(np.ndarray, bandpass_sos)

    def design_stream_filters(
        self,
        filter_order_list: list[str],
        parameters_list: list[tuple[Any, ...]],
        freq_mode: str,
    ) -> tuple[np.ndarray, ...]:
        """Design an ordered collection of SOS filters.

        Args:
            filter_order_list: Filter names in the order they will be applied.
            parameters_list: Parameters corresponding to each filter.
            freq_mode: Frequency mode used to design the filters.

        Returns:
            The designed SOS filters in application order.

        Raises:
            FilterListMismatchError: If the two input lists differ in length.
            FrequencyModeError: If the frequency mode is unsupported.
            FilterNotFoundError: If a requested filter is unsupported.
        """

        if len(filter_order_list) != len(parameters_list):
            raise FilterListMismatchError(
                f"The size of the filter_order_list {len(filter_order_list)} "
                f"!= size of parameters_list {len(parameters_list)}"
            )

        if freq_mode.lower() == "sfreq":
            sfreq = self.sfreq
        elif freq_mode.lower() == "pipeline":
            sfreq = self.pipeline_freq
        else:
            raise FrequencyModeError(
                f"Inputed frequency mode {freq_mode} is not supported. "
                f"Supported frequency modes are: {self.supported_freq_modes}"
            )

        filter_list: list[np.ndarray] = []

        for i in range(len(filter_order_list)):
            filter_name = filter_order_list[i]
            filter_parameters = parameters_list[i]

            if filter_name.lower() == "notch":
                designed_filter = self._build_notch_filter(
                    sfreq,
                    filter_parameters,
                )
            elif filter_name.lower() == "bandpass":
                designed_filter = self._build_bandpass_filter(
                    sfreq,
                    filter_parameters,
                )
            else:
                raise FilterNotFoundError(
                    "Requested filter is not supported by the class: "
                    f"{filter_order_list[i]}"
                )

            filter_list.append(designed_filter)

        return tuple(filter_list)
