"""create_buffer: independent realtime component."""

from .circular_buffer import CircularBuffer
from .config import StreamConfig
from .filters import SOSFilterBuilder
from .streaming_filter import StreamingSOSFilter
from .streaming_resampler import StreamingResampler
from .voltage_converter import VoltageConverter


def create_buffer(
    config: StreamConfig, source_exponents: tuple[int, ...]
) -> "CircularBuffer":
    """Build the same causal preprocessing components for live and offline replay.

    Args:
        config: Complete run contract.
        source_exponents: Preserved source units in canonical channel order.

    Returns:
        A reset buffer with notch, band-pass, and streaming resampling stages.
    """

    builder = SOSFilterBuilder(config.input_sfreq, config.output_sfreq)
    names = []
    parameters = []
    if config.notch_frequency is not None:
        names.append("notch")
        parameters.append((config.notch_q, config.notch_frequency))
    names.append("bandpass")
    parameters.append((*config.bandpass, config.filter_order))
    coefficients = builder.design_stream_filters(names, parameters, "sfreq")
    channels = len(config.channels)
    filters = [StreamingSOSFilter(sos, channels) for sos in coefficients]
    return CircularBuffer(
        config,
        VoltageConverter(source_exponents),
        StreamingResampler(
            config.input_sfreq, config.output_sfreq, channels, config.resample_quality
        ),
        filters,
    )
