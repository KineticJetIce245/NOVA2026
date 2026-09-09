"""Validate a connected LSL source before any sample is used.

The pipeline trusts whatever outlet it is connected to. Connecting to the wrong
stream, or to the right stream with a different rate, units or channel types,
would quietly record plausible-looking but wrong data. ``validate_source``
checks the connected inlet's metadata against the run contract and raises
before a single sample is read, recorded or processed.

Pure validation: it never pulls data, adds callbacks or changes the stream.
The source must still be connected manually (``acquisition_delay=None``) before
this is called.
"""

import math

import numpy as np

# Accepted spellings of the source's declared voltage unit, mapped to the
# power of ten of the unit (0 = volts, -3 = mV, -6 = uV, -9 = nV).
_UNIT_ALIASES = {
    "0": 0,
    "v": 0,
    "volt": 0,
    "volts": 0,
    "-3": -3,
    "mv": -3,
    "millivolt": -3,
    "millivolts": -3,
    "-6": -6,
    "uv": -6,
    "microvolt": -6,
    "microvolts": -6,
    "-9": -9,
    "nv": -9,
    "nanovolt": -9,
    "nanovolts": -9,
}


def validate_source(
    stream,
    *,
    sfreq: float,
    channels: tuple[str, ...],
    source_unit_exponent: int = 0,
    n_eeg: int | None = None,
    stream_name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
) -> None:
    """Check a connected inlet's metadata against the run contract.

    Args:
        stream: Connected ``StreamLSL`` opened with manual acquisition.
        sfreq: Expected source sampling rate in Hz.
        channels: Required channel labels, EEG first then auxiliary (EOG).
        source_unit_exponent: Expected power of ten of the source unit.
        n_eeg: Number of leading EEG channels; the rest are auxiliary. Defaults
            to ``len(channels)`` (all EEG).
        stream_name: Expected outlet name, checked when given.
        source_id: Expected outlet source ID, checked when given.
        stream_type: Expected outlet type, checked when given.

    Raises:
        RuntimeError: If any source fact does not match the run contract, or
            the outlet already has filters, callbacks or unread samples.
        ValueError: If the expectation arguments themselves are invalid.
    """

    if not getattr(stream, "connected", False) or stream.sinfo is None:
        raise RuntimeError("The source did not connect.")

    if stream_name is not None and stream.name != stream_name:
        raise RuntimeError("The source name does not match the configuration.")
    if source_id is not None and stream.source_id != source_id:
        raise RuntimeError("The source ID does not match the configuration.")
    if stream_type is not None and stream.stype != stream_type:
        raise RuntimeError("The source type does not match the configuration.")

    if not math.isfinite(float(sfreq)) or float(sfreq) <= 0:
        raise ValueError("sfreq must be finite and positive.")
    actual_sfreq = float(stream.info["sfreq"])
    if not math.isclose(actual_sfreq, float(sfreq), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"The source sampling rate ({actual_sfreq:g} Hz) does not match the "
            f"configured rate ({sfreq:g} Hz)."
        )

    dtype = getattr(stream, "dtype", None)
    if dtype is None or not np.issubdtype(np.dtype(dtype), np.number):
        raise RuntimeError("EEG samples must be numeric.")

    if stream.filters or stream.callbacks or getattr(stream, "n_new_samples", 0):
        raise RuntimeError("Source processing or acquisition started before setup.")

    names = tuple(stream.ch_names)
    # Duplicate or missing labels fail here, before any data is handled.
    if not names or any(not str(name) for name in names):
        raise RuntimeError("The source must declare non-empty channel labels.")
    if len(set(names)) != len(names):
        raise RuntimeError(f"Source channel labels are duplicated: {names!r}")
    missing = [name for name in channels if name not in names]
    if missing:
        raise RuntimeError(
            f"Source is missing required channels: {missing!r} "
            f"(source has {names!r})."
        )

    n_eeg = len(channels) if n_eeg is None else int(n_eeg)
    if not 1 <= n_eeg <= len(channels):
        raise ValueError("n_eeg must be between 1 and the channel count.")
    expected_types = ("eeg",) * n_eeg + ("eog",) * (len(channels) - n_eeg)
    actual_types = tuple(stream.get_channel_types(picks=list(channels)))
    if actual_types != expected_types:
        raise RuntimeError(
            "Source channel types must identify EEG and EOG correctly "
            f"(expected {expected_types}, got {actual_types})."
        )

    units = stream.sinfo.get_channel_units()
    if units is None or len(units) != len(names):
        raise RuntimeError("The source must declare voltage units per channel.")
    for name in channels:
        raw = units[names.index(name)]
        label = str(raw).strip().lower()
        if label not in _UNIT_ALIASES:
            raise RuntimeError(
                f"Missing or unsupported voltage units for {name}: {raw!r}."
            )
        if _UNIT_ALIASES[label] != source_unit_exponent:
            raise RuntimeError(
                f"Source voltage units for {name} do not match the configured "
                f"exponent ({source_unit_exponent})."
            )
