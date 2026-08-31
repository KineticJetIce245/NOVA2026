from collections.abc import Sequence
from math import ceil
from numbers import Real

import numpy as np
from mne.io.constants import FIFF
from mne_lsl.stream import StreamLSL

from nova2026.data.pipeline import Pipeline

FIFF_VOLT_UNIT_CODE = int(getattr(FIFF, "FIFF_UNIT_V"))
# ---------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------

ACQUISITION_DELAY: float | None = (
    None  # Specifies mne to not start automatic acquisition
)
BUFSIZE = 10.0
PROCESSING_FLAGS = ("clocksync",)
SOURCE_ID: str | None = None  # Fill when the outlet is known.
STREAM_NAME: str | None = None  # Fill when the outlet is known.
STYPE: str | None = "EEG"
TIMEOUT: float | None = 5.0

EXPECTED_N_CHANNELS = 64

# Set this once the amplifier's acquisition configuration is confirmed.
EXPECTED_SFREQ: float | None = None

# Set after confirming whether the outlet declares volts (0)
# or microvolts (-6). Metadata does not necessarily rescale samples.
EXPECTED_UNIT_MULTIPLIER: int | None = None

EXPECTED_CHANNEL_NAMES = (
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "O2",
    "EOG",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CP4",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO5",
    "PO3",
    "PO4",
    "PO6",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "Oz",
)

EXPECTED_CHANNEL_TYPES = tuple(
    "eog" if channel == "EOG" else "eeg" for channel in EXPECTED_CHANNEL_NAMES
)

ALLOWED_PROCESSING_FLAGS = {
    "clocksync",
    "dejitter",
    "monotize",
}


class DefaultStreamer(Pipeline):
    def __init__(self) -> None:
        super().__init__()

        def _validate_identifier(
            value: str | None,
            parameter_name: str,
        ) -> None:
            if value is None:
                return

            if not isinstance(value, str):
                raise TypeError(
                    f"{parameter_name} must be a string or None, "
                    f"got {type(value).__name__}."
                )

            if not value.strip():
                raise ValueError(f"{parameter_name} cannot be an empty string.")

        def _validate_connection_configuration(
            bufsize: float,
            stream_name: str | None,
            stype: str | None,
            source_id: str | None,
            acquisition_delay: float | None,
            processing_flags: Sequence[str] | None,
            timeout: float | None,
        ) -> None:
            # Stream identifiers
            _validate_identifier(stream_name, "stream_name")
            _validate_identifier(stype, "stype")
            _validate_identifier(source_id, "source_id")

            if all(
                identifier is None for identifier in (stream_name, stype, source_id)
            ):
                raise ValueError("At least one stream identifier must be provided.")

            if stype is not None and stype.casefold() != "eeg":
                raise ValueError(f"Expected an EEG stream type, got {stype!r}.")

            # Buffer size
            if (
                isinstance(bufsize, bool)
                or not isinstance(bufsize, Real)
                or not np.isfinite(bufsize)
                or bufsize <= 0
            ):
                raise ValueError(
                    f"bufsize must be finite and positive, got {bufsize!r}."
                )

            # Acquisition delay
            if acquisition_delay is not None and (
                isinstance(acquisition_delay, bool)
                or not isinstance(acquisition_delay, Real)
                or not np.isfinite(acquisition_delay)
                or acquisition_delay <= 0
            ):
                raise ValueError(
                    "acquisition_delay must be None for manual "
                    "acquisition or a finite positive number for "
                    "automatic acquisition."
                    f"acquisition delay was set to: {acquisition_delay}"
                )

            # Processing flags
            if processing_flags is not None:
                if isinstance(processing_flags, str):
                    raise TypeError(
                        "processing_flags must be a sequence of strings, "
                        "not one string."
                    )

                flags = tuple(processing_flags)

                if not all(isinstance(flag, str) for flag in flags):
                    raise TypeError("Every processing flag must be a string.")

                if len(flags) != len(set(flags)):
                    raise ValueError(f"Duplicate processing flags: {flags!r}.")

                invalid_flags = set(flags) - ALLOWED_PROCESSING_FLAGS
                if invalid_flags:
                    raise ValueError(
                        f"Invalid processing flags: {sorted(invalid_flags)}."
                    )

                if "monotize" in flags and "dejitter" not in flags:
                    raise ValueError("'monotize' should only be used with 'dejitter'.")

            # Connection timeout
            if timeout is not None and (
                isinstance(timeout, bool)
                or not isinstance(timeout, Real)
                or not np.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError(
                    f"timeout must be finite and positive or None, got {timeout!r}."
                )

        def _validate_stream_instance(
            stream: StreamLSL,
            expected_name: str | None,
            expected_type: str | None,
            expected_source_id: str | None,
        ) -> None:
            if not isinstance(stream, StreamLSL):
                raise TypeError(f"Expected StreamLSL, got {type(stream).__name__}.")

            if stream.connected:
                raise RuntimeError(
                    "A newly initialized StreamLSL is already connected."
                )

            if stream.sinfo is not None:
                raise RuntimeError(
                    "An unconnected StreamLSL unexpectedly has StreamInfo."
                )

            if stream.dtype is not None:
                raise RuntimeError("An unconnected StreamLSL unexpectedly has a dtype.")

            if stream.filters:
                raise RuntimeError("An unconnected StreamLSL unexpectedly has filters.")

            if stream.name != expected_name:
                raise RuntimeError(
                    f"Requested stream name was not retained: "
                    f"{stream.name!r} != {expected_name!r}."
                )

            if stream.stype != expected_type:
                raise RuntimeError(
                    f"Requested stream type was not retained: "
                    f"{stream.stype!r} != {expected_type!r}."
                )

            if stream.source_id != expected_source_id:
                raise RuntimeError(
                    f"Requested source ID was not retained: "
                    f"{stream.source_id!r} != {expected_source_id!r}."
                )

        def _initialize_stream(
            bufsize: float = BUFSIZE,
            stream_name: str | None = STREAM_NAME,
            stype: str | None = STYPE,
            source_id: str | None = SOURCE_ID,
        ) -> StreamLSL:
            stream = StreamLSL(
                bufsize=bufsize,
                name=stream_name,
                stype=stype,
                source_id=source_id,
            )

            _validate_stream_instance(
                stream,
                expected_name=stream_name,
                expected_type=stype,
                expected_source_id=source_id,
            )

            return stream

        def _validate_stream_connection(
            stream: StreamLSL,
            expected_bufsize: float,
            expected_name: str | None,
            expected_type: str | None,
            expected_source_id: str | None,
            manual_acquisition: bool,
        ) -> None:
            if not stream.connected:
                raise RuntimeError("StreamLSL did not report a successful connection.")

            if stream.sinfo is None:
                raise RuntimeError("Connected stream does not contain StreamInfo.")

            # Validate identifiers when explicitly requested.
            if expected_name is not None and stream.name != expected_name:
                raise RuntimeError(
                    f"Connected to stream name {stream.name!r}; "
                    f"expected {expected_name!r}."
                )

            if expected_type is not None and stream.stype != expected_type:
                raise RuntimeError(
                    f"Connected to stream type {stream.stype!r}; "
                    f"expected {expected_type!r}."
                )

            if (
                expected_source_id is not None
                and stream.source_id != expected_source_id
            ):
                raise RuntimeError(
                    f"Connected source ID {stream.source_id!r}; "
                    f"expected {expected_source_id!r}."
                )

            info = stream.info
            sfreq = float(info["sfreq"])
            n_channels = int(info["nchan"])
            channel_names = tuple(stream.ch_names)
            channel_types = tuple(stream.get_channel_types())
            channel_units = tuple(stream.get_channel_units())

            # Sampling frequency
            if not np.isfinite(sfreq) or sfreq <= 0:
                raise RuntimeError(f"Invalid sampling frequency: {sfreq!r}.")

            # Checks
            # |sfreq - EXPECTED_SFREQ| <= atol + rtol * |EXPECTED_SFREQ|
            # => |sfreq - EXPECTED_FREQ| <= atol = 10^-9
            if EXPECTED_SFREQ is not None and not np.isclose(
                sfreq,
                EXPECTED_SFREQ,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Sampling frequency is {sfreq} Hz; expected {EXPECTED_SFREQ} Hz."
                )

            # Channel count
            if n_channels != EXPECTED_N_CHANNELS:
                raise RuntimeError(
                    f"Received {n_channels} channels; expected {EXPECTED_N_CHANNELS}."
                )

            if len(channel_names) != n_channels:
                raise RuntimeError("Channel-name count does not match info['nchan'].")

            if any(not name.strip() for name in channel_names):
                raise RuntimeError("At least one channel has an empty name.")

            if len(set(channel_names)) != len(channel_names):
                duplicates = sorted(
                    {name for name in channel_names if channel_names.count(name) > 1}
                )
                raise RuntimeError(f"Duplicate channel names: {duplicates}.")

            # Exact channel set and order
            if channel_names != EXPECTED_CHANNEL_NAMES:
                missing = sorted(set(EXPECTED_CHANNEL_NAMES) - set(channel_names))
                unexpected = sorted(set(channel_names) - set(EXPECTED_CHANNEL_NAMES))

                first_mismatch = next(
                    (
                        (index, observed, expected)
                        for index, (observed, expected) in enumerate(
                            zip(
                                channel_names,
                                EXPECTED_CHANNEL_NAMES,
                                strict=False,
                            )
                        )
                        if observed != expected
                    ),
                    None,
                )

                raise RuntimeError(
                    "Channel names/order do not match the model contract. "
                    f"Missing={missing}; unexpected={unexpected}; "
                    f"first mismatch={first_mismatch}."
                )

            # Channel types
            if len(channel_types) != n_channels:
                raise RuntimeError("Channel-type count does not match channel count.")

            if channel_types != EXPECTED_CHANNEL_TYPES:
                incorrect_types = [
                    (
                        channel_names[index],
                        observed,
                        expected,
                    )
                    for index, (observed, expected) in enumerate(
                        zip(
                            channel_types,
                            EXPECTED_CHANNEL_TYPES,
                            strict=True,
                        )
                    )
                    if observed != expected
                ]

                raise RuntimeError(
                    f"Incorrect channel types: {incorrect_types}. "
                    "If the outlet labels EOG as EEG, explicitly call "
                    "stream.set_channel_types({'EOG': 'eog'}) and "
                    "validate again."
                )

            # Units
            if len(channel_units) != n_channels:
                raise RuntimeError("Channel-unit count does not match channel count.")

            non_voltage_channels = [
                channel_names[index]
                for index, (unit, _) in enumerate(channel_units)
                if int(unit) != FIFF_VOLT_UNIT_CODE
            ]

            if non_voltage_channels:
                raise RuntimeError(
                    "Expected voltage units for all EEG/EOG channels, "
                    f"but found non-voltage units on "
                    f"{non_voltage_channels}."
                )

            unit_multipliers = {
                int(unit_multiplier) for _, unit_multiplier in channel_units
            }

            if len(unit_multipliers) != 1:
                raise RuntimeError(
                    "EEG/EOG channels use inconsistent unit "
                    f"multipliers: {sorted(unit_multipliers)}."
                )

            observed_multiplier = next(iter(unit_multipliers))

            if (
                EXPECTED_UNIT_MULTIPLIER is not None
                and observed_multiplier != EXPECTED_UNIT_MULTIPLIER
            ):
                raise RuntimeError(
                    f"Unit multiplier is {observed_multiplier}; expected "
                    f"{EXPECTED_UNIT_MULTIPLIER}."
                )

            # Numeric stream format
            if stream.dtype is None:
                raise RuntimeError("Connected stream does not report a dtype.")

            try:
                stream_dtype = np.dtype(stream.dtype)
            except TypeError as error:
                raise RuntimeError(
                    f"Unsupported stream dtype: {stream.dtype!r}."
                ) from error

            if not np.issubdtype(stream_dtype, np.number):
                raise RuntimeError(f"EEG stream must be numeric, got {stream_dtype}.")

            # MNE-LSL internal buffer
            expected_buffer_samples = ceil(expected_bufsize * sfreq)

            if stream.n_buffer != expected_buffer_samples:
                raise RuntimeError(
                    f"MNE-LSL buffer holds {stream.n_buffer} samples; "
                    f"expected {expected_buffer_samples}."
                )

            # Nothing should be processing the stream yet.
            if stream.filters:
                raise RuntimeError(
                    "Stream unexpectedly has filters before queue wiring."
                )

            if stream.callbacks:
                raise RuntimeError(
                    "Stream unexpectedly has callbacks before queue wiring."
                )

            if manual_acquisition and stream.n_new_samples != 0:
                raise RuntimeError(
                    "Samples were acquired even though automatic "
                    "acquisition was disabled."
                )

            print(
                f"Correctly connected to stream: {stream.name!r}\n"
                f"Stream type: {stream.stype!r}\n"
                f"Source ID: {stream.source_id!r}\n"
                f"Sampling frequency: {sfreq} Hz\n"
                f"Channel count: {n_channels}\n"
                f"Channel names: {list(channel_names)}\n"
                f"Channel types: {list(channel_types)}\n"
                f"Channel unit multiplier: {observed_multiplier}\n"
                f"Data dtype: {stream_dtype}\n"
                f"Buffer capacity: {stream.n_buffer} samples\n"
                f"Automatic acquisition: {not manual_acquisition}"
            )

        def connect_stream(
            bufsize: float = BUFSIZE,
            stream_name: str | None = STREAM_NAME,
            stype: str | None = STYPE,
            source_id: str | None = SOURCE_ID,
            acquisition_delay: float | None = ACQUISITION_DELAY,
            processing_flags: Sequence[str] | None = PROCESSING_FLAGS,
            timeout: float | None = TIMEOUT,
        ) -> tuple[StreamLSL, None]:
            _validate_connection_configuration(
                bufsize=bufsize,
                stream_name=stream_name,
                stype=stype,
                source_id=source_id,
                acquisition_delay=acquisition_delay,
                processing_flags=processing_flags,
                timeout=timeout,
            )

            stream = _initialize_stream(
                bufsize=bufsize,
                stream_name=stream_name,
                stype=stype,
                source_id=source_id,
            )

            flags = None if processing_flags is None else list(processing_flags)

            try:
                stream.connect(
                    acquisition_delay=acquisition_delay,
                    processing_flags=flags,
                    timeout=timeout,
                )

                _validate_stream_connection(
                    stream,
                    expected_bufsize=bufsize,
                    expected_name=stream_name,
                    expected_type=stype,
                    expected_source_id=source_id,
                    manual_acquisition=acquisition_delay is None,
                )
            except Exception:
                if stream.connected:
                    stream.disconnect()
                raise

            return stream, None

        self.add_tube(connect_stream)
