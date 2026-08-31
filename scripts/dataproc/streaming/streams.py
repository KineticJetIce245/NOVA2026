from collections.abc import Sequence
from numbers import Real

import numpy as np
from mne.io.constants import FIFF
from mne_lsl.stream import StreamLSL

from scripts.dataproc.streaming.channel_selection_contract import (
    ChannelSelectionContract,
)
from nova2026.data.pipeline import Pipeline
from scripts.dataproc.streaming.streamer import Streamer


# Acquisition is started manually after the stream passes validation.
ACQUISITION_DELAY: None = None
BUFSIZE = 10.0
PROCESSING_FLAGS = ("clocksync",)
SOURCE_ID: str | None = None  # Fill when the outlet is known.
STREAM_NAME: str | None = None  # Fill when the outlet is known.
STYPE: str | None = "EEG"
TIMEOUT: float | None = 5.0

# Set once the amplifier's acquisition configuration is confirmed.
EXPECTED_SFREQ: float | None = None

# FIFF unit multiplier 0 declares volts. This metadata does not rescale samples.
VOLT_UNIT_MULTIPLIER = 0

# Channels required by the downstream pipeline, in canonical output order.
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

ALLOWED_PROCESSING_FLAGS = {
    "clocksync",
    "dejitter",
    "monotize",
}


class DefaultStreamer(Streamer):
    """Prepare and validate an EEG stream for manual acquisition.

    The streamer connects to an MNE-LSL stream, validates its configuration,
    selects the required channels, and returns the stream with its channel
    selection contract. No samples are acquired during setup.
    """

    def __init__(self) -> None:
        """Initialize the streamer with its connection setup tube."""
        super().__init__()

        def _validate_identifier(
            value: str | None,
            parameter_name: str,
        ) -> None:
            """Validate an optional stream identifier.

            Args:
                value (str | None): Identifier value to validate.
                parameter_name (str): Identifier name used in error messages.

            Raises
                TypeError: If the identifier is not a string or ``None``.
                ValueError: If the identifier is an empty string.
            """
            if value is None:
                return

            if not isinstance(value, str):
                raise TypeError(
                    f"{parameter_name} must be a string or None, "
                    f"got {type(value).__name__}."
                )

            if not value.strip():
                raise ValueError(f"{parameter_name} cannot be empty.")

        def _validate_positive_number(
            value: int | float,
            parameter_name: str,
        ) -> None:
            """Validate that a configuration value is finite and positive.

            Args:
                value (Real): Numerical value to validate.
                parameter_name (str): Parameter name used in error messages.

            Raises:
                ValueError: If ``value`` is not a finite positive real number.
            """
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{parameter_name} must be finite and positive, got {value!r}."
                )

        def _validate_connection_configuration(
            bufsize: float,
            stream_name: str | None,
            stype: str | None,
            source_id: str | None,
            acquisition_delay: None,
            processing_flags: Sequence[str] | None,
            timeout: float | None,
            expected_sfreq: float | None,
        ) -> None:
            """Validate all settings required before connecting the stream.

            Args:
                bufsize (float): Length of the stream buffer in seconds.
                stream_name (str | None): Name of the requested LSL stream.
                stype (str | None): Type of the requested LSL stream.
                source_id (str | None): Source ID of the requested LSL stream.
                acquisition_delay (None): Must be ``None`` for manual
                    acquisition.
                processing_flags (Sequence[str] | None): LSL post-processing
                    flags applied on connection.
                timeout (float | None): Maximum connection time in seconds.
                expected_sfreq (float | None): Required sampling frequency in
                    Hz.

            Raises:
                TypeError: If an identifier or processing flag has an invalid
                    type.
                ValueError: If a configuration value violates the stream
                    contract.
            """
            _validate_identifier(stream_name, "stream_name")
            _validate_identifier(stype, "stype")
            _validate_identifier(source_id, "source_id")

            if all(
                identifier is None for identifier in (stream_name, stype, source_id)
            ):
                raise ValueError("At least one stream identifier must be provided.")

            if stype is not None and stype.casefold() != "eeg":
                raise ValueError(f"Expected an EEG stream type, got {stype!r}.")

            _validate_positive_number(bufsize, "bufsize")

            # The external acquisition thread starts only after setup finishes.
            if acquisition_delay is not None:
                raise ValueError("acquisition_delay must be None.")

            if expected_sfreq is None:
                raise ValueError("EXPECTED_SFREQ must be configured.")

            _validate_positive_number(expected_sfreq, "expected_sfreq")

            if processing_flags is not None:
                if isinstance(processing_flags, str):
                    raise TypeError("processing_flags must be a sequence of strings.")

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

                # Monotization relies on the timestamps produced by dejittering.
                if "monotize" in flags and "dejitter" not in flags:
                    raise ValueError("'monotize' requires 'dejitter'.")

            if timeout is not None:
                _validate_positive_number(timeout, "timeout")

        def _validate_connected_stream(
            stream: StreamLSL,
            expected_name: str | None,
            expected_type: str | None,
            expected_source_id: str | None,
            expected_sfreq: float,
        ) -> float:
            """Validate the connected stream before channel preparation.

            Args:
                stream (StreamLSL): Connected MNE-LSL stream.
                expected_name (str | None): Requested stream name.
                expected_type (str | None): Requested stream type.
                expected_source_id (str | None): Requested source ID.
                expected_sfreq (float): Required sampling frequency in Hz.

            Returns:
                float: Validated sampling frequency in Hz.

            Raises:
                RuntimeError: If the connected stream violates the stream
                    contract or acquisition has already started.
            """
            if not stream.connected:
                raise RuntimeError("StreamLSL did not connect.")

            if stream.sinfo is None:
                raise RuntimeError("Connected stream does not contain StreamInfo.")

            if expected_name is not None and stream.name != expected_name:
                raise RuntimeError(
                    f"Connected to stream {stream.name!r}; expected {expected_name!r}."
                )

            if expected_type is not None and stream.stype != expected_type:
                raise RuntimeError(
                    f"Connected to type {stream.stype!r}; expected {expected_type!r}."
                )

            if (
                expected_source_id is not None
                and stream.source_id != expected_source_id
            ):
                raise RuntimeError(
                    f"Connected to source {stream.source_id!r}; "
                    f"expected {expected_source_id!r}."
                )

            sfreq = float(stream.info["sfreq"])

            if not np.isfinite(sfreq) or sfreq <= 0:
                raise RuntimeError(f"Invalid sampling frequency: {sfreq!r}.")

            # Checks
            # |sfreq - EXPECTED_SFREQ| <= atol + rtol * |EXPECTED_SFREQ|
            # => |sfreq - EXPECTED_SFREQ| <= atol = 10^-9
            if not np.isclose(
                sfreq,
                expected_sfreq,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Sampling frequency is {sfreq} Hz; expected {expected_sfreq} Hz."
                )

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

            # Preprocessing and queue callbacks are attached after setup.
            if stream.filters:
                raise RuntimeError("Stream has filters before preprocessing.")

            if stream.callbacks:
                raise RuntimeError("Stream has callbacks before queue setup.")

            if stream.n_new_samples != 0:
                raise RuntimeError(
                    "Samples were acquired before stream setup finished."
                )

            return sfreq

        def _prepare_channels(
            stream: StreamLSL,
        ) -> ChannelSelectionContract:
            """Select required channels and configure their metadata.

            Args:
                stream (StreamLSL): Connected stream to prepare.

            Returns:
                ChannelSelectionContract: Contract describing selected,
                dropped, and reordered channels.

            Raises:
                RuntimeError: If channel selection, types, or units do not
                    satisfy the stream contract.
            """
            channel_selection = ChannelSelectionContract(
                eeg_channel_names=tuple(stream.ch_names),
                expected_channel_names=EXPECTED_CHANNEL_NAMES,
            )

            # Keep selected channels in inlet order. The queue uses the contract
            # indices to reorder each chunk into canonical output order.
            channel_selection.get_contract()
            stream.pick(list(channel_selection.selected_channel_names))

            if tuple(stream.ch_names) != channel_selection.selected_channel_names:
                raise RuntimeError(
                    "Stream channel selection was not applied correctly."
                )

            channel_types = {
                channel: "eog" if channel == "EOG" else "eeg"
                for channel in stream.ch_names
            }

            # Set types first because changing a channel type can reset its unit.
            stream.set_channel_types(channel_types)
            stream.set_channel_units(
                {channel: VOLT_UNIT_MULTIPLIER for channel in stream.ch_names}
            )

            expected_types = tuple(
                channel_types[channel] for channel in stream.ch_names
            )
            observed_types = tuple(stream.get_channel_types())

            if observed_types != expected_types:
                raise RuntimeError(
                    f"Channel types were not configured correctly: {observed_types}."
                )

            volt_unit_code = int(getattr(FIFF, "FIFF_UNIT_V"))

            if any(
                int(unit) != volt_unit_code or int(multiplier) != VOLT_UNIT_MULTIPLIER
                for unit, multiplier in stream.get_channel_units()
            ):
                raise RuntimeError(
                    "Channel voltage units were not configured correctly."
                )

            return channel_selection

        def connect_stream(
            bufsize: float = BUFSIZE,
            stream_name: str | None = STREAM_NAME,
            stype: str | None = STYPE,
            source_id: str | None = SOURCE_ID,
            acquisition_delay: None = ACQUISITION_DELAY,
            processing_flags: Sequence[str] | None = PROCESSING_FLAGS,
            timeout: float | None = TIMEOUT,
            expected_sfreq: float | None = EXPECTED_SFREQ,
        ) -> tuple[StreamLSL, ChannelSelectionContract]:
            """Connect and prepare an EEG stream without starting acquisition.

            Args:
                bufsize (float): Length of the stream buffer in seconds.
                stream_name (str | None): Name of the requested LSL stream.
                stype (str | None): Type of the requested LSL stream.
                source_id (str | None): Source ID of the requested LSL stream.
                acquisition_delay (None): Must be ``None`` for manual
                    acquisition.
                processing_flags (Sequence[str] | None): LSL post-processing
                    flags applied on connection.
                timeout (float | None): Maximum connection time in seconds.
                expected_sfreq (float | None): Required sampling frequency in
                    Hz.

            Returns:
                tuple[StreamLSL, ChannelSelectionContract]: Connected stream
                and its validated channel selection contract.

            Raises:
                TypeError: If a configuration value has an invalid type.
                ValueError: If the connection configuration is invalid.
                RuntimeError: If the connected stream violates the stream
                    contract.
            """
            _validate_connection_configuration(
                bufsize=bufsize,
                stream_name=stream_name,
                stype=stype,
                source_id=source_id,
                acquisition_delay=acquisition_delay,
                processing_flags=processing_flags,
                timeout=timeout,
                expected_sfreq=expected_sfreq,
            )

            assert expected_sfreq is not None

            stream = StreamLSL(
                bufsize=bufsize,
                name=stream_name,
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

                sfreq = _validate_connected_stream(
                    stream=stream,
                    expected_name=stream_name,
                    expected_type=stype,
                    expected_source_id=source_id,
                    expected_sfreq=expected_sfreq,
                )

                channel_selection = _prepare_channels(stream)

                if stream.filters or stream.callbacks or stream.n_new_samples != 0:
                    raise RuntimeError("Stream changed during setup.")

            except Exception:
                # Do not leave a partially configured inlet connected.
                if stream.connected:
                    stream.disconnect()
                raise

            print(
                f"Connected to stream: {stream.name!r}\n"
                f"Sampling frequency: {sfreq} Hz\n"
                f"Selected channels: {list(stream.ch_names)}\n"
                "Manual acquisition ready."
            )

            return stream, channel_selection

        self.add_pre_processing_tube(connect_stream)
