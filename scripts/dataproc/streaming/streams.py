"""Connect and validate an LSL source without starting acquisition."""

import numpy as np
from mne.io.constants import FIFF
from mne_lsl.stream import StreamLSL

from .channel_selection_contract import ChannelSelectionContract
from .config import StreamConfig


class StreamConnection:
    """Own connection setup independently of the runner's thread architecture.

    Args:
        config: Acquisition settings and canonical channel order for this run.
    """

    def __init__(self, config: StreamConfig) -> None:
        """Store the source contract without opening an LSL connection."""

        self.config = config
        self.stream: StreamLSL | None = None
        self.channel_selection: ChannelSelectionContract | None = None

    def connect(self) -> StreamLSL:
        """Connect, verify source metadata, and select required channels.

        Returns:
            A manually acquired stream. Units and sample values are unchanged.

        Raises:
            RuntimeError: If source metadata does not match the run contract.
        """

        if self.stream is not None and self.stream.connected:
            raise RuntimeError("The source is already connected.")
        config = self.config
        stream = StreamLSL(
            bufsize=config.buffer_seconds,
            name=config.stream_name,
            stype=config.stream_type,
            source_id=config.source_id,
        )
        self.stream = stream
        try:
            stream.connect(
                acquisition_delay=None,
                processing_flags=["clocksync"],
                timeout=config.connection_timeout,
            )
            self._validate_stream(stream)
            self._prepare_channels(stream)
        except BaseException:
            self.disconnect()
            raise

        return stream

    def _validate_stream(self, stream: StreamLSL) -> None:
        """Check identity, rate, and untouched acquisition state."""

        config = self.config

        if not stream.connected or stream.sinfo is None:
            raise RuntimeError("The source did not connect.")

        if config.stream_name is not None and stream.name != config.stream_name:
            raise RuntimeError("The source name does not match the configuration.")

        if config.source_id is not None and stream.source_id != config.source_id:
            raise RuntimeError("The source ID does not match the configuration.")

        if config.stream_type is not None and stream.stype != config.stream_type:
            raise RuntimeError("The source type does not match the configuration.")

        if not np.isclose(stream.info["sfreq"], config.input_sfreq, rtol=0, atol=1e-9):
            raise RuntimeError(
                "The source sampling rate does not match the configuration."
            )

        if stream.dtype is None or not np.issubdtype(stream.dtype, np.number):
            raise RuntimeError("EEG samples must be numeric.")

        if stream.filters or stream.callbacks or stream.n_new_samples:
            raise RuntimeError("Source processing or acquisition started before setup.")

    def _prepare_channels(self, stream: StreamLSL) -> None:
        """Validate original labels, types, and units before selecting channels.

        Metadata is not rewritten: declaring volts does not convert samples.
        Reference and upstream processing are operator assertions in the config;
        ordinary LSL channel metadata cannot establish them automatically.
        """

        config = self.config
        names = tuple(stream.ch_names)

        if len(names) != len(set(names)):
            raise RuntimeError("The source contains duplicate channel labels.")
        contract = ChannelSelectionContract(names, config.channels)
        contract.get_contract()
        info = stream.sinfo

        if info is None:
            raise RuntimeError("The source is not connected.")

        units = info.get_channel_units()

        if units is None or len(units) != len(names):
            raise RuntimeError("The source must declare channel voltage units.")
        aliases = {
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

        for name in config.channels:
            raw_unit = units[names.index(name)]
            # MNE-LSL can fall back to volts for unknown text. Reject unknown
            # units explicitly instead of accepting that fallback as evidence.
            label = str(raw_unit).strip().lower()
            if label not in aliases:
                raise RuntimeError(
                    f"Missing or unsupported voltage units for {name}: {raw_unit!r}."
                )
            if aliases[label] != config.source_unit_exponent:
                raise RuntimeError(
                    f"Source voltage units for {name} do not match the config."
                )

        for name, (unit, exponent) in zip(
            config.channels, stream.get_channel_units(picks=list(config.channels))
        ):
            if int(unit) != int(FIFF["FIFF_UNIT_V"]):
                raise RuntimeError(f"{name} is not declared as a voltage channel.")
            if int(exponent) != config.source_unit_exponent:
                raise RuntimeError(f"Voltage units for {name} do not match the config.")
        expected_types = ["eeg"] * len(config.eeg_channels) + ["eog"] * len(
            config.eog_channels
        )

        if stream.get_channel_types(picks=list(config.channels)) != expected_types:
            raise RuntimeError(
                "Source channel types must identify EEG and EOG correctly."
            )
        stream.pick(list(contract.selected_channel_names))

        if tuple(stream.ch_names) != contract.selected_channel_names:
            raise RuntimeError(
                "The source did not preserve selected inlet channel order."
            )
        self.channel_selection = contract

    def disconnect(self) -> None:
        """Close the inlet after acquisition has stopped."""

        if self.stream is not None and self.stream.connected:
            self.stream.disconnect()
