import numpy as np


class ChannelDimensionError(RuntimeError):
    """Raised when required channels are missing from the source."""


class ChannelDuplicationError(RuntimeError):
    """Raised when a source or requested selection contains duplicate labels."""


class ChannelSelectionContract:
    """Validate selection and map inlet columns into canonical channel order.

    Args:
        eeg_channel_names: All original inlet labels, in inlet order.
        expected_channel_names: Required EEG and auxiliary labels, in output order.

    Notes:
        Call get_contract() before applying the mapping. First select
        selected_channel_names from the inlet, then apply_contract() to those
        columns. A None permutation means identity, not a missing contract.
    """

    def __init__(
        self,
        eeg_channel_names: tuple[str, ...],
        expected_channel_names: tuple[str, ...],
    ) -> None:
        """Store channel names; build the mapping explicitly with get_contract()."""

        self.eeg_channel_names = eeg_channel_names
        self.expected_channel_names = expected_channel_names

        self.indices_contract: tuple[int, ...] | None = None
        self.dropped_channel_names: tuple[str, ...] = ()
        self.selected_channel_names: tuple[str, ...] = ()

    def _validate_channel_count(self, selected: list[str]) -> None:
        """Reject duplicate requested/selected names and missing required channels."""

        if len(set(self.expected_channel_names)) != len(self.expected_channel_names):
            raise ChannelDuplicationError(
                "Expected channel names contain duplicates: "
                f"{self.expected_channel_names!r}"
            )

        if len(set(selected)) != len(selected):
            raise ChannelDuplicationError(
                f"Selected EEG channel names contain duplicates: {selected!r}"
            )

        missing = []
        selected_set = set(selected)

        for channel in self.expected_channel_names:
            if channel not in selected_set:
                missing.append(channel)

        if len(missing) > 0:
            raise ChannelDimensionError(
                f"Missing required EEG channels: {missing!r}. "
                f"Available selected channels: {selected!r}"
            )

    def _get_selected_indices_lookup(self) -> dict[str, int]:
        """Return each selected inlet label mapped to its column index."""

        lookup = {}

        for index in range(len(self.selected_channel_names)):
            channel = self.selected_channel_names[index]
            lookup[channel] = index

        return lookup

    def _build_queue_indices(self) -> None:
        """Build selected/dropped names and the inlet-to-output permutation."""

        selected = []
        dropped = []

        expected_set = set(self.expected_channel_names)

        for channel in self.eeg_channel_names:
            if channel in expected_set:
                selected.append(channel)
            else:
                dropped.append(channel)

        self._validate_channel_count(selected)

        self.selected_channel_names = tuple(selected)
        self.dropped_channel_names = tuple(dropped)

        selected_indices_lookup = self._get_selected_indices_lookup()
        indices_contract = []

        # For every channel in the desired output order, find the
        # column where that channel appears in the selected input.
        for channel in self.expected_channel_names:
            index = selected_indices_lookup[channel]
            indices_contract.append(index)

        indices_contract_tuple = tuple(indices_contract)
        identity_order = tuple(range(len(self.expected_channel_names)))

        if indices_contract_tuple == identity_order:
            self.indices_contract = None
        else:
            self.indices_contract = indices_contract_tuple

    def print_channel_selection_contract(self) -> None:
        """Print the source, selection, and permutation for connection diagnostics."""

        print(
            f"EEG channel names: {self.eeg_channel_names}\n"
            f"Expected EEG channel names: {self.expected_channel_names}\n"
            f"Indices contract: {self.indices_contract}\n"
            f"Selected channels: {self.selected_channel_names}\n"
            f"Dropped channels: {self.dropped_channel_names}"
        )

    def get_contract(self) -> tuple[int, ...] | None:
        """Validate and build the mapping, print it, and return its permutation."""

        self._build_queue_indices()
        self.print_channel_selection_contract()

        return self.get_indices_contract()

    def get_indices_contract(self) -> tuple[int, ...] | None:
        """Return the built permutation, or None when the order is unchanged."""

        return self.indices_contract

    def apply_contract(self, arr: np.ndarray, axis: int = 0) -> np.ndarray:
        """Copy an array into canonical order along the selected channel axis.

        Args:
            arr: Array already restricted to selected_channel_names.
            axis: Channel axis; use 1 for samples-by-channels data.

        Returns:
            A reordered copy. The source array is never modified.
        """

        if self.indices_contract is None:
            return arr.copy()

        return np.take(arr, indices=self.indices_contract, axis=axis)
