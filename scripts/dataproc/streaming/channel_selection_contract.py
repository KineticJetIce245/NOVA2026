class ChannelDimensionError(RuntimeError):
    pass


class ChannelDuplicationError(RuntimeError):
    pass


class ChannelSelectionContract:
    def __init__(
        self,
        eeg_channel_names: tuple[str, ...],
        expected_channel_names: tuple[str, ...],
    ) -> None:
        self.eeg_channel_names = eeg_channel_names
        self.expected_channel_names = expected_channel_names

        self.selected_indices_contract: tuple[int, ...] | None = None
        self.dropped_channel_names: tuple[str, ...] = ()
        self.selected_channel_names: tuple[str, ...] = ()

    def _validate_channel_count(self, selected: list[str]) -> None:
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
        lookup = {}

        for index in range(len(self.selected_channel_names)):
            channel = self.selected_channel_names[index]
            lookup[channel] = index

        return lookup

    def _build_queue_indices(self) -> None:
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
            self.selected_indices_contract = None
        else:
            self.selected_indices_contract = indices_contract_tuple

    def print_channel_selection_contract(self) -> None:
        print(
            f"EEG channel names: {self.eeg_channel_names}\n"
            f"Expected EEG channel names: {self.expected_channel_names}\n"
            f"Indices contract: {self.selected_indices_contract}\n"
            f"Selected channels: {self.selected_channel_names}\n"
            f"Dropped channels: {self.dropped_channel_names}"
        )

    def get_contract(self) -> tuple[int, ...] | None:
        self._build_queue_indices()
        self.print_channel_selection_contract()
        return self.get_selected_indices_contract()

    def get_selected_indices_contract(self):
        return self.selected_indices_contract
