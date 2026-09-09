"""Validate source channels and map columns into a canonical order.

Pure numpy only: this module never touches LSL, so it can be unit-tested
without a running stream.
"""

import numpy as np


class ChannelContract:
    """Map source data columns into a canonical ``expected_channels`` order.

    The contract is built once, right after the inlet is connected and before
    any data is acquired. It raises at construction when a required channel is
    missing or labels are duplicated, so a misconfigured source fails loudly
    before a run starts instead of silently corrupting recordings.

    Args:
        source_channels: Labels of the connected source, in its column order.
        expected_channels: Required labels, EEG first then auxiliary (e.g.
            EOG), in the exact order the preprocessing chain expects.

    Notes:
        Extra source channels are dropped. ``reorder`` is a column permutation
        and returns the input unchanged when the source already matches the
        expected columns exactly; it never edits the source array.

    Attributes:
        source_channels: Labels of the connected source.
        expected_channels: Canonical labels after ``reorder``.
        selected_channels: Labels kept from the source, in canonical order.
        dropped_channels: Source labels that are not part of the contract.
    """

    def __init__(
        self,
        source_channels: tuple[str, ...],
        expected_channels: tuple[str, ...],
    ) -> None:
        """Validate labels and build the source-to-canonical permutation."""

        source = tuple(str(name) for name in source_channels)
        expected = tuple(str(name) for name in expected_channels)

        # Reject empty label sets before anything else.
        if not source or any(not name for name in source):
            raise ValueError("source_channels must be non-empty strings.")
        if not expected:
            raise ValueError("expected_channels cannot be empty.")

        # Duplicate labels make any name-to-column mapping ambiguous.
        if len(set(source)) != len(source):
            raise ValueError(f"Source channel labels are duplicated: {source!r}")
        if len(set(expected)) != len(expected):
            raise ValueError(
                f"Expected channel labels are duplicated: {expected!r}"
            )

        # Every required channel must exist on the source; fail at setup.
        lookup = {name: index for index, name in enumerate(source)}
        missing = [name for name in expected if name not in lookup]
        if missing:
            raise ValueError(
                f"Source is missing required channels: {missing!r} "
                f"(source has {source!r})."
            )

        # Map each expected channel to its source column. When the source is
        # already exactly the expected columns, no permutation is needed.
        permutation = tuple(lookup[name] for name in expected)
        identity = permutation == tuple(range(len(permutation))) and len(
            source
        ) == len(permutation)

        self.source_channels = source
        self.expected_channels = expected
        self.selected_channels = expected
        self.dropped_channels = tuple(
            name for name in source if name not in set(expected)
        )
        # None means "copy nothing, pass the array through as-is".
        self._indices: tuple[int, ...] | None = None if identity else permutation

    def reorder(self, data: np.ndarray) -> np.ndarray:
        """Return data restricted to and ordered as ``expected_channels``.

        Args:
            data: Samples by channels matching ``source_channels``.

        Returns:
            The same array when no permutation is needed, otherwise a new
            array with the contract columns. Never modifies the input.
        """

        # Column count must match the source the contract was built for.
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")
        if data.shape[1] != len(self.source_channels):
            raise ValueError(
                f"Expected {len(self.source_channels)} source channels, "
                f"received {data.shape[1]}."
            )

        # Identity shortcut: avoid an unnecessary copy on the hot path.
        if self._indices is None:
            return data
        # Take the contract columns in canonical order (drops the extras).
        return np.take(data, self._indices, axis=1)
