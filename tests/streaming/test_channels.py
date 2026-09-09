"""Tests for the source-to-canonical channel contract."""

import unittest

import numpy as np

from nova2026.streaming.channels import ChannelContract

EXPECTED = ("F3", "Cz", "P3", "EOG")


def columns(names: tuple[str, ...], samples: int = 4) -> np.ndarray:
    """Return data whose column j is filled with its channel index."""

    return np.tile(np.arange(len(names), dtype=np.float64), (samples, 1))


class ChannelContractTests(unittest.TestCase):
    def test_exact_match_returns_data_unchanged(self) -> None:
        contract = ChannelContract(EXPECTED, EXPECTED)
        data = columns(EXPECTED)

        self.assertIs(contract.reorder(data), data)
        self.assertEqual(contract.dropped_channels, ())

    def test_reorders_to_canonical_order(self) -> None:
        source = ("EOG", "P3", "Cz", "F3")
        contract = ChannelContract(source, EXPECTED)
        data = columns(source)

        result = contract.reorder(data)

        self.assertEqual(result.shape, (4, len(EXPECTED)))
        # Column j now holds the j-th expected channel's original value.
        for j, name in enumerate(EXPECTED):
            self.assertEqual(result[0, j], source.index(name))
        # The source array is untouched.
        self.assertTrue(np.all(data[:, 0] == 0))

    def test_drops_extra_source_channels(self) -> None:
        source = ("F3", "Cz", "P3", "HEOG", "EOG", "TRIGGER")
        contract = ChannelContract(source, EXPECTED)

        self.assertEqual(contract.dropped_channels, ("HEOG", "TRIGGER"))
        result = contract.reorder(columns(source, samples=3))
        self.assertEqual(result.shape, (3, len(EXPECTED)))

    def test_missing_channel_fails_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required channels: \\['Cz'\\]"):
            ChannelContract(("F3", "P3", "EOG"), EXPECTED)

    def test_duplicate_labels_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicated"):
            ChannelContract(("F3", "F3", "Cz"), EXPECTED)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            ChannelContract(EXPECTED, ("F3", "Cz", "F3", "EOG"))

    def test_shape_validation(self) -> None:
        contract = ChannelContract(EXPECTED, EXPECTED)
        with self.assertRaises(ValueError):
            contract.reorder(columns(EXPECTED[:-1]))
        with self.assertRaises(ValueError):
            contract.reorder(np.zeros(4))

    def test_empty_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChannelContract((), EXPECTED)
        with self.assertRaises(ValueError):
            ChannelContract(EXPECTED, ())


if __name__ == "__main__":
    unittest.main()
