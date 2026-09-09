"""Tests for source pre-flight validation (B): metadata checks + prepare()."""

import unittest
from types import SimpleNamespace

import numpy as np

from nova2026.streaming import prepare, validate_source

CHANNELS = ("F3", "C3", "EOG")  # EEG, EEG, EOG


class FakeSource:
    """Minimal stand-in for a connected, manually acquired StreamLSL."""

    def __init__(
        self,
        names,
        *,
        sfreq=500.0,
        units=None,
        types=None,
        dtype=np.float64,
        name="outlet",
        source_id="dev-1",
        stype="eeg",
        filters=(),
        callbacks=(),
        n_new=0,
        connected=True,
    ) -> None:
        names = list(names)
        self.ch_names = names
        self.info = {"sfreq": float(sfreq), "nchan": len(names)}
        self.dtype = dtype
        self.name = name
        self.source_id = source_id
        self.stype = stype
        self.filters = list(filters)
        self.callbacks = list(callbacks)
        self.n_new_samples = n_new
        self.connected = connected
        self._units = units if units is not None else ["0"] * len(names)
        self._types = types if types is not None else ["eeg"] * len(names)
        self.sinfo = SimpleNamespace(get_channel_units=lambda: self._units)

    def get_channel_types(self, picks=None):
        return [self._types[self.ch_names.index(name)] for name in picks]


class SourceValidationTests(unittest.TestCase):
    def validate(self, source, **kwargs):
        options = dict(
            sfreq=500.0, channels=CHANNELS, source_unit_exponent=0, n_eeg=2
        )
        options.update(kwargs)
        return validate_source(source, **options)

    def test_valid_source_passes(self) -> None:
        source = FakeSource(CHANNELS, types=["eeg", "eeg", "eog"])
        self.validate(source)  # must not raise

    def test_all_eeg_channels_pass_when_n_eeg_matches(self) -> None:
        source = FakeSource(("F3", "C3"))
        validate_source(source, sfreq=500.0, channels=("F3", "C3"),
                        source_unit_exponent=0, n_eeg=2)

    def test_identity_mismatches_are_rejected(self) -> None:
        source = FakeSource(CHANNELS, types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "name"):
            self.validate(source, stream_name="other")
        with self.assertRaisesRegex(RuntimeError, "ID"):
            self.validate(source, source_id="other")
        with self.assertRaisesRegex(RuntimeError, "type"):
            self.validate(source, stream_type="other")

    def test_wrong_sampling_rate_is_rejected(self) -> None:
        source = FakeSource(CHANNELS, sfreq=250.0, types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "sampling rate"):
            self.validate(source)

    def test_non_numeric_samples_are_rejected(self) -> None:
        source = FakeSource(CHANNELS, dtype="S10", types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "numeric"):
            self.validate(source)

    def test_untouched_state_is_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "before setup"):
            self.validate(
                FakeSource(CHANNELS, filters=["x"], types=["eeg", "eeg", "eog"])
            )
        with self.assertRaisesRegex(RuntimeError, "before setup"):
            self.validate(
                FakeSource(CHANNELS, callbacks=[lambda *a: None],
                           types=["eeg", "eeg", "eog"])
            )
        with self.assertRaisesRegex(RuntimeError, "before setup"):
            self.validate(
                FakeSource(CHANNELS, n_new=5, types=["eeg", "eeg", "eog"])
            )

    def test_disconnected_source_is_rejected(self) -> None:
        source = FakeSource(CHANNELS, connected=False)
        with self.assertRaisesRegex(RuntimeError, "did not connect"):
            self.validate(source)

    def test_duplicate_or_missing_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            self.validate(FakeSource(("F3", "F3", "EOG"),
                                     types=["eeg", "eeg", "eog"]))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.validate(FakeSource(("F3", "P3", "EOG"),
                                     types=["eeg", "eeg", "eog"]))

    def test_declared_units_must_match_the_config(self) -> None:
        # microvolts declared but volts expected: mismatch.
        source = FakeSource(CHANNELS, units=["-6", "-6", "-6"],
                            types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "units"):
            self.validate(source)
        # Unknown unit spelling is never accepted.
        source = FakeSource(CHANNELS, units=["weird", "0", "0"],
                            types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            self.validate(source)
        # Missing unit declarations are rejected too.
        source = FakeSource(CHANNELS, units=[], types=["eeg", "eeg", "eog"])
        with self.assertRaisesRegex(RuntimeError, "declare"):
            self.validate(source)

    def test_channel_types_must_identify_eeg_and_eog(self) -> None:
        # Everything declared eeg while EOG is expected: rejected.
        source = FakeSource(CHANNELS, types=["eeg", "eeg", "eeg"])
        with self.assertRaisesRegex(RuntimeError, "types"):
            self.validate(source)

    def test_invalid_expectations_raise_value_error(self) -> None:
        source = FakeSource(CHANNELS, types=["eeg", "eeg", "eog"])
        with self.assertRaises(ValueError):
            self.validate(source, sfreq=-1.0)
        with self.assertRaises(ValueError):
            self.validate(source, n_eeg=0)
        with self.assertRaises(ValueError):
            self.validate(source, n_eeg=4)

    def test_prepare_returns_a_working_channel_contract(self) -> None:
        # The source publishes EOG first; prepare must validate the outlet and
        # hand back a contract that reorders every block to the expected order.
        source = FakeSource(("EOG", "C3", "F3"), types=["eog", "eeg", "eeg"])
        contract = prepare(
            source, sfreq=500.0, channels=CHANNELS,
            source_unit_exponent=0, n_eeg=2,
        )
        self.assertEqual(contract.expected_channels, CHANNELS)
        self.assertEqual(contract.dropped_channels, ())
        data = np.zeros((3, 3))
        data[:, 0] = 1.0  # EOG
        data[:, 1] = 2.0  # C3
        data[:, 2] = 3.0  # F3
        reordered = contract.reorder(data)
        self.assertTrue(np.allclose(reordered[:, 0], 3.0))  # F3 first
        self.assertTrue(np.allclose(reordered[:, 1], 2.0))  # C3
        self.assertTrue(np.allclose(reordered[:, 2], 1.0))  # EOG last

    def test_prepare_raises_runtime_error_on_any_mismatch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sampling rate"):
            prepare(FakeSource(CHANNELS, sfreq=250.0,
                               types=["eeg", "eeg", "eog"]),
                    sfreq=500.0, channels=CHANNELS,
                    source_unit_exponent=0, n_eeg=2)
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            prepare(FakeSource(("F3", "F3", "EOG"),
                               types=["eeg", "eeg", "eog"]),
                    sfreq=500.0, channels=CHANNELS,
                    source_unit_exponent=0, n_eeg=2)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            prepare(FakeSource(("F3", "P3", "EOG"),
                               types=["eeg", "eeg", "eog"]),
                    sfreq=500.0, channels=CHANNELS,
                    source_unit_exponent=0, n_eeg=2)


if __name__ == "__main__":
    unittest.main()
