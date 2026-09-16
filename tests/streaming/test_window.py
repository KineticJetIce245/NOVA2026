"""Tests for the EEGWindow value object."""

import unittest

import numpy as np

from nova2026.streaming.window import EEGWindow

SAMPLES = 64
EEG_CHANNELS = 3
EOG_CHANNELS = 1


def make_window(
    samples: int = SAMPLES,
    eeg: int = EEG_CHANNELS,
    eog: int = EOG_CHANNELS,
    **overrides,
) -> EEGWindow:
    """Return a valid EEGWindow with deterministic content."""

    options = dict(
        data=np.arange(samples * eeg, dtype=np.float64).reshape(samples, eeg),
        eog=np.zeros((samples, eog), dtype=np.float64),
        timestamps=np.arange(samples, dtype=np.float64) / 128.0,
        valid=True,
        reasons=(),
        start_sample=128,
        channel_names=tuple(f"E{index}" for index in range(eeg)),
        contract={"units": "uV"},
    )
    options.update(overrides)
    return EEGWindow(**options)


class EEGWindowTests(unittest.TestCase):
    def test_fields_are_kept(self) -> None:
        window = make_window()
        self.assertEqual(len(window), SAMPLES)
        self.assertEqual(window.data.shape, (SAMPLES, EEG_CHANNELS))
        self.assertEqual(window.eog.shape, (SAMPLES, EOG_CHANNELS))
        self.assertTrue(window.valid)
        self.assertEqual(window.reasons, ())
        self.assertEqual(window.start_sample, 128)
        self.assertEqual(len(window.channel_names), EEG_CHANNELS)
        self.assertEqual(window.contract, {"units": "uV"})
        # No judge reported a channel fault.
        self.assertEqual(window.bad_channels, ())

    def test_bad_channels_are_evidence_not_a_verdict(self) -> None:
        window = make_window(bad_channels=("E1", "E2"))
        self.assertEqual(window.bad_channels, ("E1", "E2"))
        # A tolerated dead electrode leaves the window usable.
        self.assertTrue(window.valid)
        self.assertEqual(window.reasons, ())
        self.assertIn("bad_channels", repr(window))

    def test_arrays_are_copies(self) -> None:
        source = np.ones((SAMPLES, EEG_CHANNELS))
        timestamps = np.arange(SAMPLES, dtype=np.float64)
        window = make_window(data=source, timestamps=timestamps)

        source[:] = -1.0
        timestamps[:] = -1.0

        self.assertTrue(np.all(window.data == 1.0))
        self.assertFalse(np.any(window.timestamps == -1.0))

    def test_auxiliary_may_be_absent(self) -> None:
        window = make_window(eog=0)
        self.assertEqual(window.eog.shape, (SAMPLES, 0))

    def test_validity_and_reasons_travel_together(self) -> None:
        window = make_window(valid=False, reasons=("warmup",))
        self.assertFalse(window.valid)
        self.assertIn("warmup", window.reasons)

    def test_validation(self) -> None:
        data = np.zeros((SAMPLES, EEG_CHANNELS))
        timestamps = np.arange(SAMPLES, dtype=np.float64)
        with self.assertRaises(ValueError):
            make_window(data=np.zeros((SAMPLES, EEG_CHANNELS, 1)))  # 3D data
        with self.assertRaises(ValueError):
            EEGWindow(  # eog rows differ from data rows
                data, np.zeros((SAMPLES - 1, EOG_CHANNELS)), timestamps,
                valid=True, reasons=(), start_sample=0,
            )
        with self.assertRaises(ValueError):
            make_window(timestamps=timestamps[:-1])  # short grid
        with self.assertRaises(ValueError):
            make_window(start_sample=-1)


if __name__ == "__main__":
    unittest.main()
