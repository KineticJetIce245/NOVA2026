"""Tests for the SoXR streaming resampler stage."""

import unittest

import numpy as np

from nova2026.streaming.preprocess import Resampler

IN_SFREQ = 500.0
OUT_SFREQ = 128.0
CHANNELS = 4


def times(count: int, start: float = 0.0, sfreq: float = IN_SFREQ) -> np.ndarray:
    return start + np.arange(count) / sfreq


def feed_all(resampler: Resampler, data: np.ndarray, chunk_size: int):
    """Feed data in chunks of ``chunk_size`` and return the whole output."""

    pieces = []
    for start in range(0, len(data), chunk_size):
        block = data[start : start + chunk_size]
        out, _ = resampler(block, times(len(block), start=start / IN_SFREQ))
        pieces.append(out)
    return np.concatenate(pieces, axis=0)


class ResamplerTests(unittest.TestCase):
    def test_dc_signal_survives_unchanged(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        data = np.ones((6000, CHANNELS))

        output = feed_all(resampler, data, chunk_size=250)

        self.assertGreater(len(output), 0)
        # The very first output samples are a 0 -> 1 step transient while SoXR
        # fills its filter history; the settled tail must equal the DC input.
        tail = output[-512:]
        self.assertTrue(np.allclose(tail, 1.0, atol=1e-3))

    def test_output_rate_matches_the_ratio(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS)
        rng = np.random.default_rng(0)
        count = 8000

        output = feed_all(resampler, rng.standard_normal((count, CHANNELS)), 200)

        # A streaming resampler retains its tail until more input arrives, so a
        # finite run under-reports the exact ratio by a bounded amount.
        expected = count * OUT_SFREQ / IN_SFREQ
        self.assertAlmostEqual(len(output) / expected, 1.0, delta=0.05)
        self.assertEqual(output.shape[1], CHANNELS)

    def test_timestamps_form_an_output_grid(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS)
        pieces = []
        for start in range(0, 4000, 500):
            block = np.ones((500, CHANNELS))
            _, output_times = resampler(block, times(len(block), start=start / IN_SFREQ))
            pieces.append(output_times)
        joined = np.concatenate(pieces)

        self.assertTrue(np.all(np.diff(joined) > 0))
        self.assertTrue(
            np.allclose(np.diff(joined), 1.0 / OUT_SFREQ, atol=1e-9)
        )

    def test_reset_restarts_the_grid(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS)
        feed_all(resampler, np.ones((1000, CHANNELS)), 500)

        resampler.reset()
        self.assertEqual(resampler.output_samples, 0)
        _, output_times = resampler(np.ones((6000, CHANNELS)), times(6000, start=10.0))
        self.assertGreater(len(output_times), 0)
        self.assertGreaterEqual(output_times[0], 10.0 - 1e-9)
        self.assertTrue(np.all(np.diff(output_times) > 0))

    def test_empty_chunk_is_a_noop(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS)
        data, output_times = resampler(np.empty((0, CHANNELS)), np.empty(0))
        self.assertEqual(data.shape, (0, CHANNELS))
        self.assertEqual(output_times.shape, (0,))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            Resampler(0.0, OUT_SFREQ, CHANNELS)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, IN_SFREQ, CHANNELS)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, 0)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="ZZ")

        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS)
        with self.assertRaises(ValueError):
            resampler(np.zeros((5, CHANNELS + 1)), times(5))
        with self.assertRaises(ValueError):
            resampler(np.zeros((5, CHANNELS)), times(4))


if __name__ == "__main__":
    unittest.main()
