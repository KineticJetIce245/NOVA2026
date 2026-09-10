"""Tests for the preprocessing stages."""

import unittest

import numpy as np
from scipy.signal import sosfilt

from nova2026.streaming.preprocess import (
    QualityMonitor,
    SosFilter,
    design_bandpass,
    design_notch,
    unit_scaler,
)

SFREQ = 500.0
CHANNELS = 4


def noise(count: int, channels: int = CHANNELS, seed: int = 0) -> np.ndarray:
    """Return deterministic noise samples by channels."""

    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, channels))


def times(count: int, start: float = 1000.0, sfreq: float = SFREQ) -> np.ndarray:
    """Return a uniform timestamp grid."""

    return start + np.arange(count) / sfreq


class ScaleTests(unittest.TestCase):
    """Unit conversion is a pure scaling stage."""

    def test_volts_to_microvolts(self) -> None:
        scale = unit_scaler(0)
        data, timestamps = scale(np.ones((10, CHANNELS)), times(10))
        self.assertTrue(np.allclose(data, 1e6))
        self.assertEqual(len(timestamps), 10)

    def test_same_units_are_identity(self) -> None:
        scale = unit_scaler(-6)
        data, _ = scale(np.full((10, CHANNELS), 3.0), times(10))
        self.assertTrue(np.allclose(data, 3.0))

    def test_invalid_exponent(self) -> None:
        with self.assertRaises(ValueError):
            unit_scaler(-2)


class SosFilterTests(unittest.TestCase):
    """Causal filtering keeps state across variable-size chunks."""

    def setUp(self) -> None:
        self.sos = design_bandpass(1.0, 45.0, 3, SFREQ)
        self.signal = noise(2000) + np.sin(2 * np.pi * 10 * np.arange(2000)[:, None] / SFREQ)

    def filter_stream(self, chunk_sizes, sos=None) -> np.ndarray:
        sos = sos or self.sos
        stage = SosFilter(sos, CHANNELS)
        pieces = []
        start = 0
        index = 0
        while start < len(self.signal):
            size = min(chunk_sizes[index % len(chunk_sizes)], len(self.signal) - start)
            block = self.signal[start : start + size]
            out, _ = stage(block, times(size, start=start / SFREQ))
            pieces.append(out)
            start += size
            index += 1
        return np.concatenate(pieces, axis=0)

    def test_chunking_does_not_change_the_output(self) -> None:
        one_shot = SosFilter(self.sos, CHANNELS)
        whole, _ = one_shot(self.signal, times(len(self.signal)))

        rng = np.random.default_rng(3)
        sizes = [int(rng.integers(1, 300)) for _ in range(30)]
        chopped = self.filter_stream(sizes)

        self.assertTrue(np.allclose(whole, chopped, atol=1e-9))

    def test_reset_restarts_from_zero(self) -> None:
        stage = SosFilter(self.sos, CHANNELS)
        first, _ = stage(self.signal, times(len(self.signal)))
        stage.reset()

        replay, _ = stage(self.signal, times(len(self.signal)))
        self.assertTrue(np.allclose(first, replay))

    def test_empty_chunk_is_a_noop(self) -> None:
        stage = SosFilter(self.sos, CHANNELS)
        data, timestamps = stage(np.empty((0, CHANNELS)), np.empty(0))
        self.assertEqual(data.shape, (0, CHANNELS))

    def test_design_validation(self) -> None:
        with self.assertRaises(ValueError):
            design_bandpass(100.0, 45.0, 3, SFREQ)
        with self.assertRaises(ValueError):
            design_bandpass(1.0, 45.0, 0, SFREQ)
        with self.assertRaises(ValueError):
            design_notch(300.0, 30.0, SFREQ)
        with self.assertRaises(ValueError):
            SosFilter(np.zeros((2, 5)), CHANNELS)

    def test_shape_validation(self) -> None:
        stage = SosFilter(design_notch(60.0, 30.0, SFREQ), CHANNELS)
        with self.assertRaises(ValueError):
            stage(np.zeros((5, CHANNELS + 1)), times(5))
        with self.assertRaises(ValueError):
            stage(np.zeros((5, CHANNELS)), times(4))


class QualityTests(unittest.TestCase):
    """Fault detection is anchored in timestamp space, not chunk boundaries."""

    def monitor(self, **overrides) -> QualityMonitor:
        options = dict(
            n_eeg=CHANNELS,
            sfreq=SFREQ,
            amplitude_limit_uv=500.0,
            saturation_limit_uv=75000.0,
            flatline_seconds=0.2,
            flatline_tolerance_uv=1e-3,
            warmup_seconds=0.0,
        )
        options.update(overrides)
        return QualityMonitor(**options)

    def test_amplitude_spike_is_reported(self) -> None:
        monitor = self.monitor()
        data = np.zeros((80, CHANNELS))
        data[30:35, 0] = 600.0  # excursion over the 500 uV limit
        monitor.feed(data, times(len(data)))

        reasons = monitor.reasons(times(80)[0], times(80)[-1])
        self.assertIn("amplitude", reasons)

    def test_spike_outside_window_is_not_reported(self) -> None:
        monitor = self.monitor()
        baseline = 20.0 * np.sin(2 * np.pi * 10 * np.arange(100)[:, None] / SFREQ)
        data = np.repeat(baseline, CHANNELS, axis=1)
        data[80:85, 0] += 600.0
        monitor.feed(data, times(len(data)))

        # A window that ends before the fault is clean, and the fault stays in
        # history for the window covering it.
        self.assertEqual(monitor.reasons(times(100)[50], times(100)[60]), ())
        self.assertIn("amplitude", monitor.reasons(times(100)[80], times(100)[84]))

    def test_saturation_is_reported(self) -> None:
        monitor = self.monitor()
        data = np.zeros((100, CHANNELS))
        data[10:12, :] = 1e5  # past the 75000 uV rail
        monitor.feed(data, times(len(data)))

        self.assertIn("saturation", monitor.reasons(times(100)[0], times(100)[-1]))

    def test_flatline_is_reported(self) -> None:
        monitor = self.monitor(flatline_seconds=0.2)
        data = np.zeros((100, CHANNELS))  # dead channels
        monitor.feed(data, times(len(data)))

        self.assertIn("flatline", monitor.reasons(times(100)[0], times(100)[-1]))

    def test_clean_data_never_reports(self) -> None:
        monitor = self.monitor()
        signal = np.repeat(
            20.0 * np.sin(2 * np.pi * 10 * np.arange(1000) / SFREQ)[:, None],
            CHANNELS,
            axis=1,
        )
        monitor.feed(signal, times(len(signal)))

        self.assertEqual(monitor.reasons(times(1000)[0], times(1000)[-1]), ())

    def test_fault_survives_chunk_boundaries(self) -> None:
        monitor = self.monitor()
        clean = np.zeros((40, CHANNELS))
        bad = np.zeros((20, CHANNELS))
        bad[:, 0] = 700.0
        tail = np.zeros((40, CHANNELS))
        times_ = times(100)

        monitor.feed(clean, times_[:40])
        monitor.feed(bad, times_[40:60])
        monitor.feed(tail, times_[60:])

        self.assertIn("amplitude", monitor.reasons(times_[0], times_[-1]))

    def test_nan_rows_do_not_crash_or_flag(self) -> None:
        monitor = self.monitor()
        data = np.zeros((50, CHANNELS))
        data[10:15, :] = np.nan
        monitor.feed(data, times(len(data)))

        self.assertEqual(monitor.reasons(times(50)[0], times(50)[-1]), ())

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.monitor(n_eeg=0)
        with self.assertRaises(ValueError):
            self.monitor(sfreq=0)
        with self.assertRaises(ValueError):
            self.monitor(saturation_limit_uv=10.0, amplitude_limit_uv=500.0)

        monitor = self.monitor()
        with self.assertRaises(ValueError):
            monitor.feed(np.zeros((5, CHANNELS - 1)), times(5))
        with self.assertRaises(ValueError):
            monitor.feed(np.zeros((5, CHANNELS)), times(4))


if __name__ == "__main__":
    unittest.main()
