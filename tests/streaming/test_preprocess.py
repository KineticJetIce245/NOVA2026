"""Tests for the preprocessing stages."""

import unittest

import numpy as np
from scipy.signal import sosfilt

from nova2026.streaming.preprocess import (
    BadChannelJudge,
    ChannelScope,
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


NAMES = ("F3", "F4", "C3", "C4")


class BadChannelTests(unittest.TestCase):
    """Channel identity survives into the verdict and into the evidence."""

    def monitor(self, **overrides) -> QualityMonitor:
        options = dict(
            n_eeg=CHANNELS,
            sfreq=SFREQ,
            channel_names=NAMES,
            flatline_seconds=0.2,
            warmup_seconds=0.0,
        )
        options.update(overrides)
        return QualityMonitor(**options)

    def flat_channel(self, index: int, count: int = 100) -> np.ndarray:
        """Deterministic signal with one dead (perfectly flat) electrode."""

        data = noise(count, seed=4) + 20.0
        data[:, index] = 0.0
        return data

    def test_faulting_channel_is_named(self) -> None:
        monitor = self.monitor()
        data = self.flat_channel(1)
        monitor.feed(data, times(len(data)))

        self.assertEqual(monitor.reasons(times(100)[0], times(100)[-1]),
                         ("flatline",))
        self.assertEqual(monitor.bad_channels(times(100)[0], times(100)[-1]),
                         ("F4",))
        self.assertEqual(
            monitor.fault_channels(times(100)[0], times(100)[-1]),
            {"flatline": ("F4",)},
        )

    def test_channels_without_names_fall_back_to_column_labels(self) -> None:
        monitor = QualityMonitor(
            n_eeg=CHANNELS, sfreq=SFREQ, flatline_seconds=0.2, warmup_seconds=0.0
        )
        data = self.flat_channel(2)
        monitor.feed(data, times(len(data)))

        self.assertEqual(monitor.bad_channels(times(100)[0], times(100)[-1]),
                         ("ch2",))

    def test_default_policy_rejects_on_one_channel(self) -> None:
        """The historical verdict: any faulty channel rejects the window."""

        monitor = self.monitor()
        self.assertEqual(monitor.max_bad_channels, 0)
        self.assertTrue(monitor.check_channels)
        data = self.flat_channel(3)
        monitor.feed(data, times(len(data)))

        self.assertIn("flatline", monitor.reasons(times(100)[0], times(100)[-1]))

    def test_max_bad_channels_tolerates_a_census(self) -> None:
        monitor = self.monitor(max_bad_channels=1)

        # One dead electrode is inside the budget: evidence, no rejection.
        first = times(100, start=1000.0)
        monitor.feed(self.flat_channel(0), first)
        self.assertEqual(monitor.reasons(first[0], first[-1]), ())
        self.assertEqual(monitor.bad_channels(first[0], first[-1]), ("F3",))

        # A second one exceeds it, so the fault type rejects again.
        second = times(120, start=1100.0)
        two = self.flat_channel(0, count=120)
        two[:, 2] = 0.0
        monitor.feed(two, second)
        self.assertEqual(monitor.reasons(second[0], second[-1]), ("flatline",))
        self.assertEqual(monitor.bad_channels(second[0], second[-1]), ("F3", "C3"))

    def test_excluded_channels_never_reject_but_stay_visible(self) -> None:
        monitor = self.monitor(exclude_channels=("F3",))

        first = times(100, start=1000.0)
        monitor.feed(self.flat_channel(0), first)
        self.assertEqual(monitor.reasons(first[0], first[-1]), ())
        # Recorded even though the operator declared it dead.
        self.assertEqual(monitor.bad_channels(first[0], first[-1]), ("F3",))

        # A second, unknown electrode still rejects.
        second = times(120, start=1100.0)
        two = self.flat_channel(0, count=120)
        two[:, 1] = 0.0
        monitor.feed(two, second)
        self.assertEqual(monitor.reasons(second[0], second[-1]), ("flatline",))
        self.assertEqual(monitor.bad_channels(second[0], second[-1]), ("F3", "F4"))

    def test_check_channels_false_records_without_rejecting(self) -> None:
        monitor = self.monitor(check_channels=False)
        data = self.flat_channel(0)
        data[:, 1] = 1e5  # rail a second channel too
        monitor.feed(data, times(100))
        span = (times(100)[0], times(100)[-1])

        self.assertEqual(monitor.reasons(*span), ())
        self.assertEqual(monitor.bad_channels(*span), ("F3", "F4"))

        # A silent monitor still prunes its history at a segment boundary.
        monitor.reset()
        self.assertEqual(monitor.bad_channels(*span), ())

    def test_channel_identity_survives_chunk_boundaries(self) -> None:
        monitor = self.monitor()
        data = self.flat_channel(2, count=120)
        stamps = times(120, start=500.0)

        monitor.feed(data[:40], stamps[:40])
        monitor.feed(data[40:80], stamps[40:80])
        monitor.feed(data[80:], stamps[80:])

        self.assertEqual(monitor.bad_channels(stamps[0], stamps[-1]), ("C3",))

    def test_only_the_faulting_window_names_the_channel(self) -> None:
        monitor = self.monitor()
        clean = noise(60, seed=1) + 20.0
        bad = self.flat_channel(0, count=120)
        monitor.feed(clean, times(60, start=500.0))
        monitor.feed(bad, times(120, start=500.0 + 60 / SFREQ))
        monitor.feed(clean, times(60, start=500.0 + 180 / SFREQ))

        # The clean span before the fault carries nothing.
        self.assertEqual(
            monitor.bad_channels(500.0, 500.0 + 50 / SFREQ), ()
        )
        # The fault needs its flatline run before it is reported, so the span
        # from its detection onward names the dead electrode.
        self.assertEqual(
            monitor.bad_channels(500.0 + 60 / SFREQ, 500.0 + 178 / SFREQ), ("F3",)
        )

    def test_policy_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.monitor(check_channels=1)
        with self.assertRaises(ValueError):
            self.monitor(max_bad_channels=-1)
        with self.assertRaises(ValueError):
            self.monitor(max_bad_channels=True)
        # Tolerating every channel would disable the guard silently.
        with self.assertRaises(ValueError):
            self.monitor(max_bad_channels=CHANNELS)
        with self.assertRaises(ValueError):
            self.monitor(channel_names=NAMES[:-1])
        with self.assertRaises(TypeError):
            self.monitor(channel_names="F3")
        with self.assertRaises(TypeError):
            self.monitor(exclude_channels="F3")
        # A misspelled exclusion must not silently exclude nothing.
        with self.assertRaises(ValueError):
            self.monitor(exclude_channels=("Fz",))
        with self.assertRaises(ValueError):
            QualityMonitor(n_eeg=CHANNELS, sfreq=SFREQ, exclude_channels=("F3",))


class BadChannelJudgeTests(unittest.TestCase):
    """The plugin judge watches by default and only blocks when asked."""

    def setUp(self) -> None:
        self.monitor = QualityMonitor(
            n_eeg=CHANNELS, sfreq=SFREQ, channel_names=NAMES,
            flatline_seconds=0.2, warmup_seconds=0.0,
        )
        data = noise(100, seed=4) + 20.0
        data[:, 1] = 0.0
        self.monitor.feed(data, times(100))
        self.span = (times(100)[0], times(100)[-1])

    def test_observe_only_by_default(self) -> None:
        judge = BadChannelJudge(self.monitor)
        self.assertEqual(judge.reasons(*self.span), ())
        self.assertEqual(judge.bad_channels(*self.span), ("F4",))

    def test_block_on_census(self) -> None:
        judge = BadChannelJudge(
            self.monitor, block_on_bad_channels=True, min_channels=2
        )
        # One bad channel is below the census threshold...
        self.assertEqual(judge.reasons(*self.span), ())
        # ...but the monitor's own evidence is always reported.
        self.assertEqual(judge.bad_channels(*self.span), ("F4",))

        strict = BadChannelJudge(
            self.monitor, block_on_bad_channels=True, min_channels=1
        )
        self.assertEqual(strict.reasons(*self.span), ("bad_channels",))

    def test_validation(self) -> None:
        with self.assertRaises(TypeError):
            BadChannelJudge("not a monitor")
        with self.assertRaises(TypeError):
            BadChannelJudge(self.monitor, block_on_bad_channels=1)
        with self.assertRaises(ValueError):
            BadChannelJudge(self.monitor, min_channels=0)
        with self.assertRaises(ValueError):
            BadChannelJudge(self.monitor, reason="")


class ChannelScopeTests(unittest.TestCase):
    """Scope resolves labels to columns and fails loudly on typos."""

    def test_mask_excludes_only_named_columns(self) -> None:
        scope = ChannelScope(NAMES, ("F4",))
        self.assertTrue(scope)
        self.assertEqual(scope.excluded_indices, frozenset({1}))
        self.assertTrue(scope.contains(1))
        self.assertFalse(scope.contains(0))
        mask = scope.mask(CHANNELS)
        self.assertEqual(list(mask), [True, False, True, True])
        self.assertEqual(list(scope.excluded_mask(CHANNELS)), [False, True, False, False])

    def test_auxiliary_columns_stay_in_scope(self) -> None:
        scope = ChannelScope(NAMES, ("F4",))
        # A column beyond the montage is not part of the exclusion.
        self.assertEqual(list(scope.mask(CHANNELS + 2)),
                         [True, False, True, True, True, True])

    def test_empty_scope_is_falsey_and_inclusive(self) -> None:
        scope = ChannelScope(NAMES)
        self.assertFalse(scope)
        self.assertTrue(scope.mask(CHANNELS).all())

    def test_validation(self) -> None:
        with self.assertRaises(TypeError):
            ChannelScope("F3")
        with self.assertRaises(TypeError):
            ChannelScope(NAMES, "F4")
        with self.assertRaises(ValueError):
            ChannelScope(NAMES, ("Fz",))
        with self.assertRaises(ValueError):
            ChannelScope(None, ("F3",))
        with self.assertRaises(ValueError):
            ChannelScope(NAMES).mask(0)


if __name__ == "__main__":
    unittest.main()
