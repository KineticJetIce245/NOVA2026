"""Tests for bounded recovery (A2): Recovery guard + UnrepairableError."""

import unittest

import numpy as np

from nova2026.streaming import EEGWindow, Recovery, StreamStats
from nova2026.streaming.preprocess import Repair, UnrepairableError

SFREQ = 500.0


class Resettable:
    """A fake stateful component that counts its resets."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def window_at(start: float, end: float, reasons=()) -> EEGWindow:
    """A minimal EEGWindow over [start, end] with the given reasons."""

    rows = 128
    timestamps = np.linspace(start, end, rows)
    return EEGWindow(
        np.zeros((rows, 2)),
        np.zeros((rows, 1)),
        timestamps,
        valid=not reasons,
        reasons=reasons,
        start_sample=0,
    )


class RecoveryTests(unittest.TestCase):
    def test_recoverable_fault_resets_and_budget_is_bounded(self) -> None:
        first, second = Resettable(), Resettable()
        guard = Recovery((first, second), max_events=2)
        guard.handle(UnrepairableError("nonfinite_run"))
        self.assertEqual((first.resets, second.resets), (1, 1))
        self.assertEqual(guard.recoveries, 1)
        self.assertEqual(guard.segment, 1)
        self.assertEqual(guard.events[0]["kind"], "nonfinite_run")

        guard.handle(UnrepairableError("irregular_timestamps"))
        self.assertEqual(guard.segment, 2)
        # The third fault exceeds the budget: the run must stop.
        with self.assertRaisesRegex(RuntimeError, "Too many data faults"):
            guard.handle(UnrepairableError("nonfinite_run"))
        self.assertEqual(guard.recoveries, 3)

    def test_fatal_gap_stops_immediately(self) -> None:
        component = Resettable()
        guard = Recovery((component,))
        with self.assertRaisesRegex(RuntimeError, "gap exceeds"):
            guard.handle(UnrepairableError("large_gap", gap_seconds=2.0))
        self.assertEqual(guard.recoveries, 0)
        self.assertEqual(component.resets, 0)

    def test_small_gap_is_recoverable(self) -> None:
        guard = Recovery(())
        guard.handle(UnrepairableError("large_gap", gap_seconds=0.2))
        self.assertEqual(guard.recoveries, 1)

    def test_persistent_faults_stop_the_run(self) -> None:
        guard = Recovery((), persistent_fault_seconds=5.0)
        guard.watch(window_at(100.0, 100.5, reasons=("amplitude",)))
        # A clean window resets the clock.
        guard.watch(window_at(101.0, 101.5))
        guard.watch(window_at(110.0, 110.5, reasons=("flatline",)))
        with self.assertRaisesRegex(RuntimeError, "persisted"):
            guard.watch(window_at(116.0, 116.5, reasons=("flatline",)))

    def test_warmup_only_windows_are_ignored(self) -> None:
        guard = Recovery((), persistent_fault_seconds=0.1)
        for start in (0.0, 0.5, 1.0):  # reasons empty: never counts as bad
            guard.watch(window_at(start, start + 0.4))
        self.assertIsNone(guard._bad_since)

    def test_reset_starts_a_fresh_run(self) -> None:
        guard = Recovery(())
        guard.handle(UnrepairableError("nonfinite_run"))
        guard.watch(window_at(1.0, 1.5, reasons=("amplitude",)))
        guard.reset()
        self.assertEqual(guard.recoveries, 0)
        self.assertEqual(guard.segment, 0)
        self.assertEqual(guard.events, [])
        self.assertIsNone(guard._bad_since)

    def test_validation_and_input_errors(self) -> None:
        with self.assertRaises(ValueError):
            Recovery((), max_events=0)
        with self.assertRaises(ValueError):
            Recovery((), persistent_fault_seconds=-1.0)
        with self.assertRaises(TypeError):
            Recovery("not-an-iterable")
        with self.assertRaises(TypeError):
            Recovery(()).handle(RuntimeError("not unrepairable"))

    def test_components_must_implement_reset(self) -> None:
        guard = Recovery((object(),))  # no reset() method
        with self.assertRaisesRegex(RuntimeError, "reset"):
            guard.handle(UnrepairableError("nonfinite_run"))

    def test_repair_and_recovery_resume_after_a_bad_chunk(self) -> None:
        repair = Repair(SFREQ)
        guard = Recovery((repair,), max_events=2)
        offset = 1000.0

        good = np.arange(20, dtype=np.float64).reshape(-1, 1)
        out, _ = repair(good, offset + np.arange(20) / SFREQ)
        self.assertEqual(len(out), 20)

        # A 15-row non-finite run (limit is 10) is unrepairable; timestamps
        # continue on the same grid as the previous feed.
        bad = good.copy()
        bad[5:, 0] = np.nan
        times = offset + 20.0 / SFREQ + np.arange(20) / SFREQ
        with self.assertRaises(UnrepairableError) as caught:
            repair(bad, times)
        self.assertEqual(caught.exception.kind, "nonfinite_run")

        # Bounded recovery resets the stage and processing can continue.
        guard.handle(caught.exception)
        self.assertEqual(guard.segment, 1)
        self.assertEqual(repair.repaired_samples, 0)  # history was cleared
        out, _ = repair(
            good, offset + 40.0 / SFREQ + np.arange(20) / SFREQ
        )
        self.assertEqual(len(out), 20)


class StatsTests(unittest.TestCase):
    def test_stream_stats_counters_and_snapshot(self) -> None:
        stats = StreamStats()
        stats.blocks += 1
        stats.samples += 500
        stats.windows += 4
        stats.valid += 3
        stats.rejected += 1
        stats.recoveries = 2
        snapshot = stats.to_dict()
        self.assertEqual(snapshot["valid"], 3)
        self.assertEqual(snapshot["recoveries"], 2)
        self.assertEqual(snapshot["max_lag"], 0.0)


if __name__ == "__main__":
    unittest.main()
