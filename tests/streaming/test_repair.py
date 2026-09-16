"""Tests for the Repair stage: short NaN/Inf repair + judge interface."""

import unittest

import numpy as np

from nova2026.streaming.preprocess import Repair

SFREQ = 500.0
INTERVAL = 1.0 / SFREQ


def ramp(count: int, channels: int = 1, start: float = 0.0) -> np.ndarray:
    """Deterministic rows where channel 0 equals 10 * row index.

    Linear rows make interpolation outcomes exact and easy to assert.
    """

    index = np.arange(count, dtype=np.float64)
    columns = [10.0 * index]
    for column in range(1, channels):
        columns.append(100.0 * column + 5.0 * index)
    return np.stack(columns, axis=1) + start


def grid(count: int, start: float = 1000.0) -> np.ndarray:
    """Uniform timestamp grid on the nominal sample interval."""

    return start + np.arange(count) / SFREQ


class RepairTests(unittest.TestCase):
    def test_clean_data_passes_unchanged(self) -> None:
        repair = Repair(SFREQ)
        data = ramp(30, channels=2)
        times = grid(30)
        out, out_times = repair(data, times)
        self.assertTrue(np.allclose(out, data))
        self.assertTrue(np.allclose(out_times, times))
        self.assertEqual(repair.repaired_samples, 0)
        # Nothing repaired: no reason over the whole span.
        self.assertEqual(repair.reasons(float(times[0]), float(times[-1])), ())
        # An empty block is a harmless no-op (stages may return zero rows).
        empty = np.empty((0, 2))
        out, out_times = repair(empty, np.empty((0,)))
        self.assertEqual(out.shape, (0, 2))

    def test_single_nan_row_repaired_linearly_per_channel(self) -> None:
        repair = Repair(SFREQ)
        times = grid(5)
        data = ramp(5, channels=2)
        data[2, 0] = np.nan  # only channel 0 is broken on that row
        out, _ = repair(data, times)
        self.assertEqual(out.shape, (5, 2))
        # Broken channel: linear midpoint between 10 and 30 -> 20.
        self.assertTrue(np.allclose(out[:, 0], [0.0, 10.0, 20.0, 30.0, 40.0]))
        # Healthy channel: untouched.
        self.assertTrue(np.allclose(out[:, 1], data[:, 1]))
        self.assertEqual(repair.repaired_samples, 1)
        self.assertEqual(repair.reasons(float(times[2]), float(times[2]) + 1e-6),
                         ("interpolated",))

    def test_run_of_three_nan_rows_is_one_repair(self) -> None:
        repair = Repair(SFREQ)
        times = grid(5)
        data = ramp(5)
        data[1:4, 0] = np.nan
        out, _ = repair(data, times)
        self.assertTrue(np.allclose(out[:, 0], [0.0, 10.0, 20.0, 30.0, 40.0]))
        self.assertEqual(repair.repaired_samples, 3)

    def test_missing_samples_are_synthesised_from_a_timestamp_gap(self) -> None:
        repair = Repair(SFREQ)
        # Two rows never arrive: the source jumps from index 0 to index 3.
        data = ramp(5)[[0, 3, 4]]
        times = grid(5)[[0, 3, 4]]
        out, out_times = repair(data, times)
        # The missing rows are recreated on the grid and linearly repaired.
        self.assertEqual(len(out), 5)
        self.assertTrue(np.allclose(out[:, 0], np.arange(5) * 10.0))
        self.assertTrue(np.allclose(out_times, grid(5)))
        self.assertEqual(repair.repaired_samples, 2)

    def test_trailing_damage_waits_for_the_right_endpoint(self) -> None:
        repair = Repair(SFREQ)
        # Feed 1: rows 0..3 finite, rows 4 and 5 damaged (no endpoint yet).
        data1 = ramp(6)
        data1[4:, 0] = np.nan
        times1 = grid(6)
        out1, out_times1 = repair(data1, times1)
        self.assertEqual(len(out1), 4)  # only the clean prefix is released
        self.assertTrue(np.allclose(out1[:, 0], [0.0, 10.0, 20.0, 30.0]))
        self.assertEqual(repair.repaired_samples, 0)

        # Feed 2: the grid continues and supplies the finite right endpoint.
        data2 = ramp(3, start=60.0)
        times2 = grid(3, start=1000.0 + 6 * INTERVAL)
        out2, out_times2 = repair(data2, times2)
        # Repaired rows 4 and 5 come out first, then the three new rows.
        self.assertEqual(len(out2), 5)
        self.assertTrue(np.allclose(out2[:, 0], [40.0, 50.0, 60.0, 70.0, 80.0]))
        self.assertEqual(repair.repaired_samples, 2)
        # Full timeline across both calls is one continuous grid.
        combined = np.concatenate([out1, out2])
        self.assertEqual(len(combined), 9)

    def test_damage_longer_than_the_limit_raises(self) -> None:
        repair = Repair(SFREQ)  # limit = 10 samples at 500 Hz
        data = ramp(18)
        data[5:, 0] = np.nan  # 13 damaged rows in a row, no endpoint
        times = grid(18)
        with self.assertRaises(RuntimeError):
            repair(data, times)

        short = Repair(SFREQ, max_seconds=0.004)  # limit = 2 samples
        data = ramp(6)
        data[3:, 0] = np.nan
        with self.assertRaises(RuntimeError):
            short(data, grid(6))

    def test_irregular_or_nonfinite_timestamps_raise(self) -> None:
        data = ramp(5)
        # Off-grid jump (1.3 samples): cannot trust where rows really belong.
        times = 1000.0 + np.array([0.0, 1.0, 2.3, 3.0, 4.0]) / SFREQ
        with self.assertRaises(RuntimeError):
            Repair(SFREQ)(data, times)
        # A non-finite timestamp can never be repaired.
        times = grid(5)
        times[1] = np.nan
        with self.assertRaises(RuntimeError):
            Repair(SFREQ)(data, times)
        # Overlapping timestamps are not on the nominal grid either.
        times = grid(5)
        times[2] = times[1]
        with self.assertRaises(RuntimeError):
            Repair(SFREQ)(data, times)

    def test_leading_damage_before_the_first_finite_row_is_dropped(self) -> None:
        repair = Repair(SFREQ)
        data = ramp(6)
        data[:3, 0] = np.nan  # damaged before any finite anchor exists
        out, _ = repair(data, grid(6))
        # The run effectively starts at the first finite row.
        self.assertEqual(len(out), 3)
        self.assertTrue(np.allclose(out[:, 0], [30.0, 40.0, 50.0]))
        self.assertEqual(repair.dropped_rows, 3)
        self.assertEqual(repair.repaired_samples, 0)

    def test_unsafe_endpoints_refuse_to_invent_signal(self) -> None:
        # Units known (volts): a 2 V jump is far beyond any EEG excursion.
        repair = Repair(SFREQ, source_unit_exponent=0)
        data = np.array([[0.0], [np.nan], [2.0]])
        times = grid(3)
        with self.assertRaises(RuntimeError):
            repair(data, times)

    def test_separate_repairs_and_expired_history(self) -> None:
        repair = Repair(SFREQ)
        data = ramp(9)
        data[2, 0] = np.nan
        data[6, 0] = np.nan
        times = grid(9)
        out, _ = repair(data, times)
        self.assertEqual(len(out), 9)
        self.assertEqual(repair.repaired_samples, 2)
        # A window overlapping either repair is flagged.
        self.assertEqual(repair.reasons(float(times[1]), float(times[3])),
                         ("interpolated",))
        # Querying far past every repair drops the expired history; the next
        # query over a past span is then empty because it was pruned.
        self.assertEqual(
            repair.reasons(float(times[-1]) + 1.0, float(times[-1]) + 2.0), ()
        )
        self.assertEqual(repair.reasons(float(times[1]), float(times[3])), ())

    def test_reset_clears_pending_rows_and_history(self) -> None:
        repair = Repair(SFREQ)
        data = ramp(6)
        data[4:, 0] = np.nan
        repair(data, grid(6))  # leaves rows 4-5 pending
        repair.reset()
        self.assertEqual(repair.repaired_samples, 0)
        self.assertEqual(repair.dropped_rows, 0)
        out, _ = repair(ramp(4), grid(4))
        self.assertEqual(len(out), 4)

    def test_excluded_channel_that_never_delivers_is_held_not_awaited(self) -> None:
        """A dead electrode must not make Repair wait for an endpoint forever."""

        names = ("A", "B")
        repair = Repair(SFREQ, channel_names=names, exclude_channels=("B",))
        data = ramp(6, channels=2)
        data[:, 1] = np.nan

        out, _ = repair(data, grid(6))

        # Everything is released immediately and nothing non-finite leaks out.
        self.assertEqual(len(out), 6)
        self.assertTrue(np.all(np.isfinite(out)))
        # The healthy channel is untouched; the dead one holds its last level.
        self.assertTrue(np.allclose(out[:, 0], data[:, 0]))
        self.assertTrue(np.allclose(out[:, 1], 0.0))
        self.assertEqual(repair.held_rows, 6)
        self.assertEqual(repair.repaired_samples, 0)
        self.assertEqual(repair.reasons(float(grid(6)[0]), float(grid(6)[-1])), ())

        # Without the scope the same data is unrepairable pending damage.
        plain = Repair(SFREQ)
        pending, _ = plain(data, grid(6))
        self.assertEqual(len(pending), 0)
        self.assertEqual(plain.held_rows, 0)

    def test_excluded_channel_cannot_make_a_repair_unsafe(self) -> None:
        """One railing dead electrode must not veto every repair around it."""

        names = ("A", "B")
        # Volts: A rails from 0 V to 2 V, far beyond any EEG excursion.
        data = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2e-5]])
        stamps = grid(3)

        with self.assertRaises(RuntimeError):
            Repair(SFREQ, source_unit_exponent=0)(data, stamps)

        scoped = Repair(
            SFREQ, source_unit_exponent=0, channel_names=names,
            exclude_channels=("A",),
        )
        out, _ = scoped(data, stamps)
        self.assertEqual(len(out), 3)
        self.assertTrue(np.all(np.isfinite(out)))
        # B is bridged linearly between its two finite endpoints.
        self.assertAlmostEqual(out[1, 1], 1e-5)

    def test_scope_validation(self) -> None:
        with self.assertRaises(ValueError):
            Repair(SFREQ, channel_names=("A", "B"), exclude_channels=("C",))
        with self.assertRaises(ValueError):
            Repair(SFREQ, exclude_channels=("A",))
        with self.assertRaises(TypeError):
            Repair(SFREQ, channel_names=("A", "B"), exclude_channels="A")


if __name__ == "__main__":
    unittest.main()
