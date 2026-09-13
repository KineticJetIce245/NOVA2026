"""Tests for the CNT-versus-recording check used to validate bring-up runs.

Everything here is hardware-free: the reference "CNT" is a synthetic array and
the recorded runs are synthetic SQLite files written in the recorder's own
schema. The real ``.cnt`` reader is exercised only through the ``.npy`` path,
so the tests do not need ``antio``.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.getlive.compare_cnt import (
    AMP_LSB_UV,
    compare,
    find_offset,
    grid_deviation,
    load_cnt,
    load_run,
    main,
)

CHANNELS = ("F3", "F4", "C3", "C4")
SFREQ = 500.0
N_CHANNELS = len(CHANNELS)


def synthetic_cnt(n_samples: int = 4000, seed: int = 7) -> np.ndarray:
    """Build a deterministic multi-channel trace with a unique shape.

    Random noise is used on purpose: a periodic reference would correlate
    equally well at several lags and could not prove the alignment is exact.
    """

    rng = np.random.default_rng(seed)
    offsets = 1000.0 * np.arange(1, N_CHANNELS + 1)[:, None]
    noise = rng.normal(0, 20, size=(N_CHANNELS, n_samples))
    return (offsets + noise).astype(np.float32)


def write_run(run_dir: Path, data: np.ndarray, *, block: int = 50) -> Path:
    """Write one run in the recorder's schema and return its database path.

    The recorder stores samples by channels (time-major), which is the layout
    :func:`load_run` transposes back into channels by samples.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{run_dir.name}.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE chunks(seq INTEGER PRIMARY KEY AUTOINCREMENT,"
        "n_samples INTEGER NOT NULL, first_timestamp REAL, data BLOB NOT NULL)"
    )
    # A recorded run always carries its metadata: the library's reader takes the
    # stored dtype from it, so a fixture without one is not a run it can read.
    connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO meta(key, value) VALUES ('dtype', ?)", ('"<f4"',))
    sizes = [block] * (data.shape[1] // block)
    remainder = data.shape[1] - sum(sizes)
    if remainder:
        sizes.append(remainder)
    start = 0
    for index, size in enumerate(sizes):
        stamp = 1_000_000.0 + index * block / SFREQ
        block_data = data[:, start : start + size].T.astype("<f4")
        connection.execute(
            "INSERT INTO chunks(n_samples, first_timestamp, data) VALUES (?,?,?)",
            (size, stamp, block_data.tobytes()),
        )
        start += size
    connection.commit()
    connection.close()
    return path


class OffsetTests(unittest.TestCase):
    def test_identical_data_aligns_exactly(self) -> None:
        cnt = synthetic_cnt()
        record = cnt[:, 1234 : 1234 + 500]
        offset, score = find_offset(record, cnt)
        self.assertEqual(offset, 1234)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_unrelated_data_does_not_align(self) -> None:
        # A signal sharing no samples with the reference must not reach a
        # convincing score, or the alignment would be trusted when it is wrong.
        cnt = synthetic_cnt()
        rng = np.random.default_rng(99)
        record = (rng.normal(0, 20, size=(N_CHANNELS, 500)) + 2500.0).astype(np.float32)
        _, score = find_offset(record, cnt)
        self.assertLess(score, 0.5)


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)
        self.cnt = synthetic_cnt()
        self.offset = 900

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, data: np.ndarray, name: str = "run-x", **kwargs) -> Path:
        run_dir = self.root / name
        write_run(run_dir, data, **kwargs)
        return run_dir

    def check(self, record: np.ndarray, name: str = "run-x", reference=None) -> dict:
        """Build a run around ``record`` and compare it with the reference."""
        run_dir = self.build(record, name)
        reference = self.cnt if reference is None else reference
        return compare(
            run_dir, np.asarray(reference, dtype=np.float64), SFREQ, CHANNELS
        )

    def test_identical_run_reports_no_outliers(self) -> None:
        record = self.cnt[:, self.offset : self.offset + 1000]
        report = self.check(record)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["offset"], self.offset)
        self.assertEqual(report["outliers"], 0)
        self.assertEqual(report["worst_channel"], "F3")
        self.assertLess(report["max_abs_uv"], AMP_LSB_UV)

    def test_amplifier_lsb_rounding_is_not_counted_as_a_mismatch(self) -> None:
        # The COUNT file quantises to one LSB while the LSL path does not, so a
        # genuine recording differs by up to half an LSB everywhere. That is
        # still the same signal and must not be reported as a mismatch.
        record = self.cnt[:, self.offset : self.offset + 1000]
        quantised = (np.round(record / AMP_LSB_UV) * AMP_LSB_UV).astype(np.float64)
        report = self.check(record, reference=quantised)
        self.assertEqual(report["outliers"], 0)
        self.assertLessEqual(report["max_abs_uv"], AMP_LSB_UV)

    def test_a_single_corrupted_sample_is_caught(self) -> None:
        record = self.cnt[:, self.offset : self.offset + 1000].copy()
        record[2, 500] += 5.0
        report = self.check(record)
        self.assertEqual(report["outliers"], 1)
        self.assertEqual(report["worst_channel"], "C3")

    def test_channel_swap_is_caught(self) -> None:
        record = self.cnt[:, self.offset : self.offset + 1000].copy()
        record[[0, 1]] = record[[1, 0]]
        report = self.check(record)
        self.assertGreater(report["outliers"], 0)

    def test_empty_run_is_reported_not_crashed(self) -> None:
        run_dir = self.root / "run-empty"
        run_dir.mkdir(parents=True)
        connection = sqlite3.connect(run_dir / "run-empty.sqlite")
        connection.execute(
            "CREATE TABLE chunks(seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "n_samples INTEGER NOT NULL, first_timestamp REAL, data BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('dtype', ?)", ('"<f4"',)
        )
        connection.commit()
        connection.close()
        report = compare(run_dir, self.cnt.astype(np.float64), SFREQ, CHANNELS)
        self.assertEqual(report["status"], "no-samples")
        self.assertIsNone(load_run(run_dir / "run-empty.sqlite")[0])

    def test_missing_database_is_reported(self) -> None:
        run_dir = self.root / "run-absent"
        run_dir.mkdir(parents=True)
        report = compare(run_dir, self.cnt.astype(np.float64), SFREQ, CHANNELS)
        self.assertEqual(report["status"], "missing-sqlite")


class GridTests(unittest.TestCase):
    def test_perfect_grid_has_no_deviation(self) -> None:
        sizes = np.full(20, 50)
        times = 1000.0 + np.arange(20) * (50 / SFREQ)
        self.assertAlmostEqual(grid_deviation(times, sizes, SFREQ), 0.0, places=9)

    def test_regridded_grid_deviation_is_measured(self) -> None:
        sizes = np.full(20, 50)
        times = 1000.0 + np.arange(20) * (50 / SFREQ) * 1.001
        deviation = grid_deviation(times, sizes, SFREQ)
        self.assertGreater(deviation, 0.5)
        # 19 intervals of 0.1 s over 1.001 -> ~0.95 samples of drift.
        self.assertAlmostEqual(deviation, 19 * 50 * 0.001, places=6)

    def test_grid_deviation_needs_a_rate_and_two_stamps(self) -> None:
        self.assertIsNone(grid_deviation(np.array([1.0]), np.array([50]), SFREQ))
        self.assertIsNone(grid_deviation(np.array([1.0, 1.1]), np.array([50, 50]), 0.0))


class LoadCntTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_npy_uses_the_meta_sidecar(self) -> None:
        data = synthetic_cnt()
        target = self.root / "ref.npy"
        np.save(target, data)
        (self.root / "ref_meta.json").write_text(
            json.dumps(
                {
                    "sfreq": SFREQ,
                    "names": list(CHANNELS),
                    "meas_date": "2026-09-12T21:13:32.550959+00:00",
                }
            )
        )
        loaded, sfreq, meta = load_cnt(target)
        self.assertEqual(sfreq, SFREQ)
        self.assertEqual(meta["names"], list(CHANNELS))
        np.testing.assert_array_equal(loaded, data)

    def test_cnt_without_antio_explains_the_fix(self) -> None:
        from unittest.mock import patch
        target = self.root / "ref.cnt"
        target.write_bytes(b"not really a cnt")
        # Test missing-dependency behavior independently of whether the optional
        # ANT reader is installed (real-recording integration needs it installed).
        with patch('mne.io.read_raw_ant', side_effect=RuntimeError('antio is required')), self.assertRaises(RuntimeError) as caught:
            load_cnt(target)
        self.assertIn("antio", str(caught.exception))


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)
        self.cnt = synthetic_cnt()
        self.reference = self.root / "ref.npy"
        np.save(self.reference, self.cnt)
        (self.root / "ref_meta.json").write_text(
            json.dumps(
                {
                    "sfreq": SFREQ,
                    "names": list(CHANNELS),
                    "meas_date": "2026-09-12T21:13:32.550959+00:00",
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_report_is_written_and_apple_files_are_skipped(self) -> None:
        records = self.root / "records"
        run_dir = records / "s1" / "a" / "run-bringup"
        write_run(run_dir, self.cnt[:, 900 : 900 + 1000])
        # exFAT volumes carry one of these next to every real file.
        (run_dir / f"._{run_dir.name}.sqlite").write_bytes(b"\x00" * 16)
        out = self.root / "report.json"

        code = main(
            [
                "--cnt-npy",
                str(self.reference),
                "--records",
                str(records),
                "--out",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        report = json.loads(out.read_text())
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["run"], "run-bringup")
        self.assertEqual(report[0]["outliers"], 0)
        self.assertEqual(report[0]["offset_utc"], "2026-09-12T21:13:34.350959+00:00")

    def test_exit_code_flags_a_mismatch(self) -> None:
        records = self.root / "records"
        record = self.cnt[:, 900 : 900 + 1000].copy()
        record[:, 100] += 100.0
        write_run(records / "s1" / "a" / "run-bad", record)

        code = main(["--cnt-npy", str(self.reference), "--records", str(records)])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
