"""Tests for the SQLite run recorder and its read-back helpers."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nova2026.streaming.recording import (
    RunRecorder,
    RunSpec,
    chunk_timestamps,
    iter_chunks,
    iter_events,
    iter_windows,
    read_metadata,
    replay_chunks,
)

CHANNELS = ("F3", "F4", "C3", "C4")
SFREQ = 500.0


class RunRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep scratch files inside the workspace.
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)
        self.spec = RunSpec("s1", "a", "trial01", role="trial")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_blocks(self):
        times = np.arange(600) / SFREQ
        data = np.vstack(
            [
                (index + 1) * 1e-6 * np.sin(2 * np.pi * 10 * times)
                for index in range(len(CHANNELS))
            ]
        ).T
        return data, times

    def test_chunks_survive_round_trip(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()

        recorder.write(data[:250], times[:250])
        recorder.write(data[250:], times[250:])
        path = recorder.close(status="completed", stats={"valid": 3})

        self.assertEqual(recorder.chunks, 2)
        self.assertEqual(recorder.samples, 600)

        rebuilt = [chunk for chunk, _ in iter_chunks(path)]
        self.assertEqual(len(rebuilt), 2)
        self.assertTrue(
            np.allclose(np.concatenate(rebuilt, axis=0), data, atol=1e-9)
        )

    def test_timestamps_rebuild_as_uniform_grids(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()
        recorder.write(data[:250], times[:250])
        path = recorder.close()

        rebuilt = list(replay_chunks(path))
        self.assertEqual(len(rebuilt), 1)
        _, grid = rebuilt[0]
        self.assertTrue(np.allclose(grid, times[:250], atol=1e-9))

    def test_fif_is_exported_next_to_the_database(self) -> None:
        import mne

        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()
        recorder.write(data, times)
        path = recorder.close()

        self.assertTrue(recorder.fif_path.exists())
        raw = mne.io.read_raw_fif(recorder.fif_path, preload=True, verbose=False)
        self.assertTrue(np.allclose(raw.get_data(), data.T, atol=1e-9))
        self.assertEqual(path, recorder.path)

    def test_events_are_recorded_in_order(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()
        recorder.write(data[:10], times[:10])
        recorder.mark("blink", timestamp=times[5])
        recorder.mark("blink", timestamp=times[8])
        path = recorder.close()

        self.assertEqual(
            iter_events(path),
            [(float(times[5]), "blink"), (float(times[8]), "blink")],
        )

    def test_metadata_records_identity_and_status(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ,
                               unit_exponent=0)
        data, times = self.build_blocks()
        recorder.write(data, times)
        path = recorder.close(status="failed", error="demo failure")

        metadata = read_metadata(path)
        self.assertEqual(metadata["subject"], "s1")
        self.assertEqual(metadata["session"], "a")
        self.assertEqual(metadata["run"], "trial01")
        self.assertEqual(metadata["role"], "trial")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"], "demo failure")
        self.assertEqual(metadata["samples"], 600)

    def test_existing_run_is_never_overwritten(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()
        recorder.write(data, times)
        recorder.close()

        with self.assertRaises(FileExistsError):
            RunRecorder(self.root, self.spec, CHANNELS, SFREQ)

    def test_close_is_idempotent_and_write_after_close_fails(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        data, times = self.build_blocks()
        recorder.write(data, times)
        recorder.close()
        recorder.close()

        with self.assertRaises(RuntimeError):
            recorder.write(data, times)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            RunSpec("s1", "a", "", role="trial")
        # Unsafe characters are stripped instead of leaking into paths.
        self.assertEqual(RunSpec("s1", "a", "../escape").run, "escape")
        with self.assertRaises(ValueError):
            RunRecorder(self.root, self.spec, CHANNELS, -1)
        with self.assertRaises(ValueError):
            RunRecorder(
                self.root, self.spec, CHANNELS, SFREQ, ch_types=("eeg",)
            )

        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        with self.assertRaises(ValueError):
            recorder.write(np.zeros((5, 3)), np.zeros(5))
        recorder.mark("bad label", timestamp=1.0)  # invalid chars are stripped
        recorder.close()

    def test_chunk_timestamps_helper(self) -> None:
        grid = chunk_timestamps(1000.0, 4, SFREQ)
        self.assertTrue(np.allclose(np.diff(grid), 1.0 / SFREQ))
        self.assertEqual(grid[0], 1000.0)
        self.assertTrue(np.all(np.isnan(chunk_timestamps(None, 4, SFREQ))))


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)
        self.spec = RunSpec("s1", "a", "prov01", role="trial")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_config_snapshot_is_stored(self) -> None:
        config = {
            "sfreq": 500.0,
            "out_sfreq": 128.0,
            "stamp": "notch60/band1-45-o3/soxr-LQ",
            "channels": list(CHANNELS),
        }
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ, config=config)
        path = recorder.close()
        self.assertEqual(read_metadata(path)["config"], config)

    def test_provenance_files_are_copied_and_hashed(self) -> None:
        source = Path(self.temporary.name) / "eye_operator.npz"
        source.write_bytes(b"fake-operator-bytes-v1")
        expected = hashlib.sha256(b"fake-operator-bytes-v1").hexdigest()

        recorder = RunRecorder(
            self.root, self.spec, CHANNELS, SFREQ,
            files={"artifact": source},
        )
        path = recorder.close()

        stored = read_metadata(path)["input:artifact"]
        self.assertEqual(stored["original"], str(source))
        self.assertEqual(stored["sha256"], expected)
        self.assertTrue((recorder.directory / stored["stored"]).exists())
        self.assertEqual(
            (recorder.directory / stored["stored"]).read_bytes(),
            b"fake-operator-bytes-v1",
        )

    def test_missing_provenance_file_fails_before_creating_the_run(self) -> None:
        missing = Path(self.temporary.name) / "absent.npz"
        with self.assertRaises(ValueError):
            RunRecorder(
                self.root, self.spec, CHANNELS, SFREQ, files={"artifact": missing}
            )
        run_dir = self.root / "s1" / "a" / "prov01"
        self.assertFalse(run_dir.exists())

    def test_window_log_round_trip(self) -> None:
        recorder = RunRecorder(
            self.root, self.spec, CHANNELS, SFREQ, track_windows=True
        )
        recorder.log_window(1000.0, True, (), segment=0)
        recorder.log_window(1000.5, False, ("warmup",), segment=0,
                            artifact_id="abc123")
        path = recorder.close()

        windows = iter_windows(path)
        self.assertEqual(
            windows,
            [
                (1000.0, True, (), 0, None),
                (1000.5, False, ("warmup",), 0, "abc123"),
            ],
        )

    def test_window_log_requires_tracking_and_open_run(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        with self.assertRaises(RuntimeError):
            recorder.log_window(1.0, True)
        # Runs without tracking read back as an empty window list.
        path = recorder.close()
        self.assertEqual(iter_windows(path), [])

    def test_not_processed_tail_is_recorded(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ)
        with self.assertRaises(ValueError):
            recorder.mark_not_processed(0, timestamp=1.0)
        recorder.write(np.zeros((10, 4)), np.arange(10) / SFREQ)
        recorder.mark_not_processed(23, timestamp=1.0)
        path = recorder.close()
        self.assertIn((1.0, "not_processed:23"), iter_events(path))

    def test_failed_close_still_keeps_provenance(self) -> None:
        recorder = RunRecorder(self.root, self.spec, CHANNELS, SFREQ,
                               config={"stamp": "x"})
        path = recorder.close(status="failed", error="calibration aborted")
        metadata = read_metadata(path)
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["config"], {"stamp": "x"})


if __name__ == "__main__":
    unittest.main()
