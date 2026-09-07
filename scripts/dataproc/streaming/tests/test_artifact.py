"""Recording reconstruction, bounded recovery, and held-out artifact checks."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.dataproc.streaming.artifact_calibration import ArtifactCalibration
from scripts.dataproc.streaming.config import StreamConfig
from scripts.dataproc.streaming.processor import RunProcessor
from scripts.dataproc.streaming.recorded_replay import calibrate_run, replay_run
from scripts.dataproc.streaming.recording import RunRecorder
from scripts.dataproc.streaming.run import RunSpec
from scripts.dataproc.streaming.run_reader import RunReader
from scripts.dataproc.streaming.spatial_operator import SpatialOperator


def config() -> StreamConfig:
    """Return a small configuration with independent neural/artifact directions."""
    return StreamConfig(
        500,
        0,
        "CPz",
        "none",
        stream_name="artifact-test",
        eeg_channels=("F3", "F4"),
        eog_channels=("EOG",),
    )


def signals(seconds: float = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create known EEG, eye artifacts, and event times without random noise."""
    times = np.arange(round(seconds * 500)) / 500
    centers = np.arange(6, seconds - 3, 2.5)
    eye = np.zeros_like(times)
    for center in centers:
        eye += np.exp(-0.5 * ((times - center) / 0.12) ** 2)
    brain = 20 * np.sin(2 * np.pi * 10 * times)
    data = np.column_stack((brain + 100 * eye, -brain + 100 * eye, 250 * eye))
    return data * 1e-6, times + 100, centers + 100


def write_run(
    root: Path, spec: RunSpec, data: np.ndarray, times: np.ndarray, markers=()
):
    """Record and process the same canonical chunks using independent components."""
    settings = config()
    operator = (
        None if spec.artifact_path is None else SpatialOperator.load(spec.artifact_path)
    )
    processor = RunProcessor(settings, operator)
    recorder = RunRecorder(root, spec, settings)
    recorder.open({"test": "known synthetic input"})
    windows = []
    try:
        for timestamp in markers:
            recorder.event(float(timestamp), "marker", {"label": "blink"})
        for start in range(0, len(data), 37):
            item = data[start : start + 37], times[start : start + 37]
            recorder.write_chunk(*item, processor.segment)
            windows.extend(processor.feed(item))
            for event in processor.events:
                recorder.event(float(times[start]), "recovery", event)
        recorder.close("completed", {"recoveries": processor.recoveries})
    finally:
        recorder.close("failed", {}, "test did not finish")
    return recorder.directory, windows


class RecordingTests(unittest.TestCase):
    def test_selected_inputs_are_copied_and_hash_checked(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.sqlite"
            model = root / "model.bin"
            baseline.write_bytes(b"selected baseline")
            model.write_bytes(b"selected model")
            spec = RunSpec(
                "s1", "a", "trial", "trial", baseline_path=baseline, model_path=model
            )
            recorder = RunRecorder(root, spec, config())
            recorder.open({})
            recorder.close("completed", {})
            model.write_bytes(b"later model")

            reader = RunReader(recorder.directory)
            try:
                saved = reader.input_path("model_path")
                assert saved is not None
                self.assertEqual(saved.read_bytes(), b"selected model")
                self.assertIsNotNone(reader.input_path("baseline_path"))
                saved.write_bytes(b"damaged snapshot")
                with self.assertRaisesRegex(ValueError, "hash"):
                    reader.input_path("model_path")
            finally:
                reader.close()

    def test_missing_input_does_not_create_a_run(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = RunSpec("s1", "a", "trial", "trial", model_path=root / "missing.bin")
            recorder = RunRecorder(root, spec, config())
            with self.assertRaises(FileNotFoundError):
                recorder.open({})
            self.assertFalse(recorder.directory.exists())

    def test_raw_roundtrip_preserves_faults_and_markers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, times, _ = signals(12)
            data[2600:2620, 0] = np.nan
            directory, windows = write_run(
                root, RunSpec("s1", "a", "r1", "baseline"), data, times, [102.5]
            )
            reader = RunReader(directory)
            try:
                restored = np.concatenate([item[0] for item in reader.chunks()])
                np.testing.assert_array_equal(restored, data)
                self.assertEqual(
                    reader.events()[0], (102.5, "marker", {"label": "blink"})
                )
                self.assertTrue(
                    any(kind == "recovery" for _, kind, _ in reader.events())
                )
            finally:
                reader.close()

            reconstructed = list(replay_run(directory))
            self.assertEqual(len(windows), len(reconstructed))
            for actual, expected in zip(windows, reconstructed):
                np.testing.assert_array_equal(actual.data, expected.data)
                np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
                self.assertEqual(
                    (actual.segment, actual.reasons),
                    (expected.segment, expected.reasons),
                )

    def test_identity_collision_and_path_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            RunSpec("../escape", "a", "r1", "trial")
        with TemporaryDirectory() as temporary:
            spec = RunSpec("s1", "a", "same", "trial")
            first = RunRecorder(Path(temporary), spec, config())
            first.open({})
            first.close("completed", {})
            with self.assertRaises(FileExistsError):
                RunRecorder(Path(temporary), spec, config()).open({})

    def test_failed_runs_cannot_masquerade_as_complete(self):
        with TemporaryDirectory() as temporary:
            recorder = RunRecorder(
                Path(temporary), RunSpec("s1", "a", "failed", "baseline"), config()
            )
            recorder.open({})
            recorder.close("failed", {}, "simulated interruption")
            with self.assertRaises(ValueError):
                list(replay_run(recorder.directory))


class RecoveryTests(unittest.TestCase):
    def test_nan_and_short_gap_resume_with_new_warmup(self):
        settings = config()
        data, times, _ = signals(24)
        data[3500:3520, 0] = np.nan
        processor = RunProcessor(settings)
        windows = []
        for start in range(0, len(data), 100):
            if start == 7500:
                continue
            windows.extend(
                processor.feed((data[start : start + 100], times[start : start + 100]))
            )
        self.assertEqual(processor.recoveries, 2)
        for segment in (0, 1, 2):
            part = [window for window in windows if window.segment == segment]
            self.assertTrue(part)
            self.assertIn("warmup", part[0].reasons)
            self.assertTrue(any(window.valid for window in part))

    def test_long_gap_and_repeated_faults_stop(self):
        data, times, _ = signals(6)
        processor = RunProcessor(config())
        processor.feed((data[:100], times[:100]))
        with self.assertRaises(RuntimeError):
            processor.feed((data[1000:1100], times[1000:1100]))
        processor = RunProcessor(config().updated(recovery_max_events=2))
        bad = np.full((10, 3), np.nan)
        processor.feed((bad, times[:10]))
        processor.feed((bad, times[10:20]))
        with self.assertRaisesRegex(RuntimeError, "Too many"):
            processor.feed((bad, times[20:30]))

    def test_persistent_saturation_stops(self):
        data, times, _ = signals(12)
        data[:, 0] = 1.0
        processor = RunProcessor(config())
        with self.assertRaisesRegex(RuntimeError, "persisted"):
            for start in range(0, len(data), 100):
                processor.feed((data[start : start + 100], times[start : start + 100]))


class ArtifactTests(unittest.TestCase):
    def test_calibration_saved_operator_and_all_run_roles(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, times, markers = signals()
            directory, _ = write_run(
                root,
                RunSpec("s1", "a", "cal", "artifact_calibration"),
                data,
                times,
                markers,
            )
            path = root / "eye_operator.npz"
            operator = calibrate_run(directory, path)
            self.assertLess(operator.report["heldout_coupling_ratio"], 0.05)

            # Independent oracle: retain the known neural direction within 2%.
            # Finite calibration data need not yield the exact ideal projector.
            neural = np.column_stack((np.arange(10), -np.arange(10)))
            np.testing.assert_allclose(
                operator.apply(neural), neural, rtol=0.02, atol=0.001
            )
            np.testing.assert_allclose(operator.apply(np.ones((10, 2))), 0, atol=0.02)
            self.assertTrue(path.with_suffix(".png").is_file())

            for role in ("baseline", "labeled_training", "trial"):
                spec = RunSpec("s1", "a", role, role, artifact_path=path)
                saved, windows = write_run(root, spec, data[:5000], times[:5000])
                reconstructed = list(replay_run(saved))
                self.assertEqual(len(windows), len(reconstructed))
                for actual, expected in zip(windows, reconstructed):
                    np.testing.assert_array_equal(actual.data, expected.data)
                    np.testing.assert_array_equal(actual.eog, expected.eog)
                    self.assertEqual(actual.artifact_id, operator.artifact_id)
                    with self.assertRaises(ValueError):
                        operator.apply_window(actual)

            with self.assertRaises(ValueError):
                operator.validate_config(config().updated(eeg_channels=("F4", "F3")))
            with self.assertRaises(ValueError):
                operator.validate_config(config().updated(input_reference="average"))
            with self.assertRaises(ValueError):
                operator.validate_config(config().updated(bandpass=(2, 40)))
            with self.assertRaises(FileExistsError):
                operator.save(path)

    def test_missing_eog_and_flat_calibration_are_rejected(self):
        with self.assertRaises(ValueError):
            ArtifactCalibration(config().updated(eog_channels=()))
        with self.assertRaises(ValueError):
            ArtifactCalibration(config()).fit(
                np.zeros((6, 128, 2)), np.zeros((6, 128, 1))
            )


if __name__ == "__main__":
    unittest.main()
