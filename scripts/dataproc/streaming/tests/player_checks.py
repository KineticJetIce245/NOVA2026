"""Opt-in lifecycle checks using real PlayerLSL outlets on this computer.

Run with ``python -m unittest scripts.dataproc.streaming.tests.player_checks -v``.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import mne
import numpy as np
from mne_lsl.player import PlayerLSL

from scripts.dataproc.streaming.config import StreamConfig
from scripts.dataproc.streaming.recorded_replay import calibrate_run, replay_run
from scripts.dataproc.streaming.replay import synthetic_recording
from scripts.dataproc.streaming.run import RunSpec
from scripts.dataproc.streaming.run_reader import RunReader
from scripts.dataproc.streaming.streamer import Streamer
from scripts.dataproc.streaming.tests.test_artifact import config, signals, write_run


class PlayerChecks(unittest.TestCase):
    def setUp(self):
        """Start a uniquely identified outlet for each independent check."""
        self.config = StreamConfig(
            input_sfreq=500,
            source_unit_exponent=0,
            input_reference="synthetic CPz contract",
            upstream_processing="synthetic sine",
            stream_name="NOVA-Lifecycle-Test",
            source_id=uuid4().hex,
            no_data_timeout=0.3,
        )
        assert self.config.source_id is not None
        self.player = PlayerLSL(
            synthetic_recording(30),
            chunk_size=10,
            n_repeat=1,
            name=self.config.stream_name,
            source_id=self.config.source_id,
            annotations=False,
        )
        self.player.start()
        self.streamer = None

    def tearDown(self):
        """Close local test resources even when an assertion fails."""
        if self.streamer is not None:
            self.streamer.close()
        if self.player.running:
            self.player.stop()

    def test_source_interruption_times_out_and_closes(self):
        self.streamer = Streamer(self.config)
        self.streamer.initialize()
        self.player.stop()
        with self.assertRaises((TimeoutError, RuntimeError)):
            self.streamer.stream(duration=5)
        assert self.streamer.acquisition_thread is not None
        self.assertFalse(self.streamer.acquisition_thread.is_alive())
        assert self.streamer.connection.stream is not None
        self.assertFalse(self.streamer.connection.stream.connected)

    def test_wrong_sampling_rate_is_rejected_during_setup(self):
        self.streamer = Streamer(self.config.updated(input_sfreq=512))
        with self.assertRaisesRegex(RuntimeError, "sampling rate"):
            self.streamer.initialize()
        assert self.streamer.connection.stream is not None
        self.assertFalse(self.streamer.connection.stream.connected)

    def test_wrong_units_are_rejected_during_setup(self):
        self.streamer = Streamer(self.config.updated(source_unit_exponent=-6))
        with self.assertRaisesRegex(RuntimeError, "voltage units"):
            self.streamer.initialize()
        assert self.streamer.connection.stream is not None
        self.assertFalse(self.streamer.connection.stream.connected)

    def test_two_runs_reset_state_and_stop_from_callback(self):
        forwarded = []
        observed = []

        def consume(window):
            forwarded.append(window)
            assert self.streamer is not None
            self.streamer.stop()

        self.streamer = Streamer(
            self.config, pipeline=consume, on_window=observed.append
        )
        for _ in range(2):
            observed.clear()
            self.streamer.initialize()
            stats = self.streamer.stream(duration=7)
            self.assertEqual(stats.valid_windows, 1)
            self.assertEqual(stats.invalid_windows, 4)
            self.assertEqual(observed[0].start_sample, 0)
            self.assertEqual(observed[0].reasons, ("warmup",))
            assert self.streamer.acquisition_thread is not None
            self.assertFalse(self.streamer.acquisition_thread.is_alive())
            assert self.streamer.connection.stream is not None
            self.assertFalse(self.streamer.connection.stream.connected)
        self.assertEqual(len(forwarded), 2)
        self.assertGreater(forwarded[1].timestamps[0], forwarded[0].timestamps[0])


class RecordedPlayerChecks(unittest.TestCase):
    def test_recording_correction_recovery_and_callback_stop(self):
        """Exercise the real transport and compare delivered windows to disk replay."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, times, markers = signals()
            directory, _ = write_run(
                root,
                RunSpec("synthetic", "a", "calibration", "artifact_calibration"),
                data,
                times,
                markers,
            )
            artifact = root / "operator.npz"
            operator = calibrate_run(directory, artifact)

            # Inject a nonfinite burst exceeding the interpolation limit after initial valid output.
            data[4500:4520, 0] = np.nan
            raw = mne.io.RawArray(
                data.T, mne.create_info(["F3", "F4", "EOG"], 500, "eeg")
            )
            raw.set_channel_types({"EOG": "eog"})
            source_id = uuid4().hex
            player = PlayerLSL(
                raw,
                chunk_size=10,
                n_repeat=1,
                name="NOVA-Recorded-Test",
                source_id=source_id,
                annotations=False,
            )
            settings = config().updated(
                stream_name="NOVA-Recorded-Test", source_id=source_id
            )
            observed = []
            forwarded = []

            def consume(window):
                forwarded.append(window)
                if window.segment == 1:
                    streamer.mark("response", float(window.timestamps[-1]))
                    streamer.stop()

            spec = RunSpec("synthetic", "a", "live", "trial", artifact_path=artifact)
            streamer = Streamer(settings, consume, observed.append, spec, root)
            try:
                player.start()
                streamer.initialize()
                stats = streamer.stream(duration=20)
            finally:
                streamer.close()
                if player.running:
                    player.stop()

            self.assertEqual(stats.recoveries, 1)
            self.assertEqual(stats.discarded_chunks, 1)
            self.assertTrue(any(window.segment == 0 for window in forwarded))
            self.assertTrue(any(window.segment == 1 for window in forwarded))
            self.assertTrue(all(window.valid for window in forwarded))
            self.assertTrue(
                any(
                    window.segment == 1 and "warmup" in window.reasons
                    for window in observed
                )
            )
            assert streamer.acquisition_thread is not None
            assert streamer.connection.stream is not None
            self.assertFalse(streamer.acquisition_thread.is_alive())
            self.assertFalse(streamer.connection.stream.connected)

            saved = root / "synthetic" / "a" / "live"
            reader = RunReader(saved)
            try:
                self.assertEqual(reader.metadata["status"], "completed")
                self.assertEqual(reader.metadata["stats"]["recoveries"], 1)
                self.assertTrue(
                    any(np.isnan(chunk).any() for chunk, _, _ in reader.chunks())
                )
                self.assertTrue(
                    any(
                        kind == "marker" and details["label"] == "response"
                        for _, kind, details in reader.events()
                    )
                )
            finally:
                reader.close()

            reconstructed = list(replay_run(saved))
            self.assertEqual(len(reconstructed), len(observed))
            for actual, expected in zip(observed, reconstructed):
                np.testing.assert_array_equal(actual.data, expected.data)
                np.testing.assert_array_equal(actual.eog, expected.eog)
                np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
                self.assertEqual(actual.artifact_id, operator.artifact_id)
                self.assertEqual(
                    (actual.segment, actual.reasons),
                    (expected.segment, expected.reasons),
                )

            print("Recorded PlayerLSL:", stats.to_dict())
            print(
                "Held-out EOG coupling ratio:",
                operator.report["heldout_coupling_ratio"],
            )


if __name__ == "__main__":
    unittest.main()
