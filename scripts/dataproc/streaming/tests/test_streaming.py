"""Deterministic checks for the realtime path, without requiring an amplifier."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
from mne_lsl.lsl import local_clock
from mne_lsl.stream import StreamLSL

from scripts.dataproc.streaming.acquisition_queue import (
    AcquisitionQueue,
    StreamDataLagError,
    StreamDataValueError,
    StreamDiscontinuityError,
)
from scripts.dataproc.streaming.buffer_factory import create_buffer
from scripts.dataproc.streaming.channel_selection_contract import (
    ChannelSelectionContract,
)
from scripts.dataproc.streaming.config import CA208_CHANNELS, EEG_CHANNELS, StreamConfig
from scripts.dataproc.streaming.quality import SignalQuality
from scripts.dataproc.streaming.run import RunSpec
from scripts.dataproc.streaming.run_reader import RunReader
from scripts.dataproc.streaming.streamer import Streamer
from scripts.dataproc.streaming.streaming_resampler import StreamingResampler
from scripts.dataproc.streaming.streams import StreamConnection
from scripts.dataproc.streaming.voltage_converter import VoltageConverter


def configuration(**changes) -> StreamConfig:
    """Return a small explicit contract for deterministic signal tests."""
    config = StreamConfig(
        input_sfreq=500.0,
        source_unit_exponent=0,
        input_reference="test reference",
        upstream_processing="none",
        stream_name="test",
        eeg_channels=("F3", "F4"),
        eog_channels=("EOG",),
    )
    return config.updated(**changes)


def signal(seconds: float = 12, sfreq: float = 500.0):
    """Return a 10 Hz signal with known channel amplitudes and regular timing."""
    timestamps = 100.0 + np.arange(round(seconds * sfreq)) / sfreq
    sine = np.sin(2 * np.pi * 10 * (timestamps - 100))
    return sine[:, None] * np.array([20, 40, 2000])[None, :] * 1e-6, timestamps


def feed_chunks(buffer, data, timestamps, size):
    """Collect windows using a chosen input partition."""
    windows = []
    for start in range(0, len(data), size):
        buffer.feed((data[start : start + size], timestamps[start : start + size]))
        windows.extend(buffer.spit())
    return windows


class ConfigurationTests(unittest.TestCase):
    def test_cap_and_shared_channels(self):
        self.assertEqual(len(CA208_CHANNELS), 64)
        self.assertEqual(len(EEG_CHANNELS), 57)
        self.assertNotIn("EOG", EEG_CHANNELS)
        self.assertNotIn("CPz", EEG_CHANNELS)
        self.assertNotIn("AFz", EEG_CHANNELS)

    def test_invalid_configuration(self):
        for changes in (
            {"input_sfreq": 0},
            {"queue_size": 0},
            {"step_seconds": 3},
            {"source_unit_exponent": 123},
            {"eog_channels": ("F3",)},
            {"output_sfreq": 64},
            {"window_seconds": 2.001},
            {"timestamp_tolerance": 0.002},
            {"upstream_processing": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configuration(**changes)

    def test_units_before_metadata_can_be_overwritten(self):
        converter = VoltageConverter((0, -6, -3))
        data = np.array([[20e-6, 40.0, 0.05]])
        np.testing.assert_allclose(converter.convert(data), [[20, 40, 50]])
        np.testing.assert_allclose(data, [[20e-6, 40.0, 0.05]])

    def test_missing_unknown_and_mismatched_units_rejected(self):
        config = configuration()
        for raw_units in (["", "0", "0"], ["banana", "0", "0"], ["-6"] * 3):
            source = SimpleNamespace(
                ch_names=list(config.channels),
                sinfo=SimpleNamespace(
                    get_channel_units=lambda raw_units=raw_units: raw_units
                ),
            )
            with self.subTest(units=raw_units), self.assertRaises(RuntimeError):
                StreamConnection(config)._prepare_channels(cast(StreamLSL, source))


class QueueTests(unittest.TestCase):
    def make_queue(self, max_chunks=3):
        contract = ChannelSelectionContract(("F4", "F3", "EOG"), ("F3", "F4", "EOG"))
        contract.get_contract()
        return AcquisitionQueue(500, 0, 1, 3, contract, max_chunks=max_chunks)

    def test_channel_order_and_array_ownership(self):
        queue = self.make_queue()
        data = np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]])
        queue.callback(data, np.array([100.0, 100.002]), None)
        data[:] = 99
        actual, _ = queue.get()
        np.testing.assert_array_equal(actual, [[1, 2, 3], [4, 5, 6]])

    def test_full_queue_is_an_error_not_a_silent_drop(self):
        queue = self.make_queue(max_chunks=1)
        queue.callback(np.ones((1, 3)), np.array([100.0]), None)
        queue.callback(np.ones((1, 3)), np.array([100.002]), None)
        with self.assertRaises(StreamDataLagError):
            queue.get()

    def test_gap_overlap_and_nonfinite_are_propagated(self):
        for timestamps in (np.array([100.0, 100.004]), np.array([100.0, 100.0])):
            queue = self.make_queue()
            queue.callback(np.ones((2, 3)), timestamps, None)
            with self.assertRaises(StreamDiscontinuityError):
                queue.get()
        queue = self.make_queue()
        queue.callback(np.full((1, 3), np.nan), np.array([100.0]), None)
        with self.assertRaises(StreamDataValueError):
            queue.get()

    def test_cross_chunk_gap_and_original_worker_error(self):
        queue = self.make_queue()
        queue.callback(np.ones((1, 3)), np.array([100.0]), None)
        queue.callback(np.ones((1, 3)), np.array([100.004]), None)
        with self.assertRaises(StreamDiscontinuityError):
            queue.get()
        queue = self.make_queue()
        error = OSError("acquisition failed")
        queue.store_error(error)
        queue.store_error(RuntimeError("later failure"))
        with self.assertRaises(OSError) as caught:
            queue.get()
        self.assertIs(caught.exception, error)


class BufferTests(unittest.TestCase):
    def test_resampler_passband_and_alias_rejection(self):
        times = np.arange(10000) / 500
        frequencies = (10, 40, 45, 70, 100, 180)
        data = np.column_stack(
            [np.sin(2 * np.pi * freq * times) for freq in frequencies]
        )
        resampler = StreamingResampler(500, 128, len(frequencies))
        output = np.concatenate(
            [resampler(data[i : i + 37]) for i in range(0, len(data), 37)]
        )
        amplitude = np.sqrt(2) * np.std(output[256:], axis=0)
        np.testing.assert_allclose(amplitude[:3], 1, atol=0.002)
        self.assertTrue(np.all(amplitude[3:] < 1e-4))

    def test_dc_offset_does_not_look_like_an_ac_artifact(self):
        config = configuration()
        data, timestamps = signal(8)
        data[:, :2] += 0.02  # 20 mV electrode offset, below the selected rail.
        windows = feed_chunks(create_buffer(config, (0, 0, 0)), data, timestamps, 23)
        self.assertTrue(any(w.valid for w in windows))
        self.assertTrue(all("amplitude" not in w.reasons for w in windows))

    def test_wrap_counts_timing_scaling_and_chunk_invariance(self):
        config = configuration()
        data, timestamps = signal()
        small = create_buffer(config, (0, 0, 0))
        large = create_buffer(config, (0, 0, 0))
        actual = feed_chunks(small, data, timestamps, 7)
        expected = feed_chunks(large, data, timestamps, 311)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(len(actual), 1 + (small.total_written - 256) // 64)
        self.assertTrue(any(window.valid for window in actual))
        for index, (a, b) in enumerate(zip(actual, expected)):
            np.testing.assert_allclose(a.data, b.data, atol=1e-9)
            np.testing.assert_allclose(a.eog, b.eog, atol=1e-9)
            np.testing.assert_allclose(
                a.timestamps, 100 + (index * 64 + np.arange(256)) / 128
            )
            self.assertEqual(a.valid, b.valid)
            self.assertEqual(a.data.shape, (256, 2))
            self.assertEqual(a.eog.shape, (256, 1))
        # Independent amplitude oracle after causal filter warm-up.
        good = actual[-1]
        amplitudes = np.sqrt(2) * np.std(good.data, axis=0)
        np.testing.assert_allclose(amplitudes, [20, 40], rtol=0.02)
        self.assertTrue(all("amplitude" not in w.reasons for w in actual))
        self.assertEqual(actual[0].reasons, ("warmup",))

    def test_reset_reproduces_a_fresh_run(self):
        config = configuration()
        buffer = create_buffer(config, (0, 0, 0))
        data, timestamps = signal(6)
        first = feed_chunks(buffer, data, timestamps, 73)
        buffer.reset()
        second = feed_chunks(buffer, data, timestamps + 50, 73)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.data, b.data)
            np.testing.assert_allclose(b.timestamps - a.timestamps, 50)
            self.assertEqual(a.reasons, b.reasons)

    def test_gap_cannot_be_hidden_by_resampling(self):
        buffer = create_buffer(configuration(), (0, 0, 0))
        data, timestamps = signal(2)
        buffer.feed((data[:100], timestamps[:100]))
        with self.assertRaises(StreamDiscontinuityError):
            buffer.feed((data[101:200], timestamps[101:200]))

    def test_pending_windows_are_bounded(self):
        config = configuration(max_ready_windows=1)
        buffer = create_buffer(config, (0, 0, 0))
        data, timestamps = signal(6)
        with self.assertRaisesRegex(RuntimeError, "Pending windows"):
            buffer.feed((data, timestamps))

    def test_artifacts_remain_invalid_after_filtering(self):
        config = configuration(saturation_limit_uv=1000)
        data, timestamps = signal(10)
        data[2000:2250, 0] = 0  # A flatline spanning multiple input chunks.
        data[3000, 1] = 0.002  # A 2000 uV spike, before filtering can attenuate it.
        windows = feed_chunks(create_buffer(config, (0, 0, 0)), data, timestamps, 31)
        reasons = {reason for window in windows for reason in window.reasons}
        self.assertTrue(
            {"flatline", "amplitude", "saturation", "warmup"}.issubset(reasons)
        )
        self.assertTrue(all(not w.valid for w in windows if w.reasons))

    def test_flatline_history_crosses_chunks_and_resets(self):
        config = configuration(warmup_seconds=0)
        quality = SignalQuality(config)
        for start in range(0, 300, 10):
            quality.feed(np.zeros((10, 3)), 100 + np.arange(start, start + 10) / 500)
        self.assertIn("flatline", quality.reasons(100, 101))
        quality.reset()
        self.assertEqual(quality.reasons(100, 101), ())

    def test_quality_does_not_fill_the_space_between_separate_spikes(self):
        config = configuration(warmup_seconds=0)
        data, timestamps = signal(1)
        data *= 1e6
        data[50, 0] = 2000
        data[450, 0] = 2000
        whole = SignalQuality(config)
        split = SignalQuality(config)
        whole.feed(data, timestamps)
        for start in range(0, len(data), 17):
            split.feed(data[start : start + 17], timestamps[start : start + 17])
        self.assertEqual(whole.reasons(100.2, 100.8), ())
        self.assertEqual(split.reasons(100.2, 100.8), ())


class FakeSource:
    """Minimal nonblocking source for deterministic lifecycle failure tests."""

    def __init__(self, mode):
        self.mode = mode
        self.name = "test-source"
        self.source_id = "test-source-id"
        self.connected = False
        self.callbacks = []
        self.n_new_samples = 0
        self.sent = False

    def get_channel_units(self, picks):
        return [(107, 0)] * len(picks)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def acquire(self):
        if self.mode == "error":
            raise OSError("test acquisition error")
        if self.mode == "disconnect":
            self.connected = False
        if self.mode == "data" and not self.sent:
            self.sent = True
            data, _ = signal(2.5)
            timestamps = local_clock() - 2.5 + np.arange(1250) / 500
            for callback in self.callbacks:
                callback(data, timestamps, None)
            self.n_new_samples = 500

    def get_data(self, **kwargs):
        self.n_new_samples = 0


class FakeConnection:
    def __init__(self, config, mode):
        self.stream = FakeSource(mode)
        self.channel_selection = ChannelSelectionContract(
            config.channels, config.channels
        )
        self.channel_selection.get_contract()

    def connect(self):
        self.stream.connected = True
        self.stream.sent = False
        self.stream.callbacks = []
        return self.stream

    def disconnect(self):
        self.stream.connected = False


class LifecycleTests(unittest.TestCase):
    def runner(self, mode, pipeline=None, observer=None):
        config = configuration(
            no_data_timeout=0.15,
            window_seconds=0.25,
            step_seconds=0.125,
            warmup_seconds=0.125,
        )
        connection = FakeConnection(config, mode)
        with patch(
            "scripts.dataproc.streaming.streamer.StreamConnection",
            return_value=connection,
        ):
            return Streamer(config, pipeline, observer)

    def assert_closed(self, streamer):
        self.assertFalse(streamer.acquisition_thread.is_alive())
        self.assertFalse(streamer.connection.stream.connected)
        self.assertFalse(streamer.initialized)

    def test_no_data_timeout_and_acquisition_failures(self):
        for mode, exception in (
            ("empty", TimeoutError),
            ("error", OSError),
            ("disconnect", RuntimeError),
        ):
            with self.subTest(mode=mode):
                streamer = self.runner(mode)
                streamer.initialize()
                with self.assertRaises(exception):
                    streamer.stream()
                self.assert_closed(streamer)

    def test_processing_error_still_closes_the_source(self):
        def fail(window):
            raise ValueError("test pipeline failure")

        streamer = self.runner("data", pipeline=fail)
        streamer.initialize()
        with self.assertRaisesRegex(ValueError, "test pipeline failure"):
            streamer.stream()
        self.assert_closed(streamer)

    def test_processing_failure_finalizes_recording(self):
        def fail(window):
            raise ValueError("test recorded pipeline failure")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            streamer = self.runner("data", pipeline=fail)
            streamer.run = RunSpec("s1", "a", "failed", "trial")
            streamer.record_root = root
            streamer.initialize()
            with self.assertRaisesRegex(ValueError, "recorded pipeline failure"):
                streamer.stream()

            self.assert_closed(streamer)
            reader = RunReader(root / "s1" / "a" / "failed")
            try:
                self.assertEqual(reader.metadata["status"], "failed")
                self.assertIn("recorded pipeline failure", reader.metadata["error"])
                self.assertTrue(list(reader.chunks()))
            finally:
                reader.close()

    def test_invalid_windows_never_reach_pipeline_and_run_can_restart(self):
        forwarded = []
        observed = []
        streamer = self.runner(
            "data", pipeline=forwarded.append, observer=observed.append
        )
        for _ in range(2):
            streamer.initialize()
            streamer.stream(duration=0.1)
            self.assert_closed(streamer)
        self.assertTrue(forwarded)
        self.assertTrue(all(window.valid for window in forwarded))
        self.assertTrue(any(not window.valid for window in observed))

    def test_explicit_stop_and_initialize_required(self):
        streamer = self.runner("data")
        with self.assertRaises(RuntimeError):
            streamer.stream()
        streamer.on_window = lambda window: streamer.stop()
        streamer.initialize()
        streamer.stream()
        self.assert_closed(streamer)


if __name__ == "__main__":
    unittest.main()
