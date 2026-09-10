"""Tests for the fixed-size acquisition layer."""

import time
import unittest
from threading import Event, Timer
from uuid import uuid4

import mne
import numpy as np
from mne_lsl.lsl import local_clock
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL

from nova2026.streaming.acquire import Acquire

SFREQ = 500.0
CHANNELS = 2


class FakeStream:
    """Minimal stand-in for a connected, manually acquired StreamLSL."""

    def __init__(
        self,
        sfreq: float = SFREQ,
        n_channels: int = CHANNELS,
        generator=None,
    ) -> None:
        self.info = {"nchan": n_channels, "sfreq": sfreq}
        self.connected = True
        self.n_new_samples = 0
        self.callbacks: list = []
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self.generator = generator
        self.acquires = 0

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def get_data(self, **kwargs) -> None:
        self.n_new_samples = 0

    def acquire(self) -> None:
        self.acquires += 1
        if not self.chunks and self.generator is not None:
            self.generator()
        if not self.chunks:
            return
        data, timestamps = self.chunks.pop(0)
        self.n_new_samples = len(data)
        for callback in list(self.callbacks):
            callback(data, timestamps, None)


class Feeder:
    """Produce continuous, uniquely identifiable samples for a FakeStream."""

    def __init__(
        self,
        block_chunks: tuple[int, ...] = (),
        *,
        sfreq: float = SFREQ,
        n_channels: int = CHANNELS,
        start: float | None = None,
        auto: bool = False,
    ) -> None:
        self.block_chunks = tuple(block_chunks)
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.time = local_clock() if start is None else start
        self.next_sample = 0
        self.generated = 0
        self._pattern = 0
        self.stream = FakeStream(
            sfreq, n_channels, generator=self._generate if auto else None
        )

    def feed(self, n_samples: int) -> None:
        """Queue one source chunk of exactly ``n_samples`` samples."""

        values = np.arange(
            self.next_sample, self.next_sample + n_samples, dtype=np.float64
        )
        data = np.repeat(values[:, None], self.n_channels, axis=1)
        timestamps = self.time + np.arange(n_samples) / self.sfreq
        self.time = float(timestamps[-1]) + 1.0 / self.sfreq
        self.next_sample += n_samples
        self.generated += n_samples
        self.stream.chunks.append((data, timestamps))

    def skip(self, n_samples: int) -> None:
        """Advance the source clock without emitting samples."""

        self.time += n_samples / self.sfreq
        self.next_sample += n_samples

    def _generate(self) -> None:
        """Emit the next chunk of the configured uneven pattern."""

        size = self.block_chunks[self._pattern % len(self.block_chunks)]
        self._pattern += 1
        self.feed(size)


def acquire_for(feeder: Feeder, block_samples: int, **kwargs) -> Acquire:
    """Build an Acquire over the feeder's fake stream.

    The fake source emits faster than real time, so the default future guard is
    relaxed here; the dedicated future test overrides it explicitly.
    """

    options = {
        "no_data_timeout": 0.2,
        "poll_interval": 0.001,
        "max_future_seconds": 60.0,
    }
    options.update(kwargs)
    return Acquire(feeder.stream, block_samples, sfreq=feeder.sfreq, **options)


class BlockAssemblyTests(unittest.TestCase):
    """Check that arbitrary source chunk sizes become exact blocks."""

    def test_uneven_chunks_are_reassembled_in_order(self) -> None:
        feeder = Feeder((37, 100, 250), auto=True)
        acquire = acquire_for(feeder, 50)

        try:
            for index in range(4):
                data, timestamps = acquire.read()

                self.assertEqual(data.shape, (50, CHANNELS))
                self.assertEqual(timestamps.shape, (50,))
                self.assertEqual(data[0, 0], index * 50)
                self.assertEqual(data[-1, 0], index * 50 + 49)
                self.assertTrue(np.all(np.diff(timestamps) > 0))
        finally:
            acquire.close()

    def test_no_sample_is_lost_or_duplicated(self) -> None:
        feeder = Feeder((13, 250, 7, 100), auto=True)
        acquire = acquire_for(feeder, 64)

        try:
            blocks = [acquire.read()[0] for _ in range(6)]
            pending = acquire.pending_samples
            blocks_read = acquire.blocks
            samples = acquire.samples
        finally:
            acquire.close()

        streamed = np.concatenate(blocks, axis=0)
        self.assertTrue(np.array_equal(streamed[:, 0], np.arange(384.0)))
        self.assertEqual(blocks_read, 6)
        self.assertEqual(samples, 384)
        self.assertEqual(pending, feeder.generated - samples)

    def test_one_large_chunk_leaves_a_remainder(self) -> None:
        feeder = Feeder((250,))
        acquire = acquire_for(feeder, 100)
        feeder.feed(250)

        try:
            first, first_times = acquire.read()
            second, second_times = acquire.read()
            pending = acquire.pending_samples
        finally:
            acquire.close()

        self.assertEqual(pending, 50)
        self.assertEqual(first[0, 0], 0)
        self.assertEqual(second[0, 0], 100)
        self.assertEqual(first_times[-1], second_times[0] - 1.0 / feeder.sfreq)

    def test_empty_chunks_are_ignored(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        feeder.stream.chunks.append(
            (np.zeros((0, CHANNELS)), np.zeros(0))
        )
        feeder.feed(10)

        try:
            data, _ = acquire.read(timeout=0.05)
        finally:
            acquire.close()

        self.assertEqual(data.shape, (10, CHANNELS))

    def test_clear_counter_is_reset(self) -> None:
        feeder = Feeder((60,))
        acquire = acquire_for(feeder, 50)
        feeder.feed(60)

        try:
            acquire.read()
        finally:
            acquire.close()

        self.assertEqual(feeder.stream.n_new_samples, 0)

    def test_returned_block_does_not_alias_the_source(self) -> None:
        feeder = Feeder((50,))
        acquire = acquire_for(feeder, 50)
        feeder.feed(50)
        source_data = feeder.stream.chunks[0][0]
        original = source_data.copy()

        try:
            data, _ = acquire.read()
        finally:
            acquire.close()

        data[:] = -1.0
        self.assertTrue(np.array_equal(source_data, original))


class FailureTests(unittest.TestCase):
    """Check every documented failure path and its exception type."""

    def test_timeout_without_data(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10, poll_interval=0.001)
        started = time.monotonic()

        try:
            with self.assertRaises(TimeoutError):
                acquire.read(timeout=0.02)
        finally:
            acquire.close()

        self.assertLess(time.monotonic() - started, 1.0)

    def test_disconnected_source(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        feeder.stream.connected = False

        try:
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_stop_before_read(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        acquire.stop()

        try:
            self.assertTrue(acquire.stopped)
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_shared_stop_event(self) -> None:
        feeder = Feeder()
        event = Event()
        acquire = acquire_for(feeder, 10, stop_event=event)
        event.set()

        try:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_stop_from_another_thread_unblocks_read(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10, poll_interval=0.001)
        timer = Timer(0.05, acquire.stop)
        timer.start()

        try:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                acquire.read(timeout=5.0)
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            timer.cancel()
            acquire.close()

    def test_wrong_channel_count(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        data = np.zeros((10, CHANNELS + 1))
        feeder.stream.chunks.append((data, feeder.time + np.arange(10) / SFREQ))

        try:
            with self.assertRaisesRegex(ValueError, "channels"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_shape_errors(self) -> None:
        cases = (
            (np.zeros(10), np.zeros(10), "samples by channels"),
            (np.zeros((10, CHANNELS)), np.zeros((10, CHANNELS)), "one timestamp"),
            (np.zeros((10, CHANNELS)), np.zeros(5), "but"),
        )
        for data, timestamps, message in cases:
            with self.subTest(message=message):
                feeder = Feeder()
                acquire = acquire_for(feeder, 10)
                feeder.stream.chunks.append((data, timestamps))

                try:
                    with self.assertRaisesRegex(ValueError, message):
                        acquire.read(timeout=0.05)
                finally:
                    acquire.close()

    def test_non_numeric_samples(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        data = np.zeros((10, CHANNELS), dtype=object)
        feeder.stream.chunks.append((data, feeder.time + np.arange(10) / SFREQ))

        try:
            with self.assertRaisesRegex(ValueError, "numeric"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_closed_instance_rejects_read(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        acquire.close()

        self.assertEqual(feeder.stream.callbacks, [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            acquire.read(timeout=0.05)

    def test_late_chunk_after_close_is_ignored(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)
        callback = feeder.stream.callbacks[0]
        acquire.close()
        feeder.feed(10)
        data, timestamps = feeder.stream.chunks[0]
        callback(data, timestamps, None)

        self.assertEqual(acquire.pending_samples, 0)

    def test_invalid_arguments(self) -> None:
        stream = FakeStream()
        bad = (
            {"block_samples": 0, "sfreq": SFREQ},
            {"block_samples": 1.5, "sfreq": SFREQ},
            {"block_samples": True, "sfreq": SFREQ},
            {"block_samples": 10, "sfreq": 0},
            {"block_samples": 10, "sfreq": float("nan")},
            {"block_samples": 10, "sfreq": SFREQ, "poll_interval": 0},
            {"block_samples": 10, "sfreq": SFREQ, "no_data_timeout": 0},
            {"block_samples": 10, "sfreq": SFREQ, "max_lag_seconds": 0},
            {"block_samples": 10, "sfreq": SFREQ, "max_future_seconds": -1},
        )
        for options in bad:
            with self.subTest(options=options):
                values = dict(options)
                block_samples = values.pop("block_samples")
                with self.assertRaises(ValueError):
                    Acquire(stream, block_samples, **values)

    def test_invalid_read_timeout(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10)

        try:
            for timeout in (-1.0, float("nan")):
                with self.subTest(timeout=timeout):
                    with self.assertRaises(ValueError):
                        acquire.read(timeout=timeout)
        finally:
            acquire.close()


class TimingTests(unittest.TestCase):
    """Check age, future and gap accounting."""

    def test_old_samples_are_rejected(self) -> None:
        feeder = Feeder(start=local_clock() - 30.0)
        acquire = acquire_for(feeder, 10, max_lag_seconds=3.0)
        feeder.feed(10)

        try:
            with self.assertRaisesRegex(RuntimeError, "seconds old"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_age_guard_can_be_disabled(self) -> None:
        feeder = Feeder(start=local_clock() - 30.0)
        acquire = acquire_for(feeder, 10, max_lag_seconds=None)
        feeder.feed(10)

        try:
            data, _ = acquire.read(timeout=0.05)
        finally:
            acquire.close()

        self.assertEqual(data.shape, (10, CHANNELS))
        self.assertGreater(acquire.max_lag, 20.0)

    def test_future_samples_are_rejected(self) -> None:
        feeder = Feeder(start=local_clock() + 30.0)
        acquire = acquire_for(feeder, 10, max_future_seconds=0.1)
        feeder.feed(10)

        try:
            with self.assertRaisesRegex(RuntimeError, "ahead"):
                acquire.read(timeout=0.05)
        finally:
            acquire.close()

    def test_missing_samples_are_counted_not_rejected(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10, max_lag_seconds=None)

        try:
            feeder.feed(10)
            first, _ = acquire.read(timeout=0.05)

            feeder.skip(5)
            feeder.feed(10)
            second, _ = acquire.read(timeout=0.05)
        finally:
            acquire.close()

        self.assertEqual(first[-1, 0], 9)
        self.assertEqual(second[0, 0], 15)
        self.assertEqual(acquire.gaps, 1)
        self.assertGreater(acquire.max_gap, 1.0 / feeder.sfreq)

    def test_nonfinite_timestamps_pass_through(self) -> None:
        feeder = Feeder()
        acquire = acquire_for(feeder, 10, max_lag_seconds=None)
        data = np.zeros((10, CHANNELS))
        timestamps = feeder.time + np.arange(10) / SFREQ
        timestamps[3] = np.nan
        feeder.stream.chunks.append((data, timestamps))

        try:
            returned, returned_times = acquire.read(timeout=0.05)
        finally:
            acquire.close()

        self.assertEqual(returned.shape, (10, CHANNELS))
        self.assertTrue(np.isnan(returned_times[3]))


def synthetic_raw(seconds: float = 3.0, sfreq: float = SFREQ) -> mne.io.RawArray:
    """Return a deterministic two-channel recording."""

    times = np.arange(round(seconds * sfreq)) / sfreq
    data = np.vstack(
        [
            20e-6 * np.sin(2 * np.pi * 10 * times),
            40e-6 * np.sin(2 * np.pi * 10 * times),
        ]
    )
    info = mne.create_info(["F3", "F4"], sfreq, ["eeg", "eeg"])
    return mne.io.RawArray(data, info, verbose=False)


class PlayerLSLTests(unittest.TestCase):
    """Run the real manual acquisition path against a PlayerLSL outlet."""

    def test_fixed_blocks_from_a_real_outlet(self) -> None:
        name = f"nova-acquire-{uuid4().hex[:8]}"
        player = PlayerLSL(synthetic_raw(), name=name, chunk_size=37)
        player.start()
        stream = StreamLSL(bufsize=4.0, name=name)
        stream.connect(
            acquisition_delay=None,
            processing_flags=["clocksync"],
            timeout=10,
        )
        acquire = Acquire(
            stream,
            block_samples=50,
            sfreq=SFREQ,
            no_data_timeout=5.0,
            max_lag_seconds=None,
        )

        try:
            blocks = [acquire.read(timeout=5.0) for _ in range(4)]
        finally:
            acquire.close()
            if stream.connected:
                stream.disconnect()
            player.stop()

        for data, timestamps in blocks:
            self.assertEqual(data.shape, (50, 2))
            self.assertTrue(np.all(np.diff(timestamps) > 0))
        self.assertEqual(acquire.blocks, 4)
        self.assertEqual(acquire.samples, 200)


if __name__ == "__main__":
    unittest.main()
