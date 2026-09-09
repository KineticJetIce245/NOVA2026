"""Tests for the fixed-capacity window buffer."""

import unittest

import numpy as np

from nova2026.streaming.circular_buffer import CircularBuffer

SFREQ = 128.0
ANCHOR = 1000.0
CHANNELS = 2


def rows(
    count: int,
    first: int = 0,
    *,
    anchor: float = ANCHOR,
    sfreq: float = SFREQ,
    channels: int = CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniquely identifiable rows and their uniform timestamps."""

    values = np.arange(first, first + count, dtype=np.float64)
    data = np.repeat(values[:, None], channels, axis=1)
    timestamps = anchor + np.arange(first, first + count) / sfreq
    return data, timestamps


def buffer_for(
    window: int, hop: int, capacity: int, **kwargs
) -> CircularBuffer:
    """Build a buffer over the test geometry."""

    options = {"sfreq": SFREQ, "n_channels": CHANNELS}
    options.update(kwargs)
    return CircularBuffer(window, hop, capacity, **options)


def first_column(windows) -> list[float]:
    """Return the first value of every window."""

    return [float(window[0][0, 0]) for window in windows]


class WindowTests(unittest.TestCase):
    """Check window geometry, contents and timestamps."""

    def test_non_overlapping_windows(self) -> None:
        buffer = buffer_for(4, 4, 16)
        data, timestamps = rows(12)

        windows = buffer.push(data, timestamps)

        self.assertEqual(len(windows), 3)
        self.assertEqual(first_column(windows), [0.0, 4.0, 8.0])
        for index, (window, window_times, start) in enumerate(windows):
            self.assertEqual(window.shape, (4, CHANNELS))
            self.assertEqual(start, index * 4)
            self.assertTrue(np.all(np.diff(window_times) > 0))
            self.assertEqual(window_times[0], ANCHOR + index * 4 / SFREQ)
        self.assertEqual(buffer.total_written, 12)
        self.assertEqual(buffer.windows, 3)

    def test_overlapping_windows(self) -> None:
        buffer = buffer_for(4, 2, 16)
        data, timestamps = rows(8)

        windows = buffer.push(data, timestamps)

        self.assertEqual(first_column(windows), [0.0, 2.0, 4.0])
        self.assertEqual([window[2] for window in windows], [0, 2, 4])
        self.assertTrue(
            np.array_equal(windows[1][0], np.array([[2.0] * 2, [3.0] * 2, [4.0] * 2, [5.0] * 2]))
        )

    def test_hop_larger_than_window_skips_rows(self) -> None:
        buffer = buffer_for(4, 6, 16)
        data, timestamps = rows(12)

        windows = buffer.push(data, timestamps)

        self.assertEqual(first_column(windows), [0.0, 6.0])
        self.assertEqual([window[2] for window in windows], [0, 6])

    def test_partial_push_emits_nothing(self) -> None:
        buffer = buffer_for(4, 4, 16)
        data, timestamps = rows(3)

        self.assertEqual(buffer.push(data, timestamps), [])
        self.assertEqual(buffer.total_written, 3)
        self.assertEqual(buffer.windows, 0)

    def test_one_push_can_complete_several_windows(self) -> None:
        buffer = buffer_for(4, 4, 8)
        data, timestamps = rows(12)

        windows = buffer.push(data, timestamps)

        self.assertEqual(first_column(windows), [0.0, 4.0, 8.0])
        self.assertEqual([window[2] for window in windows], [0, 4, 8])
        self.assertEqual(buffer.windows, 3)

    def test_capacity_equal_to_window_keeps_every_window(self) -> None:
        buffer = buffer_for(8, 8, 8)
        data, timestamps = rows(24)

        windows = buffer.push(data, timestamps)

        self.assertEqual(first_column(windows), [0.0, 8.0, 16.0])
        self.assertEqual(float(windows[-1][0][-1, 0]), 23.0)

    def test_split_pushes_match_one_push(self) -> None:
        one = buffer_for(4, 4, 8)
        split = buffer_for(4, 4, 8)
        data, timestamps = rows(12)

        expected = one.push(data, timestamps)
        received = []
        for start in range(0, 12, 5):
            received.extend(split.push(data[start : start + 5], timestamps[start : start + 5]))

        self.assertEqual(first_column(received), first_column(expected))
        self.assertEqual(
            [window[2] for window in received], [window[2] for window in expected]
        )

    def test_window_timestamps_use_the_first_finite_anchor(self) -> None:
        buffer = buffer_for(4, 4, 8)
        data, timestamps = rows(4)
        timestamps[0] = np.nan

        windows = buffer.push(data, timestamps)
        window_times = windows[0][1]

        self.assertEqual(window_times[0], ANCHOR)
        self.assertEqual(window_times[3], ANCHOR + 3.0 / SFREQ)

    def test_all_nan_timestamps_yield_nan_window_times(self) -> None:
        buffer = buffer_for(4, 4, 8)
        data, timestamps = rows(4)
        timestamps[:] = np.nan

        windows = buffer.push(data, timestamps)

        self.assertTrue(np.all(np.isnan(windows[0][1])))

    def test_empty_push_changes_nothing(self) -> None:
        buffer = buffer_for(4, 4, 8)

        windows = buffer.push(np.empty((0, CHANNELS)), np.empty(0))

        self.assertEqual(windows, [])
        self.assertEqual(buffer.total_written, 0)
        self.assertEqual(buffer.windows, 0)

    def test_returned_window_is_a_copy(self) -> None:
        buffer = buffer_for(4, 4, 8)
        data, timestamps = rows(4)
        window = buffer.push(data, timestamps)[0]

        # Wrap the ring so the rows behind the first window are overwritten.
        later, later_times = rows(8, first=4)
        buffer.push(later, later_times)

        self.assertTrue(np.array_equal(window[0][:, 0], np.arange(4.0)))

    def test_rows_are_cast_to_the_storage_dtype(self) -> None:
        buffer = buffer_for(4, 4, 8, dtype=np.float64)
        data, timestamps = rows(4)
        data = data.astype(np.float32)

        windows = buffer.push(data, timestamps)

        self.assertEqual(windows[0][0].dtype, np.float64)

    def test_reset_restarts_counting_and_reanchors(self) -> None:
        buffer = buffer_for(4, 4, 8)
        data, timestamps = rows(4)
        buffer.push(data, timestamps)

        buffer.reset()
        self.assertEqual(buffer.total_written, 0)
        self.assertEqual(buffer.windows, 0)

        moved, moved_times = rows(4, anchor=ANCHOR + 10.0)
        windows = buffer.push(moved, moved_times)

        self.assertEqual(windows[0][2], 0)
        self.assertEqual(windows[0][1][0], ANCHOR + 10.0)
        self.assertEqual(buffer.total_written, 4)


class ValidationTests(unittest.TestCase):
    """Check geometry and shape rejection."""

    def test_invalid_geometry(self) -> None:
        cases = (
            (0, 4, 8, SFREQ, CHANNELS),
            (4, 0, 8, SFREQ, CHANNELS),
            (4, 4, 0, SFREQ, CHANNELS),
            (4, 4, 2, SFREQ, CHANNELS),
            (4, 4, 8, 0, CHANNELS),
            (4, 4, 8, float("nan"), CHANNELS),
            (4, 4, 8, SFREQ, 0),
            (True, 4, 8, SFREQ, CHANNELS),
        )
        for window, hop, capacity, sfreq, channels in cases:
            with self.subTest(window=window, hop=hop, capacity=capacity):
                with self.assertRaises(ValueError):
                    CircularBuffer(window, hop, capacity, sfreq, channels)

    def test_shape_errors(self) -> None:
        buffer = buffer_for(4, 4, 8)
        cases = (
            (np.zeros(4), np.zeros(4)),
            (np.zeros((4, CHANNELS + 1)), np.zeros(4)),
            (np.zeros((4, CHANNELS)), np.zeros(3)),
            (np.zeros((4, CHANNELS)), np.zeros((4, 1))),
        )
        for data, timestamps in cases:
            with self.subTest(shape=data.shape):
                with self.assertRaises(ValueError):
                    buffer.push(data, timestamps)
        self.assertEqual(buffer.total_written, 0)


class CompositionTests(unittest.TestCase):
    """Check that Acquire blocks feed the buffer without an adapter."""

    def test_acquire_blocks_fill_overlapping_windows(self) -> None:
        from uuid import uuid4

        import mne
        from mne_lsl.player import PlayerLSL
        from mne_lsl.stream import StreamLSL

        from nova2026.streaming.acquire import Acquire

        sfreq = 500.0
        times = np.arange(int(3.0 * sfreq)) / sfreq
        raw = mne.io.RawArray(
            np.vstack(
                [
                    20e-6 * np.sin(2 * np.pi * 10 * times),
                    40e-6 * np.sin(2 * np.pi * 10 * times),
                ]
            ),
            mne.create_info(["F3", "F4"], sfreq, ["eeg", "eeg"]),
            verbose=False,
        )
        name = f"nova-buffer-{uuid4().hex[:8]}"
        player = PlayerLSL(raw, name=name, chunk_size=37)
        player.start()
        stream = StreamLSL(bufsize=4.0, name=name)
        stream.connect(
            acquisition_delay=None, processing_flags=["clocksync"], timeout=10
        )
        acquire = Acquire(
            stream,
            block_samples=50,
            sfreq=sfreq,
            no_data_timeout=5.0,
            max_lag_seconds=None,
        )
        buffer = CircularBuffer(
            window_samples=100, hop_samples=50, capacity_samples=400, sfreq=sfreq,
            n_channels=2,
        )

        try:
            windows = []
            for _ in range(4):
                data, timestamps = acquire.read(timeout=5.0)
                windows.extend(buffer.push(data, timestamps))
        finally:
            acquire.close()
            if stream.connected:
                stream.disconnect()
            player.stop()

        self.assertEqual(len(windows), 3)
        self.assertEqual([window[2] for window in windows], [0, 50, 100])
        for data, timestamps, _ in windows:
            self.assertEqual(data.shape, (100, 2))
            self.assertTrue(
                np.allclose(np.diff(timestamps), 1.0 / sfreq, atol=1e-9)
            )
        # A hop of half a window must make consecutive windows share rows.
        self.assertTrue(np.array_equal(windows[0][0][50:], windows[1][0][:50]))


if __name__ == "__main__":
    unittest.main()
