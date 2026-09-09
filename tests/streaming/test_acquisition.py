"""Tests for the streaming acquisition layer."""

import time
import unittest

import mne
import numpy as np
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL

NAME = "nova-acq-test"
SFREQ = 250.0
BUFSIZE = 4.0


def make_raw(seconds: float = 6.0) -> mne.io.RawArray:
    """Return a deterministic two-channel recording with known amplitudes."""

    n_samples = round(seconds * SFREQ)
    times = np.arange(n_samples) / SFREQ
    data = np.vstack(
        [
            20e-6 * np.sin(2 * np.pi * 10 * times),
            40e-6 * np.sin(2 * np.pi * 10 * times),
        ]
    )
    info = mne.create_info(["F3", "F4"], SFREQ, ["eeg", "eeg"])
    return mne.io.RawArray(data, info, verbose=False)


class StreamAcquisitionTests(unittest.TestCase):
    """Record what the MNE-LSL manual-acquire path actually delivers."""

    def setUp(self) -> None:
        self.player = PlayerLSL(make_raw(), name=NAME)
        self.player.start()
        self.stream = StreamLSL(bufsize=BUFSIZE, name=NAME)
        self.stream.connect(
            acquisition_delay=None,
            processing_flags=["clocksync"],
            timeout=10,
        )

    def tearDown(self) -> None:
        if self.stream.connected:
            self.stream.disconnect()
        self.player.stop()

    def acquire_until_data(self, timeout: float = 5.0):
        """Call acquire() until at least one sample arrives, then return them."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.stream.acquire()
            if self.stream.n_new_samples:
                return self.stream.get_data(exclude=())
            time.sleep(0.05)
        self.fail(f"No data arrived within {timeout} seconds.")

    def test_get_data_returns_channels_by_samples(self) -> None:
        data, timestamps = self.acquire_until_data()

        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(timestamps.shape, (data.shape[1],))
        # The dtype is the source's format, not fixed by MNE-LSL. PlayerLSL
        # publishes float64 here; a real amplifier may publish another float type.
        self.assertTrue(np.issubdtype(data.dtype, np.floating))
        self.assertTrue(np.issubdtype(timestamps.dtype, np.floating))
        self.assertGreater(data.shape[1], 0)
        # Buffer timestamps are not guaranteed to be strictly increasing without
        # the 'dejitter'/'monotize' processing flags, which we deliberately omit.
        self.assertTrue(np.all(np.isfinite(timestamps)))

    def test_callback_receives_samples_by_channels(self) -> None:
        seen = []

        def callback(data, timestamps, info):
            seen.append((data.shape, timestamps.shape))
            return data, timestamps

        self.stream.add_callback(callback)
        self.acquire_until_data()

        self.assertGreater(len(seen), 0)
        n_times, n_channels = seen[0][0]
        self.assertEqual(n_channels, 2)
        self.assertGreater(n_times, 0)
        self.assertEqual(seen[0][1], (n_times,))

    def test_get_data_resets_new_sample_counter(self) -> None:
        self.acquire_until_data()
        self.assertEqual(self.stream.n_new_samples, 0)

        self.stream.acquire()
        while not self.stream.n_new_samples:
            self.stream.acquire()
            time.sleep(0.05)

        before = self.stream.n_new_samples
        self.assertGreater(before, 0)
        self.stream.get_data(exclude=())
        self.assertEqual(self.stream.n_new_samples, 0)


if __name__ == "__main__":
    unittest.main()
