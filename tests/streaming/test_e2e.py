"""End-to-end tests: live blocking behaviour + recorded offline replay parity.

These tests exercise the *real* path used by the demo, without any mocking of
the transport:

1. ``test_live_run_keeps_up_and_records`` streams a deterministic MNE
   recording through PlayerLSL, checks the loop keeps up with real time (no
   blocking), and saves the raw data with ``RunRecorder``.
2. ``test_recorded_run_replays_identically`` reads that SQLite run back and
   pushes the exact same raw chunks through a fresh pipeline, asserting the
   offline windows match the live windows (no computation drift).
3. ``test_offline_pipeline_is_deterministic`` re-processes the same chunks
   twice without any transport and checks both runs agree.
"""

import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

import mne
import numpy as np
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL

from nova2026.streaming import Acquire, ChannelContract
from nova2026.streaming.circular_buffer import CircularBuffer
from nova2026.streaming.preprocess import (
    QualityMonitor,
    Resampler,
    SosFilter,
    design_bandpass,
    design_notch,
    unit_scaler,
)
from nova2026.streaming.recording import (
    RunRecorder,
    RunSpec,
    read_metadata,
    replay_chunks,
)

CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")
SFREQ = 500.0
OUT_SFREQ = 128.0
BLOCK = 50  # samples per acquire block (0.1 s)


def make_raw(seconds: float, sfreq: float = SFREQ) -> mne.io.RawArray:
    """Deterministic multi-channel recording in volts.

    Every channel is a sum of two sines with channel-dependent frequencies so
    that filtering, resampling and windowing all have something real to work
    on, while the result stays reproducible.
    """

    times = np.arange(round(seconds * sfreq)) / sfreq
    rows = []
    for index, _ in enumerate(CHANNELS):
        amplitude = (10.0 + index) * 1e-6
        rows.append(
            amplitude
            * (
                np.sin(2 * np.pi * 10 * times)
                + 0.3 * np.sin(2 * np.pi * (5 + index) * times)
            )
        )
    data = np.stack(rows)  # (channels, samples)
    info = mne.create_info(list(CHANNELS), sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


class ChainRunner:
    """Run the exact demo chain over raw blocks and keep every window.

    Replicates the demo's own `STAGES` tuple (scale -> quality -> filters ->
    resample) plus the warm-up/judge gate, without the recording logic, so a
    live run and an offline replay use literally the same code path.
    """

    def __init__(self, channels: int = len(CHANNELS)) -> None:
        self.scaler = unit_scaler(0, desired_exponent=-6)
        self.quality = QualityMonitor(
            n_eeg=channels, sfreq=SFREQ, warmup_seconds=0.0
        )
        self.notch = SosFilter(design_notch(60.0, 30.0, SFREQ), channels)
        self.bandpass = SosFilter(
            design_bandpass(1.0, 45.0, 3, SFREQ), channels
        )
        self.resampler = Resampler(SFREQ, OUT_SFREQ, channels, quality="LQ")
        self.buffer = CircularBuffer(
            int(2 * OUT_SFREQ),      # 2 s windows
            int(0.5 * OUT_SFREQ),    # 0.5 s hop
            int(6 * OUT_SFREQ),      # 6 s ring
            sfreq=OUT_SFREQ,
            n_channels=channels,
        )
        self.warmup = int(2 * OUT_SFREQ)
        self.records = []  # list of (window data, window times, start, valid)

    def feed(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Process one raw block exactly like the demo loop does."""

        data, timestamps = self.scaler(data, timestamps)
        self.quality.feed(data, timestamps)
        data, timestamps = self.notch(data, timestamps)
        data, timestamps = self.bandpass(data, timestamps)
        data, timestamps = self.resampler(data, timestamps)

        for window, window_times, start in self.buffer.push(data, timestamps):
            reasons = self.quality.reasons(
                float(window_times[0]), float(window_times[-1])
            )
            valid = start >= self.warmup and not reasons
            self.records.append((window.copy(), start, valid))

    def assert_matches(self, other: "ChainRunner") -> None:
        """Raise AssertionError when another runner's windows differ."""

        def check(condition: bool, message: str) -> None:
            if not condition:
                raise AssertionError(message)

        check(len(self.records) == len(other.records),
              f"window count differs: {len(self.records)} vs {len(other.records)}")
        for (window, start, valid), (other_window, other_start, other_valid) in zip(
            self.records, other.records
        ):
            check(start == other_start, "window starts differ")
            check(valid == other_valid, "window validity differs")
            check(window.shape == other_window.shape, "window shapes differ")
            check(
                np.allclose(window, other_window, rtol=1e-6, atol=1e-6),
                "offline window differs from the live window",
            )


class LiveRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def stream_raw_for(self, seconds: float):
        """Return (raw, name, player, stream) wired but not yet recording."""

        raw = make_raw(seconds + 4.0)  # extra lead time for filter warm-up
        name = f"nova-e2e-{uuid4().hex[:8]}"
        player = PlayerLSL(raw, name=name, chunk_size=37)
        player.start()
        stream = StreamLSL(bufsize=4.0, name=name)
        stream.connect(
            acquisition_delay=None,
            processing_flags=["clocksync"],
            timeout=10,
        )
        return raw, player, stream

    def test_live_run_keeps_up_and_records(self) -> None:
        duration = 8.0  # seconds of real-time streaming
        raw, player, stream = self.stream_raw_for(duration)

        recorder = RunRecorder(
            self.root,
            RunSpec("e2e", "live", "run01", role="trial"),
            CHANNELS,
            SFREQ,
            unit_exponent=0,
            dtype=np.float64,
        )
        contract = ChannelContract(stream.ch_names, CHANNELS)
        acquire = Acquire(stream, block_samples=BLOCK, sfreq=SFREQ,
                          max_lag_seconds=3.0)
        runner = ChainRunner()

        started = time.monotonic()
        try:
            while time.monotonic() - started < duration:
                data, timestamps = acquire.read()
                data = contract.reorder(data)
                recorder.write(data, timestamps)
                runner.feed(data, timestamps)
        finally:
            acquire.close()
            if stream.connected:
                stream.disconnect()
            player.stop()
            path = recorder.close(status="completed")

        elapsed = time.monotonic() - started

        # --- Blocking checks: the loop must keep up with real time. ---
        expected_samples = round(duration * SFREQ)
        self.assertLess(elapsed, duration + 1.5, "the loop fell behind wall time")
        self.assertGreaterEqual(elapsed, duration)
        # No samples lost and no giant backlog: the lag guard never fired.
        self.assertGreaterEqual(recorder.samples, expected_samples - BLOCK)
        self.assertLess(acquire.max_lag, 3.0)
        # After warm-up + SoXR latency there must be at least one valid window.
        self.assertGreaterEqual(
            sum(1 for _, _, valid in runner.records if valid), 1,
            "no valid window arrived during the live run",
        )

        # --- The recorded run must exist and be complete. ---
        metadata = read_metadata(path)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["samples"], recorder.samples)
        self.assertTrue(recorder.fif_path.exists())

        # Keep the recorded data + live windows for the offline parity test.
        self._recorded_path = path
        self._live_records = runner.records

    def test_recorded_run_replays_identically(self) -> None:
        """Feed the saved SQLite chunks offline and compare with live output."""

        # Record a short run first (reusing the same deterministic source).
        duration = 6.0
        raw, player, stream = self.stream_raw_for(duration)
        recorder = RunRecorder(
            self.root,
            RunSpec("e2e", "replay", "run01", role="trial"),
            CHANNELS,
            SFREQ,
            unit_exponent=0,
            dtype=np.float64,
        )
        contract = ChannelContract(stream.ch_names, CHANNELS)
        acquire = Acquire(stream, block_samples=BLOCK, sfreq=SFREQ)
        runner = ChainRunner()

        started = time.monotonic()
        try:
            while time.monotonic() - started < duration:
                data, timestamps = acquire.read()
                data = contract.reorder(data)
                recorder.write(data, timestamps)
                runner.feed(data, timestamps)
        finally:
            acquire.close()
            if stream.connected:
                stream.disconnect()
            player.stop()
            recorder.close(status="completed")

        # --- Offline replay: same raw chunks through a brand-new pipeline. ---
        replay = ChainRunner()
        stored = []
        for data, timestamps in replay_chunks(recorder.path):
            stored.append(data)
            replay.feed(data, timestamps)

        # Recorder round-trip: the FIF export agrees with the SQLite chunks.
        # (PlayerLSL drops its first chunk, so the recorded content is not
        # expected to start at sample 0 of the source recording.)
        saved = np.concatenate(stored, axis=0)
        fif = mne.io.read_raw_fif(recorder.fif_path, preload=True, verbose=False)
        self.assertTrue(np.allclose(fif.get_data().T, saved, rtol=1e-9, atol=1e-12))
        self.assertEqual(len(saved), recorder.samples)

        # Offline windows must match the live windows sample-for-sample.
        runner.assert_matches(replay)

    def test_offline_pipeline_is_deterministic(self) -> None:
        """Processing identical chunks twice must give identical windows."""

        raw = make_raw(3.0)
        data = raw.get_data().T  # (samples, channels), volts

        def run_offline() -> list:
            runner = ChainRunner()
            for start in range(0, len(data), BLOCK):
                block = data[start : start + BLOCK]
                runner.feed(
                    block, np.arange(len(block)) / SFREQ + 1000.0
                )
            return runner.records

        first = run_offline()
        second = run_offline()
        self.assertEqual(len(first), len(second))
        for (window, start, valid), (other, other_start, other_valid) in zip(
            first, second
        ):
            self.assertEqual(start, other_start)
            self.assertEqual(valid, other_valid)
            self.assertTrue(np.allclose(window, other, rtol=0, atol=0))


if __name__ == "__main__":
    unittest.main()
