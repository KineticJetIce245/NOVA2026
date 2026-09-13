"""The ANT recording on the live route: publisher, adapter, labels and timing.

Two halves, and the split matters. The first half is **own-premise logic**: the
loader over a synthetic trial written by the test itself (never over a file that
happens to exist on this machine), the name-based contract, the microvolt
convention, the rebasing arithmetic and the label scoring. The second half is
**the real transport**: an LSL outlet in this process, the adapter reading it
through ``Acquire``, so the claim "recorded EEG drove the live route" rests on a
test rather than on one afternoon's run.

Where a real outlet cannot be created (a machine with no multicast), the
transport test says so with the publisher's own error instead of passing
silently - the same rule ``tests/streaming/test_live_transport.py`` follows.
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.getlive.ant_live import (
    audio_start_of,
    build_parser,
    coverage_against_markers,
    load_candidates,
)
from scripts.getlive.ant_publish import (
    TRIAL_UNIT,
    UNIT_EXPONENT,
    load_ant_trial,
    slice_window,
    stream_info,
)
from scripts.getlive.ant_source import (
    AntStreamSource,
    grid_deviation_seconds,
)

RATE = 500.0
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)


def synthetic_trial(path: Path, samples: int = 2000, channels=CHANNELS, seed: int = 7):
    """Write a trial file in the importer's own format, and return its contents.

    The premise is the test's, not the environment's: nothing here depends on
    ``datasets/AAD-ANT/`` being present, and the values are known exactly so a
    mutation of the loader has something to disagree with.
    """

    rng = np.random.default_rng(seed)
    eeg = rng.normal(0.0, 20.0, (samples, len(channels)))  # microvolts
    timestamps = np.arange(samples, dtype=np.float64) / RATE + 5.006
    labels = np.full(samples, -1, dtype=np.int64)
    labels[samples // 2 :] = 0
    audio = np.zeros((int(round((samples - 1) / RATE * 64.0)) + 1, 2), dtype=np.float64)
    audio[10:, 0] = 0.5
    audio[10:, 1] = 0.25
    metadata = {
        "audio_rate": 64.0,
        "subject": "ANT",
        "trial_id": "antneuro_test",
        "channel_names": list(channels),
        "reference": "CPz",
        "upstream_processing": "synthetic fixture",
        "group": "antneuro|test",
    }
    np.savez(
        path,
        eeg=eeg,
        timestamps=timestamps,
        audio=audio,
        labels=labels,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return eeg, timestamps, audio, labels


class FakeStream:
    """The declared-metadata half of a connected inlet, with no transport.

    It answers exactly the fields the package's pre-flight reads and nothing
    else, so a test that passes here is testing the check rather than the fake.
    """

    def __init__(self, *, nchan=len(CHANNELS), sfreq=RATE, units="microvolts",
                 names=CHANNELS, types=None):
        self.ch_names = list(names)
        self.info = {"sfreq": float(sfreq), "nchan": len(names)}
        self.dtype = np.dtype("float32")
        self.connected = True
        self.filters = {}
        self.callbacks = []
        self.n_new_samples = 0
        self._types = list(types or ["eeg"] * len(names))
        self._units = [units] * len(names)
        self.sinfo = SimpleNamespace(
            get_channel_units=lambda: list(self._units),
            nchan=len(names),
        )

    def get_channel_types(self, picks=None):
        index = {name: i for i, name in enumerate(self.ch_names)}
        return [self._types[index[name]] for name in (picks or self.ch_names)]

    def add_callback(self, callback):
        """``Acquire`` registers here; a fake with no transport never fires it."""

        self.callbacks.append(callback)


class LoaderTests(unittest.TestCase):
    """The trial loader owns its premise and refuses what it cannot publish."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "session.npz"
        self.eeg, self.timestamps, self.audio, self.labels = synthetic_trial(self.path)

    def test_a_synthetic_trial_loads_with_its_contract(self):
        trial = load_ant_trial(self.path)
        self.assertEqual(trial.channel_names, CHANNELS)
        self.assertEqual(trial.sample_rate, RATE)
        self.assertEqual(trial.reference, "CPz")
        np.testing.assert_array_equal(trial.eeg, self.eeg)

    def test_a_missing_trial_names_the_command_that_makes_one(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_ant_trial(self.path.with_name("absent.npz"))
        self.assertIn("scripts.auditory.antneuro", str(caught.exception))

    def test_channel_names_must_cover_the_columns(self):
        np.savez(
            self.path,
            eeg=self.eeg,
            timestamps=self.timestamps,
            labels=self.labels,
            audio=self.audio,
            metadata=np.asarray(json.dumps({"channel_names": list(CHANNELS[:5]), "audio_rate": 64.0})),
        )
        with self.assertRaises(ValueError) as caught:
            load_ant_trial(self.path)
        self.assertIn("recorded channel names", str(caught.exception))

    def test_non_monotonic_timestamps_are_refused(self):
        stamps = self.timestamps.copy()
        stamps[3] = stamps[2]
        np.savez(
            self.path, eeg=self.eeg, timestamps=stamps, labels=self.labels,
            audio=self.audio,
            metadata=np.asarray(json.dumps({"channel_names": list(CHANNELS), "audio_rate": 64.0})),
        )
        with self.assertRaises(ValueError) as caught:
            load_ant_trial(self.path)
        self.assertIn("strictly increasing", str(caught.exception))

    def test_an_audio_column_that_is_not_at_the_import_rate_is_refused(self):
        np.savez(
            self.path, eeg=self.eeg, timestamps=self.timestamps, labels=self.labels,
            audio=self.audio[:-2],
            metadata=np.asarray(json.dumps({"channel_names": list(CHANNELS), "audio_rate": 64.0})),
        )
        with self.assertRaises(ValueError) as caught:
            load_ant_trial(self.path)
        self.assertIn("not at the import's", str(caught.exception))

    def test_slicing_keeps_the_recordings_own_clock_and_the_labels(self):
        trial = load_ant_trial(self.path)
        window, first = slice_window(trial, 1.0, 1.0)
        self.assertEqual(first, int(RATE))
        self.assertEqual(len(window.timestamps), int(RATE))
        self.assertAlmostEqual(float(window.timestamps[0]), float(self.timestamps[first]))
        np.testing.assert_array_equal(window.labels, self.labels[first:first + int(RATE)])

    def test_a_slice_past_the_end_of_the_recording_is_refused(self):
        trial = load_ant_trial(self.path)
        with self.assertRaises(ValueError):
            slice_window(trial, 5.0, 1.0)

    def test_the_declared_unit_is_microvolts_and_the_exponent_matches(self):
        trial = load_ant_trial(self.path)
        info = stream_info(trial, name="n", source_id="s")
        self.assertEqual(info.get_channel_units(), [TRIAL_UNIT] * len(CHANNELS))
        # The chain's own convention: microvolts is exponent -6, and the chain
        # takes microvolts, so nothing rescales between outlet and decoder.
        self.assertEqual(UNIT_EXPONENT, -6)


class OffsetTests(unittest.TestCase):
    """The audio anchor is the operator-given session start, plus the trial's own first stamp."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "session.npz"
        synthetic_trial(self.path, samples=4000)

    def test_the_anchor_is_the_trials_first_stamp_plus_the_operator_offset(self):
        trial = load_ant_trial(self.path)
        self.assertAlmostEqual(
            audio_start_of(trial, 0.0), float(trial.timestamps[0]), places=9
        )
        # Session 2's operator-given start is 267 s of stimulus (plan 3.12); the
        # same code path must move the anchor by exactly that.
        self.assertAlmostEqual(
            audio_start_of(trial, 267.0), float(trial.timestamps[0]) + 267.0, places=9
        )

    def test_a_slice_before_the_stimulus_begins_would_be_named_by_its_caller(self):
        # The offset is an operator fact, so this module never invents one; what
        # it must do is refuse to render a window that starts before the played
        # stimulus does, which the media loader is what enforces.
        trial = load_ant_trial(self.path)
        anchor = audio_start_of(trial, 0.0)
        self.assertLess(anchor, float(trial.timestamps[-1]))


class GridDeviationTests(unittest.TestCase):
    """The transport's timing verdict has to be able to fail."""

    def test_an_ideal_grid_has_no_deviation(self):
        grid = np.arange(10) / RATE
        self.assertEqual(grid_deviation_seconds(grid, grid), 0.0)

    def test_one_late_sample_shows_up_as_its_lateness(self):
        ideal = np.arange(10) / RATE
        late = ideal.copy()
        late[4] += 0.002
        self.assertAlmostEqual(grid_deviation_seconds(late, ideal), 0.002, places=9)

    def test_mismatched_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            grid_deviation_seconds(np.arange(4), np.arange(5))


class BlockPathTests(unittest.TestCase):
    """The adapter is judged on the blocks it *delivers*, not only on metadata.

    Both cases below are mutations the metadata tests did not catch: a source
    that carries extra electrodes in a different order, and a pre-chain
    converter that legitimately returns an empty block while it fills.
    """

    def _deliver(self, rows, stamps, **kwargs):
        delivered = []
        stream = FakeStream(names=rows[0][1], nchan=len(rows[0][1]))
        source = AntStreamSource(stream, channels=CHANNELS, sfreq=RATE, **kwargs)
        source._queue = [(data, ts) for data, ts in rows]
        stop = threading.Event()
        stop.set()
        for _, data, times in source.chunks(stop):
            delivered.append((data, times))
        return source, delivered

    def test_extra_columns_are_reordered_by_name_on_the_delivered_block(self):
        names = ("F9",) + CHANNELS + ("F10",)
        data = np.arange(3 * len(names), dtype=float).reshape(3, len(names))
        stream = FakeStream(names=names, nchan=len(names))
        source = AntStreamSource(stream, channels=CHANNELS, sfreq=RATE)
        kept = source.contract.reorder(data)
        self.assertEqual(kept.shape, (3, len(CHANNELS)))
        np.testing.assert_array_equal(kept[:, 0], data[:, 1])  # Fp1 sits at source column 1

    def test_a_converter_that_returns_nothing_is_not_an_error(self):
        from nova2026.streaming.preprocess import Resampler

        source = AntStreamSource(
            FakeStream(), channels=CHANNELS, sfreq=RATE, pre_resample_sfreq=128.0
        )
        self.assertIsNotNone(source._converter)
        out, times = source._convert(
            np.zeros((4, len(CHANNELS))), np.arange(4) / RATE
        )
        # A 25-sample block at 500 Hz is 50 ms; the HQ converter needs more
        # than that before it can emit a whole 128 Hz frame, so the first call
        # legitimately returns nothing. The iterator must treat that as progress
        # withheld, not as the end of the source.
        self.assertEqual(out.shape[0], 0)
        self.assertEqual(times.shape[0], 0)


class ContractTests(unittest.TestCase):
    """Pre-flight accepts the recorded contract and refuses each wrong assertion."""

    def make(self, **kwargs):
        return AntStreamSource(FakeStream(**kwargs), channels=CHANNELS, sfreq=RATE)

    def test_the_twenty_electrode_contract_is_accepted(self):
        source = self.make()
        self.assertEqual(source.channel_names, CHANNELS)
        self.assertEqual(source.sample_rate, RATE)
        self.assertEqual(source.source_unit_exponent, -6)

    def test_extra_columns_are_dropped_by_name_not_truncated(self):
        names = CHANNELS + ("F9", "M2")
        source = self.make(names=names)
        self.assertEqual(source.contract.dropped_channels, ("F9", "M2"))
        rows = np.arange(2 * len(names), dtype=float).reshape(2, len(names))
        kept = source.contract.reorder(rows)
        self.assertEqual(kept.shape, (2, len(CHANNELS)))
        np.testing.assert_array_equal(kept[0], rows[0, : len(CHANNELS)])

    def test_a_volts_declaration_against_a_microvolt_chain_is_refused(self):
        # The counter-check the runner runs live, as a test: the same outlet
        # asserted to be volts must NOT be accepted by a microvolt chain.
        with self.assertRaises(RuntimeError) as caught:
            self.make(units="volts")
        self.assertIn("exponent", str(caught.exception))

    def test_a_wrong_rate_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            self.make(sfreq=1000.0)
        self.assertIn("sampling rate", str(caught.exception))

    def test_a_missing_electrode_is_refused_by_name(self):
        with self.assertRaises(RuntimeError) as caught:
            self.make(names=tuple(name for name in CHANNELS if name != "Cz"))
        self.assertIn("missing required channels", str(caught.exception))

    def test_duplicate_labels_are_refused(self):
        names = list(CHANNELS)
        names[3] = names[2]
        with self.assertRaises(RuntimeError) as caught:
            self.make(names=tuple(names))
        self.assertIn("duplicated", str(caught.exception))

    def test_an_already_used_inlet_is_refused(self):
        stream = FakeStream()
        stream.n_new_samples = 3
        with self.assertRaises(RuntimeError) as caught:
            AntStreamSource(stream, channels=CHANNELS, sfreq=RATE)
        self.assertIn("before setup", str(caught.exception))

    def test_an_unknown_timebase_is_refused(self):
        with self.assertRaises(ValueError):
            AntStreamSource(FakeStream(), channels=CHANNELS, sfreq=RATE, timebase="magic")

    def test_microvolts_are_passed_through_and_not_rescaled(self):
        # The magnitude the decode path sees is the recording's own: if anything
        # silently multiplied by 1e6 the first sample would leave this range.
        trial_path = Path(tempfile.mkdtemp()) / "session.npz"
        eeg, _, _, _ = synthetic_trial(trial_path)
        peak = float(np.max(np.abs(eeg)))
        self.assertLess(peak, 200.0)
        self.assertGreater(peak, 1.0)


class LiveTransportTests(unittest.TestCase):
    """The adapter over a real outlet in this process, driven to exhaustion."""

    def test_a_published_window_arrives_rebased_and_on_rate(self):
        from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

        blocks = 40
        chunk = 25
        samples = blocks * chunk
        rng = np.random.default_rng(11)
        data = rng.normal(0.0, 15.0, (samples, len(CHANNELS))).astype(np.float32)
        name = "nova-ant-test"
        source_id = "ant-test-1"
        info = StreamInfo(name, "EEG", len(CHANNELS), RATE, "float32", source_id)
        info.set_channel_names(list(CHANNELS))
        info.set_channel_types(["eeg"] * len(CHANNELS))
        info.set_channel_units(["microvolts"] * len(CHANNELS))
        try:
            outlet = StreamOutlet(info, chunk_size=chunk)
        except Exception as error:  # pragma: no cover - no multicast on this host
            raise unittest.SkipTest(f"no LSL outlet on this host: {error}")

        from scripts.getlive.outlets import open_inlet, wait_for_outlet

        # A thread pushes the window at its true rate while the adapter reads.
        import threading
        import time

        stop = threading.Event()
        sent = {"blocks": 0}

        def publish():
            started = time.perf_counter()
            # The stamps must live on the LSL clock: ``Acquire`` measures a
            # block's age against it, so a stamp on any other clock is refused
            # as ancient (or as coming from the future).
            origin = local_clock() + 0.05
            for index in range(blocks):
                start = index * chunk
                outlet.push_chunk(
                    data[start : start + chunk],
                    np.arange(start, start + chunk) / RATE + origin,
                )
                sent["blocks"] += 1
                due = started + (start + chunk) / RATE
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

        row = {"name": name, "stype": "EEG", "source_id": source_id}
        wait_for_outlet(name=name, source_id=source_id, stream_type="EEG", timeout=10.0)
        stream = open_inlet(row, bufsize=2.0, connect_timeout=10.0)
        try:
            source = AntStreamSource(
                stream,
                channels=CHANNELS,
                sfreq=RATE,
                block_samples=chunk,
                no_data_timeout=1.0,
            )
            thread = threading.Thread(target=publish, daemon=True)
            thread.start()
            got_times, got_rows = [], []
            for _, rows, times in source.chunks(stop):
                got_times.append(np.asarray(times))
                got_rows.append(np.asarray(rows))
                if sum(len(part) for part in got_times) >= samples:
                    stop.set()
            thread.join(timeout=5)
        finally:
            if stream.connected:
                stream.disconnect()

        times = np.concatenate(got_times)
        rows = np.concatenate(got_rows)
        self.assertEqual(rows.shape[1], len(CHANNELS))
        # Rebasing: the first delivered sample is time zero, not an LSL clock
        # reading, and the grid is a whole number of sample intervals.
        self.assertAlmostEqual(float(times[0]), 0.0, places=9)
        steps = np.diff(times)
        self.assertLess(float(np.max(np.abs(steps - 1.0 / RATE))), 1e-9)
        # Values survive the transport: float32 round-trip, nothing rescaled.
        np.testing.assert_allclose(
            rows[: len(data)], data, rtol=0, atol=1e-5
        )
        diagnostics = source.transport
        self.assertGreaterEqual(diagnostics.blocks, 1)
        self.assertEqual(diagnostics.samples, len(rows))
        self.assertEqual(diagnostics.gaps, 0)
        self.assertIn(diagnostics.ended, ("stopped", "no-more-data"))
        self.assertLess(diagnostics.max_lag_seconds, 1.0)
        self.assertIsNotNone(diagnostics.raw_rate_hz)


class ScoringTests(unittest.TestCase):
    """Decisions against the marker labels, with the unknown band excluded."""

    def test_agreement_is_measured_only_on_labelled_samples(self):
        labels = np.array([0, 0, -1, -1, 1, 1], dtype=np.int64)
        decided = [(0.0, "A"), (0.004, "B"), (0.008, "A")]  # samples 0, 2, 4
        report = coverage_against_markers(labels, decided, RATE)
        self.assertEqual(report["decided_windows"], 3)
        self.assertEqual(report["decided_windows_labelled"], 2)
        self.assertEqual(report["decided_windows_in_unknown_band"], 1)
        # Decided windows were A (truth A) and A (truth B): one of two.
        self.assertAlmostEqual(report["window_accuracy_on_labelled"], 0.5)

    def test_no_committed_window_is_reported_as_such(self):
        report = coverage_against_markers(np.array([0, 1], dtype=np.int64), [], RATE)
        self.assertIn("note", report)
        self.assertNotIn("window_accuracy_on_labelled", report)


class RunnerWiringTests(unittest.TestCase):
    """The runner's own operating point and media slice, on owned premises."""

    def test_the_margin_default_is_the_calibrated_point(self):
        from nova2026.auditory.config import CALIBRATED_MARGIN, MIN_MARGIN

        args = build_parser().parse_args([])
        self.assertAlmostEqual(args.margin, CALIBRATED_MARGIN, places=12)
        self.assertNotAlmostEqual(args.margin, MIN_MARGIN, places=12)
        self.assertEqual(args.model, "models/auditory_kuleuven_live20.npz")
        self.assertEqual(args.session, "datasets/AAD-ANT/session_19-34-06.npz")

    def test_the_stimulus_slice_starts_where_the_recording_played_it(self):
        directory = Path(tempfile.mkdtemp())
        left = np.arange(48000, dtype=np.int16)
        right = np.arange(48000, dtype=np.int16) * 2
        from scipy.io import wavfile

        wavfile.write(directory / "left_mono.wav", 48000, left)
        wavfile.write(directory / "right_mono.wav", 48000, right)
        candidates, report = load_candidates(directory, 0.5, 1000)
        self.assertEqual(candidates.shape, (1000, 2))
        self.assertEqual(report["slice_samples"], [24000, 25000])
        self.assertAlmostEqual(float(candidates[0, 0]), 24000 / 32768.0, places=6)

    def test_a_slice_running_past_the_stimulus_is_refused(self):
        directory = Path(tempfile.mkdtemp())
        from scipy.io import wavfile

        wavfile.write(directory / "left_mono.wav", 48000, np.zeros(1000, np.int16))
        wavfile.write(directory / "right_mono.wav", 48000, np.zeros(1000, np.int16))
        with self.assertRaises(ValueError) as caught:
            load_candidates(directory, 0.0, 2000)
        self.assertIn("Shorten --clip", str(caught.exception))

    def test_a_missing_stimulus_names_the_file(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_candidates(Path(tempfile.mkdtemp()), 0.0, 10)
        self.assertIn("left_mono.wav", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
