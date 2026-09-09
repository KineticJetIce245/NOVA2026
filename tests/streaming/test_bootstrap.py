"""Tests for the bootstrap helpers (argument getter and session assembly)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nova2026.streaming.bootstrap import StreamSession, make_parser, parse_args
from nova2026.streaming.recording import read_metadata

CHANNELS = ("F3", "C3", "EOG")  # EEG first, then auxiliary


class StubStream:
    """Minimal stand-in for a connected, manually acquired StreamLSL."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.ch_names = tuple(names)
        self.info = {"nchan": len(names), "sfreq": 500.0}
        self.connected = True
        self.n_new_samples = 0
        self.callbacks = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)


class ArgsTests(unittest.TestCase):
    def test_parser_exposes_the_common_options(self) -> None:
        args = parse_args("test", argv=[])
        self.assertEqual(args.sfreq, 500.0)
        self.assertEqual(args.out_sfreq, 128.0)
        self.assertEqual(args.window, 2.0)
        self.assertIsNone(args.record)
        self.assertEqual(args.workers, 0)

    def test_extra_options_can_be_registered(self) -> None:
        args = parse_args(argv=[], extra=lambda p: p.add_argument("--model", default="a.pt"))
        self.assertEqual(args.model, "a.pt")

    def test_make_parser_returns_a_usable_parser(self) -> None:
        namespace = make_parser().parse_args(["--sfreq", "250"])
        self.assertEqual(namespace.sfreq, 250.0)


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.cwd() / ".tmp_tests"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_session_reorders_records_and_gates_windows(self) -> None:
        args = parse_args(argv=[])
        args.sfreq = 500.0
        args.out_sfreq = 128.0
        args.record = self.root

        # The source publishes EOG first; the session must reorder columns.
        stream = StubStream(("EOG", "C3", "F3"))
        session = StreamSession(stream, args, CHANNELS, source_unit_exponent=0)

        times = np.arange(50) / 500.0
        data = np.zeros((50, 3))
        data[:, 2] = 1e-5  # volts on "F3" (source column 2)

        reordered, _ = session.ingest(data, times)
        self.assertTrue(np.allclose(reordered[:, 0], 1e-5))  # F3 is now col 0
        self.assertEqual(session.recorder.chunks, 1)

        # The session never runs preprocessing: with no judges, the gate only
        # checks warm-up. Pre-warm-up windows are rejected, later ones valid.
        eeg_window = session.wrap(reordered, times, start_sample=0)
        self.assertFalse(eeg_window.valid)
        self.assertEqual(eeg_window.reasons, ())
        eeg_window = session.wrap(reordered, times, start_sample=session.warmup_samples)
        self.assertTrue(eeg_window.valid)

        session.close()
        metadata = read_metadata(session.recorder.path)
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(session.recorder.fif_path.exists())

    def test_wrap_unions_registered_judges(self) -> None:
        args = parse_args(argv=[])

        class Judge:
            """Minimal verdict provider standing in for QualityMonitor."""

            def __init__(self, name: str, active_from: float) -> None:
                self.name = name
                self.active_from = active_from

            def reasons(self, start: float, end: float) -> tuple[str, ...]:
                return (self.name,) if end >= self.active_from else ()

        session = StreamSession(
            StubStream(CHANNELS),
            args,
            CHANNELS,
            judges=(Judge("flatline", 0.5), Judge("amplitude", 0.7)),
        )
        window = np.zeros((10, 3))

        # No judge is active over this early window: only warm-up gates it.
        early = np.arange(10) / 500.0  # [0.0, 0.018] s
        eeg_window = session.wrap(window, early, start_sample=session.warmup_samples)
        self.assertTrue(eeg_window.valid)
        self.assertEqual(eeg_window.reasons, ())

        # A window spanning both active judges gets their unioned reasons.
        later = 0.7 + np.arange(10) / 500.0  # [0.7, 0.718] s
        eeg_window = session.wrap(window, later, start_sample=session.warmup_samples)
        self.assertFalse(eeg_window.valid)
        self.assertEqual(eeg_window.reasons, ("amplitude", "flatline"))

    def test_recorder_is_optional(self) -> None:
        args = parse_args(argv=[])
        stream = StubStream(CHANNELS)
        session = StreamSession(stream, args, CHANNELS)
        self.assertIsNone(session.recorder)
        times = np.arange(10) / 500.0
        session.ingest(np.zeros((10, 3)), times)
        session.close()

    def test_close_persists_stats_in_metadata(self) -> None:
        args = parse_args(argv=[])
        args.record = self.root
        session = StreamSession(StubStream(CHANNELS), args, CHANNELS)
        times = np.arange(10) / 500.0
        session.ingest(np.zeros((10, 3)), times)
        session.close(status="completed", stats={"valid": 4, "recoveries": 0})
        metadata = read_metadata(session.recorder.path)
        self.assertEqual(metadata["stats"], {"valid": 4, "recoveries": 0})

    def test_chain_geometry_is_derived_at_the_output_rate(self) -> None:
        args = parse_args(argv=[])
        args.out_sfreq = 128.0
        session = StreamSession(StubStream(CHANNELS), args, CHANNELS)
        self.assertEqual(session.judges, ())  # no judges by default
        self.assertEqual(session.out_sfreq, 128.0)
        self.assertEqual(session.window_samples, 256)  # 2 s @ 128 Hz
        self.assertEqual(session.hop_samples, 64)  # 0.5 s
        self.assertEqual(session.warmup_samples, 256)  # 2 s


if __name__ == "__main__":
    unittest.main()
