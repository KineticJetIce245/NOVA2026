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

    def test_session_reorders_records_and_processes(self) -> None:
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

        processed, _ = session.process(reordered, times)
        self.assertIsNotNone(processed)

        # Warm-up gate rejects the very first windows.
        accepted, reasons = session.gate(0, np.arange(10) / 128.0 + 1000.0)
        self.assertFalse(accepted)
        self.assertEqual(reasons, ())

        session.close()
        metadata = read_metadata(session.recorder.path)
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(session.recorder.fif_path.exists())

    def test_recorder_is_optional(self) -> None:
        args = parse_args(argv=[])
        stream = StubStream(CHANNELS)
        session = StreamSession(stream, args, CHANNELS)
        self.assertIsNone(session.recorder)
        times = np.arange(10) / 500.0
        session.ingest(np.zeros((10, 3)), times)
        session.close()

    def test_chain_geometry_is_derived_at_the_output_rate(self) -> None:
        args = parse_args(argv=[])
        args.out_sfreq = 128.0
        session = StreamSession(StubStream(CHANNELS), args, CHANNELS)
        self.assertEqual(session.window_samples, 256)  # 2 s @ 128 Hz
        self.assertEqual(session.hop_samples, 64)  # 0.5 s
        self.assertEqual(session.warmup_samples, 256)  # 2 s


if __name__ == "__main__":
    unittest.main()
