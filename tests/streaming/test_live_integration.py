"""The whole live path, driven end to end without an amplifier.

``test_live_loop.py`` fakes the session to pin the time base's placement. This
file goes further: ``live._prepare`` builds the **real** chain and the **real**
``StreamSession`` - real ``CircularBuffer``, real ``wrap``/``collect_verdict``,
real judges, real ``RunRecorder`` - and only the acquisition handle is swapped for
a fixture. ``_loop``, ``_finalize`` and ``_finish`` then run for real, so the
window gate, the per-electrode statistics, the acceptance rules and the recording
on disk are all exercised.

The fixture is the rig, reduced to what made it fail, with the numbers the
2026-09-12 bring-up measured: a source clock 200 ppm slow, a sub-nominal step
every 211 samples (the measured 0.48%), a whole-sample re-stamp every 500, and
samples sitting on a 12.6 mV electrode offset. Both scenarios below feed **the
same signal and the same timeline** and differ only in ``--timebase``:

* ``stamps`` (the default, and what shipped) hands those sub-nominal steps
  straight to ``Repair``, which refuses any step below one sample at any
  tolerance. Five of them exhaust the recovery budget inside two seconds of
  stream time and the run dies with "Too many data faults" - the rig's failure,
  reproduced.
* ``grid`` absorbs the same timeline and runs to the end, with the acceptance
  rules accepting it.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_live_integration -v
"""

import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from nova2026.streaming import TimeBase
from nova2026.streaming.recording import iter_chunks, iter_windows, read_metadata
from nova2026.streaming.preflight import ChannelContract
from scripts.getlive import live
from scripts.getlive.cap import select_profile

RATE = 500.0
BLOCK = 50
CLOCK_STEP = 0.01
DURATION = 1.1  # 109 iterations of the patched clock, so ~5450 samples
COUNT = 7000
CHANNELS = ("Fz", "Oz")

# The rig's measured numbers.
OFFSET_UV = 12_640.0
NOISE_UV = 20.0
DRIFT_PPM = -200.0
JITTER_EVERY = 211
JITTER_STEP_SAMPLES = 0.4
RE_STAMP_EVERY = 500


def rig_like_signal(count: int = COUNT, channels: int = len(CHANNELS), seed: int = 7):
    """Microvolt samples on the rig's scale: offset plus EEG-sized noise."""

    rng = np.random.default_rng(seed)
    data = OFFSET_UV + rng.normal(0.0, NOISE_UV, size=(count, channels))
    return np.ascontiguousarray(data, dtype="float32")


def rig_like_timeline(
    count: int = COUNT, *, base: float = 1000.0, rate: float = RATE
) -> np.ndarray:
    """The measured timeline: a slow clock, sub-nominal steps, re-stamps."""

    true_rate = rate * (1.0 + DRIFT_PPM / 1e6)
    stamps = base + np.arange(count) / true_rate
    # The sub-nominal steps: a sample stamped early, which is what Repair refuses
    # outright and what relay.py used to invent phantom samples for.
    for index in range(JITTER_EVERY, count, JITTER_EVERY):
        stamps[index] -= JITTER_STEP_SAMPLES / rate
    # The whole-sample re-stamps: the staircase the reports called "gaps".
    for index in range(RE_STAMP_EVERY, count, RE_STAMP_EVERY):
        stamps[index:] += 1.0 / rate
    return stamps


def fake_clock(step: float = CLOCK_STEP):
    """A monotonic clock that advances once per call, so the loop is finite."""

    counter = itertools.count()
    return lambda: next(counter) * step


class StubStream:
    """Enough of ``StreamLSL`` for ``StreamSession`` to be built around it.

    The session reads ``ch_names`` for the contract and ``info['nchan']`` through
    ``Acquire``; nothing here ever acquires, because the test swaps the handle.
    """

    def __init__(self, ch_names) -> None:
        self.ch_names = list(ch_names)
        self.info = {"nchan": len(ch_names)}
        self.callbacks = []
        self.connected = True
        self.n_new_samples = 0

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def acquire(self):
        return None

    def get_data(self, winsize=None, exclude=()):
        return None


class FixtureAcquire:
    """The acquisition surface ``_loop`` uses, fed straight from the fixture."""

    def __init__(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        self._data = data
        self._timestamps = timestamps
        self._next = 0
        self.pending_samples = 0
        self.max_lag = 0.0
        self.gaps = 0
        self.closed = False

    def read(self, timeout=None):
        begin, end = self._next, min(self._next + BLOCK, self._timestamps.size)
        if end <= begin:
            raise AssertionError("the fixture ran out before the loop finished")
        self._next = end
        return self._data[begin:end].copy(), self._timestamps[begin:end].copy()

    def close(self) -> None:
        self.closed = True


def build_run(root: Path, *, mode: str, session: str):
    """Assemble the run the way ``live.main`` does, around the fixture."""

    args = live.build_parser().parse_args(
        [
            "--sfreq", str(RATE),
            "--source-units", "uV",
            "--duration", str(DURATION),
            "--quiet",
            "--cap", "declared",
            "--eog", "drop",
            "--timebase", mode,
            "--record", str(root),
            "--subject", "test",
            "--session", session,
            "--run", "fixed",
            "--out", str(root / f"{session}-report.json"),
        ]
    )
    profile = select_profile("declared", CHANNELS, None)
    stream = StubStream(CHANNELS)
    source = {
        "name": "synthetic-rig",
        "stype": "EEG",
        "source_id": "synthetic-rig",
        "n_channels": len(CHANNELS),
        "sfreq": RATE,
        "channels": list(CHANNELS),
        "types": ["eeg"] * len(CHANNELS),
        "units": ["microvolts"] * len(CHANNELS),
    }
    run = live._Run(
        args=args,
        stream=stream,
        source=source,
        contract=ChannelContract(tuple(CHANNELS), CHANNELS),
        profile=profile,
        eeg=CHANNELS,
        eog=(),
        excluded=(),
    )
    return run


def drive(run, data: np.ndarray, stamps: np.ndarray) -> int:
    """Run prepare, the loop and the scoring, with acquisition swapped out."""

    with mock.patch.object(live, "monotonic", new=fake_clock()), redirect_stdout(
        io.StringIO()
    ):
        live._prepare(run)
        run.session.acquire = FixtureAcquire(data, stamps)
        live._loop(run)
        return live._finish(run)


def recorded(run) -> tuple[list[tuple[int, float]], dict]:
    """Read the run's own recording back through the library's reader."""

    path = run.session.recorder.path
    chunks = [
        (chunk.shape[0], first) for chunk, first in iter_chunks(path)
    ]
    return chunks, dict(read_metadata(path))


class RigFailureReproducedTests(unittest.TestCase):
    """The default timeline treatment fails on rig data, and the grid does not."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = rig_like_signal()
        self.stamps = rig_like_timeline()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_source_timeline_still_kills_the_run(self) -> None:
        run = build_run(self.root, mode="stamps", session="stamps")
        code = drive(run, self.data, self.stamps)

        self.assertIsNotNone(run.failure, "the rig timeline has to fail the run")
        self.assertIn(
            "Too many data faults",
            str(run.failure),
            "the rig died on the bounded-recovery budget, inside two seconds",
        )
        self.assertGreaterEqual(run.stats.recoveries, 5)
        self.assertEqual(code, 1)

    def test_the_stamps_run_records_the_source_stamps(self) -> None:
        run = build_run(self.root, mode="stamps", session="stamps_rec")
        drive(run, self.data, self.stamps)

        chunks, meta = recorded(run)
        self.assertTrue(chunks)
        self.assertEqual(meta["status"], "failed")
        np.testing.assert_array_equal(
            np.asarray([row[1] for row in chunks], dtype=float),
            self.stamps[::BLOCK][: len(chunks)],
            "the default records the source's own stamps, which is the problem",
        )

    def test_the_grid_runs_the_same_data_to_the_end(self) -> None:
        run = build_run(self.root, mode="grid", session="grid")
        code = drive(run, self.data, self.stamps)

        self.assertIsNone(run.failure)
        self.assertEqual(run.stats.recoveries, 0, "the grid absorbs the timeline")
        self.assertGreater(run.stats.windows, 0, "the window gate has to run")
        self.assertGreater(run.stats.valid, 0)
        self.assertGreater(run.stats.timebase_relocks, 0)
        self.assertGreater(run.stats.timebase_relocked_samples, 0.0)
        self.assertEqual(code, 0, "the acceptance rules should accept this run")

    def test_the_recording_carries_the_grid_the_run_used(self) -> None:
        run = build_run(self.root, mode="grid", session="grid_rec")
        drive(run, self.data, self.stamps)
        chunks, meta = recorded(run)

        # Reproduce the grid this run's time base produces for the same block
        # boundaries: the recording has to hold exactly that, not the source's
        # stamps. The anchors advance by a block (50 samples), not by one, and a
        # re-lock in progress moves them by a little more - which is the point.
        time_base = TimeBase(run.timebase_policy)
        expected = []
        for begin in range(0, run.stats.samples, BLOCK):
            times, _ = time_base.place(self.stamps[begin : begin + BLOCK])
            expected.append(float(times[0]))
        np.testing.assert_allclose(
            np.asarray([row[1] for row in chunks], dtype=float), expected, atol=1e-9
        )
        self.assertEqual(int(meta["samples"]), run.stats.samples)
        # The counters are nested inside the recorder's own stats snapshot, which
        # read_metadata hands back already decoded.
        meta_stats = meta["stats"]
        self.assertGreater(meta_stats["timebase_relocked_samples"], 0.0)
        self.assertGreater(meta_stats["timebase_relocks"], 0)
        self.assertGreater(
            meta_stats["timebase_large_steps"],
            0,
            "the re-stamps land on block boundaries and must still be reported",
        )
        self.assertEqual(meta["status"], "completed")

    def test_the_window_verdicts_reach_the_recording(self) -> None:
        run = build_run(self.root, mode="grid", session="grid_win")
        drive(run, self.data, self.stamps)

        windows = iter_windows(run.session.recorder.path)
        self.assertEqual(len(windows), run.stats.windows)
        self.assertEqual(sum(1 for _, valid, _, _, _ in windows if valid), run.stats.valid)
        # A warm-up window is one that is invalid although no judge rejected it -
        # an empty reasons tuple alone does not say that, because a valid window
        # has none either.
        warmup = sum(
            1
            for _, valid, reasons, _, _ in windows
            if not valid and not reasons
        )
        self.assertGreater(warmup, 0, "the warm-up windows have to be recorded")
        self.assertEqual(warmup, run.stats.rejected)

    def test_the_report_records_what_the_time_base_did(self) -> None:
        run = build_run(self.root, mode="grid", session="grid_json")
        drive(run, self.data, self.stamps)

        report = json.loads((self.root / "grid_json-report.json").read_text())
        stats = report["stats"]
        self.assertGreater(stats["timebase_relocks"], 0)
        self.assertGreater(stats["timebase_relocked_samples"], 0.0)
        self.assertGreater(
            stats["timebase_large_steps"],
            0,
            "a re-stamp on a block boundary must still be reported",
        )
        self.assertTrue(np.isfinite(stats["timebase_anchor_rate"]))
        self.assertEqual(report["channels"]["eeg_channels"], list(CHANNELS))
        self.assertEqual(report["arguments"]["timebase"], "grid")


if __name__ == "__main__":
    unittest.main()
