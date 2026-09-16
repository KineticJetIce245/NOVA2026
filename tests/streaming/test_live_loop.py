"""The live loop, exercised without an amplifier.

``tests/streaming/test_getlive.py`` says of the stream loop that it "is not
exercised". That loop is where the time base's placement lives, and the placement
is load-bearing: the timeline has to be rebuilt **before** the recorder sees a
block, or the recording carries the source's untrusted stamps while the run
report claims a grid. Nothing about that needs a cap, so nothing about it should
go untested.

Only the acquisition layer is faked here. ``live._loop`` and ``live._finalize``,
``TimeBase``, ``Repair`` and the run counters are the real objects, and the clock
is patched so the loop runs a fixed number of iterations instead of depending on
wall time.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_live_loop -v
"""

import io
import itertools
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nova2026.streaming import StreamStats, TimeBase
from nova2026.streaming.preprocess import Repair
from scripts.getlive import live

RATE = 500.0
BLOCK = 50
ITERATIONS = 50  # the patched clock lets the loop run exactly this many times


def rig_like_stamps(
    count: int = 5000,
    *,
    rate: float = RATE,
    drift_ppm: float = -200.0,
    jumps=((1000, 1.0), (3000, 2.0)),
) -> np.ndarray:
    """The measured rig timeline: a slow clock plus whole-sample re-stamps.

    The shape is the one frozen in ``test_timebase.py``; the count here is just
    big enough to keep the loop fed.
    """

    true_rate = rate * (1.0 + drift_ppm / 1e6)
    stamps = np.arange(count) / true_rate
    for index, extra in jumps:
        stamps[index:] += extra / rate
    return stamps


def fake_clock(step: float = 0.01):
    """A monotonic clock that advances once per call, so the loop is finite."""

    counter = itertools.count()
    return lambda: next(counter) * step


class _FakeAcquire:
    """Hand the loop the fixture, in blocks, faster than real time."""

    def __init__(self, stamps: np.ndarray, channels: int = 1) -> None:
        self._stamps = stamps
        self._channels = channels
        self._next = 0
        self.pending_samples = 0
        self.max_lag = 0.0
        self.gaps = 0
        self.closed = False

    def read(self):
        begin, end = self._next, min(self._next + BLOCK, self._stamps.size)
        if end <= begin:
            raise AssertionError("the fixture ran out before the loop finished")
        self._next = end
        data = np.zeros((end - begin, self._channels), dtype="float32")
        return data, self._stamps[begin:end]

    def close(self) -> None:
        self.closed = True


class _FakeBuffer:
    """A ring buffer that produces no windows: this is about the timeline."""

    def push(self, data, timestamps):
        return ()


class _FakeRecorder:
    """Stand in for the run recorder, keeping what it was written."""

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.not_processed = None

    def write(self, data, timestamps) -> None:
        self.blocks.append(np.asarray(timestamps, dtype=float))

    def mark_not_processed(self, tail) -> None:
        self.not_processed = tail


class _FakeSession:
    """The session surface ``_loop`` drives, with the write kept where it happens.

    The real :meth:`StreamSession.ingest` writes the raw block to the recorder on
    its way past, which is exactly why the time base has to run before it. This
    stub keeps that order so a misplaced time base shows up as a failed
    assertion rather than as a silently wrong recording.
    """

    def __init__(self, stamps: np.ndarray, recorder: _FakeRecorder) -> None:
        self.acquire = _FakeAcquire(stamps)
        self.buffer = _FakeBuffer()
        self.recorder = recorder
        self.window_samples = 256
        self.hop_samples = 64
        self.capacity_samples = 768
        self.out_sfreq = 128.0
        self.warmup_samples = 256
        self.closed: dict | None = None
        self.ingested: list[np.ndarray] = []

    def ingest(self, data, timestamps):
        self.ingested.append(np.asarray(timestamps, dtype=float))
        if self.recorder is not None:
            self.recorder.write(data, timestamps)
        return data, timestamps

    def close(self, **kwargs) -> None:
        self.closed = kwargs


class _FakeRecovery:
    """Recovery, counted the way the run report counts it."""

    def __init__(self) -> None:
        self.segment = 0
        self.recoveries = 0

    def handle(self, error) -> None:
        self.recoveries += 1
        self.segment += 1

    def watch(self, window) -> None:
        return None


def build_run(*, mode: str, stamps: np.ndarray, stages=None):
    """Assemble the pieces ``live._loop`` and ``live._finalize`` need."""

    args = live.build_parser().parse_args(
        [
            "--sfreq", str(RATE),
            "--source-units", "uV",
            "--duration", str(ITERATIONS * 0.01),
            "--quiet",
            "--timebase", mode,
        ]
    )
    policy = live.timebase_policy(args)
    repair = Repair(
        RATE,
        tolerance_seconds=policy.tolerance_seconds,
        source_unit_exponent=-6,
        n_eeg=1,
        channel_names=("Fz",),
    )
    recorder = _FakeRecorder()
    session = _FakeSession(stamps, recorder)
    run = live._Run(
        args=args,
        stream=None,
        source={},
        contract=None,
        profile=None,
        eeg=("Fz",),
        eog=(),
        excluded=(),
    )
    run.stats = StreamStats()
    run.session = session
    run.recovery = _FakeRecovery()
    run.repair = repair
    run.stages = (repair,) if stages is None else stages
    run.resampler = SimpleNamespace(
        out_sfreq=128.0, quality="LQ", startup_delay_seconds=1.88
    )
    run.timebase_policy = policy
    run.timebase = TimeBase(policy) if mode == "grid" else None
    return run, session, recorder, repair


def drive(run) -> None:
    """Run the loop with the clock patched, muting its setup banner."""

    with mock.patch.object(live, "monotonic", new=fake_clock()), redirect_stdout(
        io.StringIO()
    ):
        live._loop(run)


class LoopPlacementTests(unittest.TestCase):
    """The timeline must be rebuilt before anything records it."""

    def test_the_recorder_receives_the_grid_not_the_source_stamps(self) -> None:
        stamps = rig_like_stamps()
        run, session, recorder, repair = build_run(mode="grid", stamps=stamps)
        drive(run)

        self.assertTrue(recorder.blocks, "the loop must have recorded something")
        recorded = np.concatenate(recorder.blocks)
        steps = np.diff(recorded) * RATE
        self.assertLessEqual(
            float(np.abs(steps - 1.0).max()),
            run.timebase_policy.tolerance_samples,
            "the recording must carry the grid the run actually used",
        )
        self.assertEqual(
            repair.repaired_samples,
            0,
            "with a grid there is nothing for Repair to fabricate",
        )
        self.assertEqual(run.failure, None)
        self.assertEqual(session.closed["status"], "completed")

    def test_the_stamps_mode_leaves_the_source_timeline_alone(self) -> None:
        stamps = rig_like_stamps()
        run, session, recorder, repair = build_run(mode="stamps", stamps=stamps)
        drive(run)

        recorded = np.concatenate(recorder.blocks)
        np.testing.assert_array_equal(
            recorded,
            stamps[: recorded.size],
            "the default must pass the source's own stamps through untouched",
        )
        self.assertGreater(
            repair.repaired_samples,
            0,
            "and the source timeline is what makes Repair invent rows",
        )

    def test_the_timebase_verdict_reaches_the_run_record(self) -> None:
        stamps = rig_like_stamps()
        run, session, recorder, _ = build_run(mode="grid", stamps=stamps)
        drive(run)

        stats = run.stats
        self.assertGreater(
            stats.timebase_relocked_samples,
            0.0,
            "the absorbed drift must be reported, not inferred",
        )
        self.assertGreater(stats.timebase_relocks, 0)
        self.assertTrue(np.isfinite(stats.timebase_anchor_rate))
        self.assertLess(stats.timebase_anchor_rate, RATE)
        self.assertIn(
            "timebase_relocked_samples",
            session.closed["stats"],
            "the counters have to reach the recorder's meta",
        )
        self.assertEqual(session.closed["stats"]["blocks"], run.stats.blocks)

    def test_a_stamps_run_reports_no_timebase_numbers(self) -> None:
        run, session, _, _ = build_run(mode="stamps", stamps=rig_like_stamps())
        drive(run)
        self.assertEqual(run.stats.timebase_relocks, 0)
        self.assertEqual(run.stats.timebase_relocked_samples, 0.0)


class LoopFinalizeTests(unittest.TestCase):
    """``_finalize`` runs in a ``finally``: it must always close the run."""

    @staticmethod
    def _explode(data, timestamps):
        raise RuntimeError("stage failed")

    def test_a_failing_run_still_closes_and_records(self) -> None:
        run, session, _, _ = build_run(mode="grid", stamps=rig_like_stamps())
        run.stages = (self._explode,)
        drive(run)

        self.assertIsInstance(run.failure, RuntimeError)
        self.assertEqual(session.closed["status"], "failed")
        self.assertIn("RuntimeError", session.closed["error"])
        self.assertTrue(session.acquire.closed)
        self.assertGreater(run.elapsed, 0.0)
        self.assertTrue(
            np.isfinite(run.stats.timebase_anchor_rate),
            "the time base's verdict has to be recorded even when the run failed",
        )

    def test_the_timebase_verdict_survives_a_failure(self) -> None:
        # Fail late enough that the timeline has had something to absorb: the
        # fixture's first whole-sample re-stamp lands at sample 1000.
        calls = itertools.count()

        def explode_late(data, timestamps):
            if next(calls) >= 25:
                raise RuntimeError("stage failed late")
            return data, timestamps

        run, session, _, _ = build_run(mode="grid", stamps=rig_like_stamps())
        run.stages = (explode_late,)
        drive(run)

        self.assertIsInstance(run.failure, RuntimeError)
        self.assertGreater(
            run.stats.timebase_relocks,
            0,
            "a run that failed is exactly the run whose timeline matters",
        )
        self.assertEqual(session.closed["status"], "failed")


if __name__ == "__main__":
    unittest.main()
