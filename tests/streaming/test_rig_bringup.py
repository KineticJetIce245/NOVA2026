"""Regression tests frozen from the first real bring-up (rig data, 2026-09-12).

Every constant below was **measured on the rig**, not invented. The source was an
ANT Neuro eego EE-22x publishing the outlet
``EE511-010010-200563_on_DESKTOP-ET4GTF5``: 25 channels at a declared 500 Hz (24
EEG electrodes plus a ``Trigger`` line), the control software running on Windows
and this package consuming it over the LAN from macOS.

What the rig actually delivered:

* **A per-sample grid that jitters.** ``scripts.getlive.ts_check`` measured
  0.48-0.49% of the steps below half a sample, while the census rate stayed at
  500.00 Hz with one sample per chunk. The source stamps every sample, so it is
  *not* chunk-stamped, and the relay's block-spreading ``--regrid`` was the wrong
  tool for it: spreading preserves the span a block covers, and a 0.4-sample
  deficit spread over two samples is two 0.7-sample steps - still refused. The
  counted grid that replaced it is the fix, and it is what the live script now
  does by default.
* **Real data loss.** About one sample in a thousand never arrived (94 missing
  in 65,542, as counted at the relay). A counted grid closes up over a hole, by
  design - it gives one slot per sample received - so the loss has to stay visible
  in what the run *reports*, which is what the tests below check.
* **A declared unit that is wrong.** The outlet declares ``Volt``; the samples
  are microvolts (a 12,640 uV offset with 42 uV of noise - read as volts that is
  12,600 V, and every electrode trips the saturation rail).
* **One dead electrode.** ``O1`` sat at exactly 0.00 uV for the whole session.

Why these tests fail on the code *before* the bring-up fixes and must pass after
them: that is the whole point of freezing them. A refactor that reintroduces the
old behaviour has to fail here instead of on the rig with the cap on someone's
head. Everything is hardware-free and transport-free.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_rig_bringup -v
"""

import unittest
from pathlib import Path

import numpy as np

from nova2026.streaming import GridPolicy, TimeBase, UnrepairableError
from nova2026.streaming.preprocess import Repair
from scripts.getlive import live

# Measured on the rig, 2026-09-12.
#
# 0.48% of steps were sub-nominal, so about one step in 211; the magnitude sat
# below half a sample, which is the class ``Repair`` refuses at *any* tolerance
# (``_check_grid`` accepts a step within its tolerance of one sample and then
# rejects every step at or below one sample outright). The loss rate is one
# sample per ~1150 arriving.
RIG_RATE = 500.0
RIG_JITTER_EVERY = 211
RIG_JITTER_STEP_SAMPLES = 0.4
RIG_LOST_EVERY = 1150

# The rig's amplitude scale: a 12.6 mV electrode offset with EEG-sized noise on
# top, in microvolts. Interpreting these as volts is what made all 24 electrodes
# report amplitude + flatline + saturation at once.
RIG_DC_UV = 12_640.0
RIG_NOISE_UV = 42.0


def grid_tolerance_samples(rate: float = RIG_RATE) -> float:
    """The grid tolerance ``live._build_chain`` hands to ``Repair``, in samples."""

    return min(2e-4, 0.4 / rate) * rate


def rig_stamps(
    count: int = 6000,
    *,
    rate: float = RIG_RATE,
    jitter_every: int = RIG_JITTER_EVERY,
    jitter_step: float = RIG_JITTER_STEP_SAMPLES,
    lost: tuple[int, ...] = (),
    start: float = 1000.0,
) -> np.ndarray:
    """Timestamps shaped like the rig's: an exact grid, jittered a few tenths.

    Args:
        count: Samples to emit.
        rate: Nominal rate in Hz.
        jitter_every: Period of the short step.
        jitter_step: Size of the short step, in samples.
        lost: Indices whose *successor* is one whole sample late - data the
            source never sent, which the grid must not paper over.
        start: Timestamp of the first sample.
    """

    steps = np.ones(count)
    steps[::jitter_every] = jitter_step
    for index in lost:
        steps[index] = 2.0
    return start + np.concatenate(([0.0], np.cumsum(steps / rate)))


def repair_stage(rate: float = RIG_RATE, channels: int = 1, names=("Fz",)) -> Repair:
    """A ``Repair`` built the way ``live._build_chain`` builds it."""

    return Repair(
        rate,
        tolerance_seconds=min(2e-4, 0.4 / rate),
        source_unit_exponent=-6,
        n_eeg=channels,
        channel_names=names,
    )


class RigGridIsUnusableRawTests(unittest.TestCase):
    """The fixture really is the broken thing: it must fail before any rebuild.

    Without this, the tests below could pass on a grid that was never broken.
    """

    def test_the_raw_rig_grid_is_refused_by_repair(self) -> None:
        stamps = rig_stamps(2000)
        data = np.zeros((stamps.size, 1), dtype="float32")
        with self.assertRaises(UnrepairableError) as caught:
            repair_stage()(data, stamps)
        self.assertEqual(caught.exception.kind, "irregular_timestamps")


class GridRebuildTests(unittest.TestCase):
    """The grid the rig needs, now built by the library's time base.

    It used to be the relay's ``--regrid-jitter`` and its ``GridBuilder``. Both
    are gone: the behaviour lives in :class:`nova2026.streaming.TimeBase`, which
    the live script uses by default. The requirement has not changed, so the
    assertions have not either - only the object that satisfies them.
    """

    def rebuild(self, stamps: np.ndarray):
        """Put the rig's stamps on the library's counted grid."""

        time_base = TimeBase(GridPolicy.for_rate(RIG_RATE))
        times, state = time_base.place(stamps)
        return state, times

    def test_the_rig_timeline_becomes_a_grid_repair_accepts(self) -> None:
        stamps = rig_stamps()
        _, times = self.rebuild(stamps)
        self.assertEqual(times.size, stamps.size, "no sample may be dropped")
        steps = np.diff(times) * RIG_RATE
        self.assertGreaterEqual(
            float(steps.min()),
            1.0 - grid_tolerance_samples(),
            "a step below the grid tolerance is fatal to Repair at any setting",
        )
        # And the whole rebuilt stream has to survive the stage the live run
        # builds: this is the assertion the rig actually failed.
        repair_stage()(np.zeros((times.size, 1), dtype="float32"), times)

    def test_a_lost_sample_is_reported_rather_than_filled(self) -> None:
        # The trade this design makes on purpose: the grid gives one slot per
        # sample received, so a sample the source never sent cannot become a step
        # in it. The loss is reported as a suspicious step instead, and Repair is
        # left with nothing to fabricate - which is what used to go wrong.
        stamps = rig_stamps(lost=(1000, 2000))
        state, times = self.rebuild(stamps)

        self.assertEqual(len(state.large_steps), 2, "one report per lost sample")
        self.assertEqual(times.size, stamps.size, "and no invented row")
        steps = np.diff(times) * RIG_RATE
        self.assertLessEqual(
            float(np.abs(steps - 1.0).max()),
            grid_tolerance_samples(),
            "the grid closes up over the hole rather than carrying it",
        )
        repair = repair_stage()
        repair(np.zeros((times.size, 1), dtype="float32"), times)
        self.assertEqual(repair.repaired_samples, 0, "nothing was synthesised")

    def test_every_rebuilt_step_is_a_whole_number_of_samples(self) -> None:
        stamps = rig_stamps(4000, lost=(900,))
        _, times = self.rebuild(stamps)
        steps = np.diff(times) * RIG_RATE
        np.testing.assert_allclose(
            steps,
            np.rint(steps),
            atol=grid_tolerance_samples(),
            err_msg="the rebuilt grid must sit on the nominal grid, gaps included",
        )
        self.assertEqual(times.size, stamps.size)


class LiveOffloadWiringTests(unittest.TestCase):
    """``scripts.getlive`` must not accept offload flags and then ignore them.

    The shared parser hands every live script ``--workers``/``--compute``/
    ``--queue``. For this script they were parsed and then never used, so a
    consumer-mode load test on the rig measured nothing at all - the class of
    silent misbehaviour the package refuses everywhere else. The parser test
    below passes either way, which is exactly why nobody noticed.
    """

    def test_the_run_offload_options_reach_the_parser(self) -> None:
        args = live.build_parser().parse_args(
            [
                "--sfreq", "500",
                "--source-units", "uV",
                "--workers", "4",
                "--compute", "0.5",
                "--queue", "8",
            ]
        )
        self.assertEqual((args.workers, args.compute, args.queue), (4, 0.5, 8))

    def test_the_live_script_consumes_the_offload_options(self) -> None:
        source = Path(live.__file__).read_text(encoding="utf-8")
        for needle, why in (
            ("args.workers", "the worker count has to reach the pipeline"),
            ("TaskOffloader", "the script has to build an offloader"),
            ("offload_dropped", "the drop counter has to reach the run report"),
            ("offload_failed", "the failure counter has to reach the run report"),
        ):
            with self.subTest(needle=needle):
                self.assertTrue(
                    needle in source,
                    f"{why} (scripts/getlive/live.py never mentions {needle!r})",
                )


if __name__ == "__main__":
    unittest.main()
