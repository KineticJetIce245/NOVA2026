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
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova2026.streaming import GridPolicy, Recovery, TimeBase, UnrepairableError
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
            ("dummy_offloader(", "the script has to build the load test's consumer"),
            ("offload_dropped", "the drop counter has to reach the run report"),
            ("offload_failed", "the failure counter has to reach the run report"),
        ):
            with self.subTest(needle=needle):
                self.assertTrue(
                    needle in source,
                    f"{why} (scripts/getlive/live.py never mentions {needle!r})",
                )


def live_args(*extra: str, units: str = "uV"):
    """The live script's arguments, parsed and validated as the entry point does."""

    parser = live.build_parser()
    args = parser.parse_args(
        ["--sfreq", f"{RIG_RATE:g}", "--source-units", units, *extra]
    )
    live._validate_arguments(parser, args)
    return args


def live_run(*extra: str, units: str = "uV"):
    """A ``_Run`` whose chain the live script built itself, source untouched.

    Unlike :func:`repair_stage`, the units and both endpoint limits come from the
    parsed command line through ``live._build_chain``, which is the path under
    test here. Nothing is connected: ``_build_chain`` only builds stages.
    """

    args = live_args(*extra, units=units)
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
    live._build_chain(run)
    return run


def damaged_offset_run(rows: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """A rig-scale run with a two-sample hole in the middle.

    The values are the rig's own scale - a 12.6 mV electrode offset in
    microvolts, without noise - so the only thing the safety check can trip on
    is the unit declaration.
    """

    data = np.full((rows, 1), RIG_DC_UV, dtype="float32")
    data[rows // 2 : rows // 2 + 2] = np.nan
    stamps = np.arange(rows) / RIG_RATE
    return data, stamps


class LiveLimitWiringTests(unittest.TestCase):
    """The fault limits must reach the stages that enforce them.

    ``--repair-amplitude-uv``/``--repair-saturation-uv`` are in uV while
    ``--source-units`` decides how the source's numbers are scaled. A run against
    an outlet that declares the wrong unit therefore ends in ``unsafe_endpoints``
    - which is the honest outcome, and exactly what the rig's ``Volt``
    declaration produced before the unit was understood. The flags exist so an
    operator who knows the declaration is wrong can say so instead of losing the
    run, and so the fault budget can be widened without editing the library.
    """

    def test_the_limit_options_reach_the_parser(self) -> None:
        args = live_args()
        self.assertEqual(
            (
                args.repair_amplitude_uv,
                args.repair_saturation_uv,
                args.max_recoveries,
                args.max_fault_seconds,
            ),
            (500.0, 75_000.0, 5, 5.0),
        )

        changed = live_args(
            "--repair-amplitude-uv",
            "900",
            "--repair-saturation-uv",
            "2e10",
            "--max-recoveries",
            "12",
            "--max-fault-seconds",
            "20",
        )
        self.assertEqual(
            live.repair_limits(changed),
            {"amplitude_limit_uv": 900.0, "saturation_limit_uv": 2e10},
        )
        self.assertEqual(
            live.recovery_limits(changed),
            {"max_events": 12, "persistent_fault_seconds": 20.0},
        )

    def test_impossible_limits_are_refused_before_the_network(self) -> None:
        # The refusal has to *name the rule*, not merely exit: an unrecognised
        # flag also exits non-zero, and would let this test pass while the
        # validation was missing.
        for extra, expected in (
            (("--repair-amplitude-uv", "-1"), "must be finite and non-negative"),
            (("--repair-amplitude-uv", "nan"), "must be finite and non-negative"),
            (("--repair-saturation-uv", "100"), "must be finite and not below"),
            (("--repair-saturation-uv", "nan"), "must be finite and not below"),
            (("--max-recoveries", "0"), "--max-recoveries must be at least 1"),
            (("--max-fault-seconds", "0"), "must be finite and positive"),
            (("--max-fault-seconds", "nan"), "must be finite and positive"),
        ):
            stream = StringIO()
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                # argparse prints its usage on the way out; keep it out of the
                # suite's output.
                with redirect_stderr(stream):
                    live_args(*extra)
            self.assertIn(expected, stream.getvalue())

    def test_a_wrong_unit_declaration_trips_the_saturation_rail(self) -> None:
        data, stamps = damaged_offset_run()

        declared_uv = live_run().repair
        declared_uv(data, stamps)
        self.assertEqual(
            declared_uv.repaired_samples, 2, "uV are uV: the hole is repaired"
        )

        # The same numbers declared as volts are 1.264e10 uV, far above the
        # 75 kV rail, so the repair is refused instead of inventing 12 kV.
        declared_v = live_run(units="V").repair
        with self.assertRaises(UnrepairableError) as caught:
            declared_v(data, stamps)
        self.assertEqual(caught.exception.kind, "unsafe_endpoints")

    def test_raising_the_rail_overrides_a_wrong_declaration(self) -> None:
        repair = live_run("--repair-saturation-uv", "2e10", units="V").repair
        data, stamps = damaged_offset_run()
        repair(data, stamps)
        self.assertEqual(repair.repaired_samples, 2)

    def test_raising_the_amplitude_limit_admits_a_jump_a_repair_refused(self) -> None:
        # A one-sample pop far wider than the default 500 uV rail, with the two
        # endpoints 2 000 uV apart: refused by default, admitted when the
        # operator says so. Same data, same stage, one flag.
        rows = 40
        data = np.full((rows, 1), RIG_DC_UV, dtype="float32")
        data[rows // 2] = RIG_DC_UV + 2_000.0
        data[rows // 2 + 1] = np.nan
        stamps = np.arange(rows) / RIG_RATE

        default = live_run().repair
        with self.assertRaises(UnrepairableError) as caught:
            default(data, stamps)
        self.assertEqual(caught.exception.kind, "unsafe_endpoints")

        raised = live_run("--repair-amplitude-uv", "5000").repair
        raised(data, stamps)
        self.assertEqual(raised.repaired_samples, 1)

    def test_the_limits_reach_the_run_record(self) -> None:
        # The provenance is written on every live run, hardware or not, so a
        # wrong key here would only ever be found on the rig.
        run = live_run("--max-recoveries", "12", "--repair-saturation-uv", "2e10")
        run.source = {
            "name": "rig",
            "stype": "EEG",
            "source_id": "",
            "sfreq": RIG_RATE,
            "channels": ["Fz"],
            "types": ["eeg"],
            "units": ["uV"],
        }
        run.profile = SimpleNamespace(
            name="test",
            source="test",
            reference="",
            ground="",
            eog_channels=(),
            other_auxiliary=(),
        )

        self.assertEqual(
            live._chain_provenance(run)["limits"],
            {
                "repair": {"amplitude_limit_uv": 500.0, "saturation_limit_uv": 2e10},
                "recovery": {"max_events": 12, "persistent_fault_seconds": 5.0},
            },
        )

    def test_the_fault_budget_reaches_recovery(self) -> None:
        args = live_args("--max-recoveries", "12", "--max-fault-seconds", "20")
        recovery = Recovery(**live.recovery_limits(args))
        self.assertEqual(recovery.max_events, 12)
        self.assertEqual(recovery.persistent_fault_seconds, 20.0)

    def test_the_live_script_consumes_the_limit_options(self) -> None:
        source = Path(live.__file__).read_text(encoding="utf-8")
        for needle, why in (
            ("**repair_limits(args)", "the repair limits have to reach Repair"),
            ("**recovery_limits(args)", "the fault budget has to reach Recovery"),
            ('"limits": {', "the limits have to reach the run record"),
        ):
            with self.subTest(needle=needle):
                self.assertTrue(
                    needle in source,
                    f"{why} (scripts/getlive/live.py never mentions {needle!r})",
                )


if __name__ == "__main__":
    unittest.main()
