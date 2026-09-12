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
  *not* chunk-stamped and the ``--regrid`` block-spreading path is the wrong
  tool: spreading preserves the span a block covers, and a 0.4-sample deficit
  spread over two samples is two 0.7-sample steps - still refused.
* **Real data loss.** About one sample in a thousand never arrived (94 missing
  in 65,542, as counted at the relay). A regularised grid must keep that visible
  instead of closing it silently.
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

from nova2026.streaming import UnrepairableError
from nova2026.streaming.preprocess import Repair
from scripts.getlive import live, relay

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


class RelayPerSampleJitterTests(unittest.TestCase):
    """The relay must turn this rig's jittered grid into one ``Repair`` accepts."""

    def rebuild(self, stamps: np.ndarray, channels: int = 1):
        """Drive whatever per-sample rebuild the relay offers.

        The requirement is the behaviour, not the name. If the relay offers no
        such path, this fails with that sentence instead of an ``AttributeError``
        that says nothing about the rig.
        """

        builder = getattr(relay, "GridBuilder", None)
        if builder is None:
            self.fail(
                "the relay offers no per-sample grid rebuild: --regrid spreads a "
                "block's span and cannot remove a sub-nominal step, so a "
                "per-sample jittered source (what the rig publishes) has no "
                "usable path through this script"
            )
        instance = builder(RIG_RATE, channels)
        data = np.zeros((stamps.size, channels), dtype="float32")
        out_data, out_times = instance.feed(data, stamps)
        return instance, out_data, out_times

    def test_the_relay_cli_can_ask_for_a_uniform_grid(self) -> None:
        parser = relay.build_parser()
        argv = ["--source-name", "EE511", "--labels", "Fz", "--regrid-jitter"]
        try:
            args = parser.parse_args(argv)
        except SystemExit:
            self.fail(
                "the relay CLI must accept a per-sample jitter mode: --regrid "
                "spreads a block over its span, so a sub-nominal step stays "
                "sub-nominal and Repair still refuses it"
            )
        self.assertTrue(args.regrid_jitter)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source-name", "EE511", "--labels", "Fz",
                               "--regrid", "--regrid-jitter"])

    def test_a_per_sample_jittered_source_becomes_a_grid_repair_accepts(self) -> None:
        stamps = rig_stamps()
        _, _, times = self.rebuild(stamps)
        self.assertEqual(times.size, stamps.size, "no sample may be dropped")
        steps = np.diff(times) * RIG_RATE
        self.assertGreaterEqual(
            float(steps.min()),
            1.0 - grid_tolerance_samples(),
            "a step below the grid tolerance is fatal to Repair at any setting",
        )
        # And the whole rebuilt stream has to survive the stage the live run
        # builds: this is the assertion the rig actually failed.
        data = np.zeros((times.size, 1), dtype="float32")
        repair_stage()(data, times)

    def test_the_rebuild_keeps_real_data_loss_visible(self) -> None:
        stamps = rig_stamps(lost=(1000, 2000))
        builder, _, times = self.rebuild(stamps)
        steps = np.diff(times) * RIG_RATE
        self.assertGreater(
            float(steps.max()), 1.5, "a sample the source never sent must stay a gap"
        )
        self.assertGreaterEqual(
            builder.lost_samples, 2, "the loss must be counted, not just survived"
        )

    def test_the_rebuild_moves_timestamps_only(self) -> None:
        stamps = rig_stamps(1500)
        builder = relay.GridBuilder(RIG_RATE, 2)
        data = np.arange(stamps.size * 2, dtype="float32").reshape(-1, 2)
        out_data, out_times = builder.feed(data, stamps)
        np.testing.assert_array_equal(
            out_data, data, "samples may be re-stamped, never rewritten"
        )
        self.assertEqual(out_times.size, stamps.size)

    def test_every_rebuilt_step_is_a_whole_number_of_samples(self) -> None:
        stamps = rig_stamps(4000, lost=(900,))
        _, _, times = self.rebuild(stamps)
        steps = np.diff(times) * RIG_RATE
        np.testing.assert_allclose(
            steps,
            np.rint(steps),
            atol=grid_tolerance_samples(),
            err_msg="the rebuilt grid must sit on the nominal grid, gaps included",
        )


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
