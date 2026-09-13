"""Time-base tests: the rig's staircase must become a grid, not phantom damage.

The fixture reproduces the timeline measured on 2026-09-12 (see
``documents/TIMEBASE_DESIGN.md`` section 3): a few hundred ppm of rate error,
plus a handful of whole-sample steps, totalling 5-9 samples of drift over 30 s.
What the old code did with that timeline - read the steps as lost samples,
insert slots for them, and let ``Repair`` repair rows that never existed - is
what these tests forbid.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_timebase -v
"""

import unittest

import numpy as np

from nova2026.streaming.preprocess import Repair
from nova2026.streaming.timebase import (
    GridPolicy,
    TimeBase,
    repair_tolerance_samples,
)

RATE = 500.0

# Measured on the rig: the source's clock runs slow, and the publisher re-stamps
# occasionally, which shows up as whole-sample steps.
MEASURED_DRIFT_PPM = -200.0
MEASURED_JUMPS = ((4000, 1.0), (9000, 2.0), (13000, 1.0))


def measured_timeline(
    count: int = 15500,
    *,
    rate: float = RATE,
    drift_ppm: float = MEASURED_DRIFT_PPM,
    jumps=MEASURED_JUMPS,
    start: float = 1000.0,
) -> np.ndarray:
    """The rig's timeline: a drifting grid with whole-sample re-stamps.

    Args:
        count: Samples to emit.
        rate: Declared rate in Hz.
        drift_ppm: How far the source's clock is from the declared rate.
        jumps: ``(index, extra)`` re-stamps; ``extra`` is in nominal samples, so
            the step at that point becomes ``1 + extra`` samples.
        start: Timestamp of the first sample.
    """

    true_rate = rate * (1.0 + drift_ppm / 1e6)
    stamps = start + np.arange(count) / true_rate
    for index, extra in jumps:
        stamps[index:] += extra / rate
    return stamps


def expected_drift_samples(stamps: np.ndarray, rate: float = RATE) -> float:
    """The total disagreement the anchors carry, in samples."""

    return float((stamps[-1] - stamps[0]) * rate - (stamps.size - 1))


def feed(time_base: TimeBase, stamps: np.ndarray, chunk: int = 37):
    """Push a timestamp stream through the time base the way a pull loop does."""

    emitted = []
    for begin in range(0, stamps.size, chunk):
        times, state = time_base.place(stamps[begin : begin + chunk])
        if times.size:
            emitted.append(times)
    return np.concatenate(emitted) if emitted else np.empty(0), state


def repair_stage(policy: GridPolicy, channels: int = 1) -> Repair:
    """A ``Repair`` built from the policy, the way the live chain builds it."""

    return Repair(
        policy.nominal_sfreq,
        tolerance_seconds=policy.tolerance_seconds,
        source_unit_exponent=-6,
        n_eeg=channels,
        channel_names=("Fz",),
    )


class GridPolicyTests(unittest.TestCase):
    """One rule, defined once: the tolerance the consumer actually applies."""

    def test_the_default_tolerance_is_repairs_own_rule(self) -> None:
        for rate in (250.0, 500.0, 1000.0, 2000.0, 10000.0):
            with self.subTest(rate=rate):
                policy = GridPolicy.for_rate(rate)
                self.assertAlmostEqual(
                    policy.tolerance_samples,
                    repair_tolerance_samples(rate),
                    places=12,
                )
                self.assertAlmostEqual(
                    policy.tolerance_seconds, min(2e-4, 0.4 / rate), places=12
                )

    def test_a_policy_the_consumer_could_not_satisfy_is_refused(self) -> None:
        for bad in (
            {"nominal_sfreq": 0.0},
            {"nominal_sfreq": float("nan")},
            {"nominal_sfreq": RATE, "tolerance_samples": 0.0},
            {"nominal_sfreq": RATE, "tolerance_samples": 0.6},
            {"nominal_sfreq": RATE, "relock_samples": 0.0},
            {"nominal_sfreq": RATE, "relock_slew_samples": 0},
            {"nominal_sfreq": RATE, "max_step_samples": 1.0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GridPolicy(**bad)


class TimeBaseRigTimelineTests(unittest.TestCase):
    """The measured staircase has to become a grid ``Repair`` accepts."""

    def test_one_slot_per_sample_and_repair_accepts_the_grid(self) -> None:
        stamps = measured_timeline()
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        times, state = feed(time_base, stamps)

        self.assertEqual(
            times.size, stamps.size, "one slot per received sample, never more"
        )
        self.assertEqual(state.total_samples, stamps.size)
        self.assertTrue((np.diff(times) > 0).all(), "the grid is strictly increasing")

        # The whole point: the consumer's own rule now passes on the raw rig
        # timeline, which it refused outright before any time base existed.
        data = np.zeros((times.size, 1), dtype="float32")
        repair_stage(time_base.policy)(data, times)

    def test_no_single_step_leaves_the_tolerance(self) -> None:
        stamps = measured_timeline()
        policy = GridPolicy.for_rate(RATE)
        times, _ = feed(TimeBase(policy), stamps)
        steps = np.diff(times) * RATE
        self.assertLessEqual(
            float(np.abs(steps - 1.0).max()),
            policy.tolerance_samples,
            "a re-lock may not break the grid rule the time base exists to satisfy",
        )

    def test_the_drift_is_absorbed_and_reported(self) -> None:
        stamps = measured_timeline()
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        _, state = feed(time_base, stamps)
        drift = expected_drift_samples(stamps)
        self.assertGreater(abs(drift), 5.0, "the fixture is the rig's 5-9 samples")
        self.assertTrue(state.relocks, "the drift must be corrected, on the record")
        self.assertAlmostEqual(
            state.relocked_samples,
            abs(drift),
            delta=1.0,
            msg="every absorbed sample of drift has to be accounted for",
        )
        self.assertLessEqual(
            state.residual_samples,
            time_base.policy.relock_samples,
            "the residual is bounded by the policy by construction",
        )

    def test_the_whole_sample_steps_are_recorded_as_suspicious(self) -> None:
        stamps = measured_timeline()
        _, state = feed(TimeBase(GridPolicy.for_rate(RATE)), stamps)
        self.assertEqual(
            [round(event.size_samples, 3) for event in state.large_steps],
            [2.0, 3.0, 2.0],
            "each re-stamp is reported with the step it produced",
        )
        self.assertTrue(all(event.kind == "large_step" for event in state.large_steps))

    def test_the_anchor_rate_exposes_the_clock_error(self) -> None:
        stamps = measured_timeline()
        _, state = feed(TimeBase(GridPolicy.for_rate(RATE)), stamps)
        self.assertTrue(np.isfinite(state.anchor_rate))
        self.assertLess(state.anchor_rate, RATE, "the source's clock runs slow")
        self.assertLess(
            abs(state.anchor_rate - RATE) / RATE, 1e-3, "and it is a small error"
        )

    def test_a_real_loss_is_reported_and_never_filled(self) -> None:
        # A genuine skip of three samples: the stamps jump by four sample
        # periods and three samples never arrive. The time base may not invent
        # them, so the grid closes up - and the anomaly has to be on the record,
        # which is why the run must also be checked against the CNT.
        stamps = measured_timeline(count=6000, drift_ppm=0.0, jumps=((3000, 3.0),))
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        times, state = feed(time_base, stamps)

        self.assertEqual(times.size, stamps.size, "a lost sample is not fabricated")
        self.assertEqual(len(state.large_steps), 1)
        self.assertAlmostEqual(state.large_steps[0].size_samples, 4.0, places=6)
        self.assertTrue(state.relocks, "the skip is absorbed, and said so")
        self.assertTrue(
            (np.diff(times) > 0).all(), "even a hole leaves the grid monotonic"
        )
        # The jump is four samples, so the default 50-sample slew would move each
        # sample by 0.08 - inside the 0.1 tolerance but only just. A floor alone
        # is not enough: the slew has to be derived from the correction, which is
        # what this bigger step checks.
        steps = np.diff(times) * RATE
        self.assertLessEqual(
            float(np.abs(steps - 1.0).max()), time_base.policy.tolerance_samples
        )
        self.assertGreater(state.relocks[0].slew_samples, 50)


class TimeBaseInterfaceTests(unittest.TestCase):
    """The small contracts a pipeline needs from the class."""

    def test_an_empty_chunk_changes_nothing(self) -> None:
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        times, state = time_base.place(np.empty(0))
        self.assertEqual(times.size, 0)
        self.assertEqual(state.total_samples, 0)
        self.assertEqual(state.relocks, ())

    def test_a_chunk_anchor_is_accepted_for_a_chunk_stamped_source(self) -> None:
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        times, state = time_base.place(np.asarray([1000.0]), n_samples=50)
        self.assertEqual(times.size, 50)
        self.assertEqual(state.total_samples, 50)
        np.testing.assert_allclose(np.diff(times) * RATE, 1.0, atol=1e-9)

    def test_damaged_input_is_refused_rather_than_guessed(self) -> None:
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        with self.assertRaises(ValueError):
            time_base.place(np.asarray([1.0, np.nan]))
        with self.assertRaises(ValueError):
            time_base.place(np.asarray([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            time_base.place(np.asarray([1.0, 2.0, 3.0]), n_samples=5)
        with self.assertRaises(ValueError):
            time_base.place(np.asarray([1.0]), n_samples=0)
        with self.assertRaises(TypeError):
            TimeBase("not a policy")
        with self.assertRaises(ValueError):
            TimeBase(GridPolicy.for_rate(RATE), anchor=float("nan"))

    def test_a_clean_source_is_left_almost_untouched(self) -> None:
        stamps = 1000.0 + np.arange(2000) / RATE
        time_base = TimeBase(GridPolicy.for_rate(RATE))
        times, state = feed(time_base, stamps)
        np.testing.assert_allclose(times, stamps, atol=1e-9)
        self.assertEqual(state.relocks, ())
        self.assertEqual(state.large_steps, ())
        self.assertAlmostEqual(state.anchor_rate, RATE, places=6)


if __name__ == "__main__":
    unittest.main()
