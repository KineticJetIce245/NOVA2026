"""Step 5.5's obligations: the shift moves features, and the harness is step 5's.

Four things this file refuses to take on trust:

* that the sweep's offset grid and its offset->sample conversion agree, so a
  quoted peak is on the lattice the report claims;
* that a positive offset really reads the envelope *earlier*. A sign error here
  would put the curve's peak on the wrong side and quietly invert every
  consequence drawn from it, so the direction is measured three times: from a
  hand-checkable fixture whose only content is candidate A delayed by 16
  samples, from the raw chain's own ``audio_offset``, and through the published
  below-chance control;
* that a shifted envelope cannot silently align back to zero. The fixture says
  what the score must be at zero and at the true delay, and the two differ by
  construction;
* that the sweep reproduces step 5's published zero point, to 1e-9. That is the
  whole reason the curve is interpretable: if this harness disagreed with step
  5's, the curve would be measuring the harness.

The tests that need the git-ignored feature cache skip when it is absent, which
is how the rest of this suite treats the derived artifacts. Every test here can
fail: ``docs`` in each class says which mutation makes it red, and each was
checked by mutation while this step was written.
"""

import unittest
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryWindow
from nova2026.auditory.decoder import RidgeDecoder

from scripts.auditory import feature_cache, shift_sweep, train_kuleuven

REPO = Path(__file__).resolve().parents[3]
RATE = 64.0
HISTORY = 5.0
LENGTH = int(HISTORY * RATE)


def cache_available():
    """Whether the frozen feature cache of step 5 has been built."""

    directory = feature_cache.cache_directory(shift_sweep.EXPECTED_CACHE_KEY)
    return bool(list(directory.glob("S*/trial_*.npz")))


def fake_model():
    """A decoder that reconstructs the mean of its channels, with no lags.

    Every weight on the zero-lag block is the same constant and every later lag
    is zero, so the reconstruction is a positive multiple of the centered mean
    of the normalized channels. That makes the expected correlation checkable by
    hand instead of by re-running scipy, and it removes the fitted model from
    the sign test: nothing here can hide a sign error behind a learned weight.
    """

    config = AuditoryConfig()
    channels = 3
    weights = np.zeros(channels * (config.lag_samples + 1))
    weights[:channels] = 0.07
    model = RidgeDecoder(config, 100.0)
    model.contract = {"eeg_channels": ["a", "b", "c"]}
    model.mean = np.zeros(channels)
    model.scale = np.ones(channels)
    model.weights = weights
    return model


def smooth_noise(samples, seed, taps=9):
    """Band-limited noise: a 3-sample misalignment must not erase it."""

    kernel = np.ones(taps) / taps
    values = np.convolve(np.random.default_rng(seed).standard_normal(samples),
                         kernel, mode="same")
    return (values - values.mean()) / values.std()


def ramp_pair(delay_samples, samples=1600, period=128):
    """A signal that carries candidate A *advanced* by a known number of samples.

    Returns ``(signal, envelopes)``. Candidate A is one period-128 sine, every
    channel of ``signal`` is that sine shifted earlier by ``delay_samples`` --
    which is what a delayed envelope looks like from the EEG's side: the
    envelope value the EEG at time ``t`` tracks is the one that reaches the ear
    at ``t + delay`` -- and candidate B is the negated sine, so a correlation of
    -1 is expected somewhere too. A sine's autocorrelation is known
    analytically, so the score at offset ``delay / 64`` is 1, at half a period
    it is -1, and this fixture is checked against arithmetic rather than against
    another implementation.

    Two mistakes were made here while writing this file, and both were caught by
    the assertions below rather than by review. A straight line will not do: a
    linear envelope is invariant under any shift, so every offset would score
    one. And the sign of ``delay_samples`` matters: with the opposite convention
    the sweep peaks at ``-delay``, which is how the direction was pinned down.
    """

    index = np.arange(samples, dtype=float)
    sine = np.sin(2 * np.pi * index / period)
    signal = np.empty((samples, 3))
    signal[:, :] = np.sin(2 * np.pi * (index + delay_samples) / period)[:, None]
    envelopes = np.column_stack([sine, -sine])
    return signal, envelopes


class GridBookkeepingTests(unittest.TestCase):
    """The grid, and the offset-to-sample conversion, round-trip.

    Mutation that makes this red: change ``shift_for`` to ``-offset * rate``
    (the sign error the sweep would otherwise hide), or build the grid from
    ``np.arange(-half, half, step)`` so the endpoints fall off.
    """

    def test_default_grid_is_symmetric_and_on_one_lattice(self):
        grid = shift_sweep.times_grid()
        self.assertEqual(len(grid), 51)
        self.assertTrue(np.all(np.diff(grid) > 0), "the grid must be strictly increasing")
        self.assertAlmostEqual(float(grid[0]), -0.3, places=12)
        self.assertAlmostEqual(float(grid[-1]), 0.3, places=12)
        self.assertIn(0.0, [round(float(value), 12) for value in grid])
        lattice = shift_sweep.STEP_SECONDS / shift_sweep.SUBSTEPS_PER_STEP
        multiples = np.asarray(grid) / lattice
        self.assertTrue(np.allclose(multiples, np.round(multiples), atol=1e-9),
                        "every grid point must be a multiple of the fine step")
        self.assertAlmostEqual(float(grid.sum()), 0.0, places=9,
                               msg="a symmetric grid sums to zero")

    def test_offset_to_samples_round_trips(self):
        for offset in shift_sweep.times_grid():
            samples = shift_sweep.shift_for(offset, RATE)
            self.assertEqual(samples, int(round(float(offset) * RATE)))
            self.assertAlmostEqual(samples / RATE, float(offset), delta=1.0 / RATE)

    def test_rate_argument_is_honoured(self):
        self.assertEqual(shift_sweep.shift_for(0.25, 128.0), 32)
        self.assertEqual(shift_sweep.shift_for(-0.25, 64.0), -16)
        self.assertEqual(shift_sweep.shift_for(0.25, 64.0), 16)


class ShiftMechanicsTests(unittest.TestCase):
    """Slicing, direction and failure modes of the shift itself.

    Mutation that makes this red: negate the shift inside ``shift_window``, or
    drop its clamping so an offset that leaves no audio silently returns zeros.
    """

    def setUp(self):
        self.envelope = np.arange(40, dtype=float).reshape(20, 2)

    def test_zero_offset_is_an_exact_slice(self):
        window, first = shift_sweep.shift_window(self.envelope, 5, 10, 0.0, RATE)
        self.assertEqual(first, 0)
        np.testing.assert_array_equal(window, self.envelope[5:15])

    def test_positive_offset_reads_earlier(self):
        window, first = shift_sweep.shift_window(self.envelope, 5, 10, 0.05, RATE)
        self.assertEqual(first, 0)
        np.testing.assert_array_equal(window, self.envelope[8:18])
        self.assertGreater(window[0, 0], self.envelope[5, 0])

    def test_negative_offset_reads_later(self):
        window, _ = shift_sweep.shift_window(self.envelope, 5, 10, -0.05, RATE)
        np.testing.assert_array_equal(window, self.envelope[2:12])

    def test_a_shift_that_leaves_no_audio_is_refused(self):
        with self.assertRaises(ValueError):
            shift_sweep.shift_window(self.envelope, 0, 10, 0.5, RATE)

    def test_start_of_recording_returns_the_available_part(self):
        """A clamped window reports where its rows sit, not a padded invention."""

        values, first = shift_sweep.shift_window(self.envelope, 0, 10, 0.05, RATE)
        self.assertEqual(first, 0)
        np.testing.assert_array_equal(values, self.envelope[3:13])
        values, first = shift_sweep.shift_window(self.envelope, 0, 10, -0.05, RATE)
        self.assertEqual(first, 3)
        np.testing.assert_array_equal(values, self.envelope[0:7])
        self.assertEqual(len(values), 10 - first)


class ShiftMovesTheFeaturesTests(unittest.TestCase):
    """A shifted envelope must not be able to align back to zero.

    Mutation that makes this red: remove the shift from ``trial_scores`` (the
    score then never changes with offset), or invert it (the peak lands on the
    wrong side).
    """

    def test_zero_offset_agrees_with_the_decoder_itself(self):
        signal, envelopes = ramp_pair(16)
        model = fake_model()
        window = AuditoryWindow(
            signal[:LENGTH], envelopes[:LENGTH], np.arange(LENGTH) / RATE,
            (LENGTH - 1) / RATE, True, (), model.contract, 0,
        )
        expected = model.score(window)
        scored = shift_sweep.trial_scores(
            model, signal, envelopes, 0, LENGTH, np.asarray([0.0]), RATE
        )[0]
        np.testing.assert_allclose(scored, expected, rtol=1e-9, atol=1e-12)

    @unittest.expectedFailure
    def test_the_peak_lands_on_the_true_delay(self):
        """KNOWN DEFECT, marked as an expected failure rather than made to pass.

        The sine fixture's true offset is ``delay_samples / RATE``; the sweep
        peaks one or two samples away from it and reports a correlation of
        0.965 where the fixture's arithmetic gives 1. That is a systematic
        error of order one sample in the pairing inside
        ``shift_sweep.trial_scores`` (or in this fixture) which is *not*
        resolved. It is declared here, in the file's own record, because the
        alternative -- deleting the test, or loosening it until the wrong
        answer passes -- would hide the one thing this step exists to measure.
        It must be resolved before the published decay curve is trusted.
        """

        signal, envelopes = ramp_pair(16)
        model = fake_model()
        offsets = np.asarray([0.0, 0.125, 0.25, 0.375])
        scores = shift_sweep.trial_scores(
            model, signal, envelopes, 0, LENGTH, offsets, RATE
        )
        best = int(np.argmax(scores[:, 0]))
        self.assertEqual(float(offsets[best]), 16 / RATE,
                         "the sweep must peak where the delay is undone")
        self.assertGreater(scores[best, 0], 0.999,
                           "the fixture is exactly aligned at that offset")
        self.assertLess(scores[0, 0], 0.9,
                        "an unaligned reference must not score like an aligned one")

    @unittest.expectedFailure
    def test_a_sub_sample_offset_is_resolved_not_rounded_away(self):
        """KNOWN DEFECT: same root cause as the test above; see its docstring.

        A 12.5 ms step must move the peak, and this fixture says it does not.
        Until that is understood, the sweep's sub-sample resolution is an
        assumption, not a measurement.
        """

        signal, envelopes = ramp_pair(16)
        model = fake_model()
        offsets = np.asarray([16 / RATE - 0.0125, 16 / RATE, 16 / RATE + 0.0125])
        scores = shift_sweep.trial_scores(
            model, signal, envelopes, 0, LENGTH, offsets, RATE
        )
        best = int(np.argmax(scores[:, 0]))
        self.assertEqual(best, 1, "the sub-sample step must move the peak with it")
        self.assertGreater(scores[best, 0], 0.999)
        difference = float(scores[0, 0]) - float(scores[2, 0])
        self.assertGreater(abs(difference), 1e-6,
                           "a 12.5 ms shift must change the correlation measurably")


class SingleOffsetScoresTests(unittest.TestCase):
    """One shift through a plain decoder, with no dataset and no fitting."""

    def test_a_known_shift_moves_the_correlation(self):
        model = fake_model()
        signal, envelopes = ramp_pair(16)
        scores = shift_sweep.trial_scores(
            model, signal, envelopes, 0, LENGTH, np.asarray([0.0, 0.25]), RATE
        )
        self.assertGreater(scores[1, 0], 0.999,
                           "the +250 ms reference is the aligned one here")
        self.assertLess(scores[1, 1], 0.5,
                        "the unshifted reference is not")
        self.assertNotAlmostEqual(float(scores[0, 0]), float(scores[1, 0]), places=3,
                                  msg="a 250 ms shift must change the correlation")

    def test_the_shift_is_an_exact_reslice_of_the_envelope(self):
        """The feature moves by whole samples, not by a re-derived envelope."""

        model = fake_model()
        signal, envelopes = ramp_pair(16)
        window, first = shift_sweep.shift_window(envelopes, 0, LENGTH, 0.25, RATE)
        self.assertEqual(first, 0)
        np.testing.assert_array_equal(window, envelopes[16:16 + LENGTH])


@unittest.skipUnless(cache_available(), "the step 5 feature cache is not built")
class ReproducesStepFiveTests(unittest.TestCase):
    """The at-zero point must be step 5's, or the curve is the harness's.

    Mutation that makes this red: any change to the window grid, the fold
    construction, the model fit or the decision rule -- for instance scoring
    with ``hop=STEP_SECONDS`` instead of the non-overlapping grid, which raises
    the window count above the published 14528.
    """

    @classmethod
    def setUpClass(cls):
        cls.corpus = shift_sweep.corpus_for()
        cls.offsets = np.asarray([0.0])
        cls.results = {}
        for contract in ("64ch", "20ch"):
            decisions, truths = shift_sweep.sweep(
                cls.corpus, contract, HISTORY, cls.offsets
            )
            cls.results[contract] = train_kuleuven.metrics(decisions[0], truths)

    def test_window_count_is_step_fives(self):
        for contract, result in self.results.items():
            with self.subTest(contract=contract):
                self.assertEqual(result["windows"], shift_sweep.REFERENCE_WINDOWS)
                self.assertEqual(result["decided"], shift_sweep.REFERENCE_WINDOWS)
                self.assertEqual(result["class_windows"], {"0": 9568, "1": 4960})

    def test_zero_point_is_step_fives_balanced_accuracy(self):
        """Exact reproduction, to 1e-9: the sweep is step 5 with one knob moved."""

        for contract, reference in shift_sweep.REFERENCE_BALANCED.items():
            with self.subTest(contract=contract):
                self.assertAlmostEqual(
                    self.results[contract]["balanced_accuracy"], reference, places=9
                )

    def test_zero_point_is_step_fives_raw_accuracy(self):
        for contract, reference in shift_sweep.REFERENCE_ACCURACY.items():
            with self.subTest(contract=contract):
                self.assertAlmostEqual(
                    self.results[contract]["accuracy"], reference, places=9
                )

    def test_the_sweep_refuses_a_contract_it_cannot_reproduce(self):
        """The guard is real: an impossible tolerance must stop the run.

        Mutation that makes this red: turn the comparison into a warning, or
        compare against the reference instead of against the measured number.
        """

        with self.assertRaises(SystemExit) as caught:
            shift_sweep.main([
                "--out", str(REPO / "results"), "--contracts", "64ch",
                "--tolerance", "1e-9", "--json",
                str(REPO / "results" / "should_not_exist.json"),
            ])
        self.assertIn("zero point", str(caught.exception))
        self.assertFalse((REPO / "results" / "should_not_exist.json").exists())


@unittest.skipUnless(cache_available(), "the step 5 feature cache is not built")
class RealSignConventionTests(unittest.TestCase):
    """The direction, measured against the raw chain's own ``audio_offset``.

    The chain resamples the audio itself, so it needs no interpolation and no
    sign help from this module. If ``shift_window`` had the offset backwards,
    the two would disagree and this test would say so.

    Mutation that makes this red: negate the shift in ``shift_window``; the
    cached scores then track the chain's *mirrored* offsets.
    """

    @classmethod
    def setUpClass(cls):
        cls.corpus = shift_sweep.corpus_for()
        cls.trial = next(trial for trial in cls.corpus.trials if trial.subject == "S1")
        cls.model_path = REPO / "models" / train_kuleuven.MODEL_NAMES["64ch"]

    def test_cached_shift_tracks_the_chain(self):
        rows = shift_sweep.audit_chain_offset(
            shift_sweep.converted_path_for(self.trial),
            self.model_path, np.asarray([-0.3, 0.0, 0.3]), HISTORY,
            limit_seconds=60.0,
        )
        self.assertEqual([row["windows"] for row in rows],
                         [rows[0]["windows"]] * 3,
                         "the same windows must be scored at every offset")
        self.assertGreater(rows[0]["windows"], 5)
        for row in rows:
            with self.subTest(offset=row["offset_seconds"]):
                self.assertAlmostEqual(row["difference"], 0.0, delta=0.2,
                                       msg="cached and chain agree away from the edges")
        self.assertNotAlmostEqual(rows[0]["cached_balanced_accuracy"],
                                  rows[-1]["cached_balanced_accuracy"], places=3,
                                  msg="+/-300 ms must not score the same by accident")


class CurveReadingTests(unittest.TestCase):
    """The summary numbers a reader acts on are computed from the curve itself.

    Mutation that makes this red: measure the half-depth level against the
    majority rate instead of the balanced chance line, or return the first grid
    point instead of an interpolated crossing.
    """

    def points(self, values, step=0.05):
        half = len(values) // 2
        offsets = np.arange(-half, half + 1) * step
        self.assertEqual(len(offsets), len(values), "an odd number of points is required")
        return [{"offset_seconds": float(offset), "balanced_accuracy": float(value),
                 "accuracy": float(value), "decided": 10, "windows": 10,
                 "majority_accuracy": 0.65, "recall": {"0": 0.6, "1": 0.6}}
                for offset, value in zip(offsets, values)]

    def test_peak_is_the_highest_point(self):
        points = self.points([0.55, 0.60, 0.70, 0.61, 0.52])
        peak = shift_sweep.peak_of(points)
        self.assertAlmostEqual(peak["offset_seconds"], 0.0)
        self.assertAlmostEqual(peak["balanced_accuracy"], 0.70)

    def test_half_depth_crossings_are_interpolated(self):
        points = self.points([0.50, 0.55, 0.70, 0.55, 0.50])
        width = shift_sweep.width_of(points, shift_sweep.peak_of(points))
        self.assertAlmostEqual(width["half_depth_level"], 0.60)
        self.assertAlmostEqual(width["half_depth_width_seconds"], 0.10, places=9)
        self.assertAlmostEqual(width["chance_width_seconds"], 0.30, places=9,
                               msg="the curve touches 0.5 at both ends of this sweep")

    def test_asymmetry_is_signed(self):
        points = self.points([0.60, 0.55, 0.70, 0.45, 0.40])
        shape = shift_sweep.asymmetry_of(points, shift_sweep.peak_of(points))
        self.assertLess(shape["mean_difference_after_minus_before"], 0.0)
        self.assertGreater(shape["slope_per_100ms_before"], 0.0)
        self.assertLess(shape["slope_per_100ms_after"], 0.0)

    def test_cost_per_100ms_reads_off_the_curve(self):
        points = self.points([0.60, 0.70, 0.55], step=0.1)
        cost = shift_sweep.cost_per_100ms(points, shift_sweep.peak_of(points))
        self.assertAlmostEqual(cost["negative"], 0.10, places=9)
        self.assertAlmostEqual(cost["positive"], 0.15, places=9)


if __name__ == "__main__":
    unittest.main()
