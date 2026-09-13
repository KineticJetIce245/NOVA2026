"""Step 5.5's obligations: the pairing is measured, and the harness is step 5's.

Five things this file refuses to take on trust:

* that the sweep's offset grid and its offset->sample conversion agree, so a
  quoted peak is on the lattice the report claims;
* that a positive offset really reads the envelope *earlier* -- the direction the
  chain's own ``audio_offset`` defines. The first version of the sweep had this
  backwards and it was invisible at zero offset, which is the only point step 5
  published: every other point was the mirror image of the truth. The direction
  is now measured three times: from a hand-checkable fixture whose only content
  is candidate A advanced by 16 samples, from the raw chain's aligned envelopes
  window by window (against the mirror image as a control), and through the
  published below-chance control;
* that a shifted envelope cannot silently align back to zero. The fixture says
  what the score must be at zero and at the true delay, and the two differ by
  construction;
* that a 12.5 ms step is resolved rather than rounded to the nearest 64 Hz
  sample, checked by the fixture's symmetry about the true delay;
* that the sweep reproduces step 5's published zero point, to 1e-9. That is the
  whole reason the curve is interpretable: if this harness disagreed with step
  5's, the curve would be measuring the harness.

The tests that need the git-ignored feature cache skip when it is absent, which
is how the rest of this suite treats the derived artifacts. Every test here can
fail: ``docs`` in each class says which mutation makes it red, and each mutation
was run while this file was written.
"""

import unittest
from pathlib import Path
from unittest import mock

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


def ramp_pair(delay_samples, samples=1600, period=128):
    """A signal that carries candidate A *advanced* by a known number of samples.

    Returns ``(signal, envelopes)``. Candidate A is one period-128 sine and
    every channel of ``signal`` is that sine advanced by ``delay_samples``:
    ``signal[i] == envelopes[i + delay_samples, 0]``. Candidate B is the negated
    sine, so a correlation of -1 is expected at the same offset.

    The chain's convention is that the envelope is read *earlier* by
    ``offset * rate`` samples, so this fixture is aligned at a **negative**
    offset, ``-delay_samples / RATE``: the envelope has to be read
    ``delay_samples`` later to catch up with the advanced signal. An earlier
    version of this file asserted the opposite sign and an earlier version of
    the sweep implemented the opposite sign, so the two agreed with each other
    and disagreed with the chain. The fixture's arithmetic is exact -- ``cos`` of
    the residual phase -- and is what the assertions below compare against.
    """

    index = np.arange(samples, dtype=float)
    sine = np.sin(2 * np.pi * index / period)
    signal = np.empty((samples, 3))
    signal[:, :] = np.sin(2 * np.pi * (index + delay_samples) / period)[:, None]
    envelopes = np.column_stack([sine, -sine])
    return signal, envelopes


def fixture_score(offsets, delay_samples=16, model=None):
    """The fixture's candidate-A/B scores at one offset or a list of them."""

    signal, envelopes = ramp_pair(delay_samples)
    model = fake_model() if model is None else model
    return shift_sweep.trial_scores(model, signal, envelopes, 0, LENGTH,
                                    np.asarray(offsets), RATE)


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


class PairingMechanicsTests(unittest.TestCase):
    """Where each reconstruction row reads its envelope sample.

    Mutation that makes these red: flip the subtraction in ``positions_for``
    (the sign error that produced the first attempt's mirrored curve), round the
    positions to whole samples (the sub-sample steps then collapse), or replace
    the refusal in ``paired_envelope`` with a clamp so a window the chain calls
    ``audio_unavailable`` is scored anyway.
    """

    def setUp(self):
        # Column 0 is ``arange(20)``: linear, so interpolation at a fractional
        # position has an exact hand-checkable value equal to the position.
        self.envelope = np.column_stack([np.arange(20, dtype=float),
                                         -np.arange(20, dtype=float)])

    def test_zero_offset_is_an_exact_slice(self):
        values = shift_sweep.paired_envelope(
            self.envelope, shift_sweep.positions_for(5, 10, 0.0, RATE)
        )
        np.testing.assert_array_equal(values, self.envelope[5:15])

    def test_positive_offset_reads_earlier(self):
        """Audio that starts later is compared against what was heard before."""

        values = shift_sweep.paired_envelope(
            self.envelope, shift_sweep.positions_for(5, 10, 0.05, RATE)
        )
        expected = np.arange(5, 15, dtype=float) - 0.05 * RATE
        np.testing.assert_allclose(values[:, 0], expected, rtol=0, atol=1e-12)
        self.assertTrue(np.all(values[:, 0] < self.envelope[5:15, 0]),
                        "a positive offset must read the envelope earlier")

    def test_negative_offset_reads_later(self):
        values = shift_sweep.paired_envelope(
            self.envelope, shift_sweep.positions_for(5, 10, -0.05, RATE)
        )
        expected = np.arange(5, 15, dtype=float) + 0.05 * RATE
        np.testing.assert_allclose(values[:, 0], expected, rtol=0, atol=1e-12)

    def test_a_sub_sample_offset_is_interpolated_not_rounded(self):
        """12.5 ms is 0.8 of a 64 Hz sample, and the sweep must move by 0.8."""

        values = shift_sweep.paired_envelope(
            self.envelope, shift_sweep.positions_for(5, 10, 0.0125, RATE)
        )
        expected = np.arange(5, 15, dtype=float) - 0.8
        np.testing.assert_allclose(values[:, 0], expected, rtol=0, atol=1e-12)
        self.assertNotEqual(float(values[0, 0]), float(self.envelope[4, 0]))

    def test_a_window_the_offset_pushes_off_the_recording_is_refused(self):
        """Not clamped and not padded: ``None`` is the chain's ``audio_unavailable``."""

        positions = shift_sweep.positions_for(0, 10, 0.05, RATE)
        self.assertLess(float(positions[0]), 0.0)
        self.assertIsNone(shift_sweep.paired_envelope(self.envelope, positions))
        far = shift_sweep.positions_for(15, 10, -0.05, RATE)
        self.assertGreater(float(far[-1]), len(self.envelope) - 1)
        self.assertIsNone(shift_sweep.paired_envelope(self.envelope, far))


class ShiftMovesTheFeaturesTests(unittest.TestCase):
    """A shifted envelope must not be able to align back to zero.

    Mutation that makes these red: remove the shift from ``trial_scores`` (the
    score then never changes with offset), invert it (the peak lands on the
    wrong side), or round the positions to whole samples (the symmetry and the
    sub-sample assertions below break).
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

    def test_the_peak_lands_on_the_true_delay(self):
        """The fixture is aligned at ``-delay / rate``, and only there."""

        offsets = np.asarray([-0.375, -0.25, -0.125, 0.0])
        scores = fixture_score(offsets=offsets)
        best = int(np.argmax(scores[:, 0]))
        self.assertEqual(float(offsets[best]), -16 / RATE,
                         "the sweep must peak where the advance is undone")
        self.assertGreater(scores[best, 0], 0.999,
                           "the fixture is exactly aligned at that offset")
        self.assertLess(scores[best, 1], -0.999,
                        "and the negated candidate is exactly anti-aligned")
        self.assertLess(float(np.max(scores[offsets == 0.0, 0])), 0.75,
                        "an unaligned reference must not score like an aligned one")

    def test_a_sub_sample_offset_is_resolved_not_rounded_away(self):
        """The peak moves with a 12.5 ms step, and the fixture stays symmetric.

        A rounded implementation puts the two neighbours a whole sample either
        side of the truth instead of 0.8 of one, which both walks the peak off
        the middle and breaks the symmetry these assertions check.
        """

        offsets = np.asarray([-0.25 - 0.0125, -0.25, -0.25 + 0.0125])
        scores = fixture_score(offsets=offsets)
        best = int(np.argmax(scores[:, 0]))
        self.assertEqual(best, 1, "the sub-sample step must move the peak with it")
        self.assertGreater(scores[best, 0], 0.999)
        expected = float(np.cos(2 * np.pi * 0.8 / 128))
        for neighbour in (0, 2):
            # The ideal cosine differs from the measured correlation by the
            # fixture's finite-window centering (~1e-4). A rounded shift would
            # put these two at -0.049 and +0.049 instead.
            self.assertAlmostEqual(float(scores[neighbour, 0]), expected, delta=2e-3)
        self.assertAlmostEqual(float(scores[0, 0]), float(scores[2, 0]), delta=1e-4,
                               msg="the fixture is symmetric about the true delay; a "
                                   "rounded shift would split these by ~0.1")


class SingleOffsetScoresTests(unittest.TestCase):
    """One shift through a plain decoder, with no dataset and no fitting."""

    def test_a_known_shift_moves_the_correlation(self):
        model = fake_model()
        scores = fixture_score(offsets=[0.0, -0.25], model=model)
        self.assertGreater(scores[1, 0], 0.999,
                           "the reference read 16 samples later is the aligned one")
        self.assertLess(scores[1, 1], -0.999,
                        "the negated candidate is anti-aligned there")
        self.assertLess(scores[0, 0], 0.75,
                        "the unshifted reference is a quarter period out")
        self.assertNotAlmostEqual(float(scores[0, 0]), float(scores[1, 0]), places=3,
                                  msg="a 250 ms shift must change the correlation")

    def test_a_window_the_offset_pushes_off_the_recording_is_not_scored(self):
        """The whole window abstains rather than correlating over fewer rows."""

        scores = fixture_score(offsets=[0.3])
        self.assertTrue(np.all(np.isnan(scores)),
                        "a window whose first rows precede the audio has no score")
        scores = fixture_score(offsets=[0.0])
        self.assertTrue(np.all(np.isfinite(scores)))


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
            cls.results[contract] = train_kuleuven.metrics(decisions[0], truths[0])

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
        """The guard is real: a zero point that is not step 5's stops the run.

        The reference is injected rather than approached by a tolerance, because
        the reproduction is too exact to fail honestly: at ``--tolerance 1e-9``
        this harness reproduces step 5 to the last digit, so no honest tolerance
        makes it raise. Injecting the *wrong* reference is therefore the only way
        to exercise the guard, and it exercises exactly the code path a real
        harness mismatch would take.

        Mutation that makes this red: turn the comparison into a warning, or
        compare against the reference instead of against the measured number.
        """

        report = REPO / "results" / "should_not_exist.json"
        report.unlink(missing_ok=True)
        with mock.patch.dict(shift_sweep.REFERENCE_BALANCED, {"64ch": 0.111111}):
            with self.assertRaises(SystemExit) as caught:
                shift_sweep.main([
                    "--out", str(REPO / "results"), "--contracts", "64ch",
                    "--json", str(report),
                ])
        self.assertIn("zero point", str(caught.exception))
        self.assertFalse(report.exists())


@unittest.skipUnless(cache_available(), "the step 5 feature cache is not built")
class RealSignConventionTests(unittest.TestCase):
    """The direction, measured window by window against the raw chain.

    The chain re-reads the audio itself, so ``aligned.envelopes`` is what the
    offset means and needs no interpolation from this module. The mirror column
    is the control: if the sweep's sign were inverted, the two columns would
    swap, which is exactly what the first attempt's audit showed as a 0.09-0.20
    accuracy gap at non-zero offsets.

    Mutation that makes this red: negate the shift in ``positions_for``; the
    cached envelope then tracks the chain's mirror image and the "mirror" column
    becomes the small one.
    """

    @classmethod
    def setUpClass(cls):
        cls.corpus = shift_sweep.corpus_for()
        cls.trial = next(trial for trial in cls.corpus.trials if trial.subject == "S1")
        cls.model_path = REPO / "models" / train_kuleuven.MODEL_NAMES["64ch"]

    def test_cached_pairing_reproduces_the_chain_window_by_window(self):
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
                self.assertGreater(row["chain_envelope_mean_abs"], 1e-4,
                                   "a comparison of near-zero envelopes proves nothing")
                self.assertLess(row["envelope_max_difference"], 1e-8,
                                "the cached envelope is the chain's envelope to the "
                                "cache's own float32 storage precision (<= 8.6e-10)")
                self.assertLess(row["score_max_difference"], 1e-6,
                                "and the scores built from it are the chain's scores "
                                "(<= 8.6e-08; zero offset mixes float32 and float64)")
                self.assertAlmostEqual(row["difference"], 0.0, places=9,
                                       msg="cached and chain agree on the decisions")
                if row["offset_seconds"] != 0.0:
                    # The mirror of zero offset *is* zero offset, so the control
                    # only says anything where there is a sign to get wrong.
                    self.assertGreater(row["envelope_mirror_max_difference"], 1e-4,
                                       "the mirror must NOT be the chain's envelope")
                    self.assertGreater(row["score_mirror_max_difference"], 0.05,
                                       "and it must score materially differently")
        self.assertNotAlmostEqual(rows[0]["cached_balanced_accuracy"],
                                  rows[-1]["cached_balanced_accuracy"], places=3,
                                  msg="+/-300 ms must not score the same by accident")


class CurveReadingTests(unittest.TestCase):
    """The summary numbers a reader acts on are computed from the curve itself.

    Mutation that makes this red: measure the half-depth level against the
    majority rate instead of the balanced chance line, or return the first grid
    point past the level instead of the interpolated crossing.
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
        """Half depth is 0.60 between a 0.55 and a 0.70 point, so 1/3 of the way.

        The level sits between grid points by construction, which is what makes
        this test about interpolation: a reader that took the first grid point
        past the level would report 0.10 s of width instead of 1/15 s. (An
        earlier version of this test asserted 0.10 s and a 0.30 s chance width --
        a width no five-point grid spanning +/-0.1 s can have, so it could never
        have passed.)
        """

        points = self.points([0.50, 0.55, 0.70, 0.55, 0.50])
        width = shift_sweep.width_of(points, shift_sweep.peak_of(points))
        self.assertAlmostEqual(width["half_depth_level"], 0.60)
        self.assertAlmostEqual(width["left_half_depth_seconds"], -1 / 30, places=9)
        self.assertAlmostEqual(width["right_half_depth_seconds"], 1 / 30, places=9)
        self.assertAlmostEqual(width["half_depth_width_seconds"], 1 / 15, places=9)
        self.assertAlmostEqual(width["chance_width_seconds"], 0.20, places=9,
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
