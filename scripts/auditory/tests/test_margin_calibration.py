"""Margin calibration's obligations: the sweep is the controller, and it can fail.

Six things this file refuses to take on trust:

* that the sweep drives a real :class:`AttentionController` rather than a
  restatement of its rules. The states in the report come out of
  ``AttentionController.update``; the test drives the controller by hand on the
  same streams and compares, and it also shows that the controller's
  ``min_switch_windows`` *changes the verdict*, which a private reimplementation
  of "commit when the difference exceeds the margin" would not;
* that the fast scoring path is ``RidgeDecoder.score``. The calibration scores a
  whole trial in one lag-weighted pass and reads windows off it; if that pass
  disagreed with the decoder, every margin in the report would be measuring the
  harness instead of the model, so the equivalence is asserted against the
  decoder itself on a fitted model;
* that ``MIN_MARGIN`` still defaults to ``0.5`` and that the calibrated value is
  a *different* name nothing defaults to. The calibration's whole point is that
  the default is not moved quietly;
* that the ladder contains the value it judges (``main`` refuses to run
  otherwise, and this pins the ladder itself);
* that coverage/accuracy on decided frames can *fail* as a pair -- a margin that
  commits only on class A must read 1.0 raw and 0.5 balanced, or the two columns
  in the report would be the same column twice;
* that the published frames follow the session's rules: warm-up and the wait for
  the first full window are ``unavailable``, stale evidence is ``unavailable``,
  and neither is silently counted as ``uncertain``.

Every test here can fail: each class's ``docs`` says which mutation makes it red,
and each mutation was run while this file was written.
"""

import unittest

import numpy as np

from nova2026.auditory.config import (
    CALIBRATED_HISTORY_SECONDS, CALIBRATED_MARGIN, MIN_MARGIN, AuditoryConfig,
)
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import AttentionEstimate, AuditoryWindow
from nova2026.auditory.evaluation import selection_metrics

from scripts.auditory import aad_ridge, margin_calibration as mc

RATE = 64.0
CHANNELS = 4
CONTRACT = {"eeg_channels": ["a", "b", "c", "d"]}


def synthetic_trial(seconds=20.0, seed=20260914, label=0):
    """A short recording with real structure, so a fit and a score mean something."""

    random = np.random.default_rng(seed)
    samples = int(seconds * RATE)
    times = np.arange(samples) / RATE
    carrier = np.sin(2 * np.pi * 3.0 * times)
    signal = random.normal(0.0, 20.0, size=(samples, CHANNELS))
    envelopes = np.column_stack([
        carrier + random.normal(0.0, 0.2, size=samples),
        np.sin(2 * np.pi * 5.0 * times) + random.normal(0.0, 0.2, size=samples),
    ])
    signal[:, 0] += 8.0 * envelopes[:, label]
    return signal, envelopes, times


def fitted_model(history=5.0):
    """A model fitted the way the calibration fits folds, on synthetic data."""

    signal, envelopes, _ = synthetic_trial()
    length = int(history * RATE)
    starts = [start for start in range(0, len(signal) - length + 1, length)]
    moments = aad_ridge.trial_moments(signal, envelopes, 0, starts, length,
                                      AuditoryConfig().lag_samples)
    return aad_ridge.fit(moments, CONTRACT, 100.0), signal, envelopes, starts, length


def window_of(signal, envelopes, times, start, length):
    return AuditoryWindow(
        signal[start : start + length], envelopes[start : start + length],
        times[start : start + length], float(times[start + length - 1]), True, (),
        CONTRACT, 0,
    )


class ReplayIsTheController(unittest.TestCase):
    """docs: replace ``replay_windows``' controller with ``abs(diff) >= margin``.

    Mutating the loop to a local threshold makes ``states`` commit on the first
    crossing window, so ``test_min_switch_windows_delays_the_commit`` and the
    hand-driven comparison both go red.
    """

    def streams(self):
        arrivals = np.arange(0.0, 12.0, 1.0)
        scores = np.tile(np.array([0.30, 0.10]), (len(arrivals), 1))
        return scores, arrivals

    def test_states_come_from_the_controller(self):
        scores, arrivals = self.streams()
        kwargs = {"max_age": mc.MAX_AGE, "min_switch_windows": mc.MIN_SWITCH_WINDOWS,
                  "attenuation_db": mc.ATTENUATION_DB}
        states = mc.replay_windows(scores, arrivals, 0.1, controller_kwargs=kwargs)

        controller = AttentionController(margin=0.1, **kwargs)
        expected = []
        for index, evidence_end in enumerate(arrivals):
            controller.update(
                AttentionEstimate(scores[index], evidence_end, evidence_end, True, ()),
                evidence_end,
            )
            expected.append(-1 if controller.selected is None else controller.selected)
        self.assertEqual(states.tolist(), expected)
        self.assertEqual(states[0], mc.UNCERTAIN)
        self.assertEqual(states[-1], 0)

    def test_min_switch_windows_delays_the_commit(self):
        scores, arrivals = self.streams()
        slow = mc.replay_windows(scores, arrivals, 0.1,
                                 controller_kwargs={"min_switch_windows": 3})
        fast = mc.replay_windows(scores, arrivals, 0.1,
                                 controller_kwargs={"min_switch_windows": 1})
        self.assertEqual(slow[:2].tolist(), [mc.UNCERTAIN, mc.UNCERTAIN])
        self.assertEqual(fast[:2].tolist(), [0, 0])
        self.assertGreater(int(np.count_nonzero(slow == mc.UNCERTAIN)),
                           int(np.count_nonzero(fast == mc.UNCERTAIN)))

    def test_margin_decides_what_is_committed(self):
        scores, arrivals = self.streams()
        kwargs = {"min_switch_windows": 1}
        committed = mc.replay_windows(scores, arrivals, 0.05, controller_kwargs=kwargs)
        abstained = mc.replay_windows(scores, arrivals, 0.5, controller_kwargs=kwargs)
        self.assertTrue(np.all(committed == 0))
        self.assertTrue(np.all(abstained == mc.UNCERTAIN))


class FastPathIsTheDecoder(unittest.TestCase):
    """docs: read the envelope at ``start + lag`` instead of ``start``.

    That one-sample pairing slip leaves the scores plausible -- they are still
    correlations, still in [-1, 1], still the right sign -- and moves them well
    past the 1e-9 tolerance.
    """

    def test_series_scores_equal_model_score(self):
        model, signal, envelopes, starts, length = fitted_model()
        series = mc.reconstruction_series(model, signal)
        worst = 0.0
        for start in starts[1:]:
            window = window_of(signal, envelopes, np.arange(len(signal)) / RATE,
                               start, length)
            reference = np.asarray(model.score(window), dtype=float)
            fast = mc.correlation_scores(series, envelopes, start, length,
                                         model.config.lag_samples)
            worst = max(worst, float(np.max(np.abs(reference - fast))))
        self.assertLess(worst, mc.PIN_TOLERANCE)
        self.assertGreater(len(starts), 1)

    def test_scores_are_correlations_that_can_be_negative(self):
        model, signal, envelopes, starts, length = fitted_model()
        series = mc.reconstruction_series(model, signal)
        scores = mc.correlation_scores(series, envelopes, starts[0], length,
                                       model.config.lag_samples)
        self.assertTrue(np.all(np.abs(scores) <= 1.0 + 1e-12))


class DefaultsAreUnchanged(unittest.TestCase):
    """docs: set ``MIN_MARGIN = CALIBRATED_MARGIN`` in ``nova2026.auditory.config``.

    That is exactly the silent edit this calibration exists to prevent, and both
    ``test_default_margin_is_still_half`` and
    ``test_calibrated_margin_is_a_separate_knob`` go red.
    """

    def test_default_margin_is_still_half(self):
        self.assertEqual(MIN_MARGIN, 0.5)
        self.assertEqual(AttentionController().margin, 0.5)
        self.assertEqual(AttentionController().margin, MIN_MARGIN)

    def test_calibrated_margin_is_a_separate_knob(self):
        self.assertIsInstance(CALIBRATED_MARGIN, float)
        self.assertGreater(CALIBRATED_MARGIN, 0.0)
        self.assertNotEqual(CALIBRATED_MARGIN, MIN_MARGIN)
        self.assertEqual(AttentionController(margin=CALIBRATED_MARGIN).margin,
                         CALIBRATED_MARGIN)
        self.assertEqual(AttentionController().margin, 0.5)
        self.assertGreater(CALIBRATED_HISTORY_SECONDS, 0.0)

    def test_calibrated_window_is_one_the_cache_carries(self):
        from scripts.auditory import feature_cache

        self.assertIn(CALIBRATED_HISTORY_SECONDS, feature_cache.HISTORIES)


class LadderJudgesTheDefault(unittest.TestCase):
    """docs: drop ``0.5`` from ``MARGIN_LADDER``.

    A ladder that does not contain the value it is judging cannot calibrate it,
    which is the mistake the report's first table exists to avoid.
    """

    def test_ladder_contains_the_current_default(self):
        self.assertIn(MIN_MARGIN, mc.MARGIN_LADDER)
        self.assertEqual(list(mc.MARGIN_LADDER), sorted(mc.MARGIN_LADDER))
        self.assertTrue(all(margin > 0 for margin in mc.MARGIN_LADDER))
        self.assertGreaterEqual(len(mc.MARGIN_LADDER), 6)

    def test_main_refuses_a_ladder_without_the_default(self):
        with self.assertRaises(ValueError):
            mc.main(["--margins", "0.05", "0.10", "--json", "tmp/unused.json"])


class DecidedMetricsCanFail(unittest.TestCase):
    """docs: two mutations, both caught.

    ``decision_metrics`` mutated to divide ``accuracy_decided`` by *all* frames
    instead of the decided ones turns 1.0 into 0.5 and reddens
    ``test_majority_class_only_margin_reads_half_balanced``. Mutating the
    per-class rule to score a class with no decided frames as 1.0 instead of 0.0
    turns the balanced accuracy into 1.0 and reddens the same test.
    """

    def test_majority_class_only_margin_reads_half_balanced(self):
        truths = np.array([0] * 40 + [1] * 40)
        decisions = np.concatenate([np.zeros(40, dtype=np.int8),
                                    np.full(40, mc.UNCERTAIN, dtype=np.int8)])
        record = mc.decision_metrics(decisions, truths)
        self.assertEqual(record["accuracy_decided"], 1.0)
        self.assertEqual(record["majority_decided"], 1.0)
        self.assertEqual(record["balanced_accuracy_decided"], 0.5)
        self.assertEqual(record["recall_decided"]["1"], None)
        self.assertEqual(record["recall_over_all"]["1"], 0.0)
        self.assertEqual(record["coverage"], 0.5)
        self.assertEqual(record["uncertain"], 40)

    def test_unavailable_is_not_counted_as_uncertain(self):
        truths = np.array([0] * 10)
        decisions = np.concatenate([np.full(4, mc.UNAVAILABLE, dtype=np.int8),
                                    np.full(6, mc.UNCERTAIN, dtype=np.int8)])
        record = mc.decision_metrics(decisions, truths)
        self.assertEqual(record["unavailable"], 4)
        self.assertEqual(record["uncertain"], 6)
        self.assertEqual(record["coverage"], 0.0)
        self.assertIsNone(record["accuracy_decided"])


class FramesFollowTheSession(unittest.TestCase):
    """docs: drop the warm-up mask from ``frame_decisions``.

    Without it the frames before the first window report ``uncertain`` instead of
    ``unavailable``, and the two tests below see the wrong census.
    """

    def test_warmup_and_wait_are_unavailable(self):
        frames = np.arange(0.0, 8.0, 0.25)
        # Evidence exists from the first sample: without the warm-up rule these
        # frames would be decided, which is what the mutation produces.
        early = np.arange(0.0, 8.0, 1.0)
        decisions = mc.frame_decisions(np.zeros(len(early), dtype=np.int8), early,
                                       frames, max_age=3.0, warmup_seconds=2.0)
        self.assertTrue(np.all(decisions[frames < 2.0] == mc.UNAVAILABLE))
        self.assertTrue(np.all(decisions[frames >= 2.0] == 0))
        # And a window that is still filling publishes nothing, not a guess.
        late = np.array([5.0, 6.0, 7.0])
        waiting = mc.frame_decisions(np.zeros(len(late), dtype=np.int8), late, frames,
                                     max_age=3.0, warmup_seconds=2.0)
        self.assertTrue(np.all(waiting[frames < 5.0] == mc.UNAVAILABLE))
        self.assertTrue(np.all(waiting[frames >= 5.0] == 0))

    def test_stale_evidence_is_unavailable_not_uncertain(self):
        arrivals = np.array([0.0])
        states = np.array([-1], dtype=np.int8)
        frames = np.array([0.5, 1.0, 4.0, 6.0])
        decisions = mc.frame_decisions(states, arrivals, frames, max_age=3.0,
                                       warmup_seconds=0.0)
        self.assertEqual(decisions.tolist(),
                         [mc.UNCERTAIN, mc.UNCERTAIN, mc.UNAVAILABLE, mc.UNAVAILABLE])

    def test_switching_metrics_are_the_library_ones(self):
        """docs: swap the ``choices`` and ``labels`` arguments to ``selection_metrics``.

        The measured stream would then be scored against one whose labels change
        mid-recording, so the false-selection count collapses and the equality
        with the direct library call breaks.
        """

        frame_times = np.arange(0.0, 12.0, mc.FRAME_SECONDS)
        count, half = len(frame_times), len(frame_times) // 2
        scores = np.tile(np.array([0.90, 0.10]), (count, 1))
        scores[half:] = np.array([0.10, 0.90])
        truths = np.zeros(count, dtype=int)
        trial = {"scores": scores, "arrivals": frame_times.copy(),
                 "frame_times": frame_times, "frame_truths": truths, "label": 0}
        outcome = mc.trial_outcome(trial, 0.05)
        decisions = outcome["decisions"]
        # Warm-up is unavailable, the first three windows are pending, and the
        # stream then commits -- to the right candidate and then to the wrong one.
        self.assertEqual(decisions[0], mc.UNAVAILABLE)
        self.assertEqual(decisions[10], 0)
        self.assertEqual(decisions[-1], 1)
        choices = np.where(decisions == mc.UNAVAILABLE, mc.UNCERTAIN, decisions)
        direct = selection_metrics(frame_times, choices, truths,
                                   end_time=float(frame_times[-1] + mc.FRAME_SECONDS))
        self.assertEqual(outcome["switching"], direct)
        self.assertGreater(direct["coverage"], 0.8)
        self.assertGreaterEqual(direct["false_selection_changes"], 1)
        self.assertGreaterEqual(outcome["flips"], 1)


if __name__ == "__main__":
    unittest.main()
