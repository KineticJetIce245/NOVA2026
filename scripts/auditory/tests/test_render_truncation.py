"""The rendered media may not outlast the session it is played against.

The defect these tests lock: the conversion drops the candidate tail past the
last EEG sample (``scripts/auditory/kuleuven_contract.py::usable_audio_samples``,
plan section 3.3), but ``AuditoryTrial.audio`` keeps it, and every render call
site passed ``trial.audio`` straight through. For ``S1/trial_004`` that produced
a 394.00 s stereo file for a 389 s replay. The browser reads ``duration`` from
that file, so the file - not the trial - is what the frontend's window check,
the gain gate's position clause and the operator's progress bar all believe.

Two behaviours are pinned here:

* candidates **longer** than the EEG are cut, before the mix, by section 3.3's
  rule, and the file's duration is the EEG span's;
* candidates **already shorter** than the EEG are rendered untouched, because a
  clamp that always shortens would silently delete audio from a legitimate trial.

Each test names the mutation it was shown to fail under (plan section 6.4).
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.data import AuditoryTrial
from nova2026.auditory.render import (
    eeg_prefix_samples,
    render_stereo,
    truncate_to_eeg,
)

EEG_RATE = 128.0
AUDIO_RATE = 44100.0


def make_trial(eeg_samples: int, audio_samples: int, *, rate: float = AUDIO_RATE):
    """A trial with ``eeg_samples`` of EEG and ``audio_samples`` of candidates.

    The candidates are a constant, so the rendered file's length is the only
    thing under test and no assertion here depends on the signal.
    """

    timestamps = np.arange(eeg_samples, dtype=float) / EEG_RATE
    eeg = np.zeros((eeg_samples, 2), dtype=float)
    audio = np.stack(
        (np.full(audio_samples, 0.5), np.full(audio_samples, -0.5)), axis=1
    )
    labels = np.zeros(eeg_samples, dtype=int)
    return AuditoryTrial(
        eeg,
        timestamps,
        audio,
        rate,
        labels,
        subject="S1",
        trial_id="trial_004",
        channel_names=["A1", "A2"],
        reference="unit test",
        upstream_processing="unit test",
    )


def media_args(out: Path):
    """The four arguments every render call site reads off its own CLI."""

    return types.SimpleNamespace(
        media_out=str(out),
        media_title="unit-test trial",
        presentation="dichotic",
        crossmix_weight=0.25,
    )


class TruncationRuleTests(unittest.TestCase):
    """The rule is section 3.3's, not a second opinion about it."""

    def test_the_restated_rule_agrees_with_the_frozen_contract(self):
        from scripts.auditory.kuleuven_contract import usable_audio_samples

        # The converted trials are 128 Hz EEG against 44.1 kHz audio; the table
        # also covers a 500 Hz live rig (plan section 3.11) and an exactly
        # divisible pair, where an off-by-one would hide.
        cases = (
            (49792, 128.0, 44100.0),
            (250000, 500.0, 48000.0),
            (1000, 128.0, 44100.0),
            (1280, 128.0, 44100.0),
            (500, 500.0, 500.0),
        )
        for samples, eeg_rate, audio_rate in cases:
            with self.subTest(samples=samples, eeg_rate=eeg_rate, audio_rate=audio_rate):
                self.assertEqual(
                    eeg_prefix_samples(samples, eeg_rate, audio_rate),
                    usable_audio_samples(samples, eeg_rate, audio_rate),
                )
        # Mutation: counting from `eeg_samples` instead of the last EEG sample -
        # `floor(eeg_samples * audio_rate / eeg_rate) + 1`, i.e. dropping the
        # `- 1` - returns 17156108 for (49792, 128, 44100) where the frozen
        # contract returns 17154556, and fails on the first case.

    def test_a_session_with_no_samples_is_refused(self):
        with self.assertRaises(ValueError):
            eeg_prefix_samples(0, EEG_RATE, AUDIO_RATE)
        with self.assertRaises(ValueError):
            eeg_prefix_samples(100, 0.0, AUDIO_RATE)
        with self.assertRaises(ValueError):
            eeg_prefix_samples(100, EEG_RATE, 0.0)


class TruncationArithmeticTests(unittest.TestCase):
    """What the cut keeps, drops and reports."""

    def test_longer_candidates_are_cut_to_the_last_eeg_sample(self):
        trial = make_trial(eeg_samples=200, audio_samples=88200)
        prefix, report = truncate_to_eeg(
            trial.audio,
            eeg_samples=len(trial.timestamps),
            eeg_rate=float(trial.sample_rate),
            audio_rate=float(trial.audio_rate),
        )
        expected = int(np.floor(199 * AUDIO_RATE / EEG_RATE)) + 1
        self.assertEqual(prefix.shape[0], expected)
        self.assertEqual(report["samples_dropped"], 88200 - expected)
        self.assertTrue(report["truncated"])
        self.assertAlmostEqual(report["eeg_seconds"], 199 / EEG_RATE, places=9)
        self.assertAlmostEqual(report["candidate_seconds"], 2.0, places=9)
        self.assertAlmostEqual(report["kept_seconds"], expected / AUDIO_RATE, places=9)
        self.assertAlmostEqual(
            report["dropped_seconds"], (88200 - expected) / AUDIO_RATE, places=9
        )
        # The kept prefix is the *first* samples, not an arbitrary window.
        self.assertTrue(np.all(prefix[:, 0] == trial.audio[0, 0]))
        # Mutation: dropping the slice and keeping the whole array - the code as
        # it stood before this change - fails here and in the render test below.

    def test_shorter_candidates_are_left_alone(self):
        trial = make_trial(eeg_samples=200, audio_samples=1000)
        prefix, report = truncate_to_eeg(
            trial.audio,
            eeg_samples=len(trial.timestamps),
            eeg_rate=float(trial.sample_rate),
            audio_rate=float(trial.audio_rate),
        )
        self.assertEqual(prefix.shape[0], 1000)
        self.assertEqual(report["samples_dropped"], 0)
        self.assertFalse(report["truncated"])
        self.assertAlmostEqual(report["kept_seconds"], report["candidate_seconds"])
        # Mutation: clamping unconditionally to the EEG prefix (min() replaced by
        # the rule's own count) still passes this - which is why the assertion is
        # `equal` and not `less_equal`: the whole 1000 samples must survive.

    def test_candidates_must_be_two_columns(self):
        with self.assertRaises(ValueError):
            truncate_to_eeg(
                np.zeros((10, 3)),
                eeg_samples=200,
                eeg_rate=EEG_RATE,
                audio_rate=AUDIO_RATE,
            )


class RenderedFileTests(unittest.TestCase):
    """The file on disk, which is the thing the browser measures."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "media.wav"

    def test_the_file_equals_the_eeg_span_and_not_the_candidate_span(self):
        trial = make_trial(eeg_samples=200, audio_samples=88200)
        expected = int(np.floor(199 * AUDIO_RATE / EEG_RATE)) + 1
        prefix, _ = truncate_to_eeg(
            trial.audio,
            eeg_samples=len(trial.timestamps),
            eeg_rate=float(trial.sample_rate),
            audio_rate=float(trial.audio_rate),
        )
        report = render_stereo(prefix, self.out, sample_rate=int(AUDIO_RATE))
        rate, pcm = wavfile.read(self.out)
        self.assertEqual(rate, int(AUDIO_RATE))
        self.assertEqual(pcm.shape[0], expected)
        self.assertAlmostEqual(
            pcm.shape[0] / rate, 199 / EEG_RATE, delta=1.0 / AUDIO_RATE
        )
        self.assertLess(report["seconds"], 2.0)
        # Mutation: rendering `trial.audio` instead of `prefix` - the call site as
        # it stood before this change - writes 88200 samples and fails here.

    def test_a_short_candidate_set_renders_every_sample_it_has(self):
        trial = make_trial(eeg_samples=200, audio_samples=1000)
        prefix, _ = truncate_to_eeg(
            trial.audio,
            eeg_samples=len(trial.timestamps),
            eeg_rate=float(trial.sample_rate),
            audio_rate=float(trial.audio_rate),
        )
        render_stereo(prefix, self.out, sample_rate=int(AUDIO_RATE))
        _, pcm = wavfile.read(self.out)
        self.assertEqual(pcm.shape[0], 1000)


class CallSiteTests(unittest.TestCase):
    """The call sites are where the defect lived, so the defect is locked there.

    Every one of them had ``trial.audio`` in its hand and passed it straight to
    ``render_stereo``. Testing only :func:`truncate_to_eeg` would leave the next
    call site free to make the same mistake, which is exactly what happened here.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "media.wav"
        self.trial = make_trial(eeg_samples=200, audio_samples=88200)
        self.eeg_span = float(
            self.trial.timestamps[-1] - self.trial.timestamps[0]
        )

    def assert_matches_the_eeg(self, report):
        rate, pcm = wavfile.read(self.out)
        self.assertAlmostEqual(pcm.shape[0] / rate, self.eeg_span, delta=1.0 / rate)
        self.assertEqual(report["truncation"]["samples_dropped"], 88200 - pcm.shape[0])
        self.assertGreater(report["truncation"]["samples_dropped"], 0)
        self.assertLess(report["seconds"], 2.0)

    def test_the_demo_call_site_renders_the_eeg_span(self):
        from scripts.auditory_ui.demo import prepare_media

        _, report = prepare_media(self.trial, media_args(self.out))
        self.assert_matches_the_eeg(report)
        # Mutation: passing `trial.audio` to `render_stereo` in
        # `scripts/auditory_ui/demo.py::prepare_media` - the pre-change code -
        # writes 88200 samples (2.0 s) and fails the first assertion.

    def test_the_session_call_site_renders_the_eeg_span(self):
        from scripts.auditory_ui.session import render_media

        _, report = render_media(self.trial, media_args(self.out))
        self.assert_matches_the_eeg(report)
        # Mutation: the same, in `scripts/auditory_ui/session.py::render_media`,
        # whose docstring claimed the cut while the code did not make it.

    def test_the_demorun_call_site_renders_the_eeg_span(self):
        from scripts.auditory_ui.demorun import render_trial_stereo

        report = render_trial_stereo(self.trial, self.out)
        self.assert_matches_the_eeg(report)
        # Mutation: `render_trial_stereo` calling `render_stereo(trial.audio, ...)`
        # puts all four perturbation cases back on a file longer than the session.


if __name__ == "__main__":
    unittest.main()
