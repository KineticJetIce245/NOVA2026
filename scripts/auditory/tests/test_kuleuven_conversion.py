"""The KU Leuven conversion contract, with and without the converted data.

The dataset authors' own recipe takes the reference envelope from the *dry*
stimulus: ``README.txt.txt`` says envelopes "were extracted only from
part{part}_track{track}_dry.wav files", and ``preprocess_data.m`` loads
``<name>_dry.mat`` after truncating the name it found in ``trials[i].stimuli``,
which names the ``*_hrtf.wav`` rendering that was actually presented.

The first three tests need no converted trial, so they run on a fresh clone and
pin the mapping itself. The fourth inspects a converted trial and skips when the
conversion has not been run: ``datasets/`` is git-ignored by design, and a suite
that failed merely because a 15 GB derived artifact is absent would train people
to ignore it. ``scripts/run_tests.py`` only rejects a suite where *every* test
skipped, so a partial skip here is reported honestly and stays green.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat, wavfile

from nova2026.auditory.data import dry_stimulus_name, load_kuleuven

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "datasets" / "AAD-KULeuven"
METADATA = DATASET / "metadata.json"
S1_TRIAL = DATASET / "converted" / "S1" / "trial_000.npz"

# 16-bit PCM is scaled by 32768 on the way in, so these are the exact values the
# loader must produce for the fixture constants below.
def pcm(value):
    """The float a 16-bit PCM constant becomes after ``read_audio`` scales it."""
    return value / (np.iinfo(np.int16).max + 1)


class StimulusNameTests(unittest.TestCase):
    """The dry mapping, which is the published recipe and not a preference."""

    def test_maps_hrtf_to_dry_and_preserves_the_rep_prefix(self):
        self.assertEqual(dry_stimulus_name("part1_track2_hrtf.wav"), "part1_track2_dry.wav")
        self.assertEqual(
            dry_stimulus_name("rep_part4_track1_hrtf.wav"), "rep_part4_track1_dry.wav"
        )

    def test_is_idempotent_because_some_trials_already_name_the_dry_file(self):
        # Half the published trials carry dry names in `stimuli`, so a mapping
        # that only handled `_hrtf` would leave those trials inconsistent.
        self.assertEqual(dry_stimulus_name("part3_track1_dry.wav"), "part3_track1_dry.wav")
        self.assertEqual(
            dry_stimulus_name("rep_part2_track2_dry.wav"), "rep_part2_track2_dry.wav"
        )

    def test_refuses_a_name_it_cannot_classify(self):
        # Silently passing an unrecognised name through would let the hrtf file be
        # used as the reference, which is the defect this mapping exists to stop.
        with self.assertRaises(ValueError):
            dry_stimulus_name("part1_track1.wav")

    def test_refuses_a_name_that_would_leave_the_stimulus_root(self):
        # The name is joined onto the stimuli directory, so a relative path in the
        # .mat must not be able to reach past it.
        with self.assertRaises(ValueError):
            dry_stimulus_name("../part1_track1_hrtf.wav")


class LoaderResolutionTests(unittest.TestCase):
    """The trap: the hrtf name need be neither present nor unique on disk."""

    def test_loader_ignores_absent_and_duplicated_hrtf_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stimuli = root / "stimuli"
            (stimuli / "decoy").mkdir(parents=True)
            wavfile.write(stimuli / "part1_track1_dry.wav", 8000, np.full(8000, 1000, np.int16))
            wavfile.write(stimuli / "part1_track2_dry.wav", 8000, np.full(8000, 2000, np.int16))
            # `part1_track1_hrtf.wav` is deliberately absent from the flat layout
            # and deliberately duplicated in two other places. The loader must
            # resolve neither: it derives the dry name and never reads an hrtf.
            wavfile.write(stimuli / "part1_track1_hrtf.wav", 8000, np.full(8000, 3000, np.int16))
            wavfile.write(
                stimuli / "decoy" / "part1_track1_hrtf.wav", 8000, np.full(8000, 4000, np.int16)
            )
            record = {
                "RawData": {"EegData": np.ones((128, 2))},
                "FileHeader": {"SampleRate": 128},
                "stimuli": np.array(
                    ["part1_track2_hrtf.wav", "part1_track1_hrtf.wav"], dtype=object
                ),
                "attended_track": 2,
            }
            path = root / "S1.mat"
            savemat(path, {"trials": np.array([record], dtype=object)})

            trial = load_kuleuven(path, stimuli, ("F3", "F4"), "reference", "release")[0]

            # Candidate 0 is the sorted track1 excerpt, so the attended track 2 is
            # label 1 and the two columns are the two dry files, in name order.
            np.testing.assert_equal(trial.labels, 1)
            self.assertEqual(trial.group, "part1_track1_dry.wav|part1_track2_dry.wav")
            np.testing.assert_allclose(trial.audio[:, 0], pcm(1000))
            np.testing.assert_allclose(trial.audio[:, 1], pcm(2000))


class MetadataTests(unittest.TestCase):
    """The channel order is a derivation, and it is pinned to the montage."""

    def test_metadata_declares_the_standard_biosemi64_order(self):
        self.assertTrue(METADATA.is_file(), f"missing {METADATA}")
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        self.assertEqual(
            set(metadata), {"channel_names", "reference", "upstream_processing"}
        )
        try:
            import mne
        except ImportError as error:  # pragma: no cover - dependency is declared
            self.skipTest(f"mne is required to check the montage: {error}")
        names = list(mne.channels.make_standard_montage("biosemi64").ch_names)
        self.assertEqual(list(metadata["channel_names"]), names)
        # The authors re-reference by subtracting column 48, which only lines up
        # if the standard order is right.
        self.assertEqual(names[47], "Cz")
        self.assertIn("no channel-location file", metadata["upstream_processing"].lower())


class ConvertedTrialTests(unittest.TestCase):
    """What the conversion actually wrote, when it has been run."""

    @unittest.skipUnless(
        S1_TRIAL.is_file(),
        "converted KU Leuven trials are absent (datasets/ is git-ignored); "
        "run `python -B -m scripts.auditory.convert --kind kuleuven ...` first",
    )
    def test_converted_s1_trial_satisfies_the_contract(self):
        from nova2026.auditory.data import load_trial

        trial = load_trial(S1_TRIAL)
        self.assertEqual(trial.eeg.ndim, 2)
        self.assertEqual(trial.eeg.shape[1], 64)
        # The invariant the conversion plan pre-registered.
        self.assertEqual(trial.eeg.shape[0], len(trial.timestamps))
        self.assertEqual(trial.eeg.shape[0], len(trial.labels))
        self.assertEqual(trial.audio.ndim, 2)
        self.assertEqual(trial.audio.shape[1], 2)
        self.assertGreater(trial.audio_rate, 40)
        self.assertTrue(set(np.unique(trial.labels)).issubset({0, 1}))
        self.assertTrue(np.all(np.diff(trial.timestamps) > 0))
        self.assertAlmostEqual(trial.sample_rate, 128.0, places=6)
        # Candidate order is the sorted dry names, and the label names one of them.
        self.assertEqual(
            trial.group.split("|"),
            sorted(trial.group.split("|")),
        )
        self.assertTrue(trial.group.endswith("_dry.wav"))

        try:
            import mne
        except ImportError as error:  # pragma: no cover - dependency is declared
            self.skipTest(f"mne is required to check the montage: {error}")
        expected = list(mne.channels.make_standard_montage("biosemi64").ch_names)
        self.assertEqual(list(trial.channel_names), expected)
        self.assertEqual(trial.channel_names[47], "Cz")


if __name__ == "__main__":
    unittest.main()
