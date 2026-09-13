"""Step 5's obligations, asserted rather than promised.

Four things this file refuses to take on trust: that a split is group-disjoint
across the ``rep_`` replays *and* carries both classes, that a feature cache
cannot be reused after the frozen contract moves, that a saved model's contract
survives a round trip and still rejects a window from another contract, and that
no ground-truth label reaches the decoder. The last one is measured by scoring
the same windows twice from two caches that differ only in their labels.

The tests that need a real converted trial skip when ``datasets/`` has not been
built, which is how the rest of this suite treats the git-ignored artifacts.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryWindow
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import check_split

from scripts.auditory import aad_ridge, feature_cache, train_kuleuven
from scripts.auditory.kuleuven_contract import CONVERTED, canonical_group

REPO = Path(__file__).resolve().parents[3]
S1_TRIAL = CONVERTED / "S1" / "trial_008.npz"

GROUPS = [
    "part1_track1_dry.wav|part1_track2_dry.wav",
    "part2_track1_dry.wav|part2_track2_dry.wav",
    "part3_track1_dry.wav|part3_track2_dry.wav",
    "part4_track1_dry.wav|part4_track2_dry.wav",
]


def synthetic_corpus(subjects=("S1", "S2", "S3", "S4")):
    """Two trials per (subject, story), one of each label, as the dataset has."""

    trials = []
    for subject in subjects:
        for group in GROUPS:
            base = group.split("|")[0]
            for label in (0, 1):
                stored = group if label == 0 else "|".join(
                    f"rep_{name}" for name in group.split("|"))
                trials.append(train_kuleuven.TrialRef(
                    subject, f"{group[:5]}_{label}", canonical_group(stored), label,
                    Path("unused.npz")))
    return trials


class SplitTests(unittest.TestCase):
    """Both schemes must be group-disjoint and stratified."""

    def corpus(self, trials):
        corpus = train_kuleuven.Corpus.__new__(train_kuleuven.Corpus)
        corpus.key = "test"
        corpus.contracts = {}
        corpus.trials = trials
        return corpus

    def test_story_folds_hold_out_whole_groups(self):
        corpus = self.corpus(synthetic_corpus())
        folds = train_kuleuven.held_out_story_folds(corpus)
        self.assertEqual(len(folds), len(GROUPS))
        for fold in folds:
            self.assertNotIn(fold["name"], {t.group for t in fold["training"]})
            self.assertEqual({t.group for t in fold["validation"]}, {fold["name"]})
            self.assertEqual({t.label for t in fold["validation"]}, {0, 1})

    def test_subject_folds_hold_out_whole_subjects(self):
        corpus = self.corpus(synthetic_corpus())
        folds = train_kuleuven.leave_one_subject_folds(corpus)
        self.assertEqual(len(folds), 4)
        for fold in folds:
            held = {t.subject for t in fold["validation"]}
            self.assertEqual(held, {fold["name"]})
            self.assertFalse(held & {t.subject for t in fold["training"]})
            self.assertEqual({t.label for t in fold["validation"]}, {0, 1})

    def test_the_stored_name_guard_accepts_a_split_that_replays_the_same_audio(self):
        # The trap results/kuleuven_audit.md section 4 records: rep_part1_track1
        # replays part1_track1 sample for sample, so a split drawn on the stored
        # names trains and validates on the same 125 s of audio.
        base = train_kuleuven.TrialRef("S1", "a", "part1_track1_dry.wav|part1_track2_dry.wav",
                                       0, Path("unused.npz"))
        repeat = train_kuleuven.TrialRef(
            "S2", "b", "rep_part1_track1_dry.wav|rep_part1_track2_dry.wav", 0,
            Path("unused.npz"))
        naive = train_kuleuven.TrialRef("S2", "b",
                                        canonical_group(repeat.group), 0, Path("unused.npz"))
        check_split([base], [repeat])  # the stored names look disjoint
        with self.assertRaises(ValueError):
            train_kuleuven.assert_group_disjoint([base], [naive])

    def test_a_validation_side_with_one_class_is_refused(self):
        corpus = self.corpus(synthetic_corpus())
        only_a = [t for t in corpus.trials if t.label == 0]
        with self.assertRaises(ValueError):
            train_kuleuven.assert_group_disjoint(corpus.trials, only_a)


class CacheKeyTests(unittest.TestCase):
    """A contract change must move the key, and an old file must be refused."""

    def test_the_key_moves_with_the_contract_material(self):
        material = feature_cache.key_material()
        reference, _ = feature_cache.cache_key(material)
        for field, value in (
            ("auditory_config", dict(material["auditory_config"], band=[2.0, 9.0])),
            ("channel_contracts", dict(material["channel_contracts"], **{"20ch": ["Fp1"]})),
            ("histories", [5.0]),
            ("envelope_generator_version", "999"),
            ("stage", "another-stage"),
        ):
            changed = dict(material, **{field: value})
            self.assertNotEqual(feature_cache.cache_key(changed)[0], reference, field)

    def test_a_cache_from_another_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._synthetic_cache(Path(directory) / "trial.npz", "aaaaaaaaaaaaaaaa")
            self.assertEqual(len(feature_cache.load(path, "aaaaaaaaaaaaaaaa").times), 800)
            with self.assertRaises(ValueError):
                feature_cache.load(path, "bbbbbbbbbbbbbbbb")

    def _synthetic_cache(self, path, key, labels=None):
        config = AuditoryConfig()
        samples, rate = 800, config.sample_rate
        rng = np.random.default_rng(11)
        contracts = feature_cache.channel_contracts()
        arrays = {
            "signal64": rng.standard_normal((samples, len(contracts["64ch"]))).astype(np.float32),
            "signal20": rng.standard_normal((samples, len(contracts["20ch"]))).astype(np.float32),
            "times": np.arange(samples) / rate,
            "envelopes": np.abs(rng.standard_normal((samples, 2))).astype(np.float32),
            "labels": np.full(samples, 0, dtype=np.int8) if labels is None
            else np.asarray(labels, dtype=np.int8),
        }
        metadata = {
            "cache_version": feature_cache.CACHE_VERSION, "contract_key": key,
            "key_material": feature_cache.key_material(), "subject": "S1",
            "trial_id": "trial_008", "group_stored": GROUPS[0],
            "group_key": GROUPS[0], "candidates": GROUPS[0].split("|"),
            "eeg_samples": samples, "eeg_sha256": "0" * 64,
            "sample_rate": 128.0, "audio_rate": 44100.0,
            "channel_contracts": contracts, "contracts": {}, "histories": list(feature_cache.HISTORIES),
            "step_seconds": feature_cache.STEP_SECONDS,
            "warmup_seconds": feature_cache.WARMUP_SECONDS,
            "lag_samples": config.lag_samples, "fault_spans": {}, "envelopes": {},
            "windows": {},
        }
        for name in feature_cache.CONTRACTS:
            metadata["fault_spans"][name] = {}
            metadata["contracts"][name] = {
                feature_cache.history_tag(history): {
                    "eeg_channels": contracts[name], "output_sfreq": rate,
                    "input_sfreq": 128.0, "units": "uV", "bandpass": [1.0, 9.0],
                    "filter_order": 3, "stage": feature_cache.STAGE,
                    "resample_quality": feature_cache.RESAMPLE_QUALITY,
                    "input_reference": "none", "upstream_processing": "synthetic",
                    "window_seconds": history, "step_seconds": feature_cache.STEP_SECONDS,
                }
                for history in feature_cache.HISTORIES
            }
            for history in feature_cache.HISTORIES:
                _, valid, _ = feature_cache.window_validity(samples, history, [])
                arrays[f"valid_{name}_{feature_cache.history_tag(history)}"] = valid
                metadata["fault_spans"][name][feature_cache.history_tag(history)] = []
        np.savez(path, metadata=json.dumps(metadata), **arrays)
        return path


class FastFitTests(unittest.TestCase):
    """The moment path is an optimization, so it must equal the reference."""

    def setUp(self):
        config = AuditoryConfig()
        rng = np.random.default_rng(7)
        self.signal = rng.standard_normal((2400, 5)) * 12 + 3
        envelope = np.abs(rng.standard_normal((2400, 2))) + 0.1
        self.envelopes = np.ascontiguousarray(
            np.convolve(envelope[:, 0], np.ones(9) / 9, "same")[:, None]
            * np.array([[1.0, 0.8]]))
        self.times = np.arange(2400) / config.sample_rate
        self.length = 320
        self.starts = list(range(0, 2400 - self.length + 1, self.length))
        self.contract = {
            "eeg_channels": [f"ch{i}" for i in range(5)], "output_sfreq": 64,
            "input_sfreq": 64.0, "units": "uV", "bandpass": [1.0, 9.0],
            "filter_order": 3, "stage": "synthetic", "window_seconds": 5.0,
            "step_seconds": 1.0, "resample_quality": "QQ", "input_reference": "none",
            "upstream_processing": "synthetic",
        }

    def examples(self):
        built = []
        for start in self.starts:
            stop = start + self.length
            built.append((
                AuditoryWindow(self.signal[start:stop], self.envelopes[start:stop],
                               self.times[start:stop], float(self.times[stop - 1]),
                               True, (), dict(self.contract), 0),
                np.full(self.length, 0),
            ))
        return built

    def test_the_moment_fit_equals_the_reference_fit(self):
        reference = RidgeDecoder(AuditoryConfig(), 100.0)
        reference.fit(self.examples(), {"training": [], "validation": []})
        fast = aad_ridge.fit(
            aad_ridge.trial_moments(self.signal, self.envelopes, 0, self.starts,
                                    self.length, AuditoryConfig().lag_samples),
            self.contract, 100.0)
        self.assertTrue(np.allclose(reference.mean, fast.mean, rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(reference.scale, fast.scale, rtol=0, atol=1e-11))
        self.assertTrue(np.allclose(reference.weights, fast.weights, rtol=1e-10, atol=1e-12))
        for window, _ in self.examples():
            self.assertTrue(np.allclose(reference.score(window), fast.score(window)))

    def test_the_model_contract_round_trips_through_save_and_load(self):
        model = aad_ridge.fit(
            aad_ridge.trial_moments(self.signal, self.envelopes, 0, self.starts,
                                    self.length, AuditoryConfig().lag_samples),
            self.contract, 100.0,
            training_info={"training": [("S1", "trial_000", "g")], "validation": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = RidgeDecoder.load(path)
        self.assertEqual(loaded.contract, model.contract)
        # JSON turns the provenance tuples into lists; the content must survive.
        self.assertEqual(loaded.training_info["training"],
                         [list(entry) for entry in model.training_info["training"]])
        self.assertTrue(np.allclose(loaded.weights, model.weights))
        self.assertTrue(np.allclose(loaded.mean, model.mean))

    def test_a_window_from_another_contract_is_refused(self):
        model = aad_ridge.fit(
            aad_ridge.trial_moments(self.signal, self.envelopes, 0, self.starts,
                                    self.length, AuditoryConfig().lag_samples),
            self.contract, 100.0)
        other = dict(self.contract, window_seconds=10.0)
        window = AuditoryWindow(self.signal[: self.length], self.envelopes[: self.length],
                                self.times[: self.length], float(self.times[self.length - 1]),
                                True, (), other, 0)
        with self.assertRaises(ValueError):
            model.score(window)


class LabelIsolationTests(unittest.TestCase):
    """No label reaches the decoder: only the score does, and only afterwards."""

    def test_flipping_every_label_changes_the_truths_and_not_one_decision(self):
        helper = CacheKeyTests("test_a_cache_from_another_contract_is_refused")
        key = "cccccccccccccccc"
        with tempfile.TemporaryDirectory() as directory:
            a = helper._synthetic_cache(Path(directory) / "a.npz", key,
                                        labels=np.zeros(800, dtype=np.int8))
            b = helper._synthetic_cache(Path(directory) / "b.npz", key,
                                        labels=np.ones(800, dtype=np.int8))
            features = feature_cache.load(a, key)
            starts = [start for start, _ in features.windows("64ch", 5.0)]
            moments = aad_ridge.trial_moments(
                features.signal("64ch").astype(float), features.envelopes.astype(float), 0,
                starts, round(5.0 * feature_cache.sample_rate()),
                features.metadata["lag_samples"])
            model = aad_ridge.fit(moments, features.contract["64ch"]["5s"], 100.0)
            ref_a = train_kuleuven.TrialRef("S1", "trial_008", GROUPS[0], 0, a)
            ref_b = train_kuleuven.TrialRef("S1", "trial_008", GROUPS[0], 1, b)
            decisions_a, truths_a = train_kuleuven.score_trials(
                model, [ref_a], "64ch", 5.0, key)
            decisions_b, truths_b = train_kuleuven.score_trials(
                model, [ref_b], "64ch", 5.0, key)
        self.assertEqual(len(decisions_a), len(decisions_b))
        self.assertTrue(np.array_equal(decisions_a, decisions_b))
        self.assertEqual(set(truths_a.tolist()), {0})
        self.assertEqual(set(truths_b.tolist()), {1})


class CacheAgainstChainTests(unittest.TestCase):
    """The cache must hold exactly what the replay chain produces."""

    def test_cached_signal_and_verdicts_match_the_replay_chain(self):
        if not S1_TRIAL.is_file():
            self.skipTest("the converted KU Leuven trials are not present")
        from nova2026.auditory.data import load_trial
        from scripts.auditory.runner import replay_windows

        trial = load_trial(S1_TRIAL)
        config = AuditoryConfig()
        replayed = list(replay_windows(trial, config, 5.0, 1.0))
        with tempfile.TemporaryDirectory() as directory:
            key = "dddddddddddddddd"
            feature_cache.CACHE_ROOT = Path(directory)
            try:
                feature_cache.build_trial(S1_TRIAL, key)
                features = feature_cache.load(
                    feature_cache.cache_path("S1", "trial_008", key), key)
            finally:
                feature_cache.CACHE_ROOT = REPO / "datasets" / "auditory_features"
        starts, valid, _ = features.grid("64ch", 5.0)
        self.assertEqual(len(starts), len(replayed))
        signal = features.signal("64ch").astype(float)
        for position, window in enumerate(replayed):
            start = int(round(window.timestamps[0] * config.sample_rate))
            self.assertEqual(start, int(starts[position]))
            self.assertEqual(window.valid, bool(valid[position]))
            if not window.valid:
                continue
            stop = start + len(window.eeg)
            self.assertTrue(np.array_equal(window.eeg, signal[start:stop]))
            # The reference envelope comes from the precomputed record here and
            # from the extractor in the replay chain, so agreeing to float32 is
            # evidence that the two paths are the same reference.
            self.assertTrue(np.allclose(
                window.envelopes, features.envelopes[start:stop].astype(float),
                rtol=0, atol=1e-5))

    def test_the_training_grid_is_the_non_overlapping_one(self):
        if not S1_TRIAL.is_file():
            self.skipTest("the converted KU Leuven trials are not present")
        from nova2026.auditory.data import load_trial
        from scripts.auditory.runner import replay_windows

        trial = load_trial(S1_TRIAL)
        replayed = list(replay_windows(trial, AuditoryConfig(), 5.0, 1.0))
        # The policy scripts/auditory/train.py::prepare applies: walk the 1 s
        # grid and keep a window only once the previous one's history has passed.
        expected = []
        next_start = -np.inf
        for window in replayed:
            if window.valid and window.timestamps[0] >= next_start:
                expected.append(int(round(window.timestamps[0] * 64)))
                next_start = window.timestamps[0] + 5.0
        with tempfile.TemporaryDirectory() as directory:
            key = "eeeeeeeeeeeeeeee"
            feature_cache.CACHE_ROOT = Path(directory)
            try:
                feature_cache.build_trial(S1_TRIAL, key)
                features = feature_cache.load(
                    feature_cache.cache_path("S1", "trial_008", key), key)
            finally:
                feature_cache.CACHE_ROOT = REPO / "datasets" / "auditory_features"
        chosen = [start for start, _ in features.windows("64ch", 5.0, hop=5.0)]
        self.assertEqual(chosen, expected)
        # Non-overlapping means the whole grid is about one window per five
        # seconds of signal, not one per second of hop.
        self.assertLessEqual(len(chosen) * (320 - AuditoryConfig().lag_samples),
                             features.samples)
