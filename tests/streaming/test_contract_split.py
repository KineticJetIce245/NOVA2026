"""The contract check gates on processing and records provenance, and a scoring
failure names its cause.

Measured on the operator's ANT session through the live route: 121 windows, 0
scored, 116 failures, every one of them a ``ValueError: EEG preprocessing contract
does not match the model`` raised by ``RidgeDecoder.validate``. The chain and the
model agreed on eight of the ten processing-deciding keys; the two that differed
were *provenance*: ``input_reference`` (the rig declares CPz, the KU Leuven model
records an unknown/Cz derivation) and ``upstream_processing`` (each loader's own
paragraph about where its data came from). Comparing the whole dict for equality
therefore refused every honest run on a different rig - and copying the model's
text onto another rig's recording would have been a false statement about the
data, not a fix.

What these tests pin down:

* a **processing** difference still refuses - band, rate, units, channel set,
  order, stage, resample quality, or a key missing from either side;
* a **provenance** difference is accepted and, in the same motion, *recorded*,
  with both texts, so a cross-rig run is explicit rather than silent;
* the record is what a run record can print: JSON-safe, and naming what gated;
* a scoring failure carries the exception's own text into the ``scoring_failed``
  reason and into ``SessionSummary.scoring_failures``, so ``failed 116`` can no
  longer be the whole story;
* the chain's contract gains no key (a new key would invalidate every trained
  decoder), and nothing in the decoder rewrites either contract.

The window is a real chain's own window over a synthetic microvolt trace, and the
model is fitted on data this chain produced, so the premise is built here rather
than inherited from ``models/`` or ``datasets/`` (plan section 6.4 item 1).
"""

import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryWindow
from nova2026.auditory.decoder import GATE_KEYS, PROVENANCE_KEYS, RidgeDecoder
from nova2026.auditory.session import (
    FAILURE_REASON_LIMIT,
    AttentionSession,
    RunPolicy,
    _failure_reason,  # private on purpose: its text is what the record carries
)
from nova2026.auditory.streaming import AuditoryProcessor, stream_config

REPO = Path(__file__).resolve().parents[2]
RATE = 128.0
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)
SECONDS = 12.0
"""Long enough for two windows at the model's one-second step and 5 s history."""

MODEL_REFERENCE = "KU Leuven: no reference channel shipped; column 48 is Cz."
SOURCE_REFERENCE = "CPz, declared in the .cnt header's Basic Channel Data (REF:CPz)."
MODEL_UPSTREAM = "KU Leuven: the authors' 0.5 Hz high-pass and 128 Hz downsampling."
SOURCE_UPSTREAM = "this rig: 0.5 Hz high-pass, recorded at 500 Hz."
"""The two texts that actually differed on the ANT run, shortened to their claims."""


def settings(**overrides):
    """Chain settings for this fixture: 20 named electrodes at 128 Hz."""

    trial = SimpleNamespace(
        sample_rate=RATE, channel_names=CHANNELS, reference=SOURCE_REFERENCE,
        upstream_processing=SOURCE_UPSTREAM,
    )
    config = AuditoryConfig()
    return stream_config(
        trial, config, history=5.0, step=1.0,
        check_channels=bool(overrides.pop("check_channels", False)),
        exclude_channels=overrides.pop("exclude_channels", ()),
        source_unit_exponent=-6, **overrides,
    )


def trace(seconds=SECONDS, rate=RATE, channels=len(CHANNELS), seed=7):
    """A deterministic microvolt trace: enough movement to fit a decoder on."""

    rng = np.random.default_rng(seed)
    count = int(round(seconds * rate))
    time = np.arange(count) / rate
    rows = [
        (12.0 + index) * np.sin(2 * np.pi * (2.0 + index / 8.0) * time + index)
        + rng.normal(0.0, 2.0, count)
        for index in range(channels)
    ]
    return np.stack(rows, axis=1), time


def envelopes_for(timestamps, seed=11):
    """Two candidate envelopes on the window's own time grid.

    Synthetic and independent of the EEG - this module tests the contract check,
    not the decoder's accuracy - but shaped like the real thing: nonnegative,
    varying, and at the feature rate the chain's windows are on.
    """

    rng = np.random.default_rng(seed)
    time = np.asarray(timestamps, dtype=float)
    return np.stack(
        [1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * time),
         1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * time + 1.0)],
        axis=1,
    ) + rng.normal(0.0, 0.01, (len(time), 2))


def chain_windows(processor, eeg, time, block=256):
    """Every window a real chain produces from one trace, in order."""

    windows = []
    for start in range(0, len(eeg), block):
        windows.extend(processor.feed((eeg[start : start + block],
                                       time[start : start + block])))
    return windows


class Fixture(unittest.TestCase):
    """One fitted model and one real window, built per class.

    The windows come from a real :class:`AuditoryProcessor` over a synthetic
    microvolt trace, and the contract under test is that chain's own, so the
    only thing this module changes is the key it names in each test.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = AuditoryConfig()
        eeg, time = trace()
        cls.processor = AuditoryProcessor(settings())
        chain = chain_windows(cls.processor, eeg, time)
        cls.assertTrue(cls, chain, "the chain produced no window to test on")
        cls.windows = [
            AuditoryWindow(
                window.data, envelopes_for(window.timestamps), window.timestamps,
                float(window.timestamps[-1]), bool(window.valid), tuple(window.reasons),
                dict(cls.processor.contract), int(window.segment),
            )
            for window in chain
            if window.valid
        ]
        cls.assertTrue(cls, cls.windows, "no valid window to test on")
        # The model is fitted on windows carrying the *model's* provenance text
        # while the chain declares the rig's: that is the ANT run's measured shape,
        # and it is what lets these tests tell the two sides of the comparison
        # apart. Nothing else differs - every gate key is identical.
        cls.model = RidgeDecoder(cls.config)
        cls.model.fit(_examples(Fixture.training_windows(cls)))
        cls.window = cls.windows[-1]

    @staticmethod
    def training_windows(cls):
        """The fixture's windows carrying the model's own provenance text."""

        model_contract = dict(cls.processor.contract, input_reference=MODEL_REFERENCE,
                              upstream_processing=MODEL_UPSTREAM)
        return [Fixture._with(window, model_contract) for window in cls.windows]

    @staticmethod
    def _with(window, contract):
        return AuditoryWindow(
            window.eeg, window.envelopes, window.timestamps,
            float(window.available_at), True, tuple(window.reasons),
            dict(contract), int(window.segment),
        )

    def altered(self, **contract_changes):
        """The chain's own window with a copy of its contract changed by name."""

        return self._with(self.window, dict(self.window.contract, **contract_changes))

    def without(self, key):
        contract = dict(self.window.contract)
        contract.pop(key, None)
        return self._with(self.window, contract)

    def model_without(self, key):
        """A copy of the fitted model whose contract lacks ``key``.

        A copy, not the fixture's own decoder: one model is fitted per class
        here, so editing it in place would leak this test's contract into every
        other test in the module.
        """

        model = copy.copy(self.model)
        model.contract = {
            name: value for name, value in self.model.contract.items()
            if name != key
        }
        return model


def _examples(windows):
    """``(window, per-sample labels)`` pairs for fitting, from the chain itself."""

    examples = []
    for window in windows:
        labels = np.zeros(len(window.eeg), dtype=int)
        labels[len(labels) // 2 :] = 1
        examples.append((window, labels))
    return examples


class GatedKeysTests(Fixture):
    """Anything that changes what the decoder is fed still refuses."""

    def test_a_matching_contract_is_accepted_on_every_key(self):
        # The premise of the rest of this module: every gate key matches, so a
        # refusal below is caused by the change the test made - and the two
        # provenance keys differ, which is what this split exists for.
        self.assertIsNotNone(self.model.weights)
        self.assertEqual(
            len([key for key in GATE_KEYS if key in self.window.contract]),
            len(GATE_KEYS),
        )
        self.model.validate(self.window)
        record = self.model.contract_provenance()
        self.assertEqual(sorted(record["provenance"]),
                         ["input_reference", "upstream_processing"])
        self.assertEqual(record["gated"], {})

    def test_a_different_band_still_refuses(self):
        with self.assertRaises(ValueError) as caught:
            self.model.validate(self.altered(bandpass=[0.5, 32.0]))
        self.assertIn("bandpass", str(caught.exception))
        self.assertIn("processing keys", str(caught.exception))

    def test_a_different_rate_still_refuses(self):
        # Both rate keys are gated: `input_sfreq` is what the chain was fed and
        # `output_sfreq` is the feature rate the decoder was fitted at.
        for key, value in (("input_sfreq", 64.0), ("output_sfreq", 128.0)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    self.model.validate(self.altered(**{key: value}))
                self.assertIn(key, str(caught.exception))

    def test_a_different_channel_set_or_order_refuses(self):
        swapped = list(self.window.contract["eeg_channels"])
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaises(ValueError) as caught:
            self.model.validate(self.altered(eeg_channels=swapped))
        self.assertIn("eeg_channels", str(caught.exception))

    def test_the_remaining_gated_keys_refuse(self):
        for key, value in (
            ("units", "mV"), ("filter_order", 4), ("stage", "some-other-chain"),
            ("resample_quality", "fast"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    self.model.validate(self.altered(**{key: value}))
                self.assertIn(key, str(caught.exception))

    def test_a_gate_key_missing_from_a_side_refuses(self):
        # Keys are never dropped from the comparison: an absent key is reported as
        # absent rather than skipped, which is what would let a contract shrink
        # into agreement.
        for key in ("units", "bandpass", "stage"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    self.model.validate(self.without(key))
                self.assertIn(key, str(caught.exception))
                self.assertIn("absent", str(caught.exception))

    def test_a_gate_key_absent_from_both_sides_is_accepted(self):
        # The case this suite shipped without, and with it the correction that
        # was not implemented: a key *neither* side declares is not a processing
        # difference. Nothing can disagree when neither contract carries it, so
        # refusing it refuses a contract that decides nothing - the same false
        # refusal that cost the ANT session 116 of 116 windows over a number
        # that could not differ. The old line (``key not in incoming or key not
        # in model``) refused exactly this case and called it "absent from one
        # side", which was false twice over.
        for key in ("stage", "resample_quality", "filter_order", "units"):
            with self.subTest(key=key):
                model = self.model_without(key)
                contract = dict(model.contract)
                window = self._with(self.window, contract)
                # Byte-identical and short exactly one gate key: this is an
                # absent key, not a difference dressed up as one.
                self.assertNotIn(key, contract)
                self.assertEqual(model.contract, dict(window.contract))
                model.validate(window)  # must not raise
                record = model.contract_provenance()
                self.assertEqual(record["gated"], {})
                # Nor is it smuggled out as a key present on one side only.
                self.assertEqual(record["only_in_model"], [])
                self.assertEqual(record["only_in_source"], [])

    def test_the_absent_from_one_side_message_is_true_where_it_refuses(self):
        # The other half of the defect: where an absent key *is* a difference,
        # "absent from one side" has to be a fact about the two contracts rather
        # than a phrase. Every case below has the key on exactly one side.
        for key in ("units", "bandpass", "stage"):
            with self.subTest(key=key, missing_from="window"):
                window = self.without(key)
                self.assertNotIn(key, window.contract)
                self.assertIn(key, self.model.contract)
                with self.assertRaises(ValueError) as caught:
                    self.model.validate(window)
                self.assertIn(f"processing keys {key}", str(caught.exception))
                self.assertIn("absent from one side", str(caught.exception))
            with self.subTest(key=key, missing_from="model"):
                model = self.model_without(key)
                self.assertNotIn(key, model.contract)
                self.assertIn(key, self.window.contract)
                with self.assertRaises(ValueError) as caught:
                    model.validate(self.window)
                self.assertIn(f"processing keys {key}", str(caught.exception))
                self.assertIn("absent from one side", str(caught.exception))


class RecordedProvenanceTests(Fixture):
    """A provenance difference is accepted, and recorded in the same motion."""

    def test_a_provenance_only_difference_is_accepted_and_recorded(self):
        # The fitted model records the KU Leuven provenance; the chain declares
        # this rig's. That is the shape the ANT run measured, and it must not
        # refuse - but it must be recorded.
        scores = self.model.score(self.window)
        self.assertEqual(scores.shape, (2,))
        record = self.model.contract_provenance()
        self.assertEqual(record["provenance"], {
            "input_reference": {"model": MODEL_REFERENCE, "source": SOURCE_REFERENCE},
            "upstream_processing": {"model": MODEL_UPSTREAM, "source": SOURCE_UPSTREAM},
        })
        self.assertEqual(record["gated"], {})

    def test_both_provenance_keys_are_reported_when_both_differ(self):
        record = self.model.record_contract_difference(self.window.contract)
        self.assertEqual(sorted(record["provenance"]),
                         ["input_reference", "upstream_processing"])
        # The model's own text is carried beside the rig's, so a reader can see
        # exactly what was not equalised.
        self.assertIn("KU Leuven", record["provenance"]["input_reference"]["model"])
        self.assertIn("CPz", record["provenance"]["input_reference"]["source"])
        json.dumps(record)  # the run record has to be able to write it

    def test_a_matching_contract_records_nothing_and_a_missing_key_still_reports(self):
        # Nothing differs when the model is asked about its own contract text.
        self.assertEqual(self.model.contract_provenance()["provenance"], {})
        self.assertEqual(
            self.model.record_contract_difference(self.model.contract)["provenance"], {}
        )
        # But a key that vanished from one side is reported even when its content
        # would have matched: keys are never dropped from the comparison.
        record = self.model.record_contract_difference(
            {key: value for key, value in self.model.contract.items()
             if key != "upstream_processing"}
        )
        self.assertIn("upstream_processing", record["provenance"])
        self.assertIsNone(record["provenance"]["upstream_processing"]["source"])
        self.assertIn("upstream_processing", record["only_in_model"])

    def test_an_extra_source_key_is_reported_and_does_not_refuse(self):
        record = self.model.record_contract_difference(
            dict(self.window.contract, rig_note="not a processing key")
        )
        self.assertEqual(record["only_in_source"], ["rig_note"])
        self.model.validate(self.altered(rig_note="not a processing key"))

    def test_the_chain_contract_gains_no_key_and_the_model_is_not_edited(self):
        contract = self.processor.contract
        self.assertEqual(
            sorted(contract),
            sorted(list(GATE_KEYS) + list(PROVENANCE_KEYS) + ["step_seconds"]),
        )
        before = json.dumps(self.model.contract, sort_keys=True)
        self.model.validate(self.altered(input_reference="another rig entirely"))
        # The difference is recorded, never applied: the model keeps the contract
        # it was fitted under, and the chain keeps the one it declared.
        self.assertEqual(json.dumps(self.model.contract, sort_keys=True), before)
        self.assertEqual(
            self.altered(input_reference=SOURCE_REFERENCE).contract["input_reference"],
            SOURCE_REFERENCE,
        )


class FailureCauseTests(unittest.TestCase):
    """``failed 116`` must not be the whole story; the cause travels with it."""

    def test_the_reason_names_the_type_and_the_message(self):
        self.assertEqual(
            _failure_reason(ValueError("EEG preprocessing contract does not match")),
            "ValueError: EEG preprocessing contract does not match",
        )
        self.assertEqual(_failure_reason(RuntimeError()), "RuntimeError")
        folded = _failure_reason(ValueError("first line\nsecond\tline"))
        self.assertEqual(folded, "ValueError: first line second line")
        long = _failure_reason(ValueError("x" * 500))
        self.assertLessEqual(len(long), FAILURE_REASON_LIMIT)
        self.assertTrue(long.endswith("..."))
        self.assertTrue(long.startswith("ValueError: "), "the type survives clipping")

    def test_a_scoring_failure_records_its_cause_in_the_summary(self):
        # A session whose decoder always refuses is the measured ANT shape:
        # nothing scored, every window failed - and now a named cause.
        session, frames, summary = _failing_session()
        self.assertGreater(summary["failed"], 0)
        self.assertEqual(summary["scored"], 0)
        reasons = [
            reason for frame in frames for reason in frame.reasons
            if reason.startswith("ValueError: ")
        ]
        self.assertTrue(reasons, "the cause must be visible in the published frame")
        self.assertIn("on purpose", reasons[0])
        self.assertEqual(len(summary["scoring_failures"]), 1)
        entry = summary["scoring_failures"][0]
        self.assertEqual(entry["type"], "ValueError")
        self.assertEqual(entry["count"], session.failed)
        self.assertIn("scoring failed on purpose", entry["reason"])
        json.dumps(summary)


def _failing_session():
    """A real session whose decoder raises; the chain and envelopes are real.

    Built from ``test_session``'s own fixture in a temporary directory, so the
    cause this test asserts on is produced by the session's real offload path
    rather than by calling the private handler directly.
    """

    from scripts.auditory.tests.test_session import BrokenModel, FixtureCase

    FixtureCase.setUpClass()
    try:
        session = FixtureCase.session(
            FixtureCase, model=BrokenModel(FixtureCase.model, delay=0.0)
        )
        frames = []
        summary = session.run(frames.append, threading.Event()).to_dict()
    finally:
        FixtureCase.tearDownClass()
    return session, frames, summary


MUTATIONS = (
    {
        "name": "the-gate-stops-refusing-a-processing-difference",
        "file": "src/nova2026/auditory/decoder.py",
        "before": '            if differing:\n'
                  '                raise ValueError(\n'
                  '                    "EEG preprocessing contract does not match the model: the "\n'
                  '                    "processing keys " + ", ".join(differing) + " differ."\n'
                  '                )\n',
        "after": '            if differing:  # the processing difference is no longer refused\n'
                 '                pass\n',
        "test": "GatedKeysTests.test_a_different_band_still_refuses",
    },
    {
        "name": "the-provenance-difference-is-not-recorded",
        "file": "src/nova2026/auditory/decoder.py",
        "before": '        record = {"provenance": {}, "gated": {}, '
                  '"only_in_model": [], "only_in_source": []}',
        "after": '        record = {"gated": {}, "only_in_model": [], "only_in_source": []}',
        "test": "RecordedProvenanceTests."
                "test_a_provenance_only_difference_is_accepted_and_recorded",
    },
    {
        "name": "a-scoring-failure-loses-its-cause",
        "file": "src/nova2026/auditory/session.py",
        "before": '                None, self._now, self._now, False, ("scoring_failed", reason)',
        "after": '                None, self._now, self._now, False, ("scoring_failed",)',
        "test": "FailureCauseTests.test_a_scoring_failure_records_its_cause_in_the_summary",
    },
    {
        "name": "the-gate-refuses-a-key-absent-from-both-sides",
        "file": "src/nova2026/auditory/decoder.py",
        "before": '            missing = [key for key in GATE_KEYS if (key in incoming) != (key in model)]\n',
        "after": '            missing = [key for key in GATE_KEYS if key not in incoming or key not in model]\n',
        "test": "GatedKeysTests.test_a_gate_key_absent_from_both_sides_is_accepted",
    },
)
"""Each removes one piece of the mechanism; each must turn its named test red.

Kept focused rather than a suite: one for the gate, one for the record, one for
the cause - and one for the line that shipped late, a gate key absent from both
sides, because a mutation nobody wrote a case for is a correction nobody sees
revert. They are applied to a **copy** of the tree, never to the working tree.
"""


def run_mutation_check(repo: Path, *, timeout: float = 600.0) -> dict:
    """Apply each mutation in a copied tree and report whether its test went red.

    The four conditions this repository learned to require (plan section 6.4 and
    ``test_declared_amplitude``): a **copy of the tree** so the real one is only
    read; the child's own ``src`` **pinned at the front of ``sys.path``**, because
    ``.venv``'s editable install would otherwise put the unmutated module back;
    every ``__pycache__`` in the copy **cleared before each run**, because a stale
    ``.pyc`` can hide a mutation whose mtime lands in the same second; the
    mutated file **compiled** before it runs, because a syntax error also makes a
    test "fail" while proving nothing; and the child **imports this module by
    name**, because ``unittest.main(module=None, ...)`` resolves a bare
    ``Class.test`` name as a top-level module, loads a failed-import test
    instead, and so reports every mutation "caught" (``errors=1``, ``Ran 1 test
    in 0.000s``) while no test ran at all.
    """

    import shutil

    report = {"kind": "mutation check for the contract split and the failure cause",
              "results": []}
    with tempfile.TemporaryDirectory(prefix="nova-contract-mutation-") as scratch:
        copy = Path(scratch) / "repo"
        shutil.copytree(
            repo, copy,
            ignore=shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__"),
        )
        for cache in copy.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        runner = (
            "import sys\n"
            "sys.path[:0] = ['', {root!r}, {src!r}]\n"
            "import unittest\n"
            "unittest.main(module={module!r}, "
            "argv=['unittest', '-v', {test!r}])\n"
        )
        for mutation in MUTATIONS:
            target = copy / mutation["file"]
            original = target.read_text(encoding="utf-8")
            if original.count(mutation["before"]) != 1:
                report["results"].append({
                    "mutation": mutation["name"], "caught": False,
                    "detail": (
                        f"the mutation text appears {original.count(mutation['before'])} "
                        "times in the source, not once: the mutation itself is stale"
                    ),
                })
                continue
            mutated = original.replace(mutation["before"], mutation["after"], 1)
            target.write_text(mutated, encoding="utf-8")
            for cache in copy.rglob("__pycache__"):
                shutil.rmtree(cache, ignore_errors=True)
            try:
                compile(mutated, str(target), "exec")
            except SyntaxError as error:
                target.write_text(original, encoding="utf-8")
                report["results"].append({
                    "mutation": mutation["name"], "caught_by": mutation["test"],
                    "caught": False, "invalid_mutation": True,
                    "detail": f"the mutated source does not parse: {error}",
                })
                continue
            result = subprocess.run(
                [sys.executable, "-B", "-c", runner.format(
                    root=str(copy), src=str(copy / "src"),
                    module="tests.streaming.test_contract_split",
                    test=mutation["test"],
                )],
                cwd=str(copy), capture_output=True, text=True, timeout=timeout,
            )
            target.write_text(original, encoding="utf-8")
            report["results"].append({
                "mutation": mutation["name"], "caught_by": mutation["test"],
                "caught": result.returncode != 0,
                "returncode": result.returncode,
                "detail": (result.stderr or result.stdout).strip().splitlines()[-1:],
            })
    report["survivors"] = [
        row["mutation"] for row in report["results"] if not row["caught"]
    ]
    return report


class MutationEvidenceTests(unittest.TestCase):
    """Each mutation must turn its named test red, run here rather than claimed."""

    def test_every_mutation_is_caught_by_its_named_test(self):
        report = run_mutation_check(REPO)
        out = REPO / "results" / "antneuro_contract_split_mutations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Written before the assertions on purpose: when a mutation survives, the
        # report is the evidence of which one.
        out.write_text(json.dumps(report, indent=2))
        self.assertEqual(
            [row["mutation"] for row in report["results"] if not row["caught"]], []
        )
        self.assertEqual(len(report["results"]), len(MUTATIONS))


if __name__ == "__main__":  # pragma: no cover - manual runs
    unittest.main()
