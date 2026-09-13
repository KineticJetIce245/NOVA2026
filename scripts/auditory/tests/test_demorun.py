"""Tests for the perturbation run's orchestration and for what its record carries.

These are not tests of the matrix's criteria - running the matrix is what tests
those, and its report states each verdict. They pin the three properties the
matrix's *third* question depends on, plus the run record's new fields:

1. every case runs in its own process and writes only under its own directory, so
   one case cannot reach another;
2. a case whose process dies without a verdict is recorded as a crash and the run
   continues to the next case, instead of aborting the whole matrix;
3. a verdict distinguishes "the plan's named observable did not appear" from "the
   invariant was violated", because only the second one is dangerous;
4. the session's run record carries the chain's own recovery and repair
   diagnostics, so a fault the chain handled - which leaves no window behind - is
   still auditable.

Everything builds its own premise (plan section 6.4): the metrics case (C16) is
the subject because it is the one row of the matrix about labels rather than
signals, so it runs with ``--trial`` pointed at a path that does not exist. No
dataset, no model file, no browser and no network beyond loopback are involved.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:  # allow `python -m unittest` from anywhere
    sys.path.insert(0, str(REPO))

from nova2026.auditory.evaluation import inject_fault  # noqa: E402
from nova2026.auditory.session import AttentionSession, RunPolicy  # noqa: E402
from nova2026.auditory.sources import ReferenceEnvelopes, ReplaySource  # noqa: E402
from scripts.auditory.tests.session_fixture import build_fixture  # noqa: E402
from scripts.auditory_ui import demorun  # noqa: E402


def cli(argv):
    """Parse a command line exactly as ``main`` does, stamp and output names included."""

    return demorun.resolve(demorun.build_parser().parse_args(argv))


class OrchestrationTest(unittest.TestCase):
    """The parent process's own behaviour: isolation, crash recording, reporting."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scratch = Path(self.directory.name) / "scratch"

    def args(self, **overrides):
        """A real command line for the metrics case, with no dataset behind it."""

        argv = [
            "--only", "C16",
            "--trial", str(Path(self.directory.name) / "there-is-no-trial.npz"),
            "--model", str(Path(self.directory.name) / "there-is-no-model.npz"),
            "--scratch-root", str(self.scratch),
            "--stamp", "teststamp",
            "--out", str(Path(self.directory.name) / "perturbation.json"),
            "--md", str(Path(self.directory.name) / "perturbation.md"),
        ]
        for key, value in overrides.items():
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return cli(argv)

    def test_a_case_runs_without_a_dataset_and_records_its_own_verdict(self):
        """C16 is about labels, so it must not need a trial file to say something."""

        row = demorun.run_case_subprocess(self.args(), demorun.BY_ID["C16"])
        self.assertFalse(row["answers"]["crashed"], row["judgement"]["detail"])
        self.assertEqual(row["judgement"]["verdict"], "PASS", row["judgement"]["detail"])
        self.assertEqual(row["process"]["exit_code"], 0)
        self.assertTrue(row["answers"]["explicit"])
        written = Path(row["process"]["scratch"]) / "result.json"
        self.assertTrue(written.is_file(), row["process"]["scratch"])
        beacon = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(beacon["case"], "C16")

    def test_each_case_gets_its_own_process_and_its_own_directory(self):
        """Two runs of one case are two processes writing into two directories."""

        first = demorun.run_case_subprocess(self.args(), demorun.BY_ID["C16"])
        second = demorun.run_case_subprocess(
            self.args(stamp="teststamp2"), demorun.BY_ID["C16"]
        )
        self.assertNotEqual(first["process"]["pid"], second["process"]["pid"])
        self.assertNotEqual(first["process"]["scratch"], second["process"]["scratch"])
        for row in (first, second):
            scratch = Path(row["process"]["scratch"]).resolve()
            log = Path(row["process"]["log"]).resolve()
            self.assertEqual(log.parent, scratch)
            self.assertTrue(log.is_file())

    def test_a_case_whose_process_fails_is_recorded_instead_of_aborting_the_run(self):
        """A scratch root under a regular file makes the child die with no beacon."""

        blocked = Path(self.directory.name) / "not-a-directory"
        blocked.write_text("this file is in the way of the scratch root", encoding="utf-8")
        row = demorun.run_case_subprocess(
            self.args(scratch_root=str(blocked)), demorun.BY_ID["C16"]
        )
        self.assertTrue(row["answers"]["crashed"])
        self.assertEqual(row["judgement"]["verdict"], "FINDING")
        self.assertFalse(row["judgement"]["invariant_held"])
        self.assertIn("produced a verdict beacon", row["judgement"]["missing_observables"][0])
        self.assertIn("unavailable", row["process"]["log"])

    def test_orchestration_writes_one_json_and_one_markdown_report(self):
        """The whole parent path, over a single cheap case."""

        code = demorun.main(
            [
                "--only", "C16",
                "--trial", str(Path(self.directory.name) / "there-is-no-trial.npz"),
                "--model", str(Path(self.directory.name) / "there-is-no-model.npz"),
                "--scratch-root", str(self.scratch),
                "--stamp", "reported",
                "--out", str(Path(self.directory.name) / "perturbation.json"),
                "--md", str(Path(self.directory.name) / "perturbation.md"),
            ]
        )
        self.assertEqual(code, 0)
        report = json.loads(
            (Path(self.directory.name) / "perturbation_reported.json").read_text(encoding="utf-8")
        )
        markdown = (Path(self.directory.name) / "perturbation_reported.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(report["summary"]["cases"], 1)
        self.assertEqual(report["summary"]["exit_code"], 0)
        self.assertEqual(report["cases"][0]["case"], "C16")
        self.assertIn("| case | fault | expected observable | observed | verdict | evidence |", markdown)
        self.assertIn("C16", markdown)
        self.assertTrue(report["isolation"]["untouched"])

    def test_an_unknown_case_id_is_refused(self):
        """--only names cases; a typo must not silently run nothing."""

        with self.assertRaises(SystemExit):
            demorun.parse_only("C1,Z9")


class VerdictTest(unittest.TestCase):
    """A verdict has to say which of the two things went wrong."""

    def test_a_missing_observable_with_the_invariant_intact_is_a_finding(self):
        case = demorun.BY_ID["E4"]
        judgement = demorun.verdict_of(
            case,
            [demorun.named("the mechanism the plan names", False, "absent")],
            invariant_held=True,
            explicit=True,
            detail="the behaviour was safe, the mechanism is not there",
        )
        self.assertEqual(judgement["verdict"], "FINDING")
        self.assertTrue(judgement["invariant_held"])
        self.assertEqual(judgement["missing_observables"], ["the mechanism the plan names"])

    def test_a_violated_invariant_is_marked_as_such_in_the_report(self):
        case = demorun.BY_ID["E7"]
        judgement = demorun.verdict_of(
            case,
            [demorun.named("no confident decision on railed data", False, "A published")],
            invariant_held=False,
            explicit=False,
            detail="a confident answer was published",
        )
        report = {
            "command": ["demorun"],
            "started_at": "now",
            "inputs": {"trial": "t", "model": "m"},
            "settings": {
                "seconds": 24.0, "speed": 4.0, "margin": 0.05,
                "scratch_root": "s", "stamp": "x",
            },
            "isolation": {"untouched": True, "changed": []},
            "summary": {"cases": 1, "passed": 0, "findings": 1, "verdict": "FAIL", "exit_code": 1},
            "cases": [
                {
                    "case": case.id,
                    "fault": case.fault,
                    "expected": case.expected,
                    "judgement": judgement,
                    "answers": {"crashed": False, "explicit": False, "contaminated": False},
                    "process": {"seconds": 1.0},
                }
            ],
        }
        markdown = demorun.markdown_report(report)
        self.assertIn("INVARIANT VIOLATED", markdown)
        self.assertIn("did not appear: no confident decision on railed data", markdown)

    def test_the_three_answers_are_recorded_per_case(self):
        """Crash, explicitness and contamination are separate fields, not one word."""

        row = demorun.run_case_subprocess(
            cli(
                [
                    "--only", "C16",
                    "--trial", "missing.npz",
                    "--model", "missing.npz",
                    "--scratch-root", str(Path(tempfile.mkdtemp())),
                    "--stamp", "answers",
                ]
            ),
            demorun.BY_ID["C16"],
        )
        self.assertEqual(set(row["answers"]), {"crashed", "explicit", "contaminated"})


class RunRecordTest(unittest.TestCase):
    """The chain's own diagnostics reach the run record (plan section 3.16)."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model, self.trial, self.envelopes = build_fixture(
            self.directory.name, seconds=12
        )

    def run_session(self, source):
        """Run one real session over the fixture and return its summary."""

        references = ReferenceEnvelopes.load(self.envelopes, self.model.config)
        session = AttentionSession(
            source=source,
            decoder=self.model,
            references=references,
            policy=RunPolicy(check_channels=False, max_bad_channels=0, margin=0.05),
        )
        frames = []
        summary = session.run(frames.append, threading.Event())
        return session, summary, frames

    def test_a_duplicated_block_is_recorded_as_a_recovery_not_as_silence(self):
        source = demorun.PerturbedSource(
            self.trial, speed=50.0, fault=demorun.fault_repeat(3.0, 2)
        )
        session, summary, _ = self.run_session(source)
        kinds = [event["kind"] for event in summary.recovery_events]
        self.assertIn("irregular_timestamps", kinds, kinds)
        self.assertGreaterEqual(summary.recovery_segment, 1)
        self.assertIsNotNone(session.processor)
        self.assertEqual(
            summary.to_dict()["recovery_events"], summary.recovery_events
        )

    def test_a_repairable_nan_run_is_counted_in_the_run_record(self):
        damaged = inject_fault(self.trial, "dropout", start=4.0, duration=0.015)
        _, summary, frames = self.run_session(
            ReplaySource(damaged, speed=50.0, kind="test_replay")
        )
        self.assertGreater(summary.repaired_samples, 0)
        self.assertEqual(summary.to_dict()["repaired_samples"], summary.repaired_samples)
        reasons = {reason for frame in frames for reason in frame.reasons}
        self.assertIn("interpolated", reasons, reasons)


if __name__ == "__main__":
    unittest.main()
