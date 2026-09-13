"""The one-command entry point: one run, one URL, one record that says the margin.

The demo command is the plan's V1 deliverable, so what is tested here is exactly
what V1 asks for and what decision D-29 added:

* the command takes a margin from the policy when it is not told one, and the
  documented default (0.5) is what an untold session gets - never the calibrated
  0.05 by accident;
* the run record states the **effective** margin beside the quality policy, and
  the calibration's own record is what a reader compares it against;
* the built frontend is served from the same origin as ``/api`` and ``/ws``, not
  a second server;
* the end of a run is a clean exit, and the record exists on disk either way.

Nothing here needs a dataset, a model file or a browser: the fixture is generated
in a temporary directory, exactly as ``scripts/auditory_ui/session``'s tests do.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from nova2026.auditory.config import CALIBRATED_MARGIN, MIN_MARGIN
from nova2026.auditory.session import RunPolicy
from scripts.auditory_ui import demo

REPO = Path(__file__).resolve().parents[3]
REAL_FRONTEND = REPO / "apps" / "attune-ui" / "dist" / "index.html"


class MarginSelectionTests(unittest.TestCase):
    """``--margin`` means "the policy default" when it is absent, and only then."""

    def test_the_demos_default_is_the_calibrated_point(self):
        parsed = parse(["--synthetic", "--seconds", "12"])
        self.assertEqual(parsed.margin, CALIBRATED_MARGIN)
        self.assertNotEqual(parsed.margin, MIN_MARGIN)

    def test_an_explicit_margin_is_kept(self):
        self.assertEqual(
            parse(["--synthetic", "--seconds", "12", "--margin", "0.5"]).margin,
            MIN_MARGIN,
        )
        self.assertEqual(
            parse(["--synthetic", "--seconds", "12", "--margin", "0.2"]).margin, 0.2
        )

    def test_the_evidence_commands_own_margin_is_the_policy_default(self):
        runner = __import__("scripts.auditory_ui.session", fromlist=["session"])
        self.assertEqual(runner.policy_margin(None), MIN_MARGIN)
        self.assertEqual(runner.policy_margin(0.05), 0.05)
        self.assertEqual(runner.policy_margin(0.5), MIN_MARGIN)

    def test_the_run_policy_default_is_the_documented_one(self):
        policy = RunPolicy(check_channels=False, max_bad_channels=0)
        self.assertEqual(policy.margin, MIN_MARGIN)
        self.assertEqual(policy.to_dict()["margin"], MIN_MARGIN)


def parse(argv):
    """Parse demo arguments without running anything.

    ``main`` parses and then calls ``_execute``, which is where the event loop and
    the run live; replacing that for the duration of the call reads the parsed
    namespace without starting a server or a session.
    """

    captured = {}
    original = demo._execute

    def capture(args):
        captured["args"] = args
        return 0

    demo._execute = capture
    try:
        code = demo.main(argv)
    finally:
        demo._execute = original
    assert code == 0
    return captured["args"]


class RunRecordTests(unittest.TestCase):
    """One real run of the fixture: exit code, record, and both numbers in it."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="demo-test-"))
        cls.out = cls.directory / "demo_run.json"
        cls.media = cls.directory / "stereo.wav"
        cls.code = demo.main(
            [
                "--synthetic",
                "--seconds",
                "12",
                "--speed",
                "4",
                "--fixture-dir",
                str(cls.directory / "fixture"),
                "--media-out",
                str(cls.media),
                "--static-dir",
                str(REPO / "apps" / "attune-ui" / "dist"),
                "--serve-seconds",
                "3",
                "--out",
                str(cls.out),
            ]
        )
        cls.record = json.loads(cls.out.read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_the_command_exits_zero_and_writes_a_record(self):
        self.assertEqual(self.code, 0)
        self.assertTrue(self.out.is_file())

    def test_the_record_states_the_effective_margin_and_policy(self):
        point = self.record["operating_point"]
        self.assertEqual(point["margin"], CALIBRATED_MARGIN)
        self.assertEqual(point["documented_default_margin"], MIN_MARGIN)
        self.assertTrue(point["margin_overridden"])
        policy = self.record["session"]["policy"]
        self.assertEqual(policy["margin"], CALIBRATED_MARGIN)
        self.assertEqual(policy["check_channels"], False)
        self.assertEqual(policy["max_bad_channels"], 0)
        self.assertEqual(point["window_seconds"], 5.0)

    def test_the_record_matches_the_calibration_that_chose_the_point(self):
        """The margin in the record is the one the calibration report recommends."""

        report = sorted(
            (REPO / "results").glob("aad_margin_calibration_*.md"),
            key=lambda path: path.stat().st_mtime,
        )[-1].read_text(encoding="utf-8")
        self.assertIn("margin 0.05 at 5s", report)
        self.assertEqual(self.record["operating_point"]["margin"], 0.05)

    def test_the_session_ended_cleanly_with_the_relaxed_policy(self):
        summary = self.record["session"]
        self.assertIsNone(summary["failure"])
        self.assertEqual(self.record["session_record"]["status"], "stopped")
        self.assertGreater(summary["frames"], 0)
        self.assertEqual(
            set(summary["decisions"]),
            {"A", "B", "uncertain", "unavailable"} & set(summary["decisions"]),
        )

    def test_the_record_is_a_replay_and_says_so(self):
        self.assertIn("not a live participant", self.record["kind"])
        self.assertEqual(self.record["operating_point"]["presentation"], "dichotic")
        self.assertEqual(self.record["trial"]["subject"], "synthetic")


class StaticServingTests(unittest.TestCase):
    """The built frontend comes from the same origin, as section 3.6 requires."""

    def test_the_server_serves_the_built_frontend_from_the_same_port(self):
        self.assertTrue(REAL_FRONTEND.is_file(), "the built frontend must exist")
        directory = Path(tempfile.mkdtemp(prefix="demo-static-"))
        seen: dict = {}
        try:
            original = demo.drive

            async def probe(port, args, session, media_report, stop):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("GET", "/")
                response = connection.getresponse()
                seen["status"] = response.status
                seen["body"] = response.read().decode("utf-8", "replace")
                connection.request("GET", "/api/health")
                health = connection.getresponse()
                seen["health"] = health.status
                health.read()
                asset = seen["body"].split('src="', 1)[1].split('"', 1)[0]
                connection.request("GET", asset)
                bundle = connection.getresponse()
                seen["bundle_status"] = bundle.status
                seen["bundle"] = bundle.read().decode("utf-8", "replace")
                connection.close()
                return await original(port, args, session, media_report, stop)

            demo.drive = probe
            try:
                code = demo.main(
                    [
                        "--synthetic",
                        "--seconds",
                        "12",
                        "--speed",
                        "12",
                        "--fixture-dir",
                        str(directory / "fixture"),
                        "--media-out",
                        str(directory / "stereo.wav"),
                        "--static-dir",
                        str(REPO / "apps" / "attune-ui" / "dist"),
                        "--out",
                        str(directory / "demo_run.json"),
                    ]
                )
            finally:
                demo.drive = original
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self.assertEqual(code, 0)
        self.assertEqual(seen["status"], 200)
        self.assertEqual(seen["health"], 200)
        self.assertIn("<div id=\"root\">", seen["body"])
        # The page is the built frontend, not a placeholder: its own asset is
        # served from the same origin, and that asset is the real bundle (it
        # carries the banner the plan requires), not an error page.
        self.assertIn('src="/assets/', seen["body"])
        self.assertEqual(seen["bundle_status"], 200)
        self.assertIn("RESEARCH PROTOTYPE", seen["bundle"])


class RefusalTests(unittest.TestCase):
    """A demo that cannot be faithful refuses before it binds a port."""

    def test_a_missing_model_is_refused_with_a_message(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = demo.main(["--trial", "does/not/exist.npz", "--model", "also/missing.npz"])
        self.assertEqual(code, 2)

    def test_a_non_loopback_host_is_refused(self):
        directory = Path(tempfile.mkdtemp(prefix="demo-host-"))
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                code = demo.main(
                    [
                        "--synthetic",
                        "--seconds",
                        "12",
                        "--fixture-dir",
                        str(directory / "fixture"),
                        "--media-out",
                        str(directory / "stereo.wav"),
                        "--host",
                        "0.0.0.0",
                    ]
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

