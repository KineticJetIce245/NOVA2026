"""The live path over the real LSL transport, from another process.

Everything else in this suite fakes acquisition. This file does not: it starts
the repository's own fixture publisher (``scripts.getlive.publish_raw``) in a
separate process, then runs ``live.main`` against the outlet it publishes. What
that exercises, and nothing else in the suite does:

* outlet resolution on the network (``outlets.wait_for_outlet``);
* connecting to a real inlet and reading its declared metadata
  (``outlets.open_inlet``, ``outlets.channel_facts``);
* the package's pre-flight check against that metadata
  (``preflight.prepare``, ``validate_source``);
* the real ``Acquire``: MNE-LSL callbacks, fixed-size blocks, and its lag and gap
  counters;
* the whole chain on a real clock, and the acceptance report written to disk.

It is slower than the rest of the suite (about ten seconds) because LSL discovery
and the fixture both work in real time. If the publisher cannot start - no LSL on
this machine, or a blocked multicast - the file skips with the publisher's own
output as the reason, so the skip is visible rather than silent.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_live_transport -v
"""

import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.getlive import live

REPO = Path(__file__).resolve().parents[2]
OUTLET_NAME = "nova-transport-fixture"
SOURCE_ID = "nova-transport"
SFREQ = 500.0  # the rig's rate: at 250 Hz the LQ resampler needs 3.85 s to
# start, which the acceptance rules refuse - see the note in the docstring
CHANNELS = 2
DURATION = 6.0
WARMUP = 1.0  # the default 2 s warm-up plus the resampler's 1.88 s
# startup leaves 5 s of source with no window past it at all
PUBLISH_SECONDS = 60.0
START_TIMEOUT = 25.0


def start_publisher() -> subprocess.Popen:
    """Publish the fixture outlet in its own process, with metadata declared."""

    return subprocess.Popen(
        [
            sys.executable, "-B", "-m", "scripts.getlive.publish_raw",
            "--name", OUTLET_NAME,
            "--source-id", SOURCE_ID,
            "--channels", str(CHANNELS),
            "--sfreq", str(SFREQ),
            "--seconds", str(PUBLISH_SECONDS),
            "--metadata",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def wait_until_publishing(process: subprocess.Popen, timeout: float = START_TIMEOUT):
    """Return the publisher's banner once it is live, else its output so far."""

    output: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                return None, "".join(output)
            time.sleep(0.05)
            continue
        output.append(line)
        if "publishing" in line:
            return line.strip(), "".join(output)
    return None, "".join(output)


def run_live(argv: list[str]):
    """Run the acceptance script, returning its exit code and what it printed."""

    printed = io.StringIO()
    with redirect_stdout(printed):
        code = live.main(argv)
    return code, printed.getvalue()


class TransportIntegrationTests(unittest.TestCase):
    """``live.main`` against a real outlet, published by another process."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._publisher = start_publisher()
        cls._banner, cls._output = wait_until_publishing(cls._publisher)
        if cls._banner is None:
            cls._publisher.terminate()
            raise unittest.SkipTest(
                "the fixture publisher never came up, so the transport cannot be "
                f"tested here. Its output was:\n{cls._output}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._publisher.terminate()
        try:
            cls._publisher.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            cls._publisher.kill()
        if cls._publisher.stdout is not None:
            cls._publisher.stdout.close()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.report = self.root / "transport.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_default_path_runs_against_a_real_outlet(self) -> None:
        code, printed = run_live(
            [
                "--stream-name", OUTLET_NAME,
                "--source-id", SOURCE_ID,
                "--sfreq", str(SFREQ),
                "--source-units", "uV",
                "--cap", "declared",
                "--eog", "drop",
                "--duration", str(DURATION),
                "--warmup", str(WARMUP),
                "--min-valid-windows", "1",
                "--quiet",
                "--resolve-timeout", "15",
                "--out", str(self.report),
            ]
        )
        self.assertTrue(self.report.exists(), printed)
        report = json.loads(self.report.read_text())

        # The default timeline treatment is what shipped, so this test is about
        # the transport and not about the grid.
        self.assertEqual(report["arguments"]["timebase"], "stamps")
        self.assertEqual(report["source"]["name"], OUTLET_NAME)
        self.assertEqual(report["source"]["source_id"], SOURCE_ID)
        self.assertEqual(report["source"]["channels"], ["E1", "E2"])
        self.assertEqual(report["source"]["units"], ["microvolts"] * CHANNELS)

        # Real acquisition produced real blocks, windows and a verdict.
        stats = report["stats"]
        self.assertGreater(stats["blocks"], 0)
        self.assertGreater(stats["samples"], 0)
        self.assertGreater(stats["windows"], 0)
        self.assertGreater(stats["valid"], 0)
        self.assertEqual(stats["recoveries"], 0)
        self.assertLess(
            stats["max_lag"], 3.0, "the age guard is what stops a stalling source"
        )
        self.assertTrue(report["accepted"], printed)
        self.assertEqual(code, 0, printed)

    def test_an_unknown_outlet_is_reported_rather_than_guessed(self) -> None:
        # A named outlet that is not publishing has to fail loudly: the script
        # never falls back to whatever else is on the network.
        code, printed = run_live(
            [
                "--stream-name", "no-such-outlet-is-publishing",
                "--sfreq", str(SFREQ),
                "--source-units", "uV",
                "--duration", "1",
                "--quiet",
                "--resolve-timeout", "3",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("No outlet matches the requested identity", printed)
        self.assertIn(
            OUTLET_NAME,
            printed,
            "the refusal has to list what is on the network",
        )


if __name__ == "__main__":
    unittest.main()
