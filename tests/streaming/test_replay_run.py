"""Tests for replaying a recorded run through the live chain.

The recordings are made with the hardware-free harness in
``test_live_integration``, so both files drive one chain: that file pins what a
live run does, and this one pins what the replay tool does with the recording it
left behind.

Run from the repository root:

    .venv/bin/python -B -m unittest tests.streaming.test_replay_run -v
"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nova2026.streaming.recording import read_metadata
from scripts.getlive import replay_run
from tests.streaming.test_live_integration import (
    build_run,
    drive,
    rig_like_signal,
    rig_like_timeline,
)


def record(root: Path, *, mode: str, session: str) -> tuple[Path, object]:
    """Write one recording with the live harness and return its path."""

    run = build_run(root, mode=mode, session=session)
    drive(run, rig_like_signal(), rig_like_timeline())
    return Path(run.session.recorder.path), run


class ReplayToolTests(unittest.TestCase):
    """A recording is the only amplifier this repository still has."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def replay(self, path: Path, *flags: str) -> tuple[int, str]:
        """Run the tool as ``__main__`` does, and keep what it printed."""

        out = io.StringIO()
        with redirect_stdout(out):
            code = replay_run.main([str(path), *flags])
        return code, out.getvalue()

    def test_the_grid_replay_keeps_a_usable_run_usable(self) -> None:
        path, _ = record(self.root, mode="grid", session="grid")

        code, text = self.replay(path)

        self.assertEqual(code, 0, text)
        self.assertIn("verdict: USABLE", text)
        self.assertIn("the chain never had to restart", text)
        self.assertIn("no damaged source row", text)
        self.assertNotIn("no valid window out of 0", text)

    def test_the_stamps_replay_refuses_what_the_grid_absorbed(self) -> None:
        # The recorded anchors keep the source's steps at the block seams, so a
        # replay in stamps mode has to fail where grid mode did not. If this ever
        # passes, the replay stopped exercising placement.
        path, _ = record(self.root, mode="grid", session="grid_again")

        code, text = self.replay(path, "--timebase", "stamps")

        self.assertEqual(code, 1, text)
        self.assertIn("verdict: NOT USABLE", text)
        self.assertNotIn("the chain never had to restart", text)

    def test_a_recording_of_a_failed_run_replays_with_a_clean_timeline(self) -> None:
        # This recording is only as long as the run survived, so no window can
        # form from it. What the replay still proves is the point of the tool:
        # the chain never restarted and Repair repaired nothing.
        path, recorded_run = record(self.root, mode="stamps", session="stamps")
        self.assertIsNotNone(recorded_run.failure, "the fixture has to have failed")

        code, text = self.replay(path, "--timebase", "grid")

        self.assertEqual(code, 1, "too few samples for a valid window, and it says so")
        self.assertIn("status=failed", text, "the replay reports what it replayed")
        self.assertIn("recovery: 0 restart(s), 0 repaired row(s)", text)

    def test_the_replay_says_which_rules_it_cannot_measure(self) -> None:
        path, _ = record(self.root, mode="grid", session="grid_note")

        _, text = self.replay(path)

        for rule in ("input_lag", "timing_gaps"):
            # The acceptance row, not the header note that says the same thing:
            # matching the note would let this pass with the rules untouched.
            line = next(
                (
                    row
                    for row in text.splitlines()
                    if row.split()[:1] in (["PASS"], ["WARN"], ["FAIL"])
                    and rule in row.split()
                ),
                "",
            )
            self.assertTrue(line, f"{rule} is missing from the acceptance table")
            self.assertIn("not measured", line)

    def test_a_replay_is_recorded_as_a_replay(self) -> None:
        path, _ = record(self.root, mode="grid", session="grid_rec")
        again = self.root / "again"

        code, text = self.replay(
            path,
            "--record",
            str(again),
            "--subject",
            "replay",
            "--session",
            "again",
            "--run",
            "fixed",
        )

        self.assertEqual(code, 0, text)
        written = [p for p in again.rglob("*.sqlite") if not p.name.startswith("._")]
        self.assertEqual(len(written), 1, "the replay has to be recorded once")
        metadata = read_metadata(written[0])
        self.assertEqual(metadata["status"], "completed")
        self.assertGreater(metadata["samples"], 0)
        self.assertEqual(metadata["role"], "replay")
        self.assertIn("limits", metadata["config"], "the limits belong in the record")
        self.assertEqual(metadata["config"]["timebase"]["mode"], "grid")

    def test_the_recorded_policy_is_used_unless_the_flags_say_otherwise(self) -> None:
        path, _ = record(self.root, mode="grid", session="policy")

        _, recorded = self.replay(path)
        _, overridden = self.replay(path, "--no-channel-check")

        self.assertIn("channel check on", recorded, "as the recording was made")
        self.assertIn("channel check off", overridden)
        self.assertIn("excluded=none", recorded, "the recording excluded nothing")

    def test_a_path_without_a_recording_is_refused(self) -> None:
        with self.assertRaises(SystemExit), redirect_stdout(
            io.StringIO()
        ), redirect_stderr(io.StringIO()):
            replay_run.main([str(self.root)])

    def test_the_limits_default_to_the_live_scripts_own(self) -> None:
        args = replay_run.build_parser().parse_args(["some/run"])

        self.assertEqual(args.timebase, "grid")
        self.assertIsNone(args.exclude_channels, "unset means: as recorded")
        self.assertIsNone(args.no_channel_check, "unset means: as recorded")
        for dest in (
            "repair_amplitude_uv",
            "repair_saturation_uv",
            "max_recoveries",
            "max_fault_seconds",
        ):
            with self.subTest(dest=dest):
                self.assertIsNone(
                    getattr(args, dest), "unset means: the live script's default"
                )
                self.assertIsNotNone(replay_run._live_default(dest))

    def test_the_recording_path_survives_a_run_name_flag(self) -> None:
        # ``--run`` is the live script's name for a recording's run, and the
        # positional must not share its destination: that collision silently
        # replaced the path with a string once.
        args = replay_run.build_parser().parse_args(["some/run", "--run", "named"])

        self.assertEqual(args.recording, Path("some/run"))
        self.assertEqual(args.run, "named")


if __name__ == "__main__":
    unittest.main()
