"""Contract tests for the observed media timeline and its acknowledgement shape.

The browser owns the playback position (decision D-02). These tests pin what the
transport is allowed to record, when it must refuse, and when it must report
``desynchronized`` instead of claiming a synchronized measurement.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova2026.transport.media import MediaTimeline


class TimelineTestCase(unittest.TestCase):
    """A timeline over a throwaway file with a hand-driven clock."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.wav"
        self.path.write_bytes(b"RIFF" + bytes(64))
        self.now = 100.0
        self.timeline = MediaTimeline(self.path, "Conversation", clock=lambda: self.now)
        self.timeline.bind("session")
        self.request = 0

    def command(self, action, position=0.0, **extra):
        """Send one controller command with a fresh request id."""

        self.request += 1
        body = dict(
            session_id="session",
            media_id=self.timeline.media_id,
            client_id="browser",
            request_id=self.request,
            action=action,
            media_time_s=position,
            duration_s=90.0,
        )
        body.update(extra)
        return self.timeline.control(body)


class HandshakeTests(TimelineTestCase):
    """Prepare, acknowledge, and the descriptor that hides the filesystem."""

    def test_prepare_acknowledges_a_revision_and_observed_sync(self):
        before = self.timeline.snapshot()
        after = self.command("prepare")
        self.assertEqual(after["revision"], before["revision"] + 1)
        self.assertEqual(after["sync_status"], "observed")
        self.assertEqual(after["playback_state"], "paused")
        self.assertEqual(after["media_time_s"], 0.0)
        self.assertEqual(after["server_reference_s"], self.now)
        self.assertEqual(after["server_received_s"], self.now)
        self.assertEqual(after["duration_s"], 90.0)

    def test_prepare_needs_a_stopped_timeline_at_zero(self):
        self.command("prepare")
        with self.assertRaises(ValueError):
            self.command("prepare")

    def test_actions_before_prepare_are_refused_except_stopping(self):
        with self.assertRaises(ValueError):
            self.command("playing")
        with self.assertRaises(ValueError):
            self.command("report", 0.0)
        self.assertEqual(self.command("stopped")["playback_state"], "stopped")

    def test_descriptor_never_exposes_the_media_path(self):
        descriptor = self.timeline.descriptor()
        self.assertEqual(descriptor["url"], "/api/media/file")
        self.assertEqual(descriptor["kind"], "audio")
        self.assertNotIn(str(self.path), json.dumps(descriptor))

    def test_unusable_media_files_are_refused(self):
        with self.assertRaises(ValueError):
            MediaTimeline(self.path.parent / "missing.wav")
        unsupported = Path(self.temp.name) / "sample.txt"
        unsupported.write_bytes(b"x")
        with self.assertRaises(ValueError) as caught:
            MediaTimeline(unsupported)
        self.assertIn("suffix", str(caught.exception))


class ProgressionTests(TimelineTestCase):
    """A report may not move faster than time did, and a violation is sticky."""

    def test_reports_may_advance_by_the_elapsed_time_plus_half_a_second(self):
        self.command("prepare")
        self.command("playing")
        self.now += 1.0
        self.assertEqual(self.command("report", 1.0)["media_time_s"], 1.0)
        self.now += 0.5
        self.assertEqual(self.command("report", 1.5)["media_time_s"], 1.5)

    def test_a_jump_faster_than_half_a_second_per_report_is_refused(self):
        self.command("prepare")
        self.command("playing")
        self.now += 0.1
        with self.assertRaises(ValueError) as caught:
            self.command("report", 8.0)
        self.assertIn("elapsed", str(caught.exception))
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")

    def test_a_refused_seek_cannot_be_repaired_by_a_later_good_report(self):
        self.command("prepare")
        self.command("playing")
        self.now += 0.1
        with self.assertRaises(ValueError):
            self.command("report", 40.0)
        # Six seconds later a report of 6.0 s is plausible in isolation (elapsed
        # 6.1 s plus the 0.5 s allowance), yet the timeline stays desynchronized
        # until a fresh prepare starts a new revision.
        self.now += 6.0
        self.assertEqual(self.command("report", 6.0)["media_time_s"], 6.0)
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")
        self.assertEqual(self.command("stopped")["sync_status"], "desynchronized")
        self.assertEqual(self.command("prepare")["sync_status"], "observed")

    def test_paused_reports_have_a_tighter_allowance_and_no_backward_moves(self):
        self.command("prepare")
        self.now += 30
        self.assertEqual(self.command("report", 0.05)["media_time_s"], 0.05)
        with self.assertRaises(ValueError):
            self.command("report", 5.0)  # paused: only 0.1 s of advance is allowed
        with self.assertRaises(ValueError):
            self.command("report", 0.0)  # 0.05 s backwards is beyond the 20 ms tolerance

    def test_playing_can_resume_from_a_paused_position(self):
        self.command("prepare")
        self.command("playing")
        self.now += 8.0
        self.command("paused", 8.0)
        self.now += 30.0
        self.assertEqual(self.command("report", 8.0)["media_time_s"], 8.0)
        self.assertEqual(self.command("playing", 8.0)["playback_state"], "playing")
        self.now += 0.25
        self.assertEqual(self.command("report", 8.25)["media_time_s"], 8.25)

    def test_a_stale_timeline_is_desynchronized_and_a_report_makes_it_fresh(self):
        self.command("prepare")
        self.command("playing")
        self.now += 2.0
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")
        self.now += 0.1
        self.assertEqual(self.command("report", 0.0)["sync_status"], "observed")


class RefusalTests(TimelineTestCase):
    """Identity, controller ownership and value checks."""

    def test_a_mismatched_media_or_session_is_refused_and_changes_nothing(self):
        self.command("prepare")
        original = self.timeline.snapshot()
        base = dict(
            session_id="session",
            media_id=self.timeline.media_id,
            client_id="browser",
            request_id=self.request + 1,
            action="report",
            media_time_s=0.0,
            duration_s=90.0,
        )
        cases = [
            dict(session_id="other-session"),
            dict(media_id="other-media"),
            dict(client_id="other-browser"),
            dict(request_id=self.request),
            dict(request_id=True),
            dict(client_id=""),
            dict(action="seek"),
            dict(action=None),
            dict(media_time_s=float("nan")),
            dict(media_time_s=float("inf")),
            dict(media_time_s=-1),
            dict(media_time_s=91.0),
            dict(duration_s=0),
            dict(duration_s=float("nan")),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.timeline.control({**base, **case})
        self.assertEqual(self.timeline.snapshot(), original)

    def test_stopping_releases_the_controller_and_resets_the_position(self):
        self.command("prepare")
        self.command("playing")
        self.now += 2.0
        self.command("report", 2.0)
        stopped = self.command("stopped")
        self.assertEqual(stopped["playback_state"], "stopped")
        self.assertEqual(stopped["media_time_s"], 0.0)
        self.assertEqual(stopped["sync_status"], "desynchronized")
        # A new controller must prepare before it can do anything else.
        self.request += 1
        with self.assertRaises(ValueError):
            self.timeline.control(
                dict(
                    session_id="session",
                    media_id=self.timeline.media_id,
                    client_id="second-browser",
                    request_id=self.request,
                    action="report",
                    media_time_s=0.0,
                    duration_s=90.0,
                )
            )
        after = self.timeline.control(
            dict(
                session_id="session",
                media_id=self.timeline.media_id,
                client_id="second-browser",
                request_id=self.request + 1,
                action="prepare",
                media_time_s=0.0,
                duration_s=90.0,
            )
        )
        self.assertEqual(after["sync_status"], "observed")

    def test_rebinding_to_a_new_session_stops_the_previous_controller(self):
        self.command("prepare")
        before = self.timeline.snapshot()
        self.timeline.bind("next-session")
        after = self.timeline.snapshot()
        self.assertEqual(after["session_id"], "next-session")
        self.assertEqual(after["playback_state"], "stopped")
        self.assertEqual(after["media_time_s"], 0.0)
        self.assertGreater(after["revision"], before["revision"])
        self.timeline.bind("next-session")  # idempotent while the session matches
        self.assertEqual(self.timeline.snapshot()["revision"], after["revision"])


if __name__ == "__main__":
    unittest.main()
