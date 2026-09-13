"""Contract tests for the media timeline the browser's gain path depends on.

Step 9's acceptance criterion is that the vendored frontend's ``mediaFocusReady``
*can* hold, and every clause of it is decided by something the transport
publishes. These tests pin the three pieces this step added on the Python side:

* :func:`nova2026.auditory.render.render_stereo` - the stereo file is candidate A
  on the left and candidate B on the right, in the requested presentation;
* :meth:`nova2026.transport.media.MediaTimeline.reference` - the echo guarantee.
  It may only return values the controller reported, and only while the
  acknowledgement is fresh (decision D-02);
* :class:`nova2026.auditory.producer.AttentionProducer` - the attention and gain
  packets of one frame carry the *same* reference, which ``playbackGains``
  compares to the bit;
* :class:`nova2026.transport.media.MediaBroadcaster` - the ``media`` packet the
  frontend reads ``syncStatus``/``revision``/``mediaId`` from.

Each test names the mutation it was shown to fail under (plan section 6.4).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.data import AttentionEstimate
from nova2026.auditory.producer import AttentionProducer
from nova2026.auditory.render import (
    PRESENTATION_MODES,
    mix_candidates,
    presentation_mode,
    render_stereo,
)
from nova2026.transport.media import MediaBroadcaster, MediaTimeline
from nova2026.transport.publisher import Publisher


class RenderTests(unittest.TestCase):
    """The stereo file: which candidate ends up in which ear, and how loud."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "media.wav"

    def render(self, candidates, **kwargs):
        return render_stereo(candidates, self.out, sample_rate=8000, **kwargs)

    def test_dichotic_puts_candidate_a_on_the_left_and_b_on_the_right(self):
        candidates = np.stack((np.full(16, 0.5), np.full(16, -0.25)), axis=1)
        self.render(candidates)
        rate, pcm = wavfile.read(self.out)
        self.assertEqual(rate, 8000)
        self.assertEqual(pcm.shape, (16, 2))
        self.assertTrue(np.all(pcm[:, 0] > 0))
        self.assertTrue(np.all(pcm[:, 1] < 0))
        # Mutation: swapping the two columns in `mix_candidates` fails here.
        self.assertAlmostEqual(float(pcm[0, 0]), 32767 * 0.5, delta=2)
        self.assertAlmostEqual(float(pcm[0, 1]), -32767 * 0.25, delta=2)

    def test_crossmix_keeps_the_leading_candidate_louder_in_each_ear(self):
        candidates = np.stack((np.ones(8), np.full(8, 0.5)), axis=1)
        mixed = mix_candidates(
            candidates, presentation="crossmix", crossmix_weight=0.25
        )
        self.assertTrue(np.all(mixed[:, 0] > mixed[:, 1]))
        # Mutation: a weight of 0.25 must not be applied as if it were 4.
        self.assertAlmostEqual(float(mixed[0, 0]), (1.0 + 0.25 * 0.5) / 1.25, places=6)
        self.assertAlmostEqual(float(mixed[0, 1]), (0.5 + 0.25 * 1.0) / 1.25, places=6)

    def test_the_two_modes_are_selectable_and_an_unknown_mode_is_refused(self):
        self.assertEqual(set(PRESENTATION_MODES), {"dichotic", "crossmix"})
        self.assertEqual(presentation_mode("crossmix"), "crossmix")
        # Mutation: making `presentation_mode` return the default instead of
        # raising lets a typo render an unnamed route.
        with self.assertRaises(ValueError):
            presentation_mode("binaural")

    def test_the_report_names_the_mode_and_hashes_each_channel(self):
        report = self.render(np.ones((4, 2)) * 0.1)
        self.assertEqual(report["presentation"], "dichotic")
        self.assertIsNone(report["crossmix_weight"])
        self.assertEqual(report["channels"], ["A", "B"])
        self.assertEqual(report["samples"], 4)
        self.assertEqual(set(report["sha256"]), {"left", "right"})
        self.assertTrue(self.out.is_file())

    def test_a_mismatched_or_nonfinite_candidate_array_is_refused(self):
        # Mutation: dropping the shape check renders a silence-or-crash instead
        # of refusing, and the failure would surface as a bad demo rather than here.
        with self.assertRaises(ValueError):
            mix_candidates(np.ones((4, 3)))
        with self.assertRaises(ValueError):
            mix_candidates(np.array([[np.nan, 0.0]]))
        with self.assertRaises(ValueError):
            mix_candidates(np.ones((0, 2)))

    def test_silence_stays_silent_instead_of_dividing_by_zero(self):
        report = self.render(np.zeros((4, 2)))
        self.assertEqual(report["peak_int16"], {"left": 0, "right": 0})
        _, pcm = wavfile.read(self.out)
        self.assertTrue(np.all(pcm == 0))


class ReferenceTests(unittest.TestCase):
    """The echo guarantee: the reference is what the controller said, or nothing."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.wav"
        self.path.write_bytes(b"RIFF" + bytes(64))
        self.now = 500.0
        self.timeline = MediaTimeline(self.path, "Conversation", clock=lambda: self.now)
        self.timeline.bind("session")
        self.request = 0

    def command(self, action, position=0.0):
        self.request += 1
        return self.timeline.control(
            {
                "session_id": "session",
                "media_id": self.timeline.media_id,
                "client_id": "browser",
                "request_id": self.request,
                "action": action,
                "media_time_s": position,
                "duration_s": 300.0,
            }
        )

    def test_no_reference_before_a_controller_prepares_the_timeline(self):
        # Mutation: returning a zeroed reference here would let a producer stamp
        # `media_time_s = 0` on every packet before a browser ever connected.
        self.assertIsNone(self.timeline.media_reference())
        self.command("prepare")
        self.assertIsNotNone(self.timeline.media_reference())

    def test_the_reference_is_the_reported_position_and_the_acknowledged_revision(self):
        self.command("prepare")
        self.now += 0.25
        snapshot = self.command("playing")
        self.now += 4.0
        self.command("report", 4.0)
        reference = self.timeline.media_reference()
        self.assertEqual(reference["media_id"], self.timeline.media_id)
        self.assertEqual(reference["media_revision"], snapshot["revision"])
        # Mutation: reading `self.position` before applying the command, or
        # rounding it, fails here.
        self.assertEqual(reference["media_time_s"], 4.0)

    def test_a_stale_report_withdraws_the_reference(self):
        self.command("prepare")
        self.assertIsNotNone(self.timeline.media_reference())
        self.now += 1.4
        self.assertIsNotNone(self.timeline.media_reference())
        self.now += 0.2  # 1.6 s without a report
        # Mutation: dropping the freshness test makes a dead tab keep looking
        # synchronized, which is the failure the gate's 0.75 s window exists for.
        self.assertIsNone(self.timeline.media_reference())
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")

    def test_a_refused_move_withdraws_the_reference_until_a_prepare(self):
        self.command("prepare")
        self.now += 0.25
        self.command("playing")
        self.now += 0.25
        with self.assertRaises(ValueError):
            # Inside the asset, but far beyond the elapsed time: this is the
            # refusal that invalidates the timeline.
            self.command("report", 250.0)
        self.assertIsNone(self.timeline.media_reference())
        self.now += 0.25
        self.command("stopped")
        self.now += 0.25
        self.command("prepare")
        self.assertIsNotNone(self.timeline.media_reference())

    def test_the_reference_carries_no_other_field(self):
        self.command("prepare")
        # Mutation: adding a server-computed position under another name would
        # widen the echo into an invention; the whole mapping is pinned.
        self.assertEqual(
            set(self.timeline.media_reference()),
            {"media_id", "media_revision", "media_time_s"},
        )


class _FakeSession:
    """The one session summary the broadcaster reads."""

    def __init__(self, session_id="session", status="running"):
        self.records = [{"id": session_id, "status": status}]

    def list(self):
        return list(self.records)


class BroadcasterTests(unittest.TestCase):
    """The `media` packet the dashboard needs before its gate can open."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.wav"
        self.path.write_bytes(b"RIFF" + bytes(64))
        self.now = 100.0
        self.timeline = MediaTimeline(self.path, "Conversation", clock=lambda: self.now)
        self.timeline.bind("session")
        self.publisher = Publisher()
        self.publisher.begin("session")
        self.sessions = _FakeSession()
        self.broadcaster = MediaBroadcaster(
            self.publisher, self.timeline, self.sessions, clock=lambda: self.now
        )

    def prepare(self):
        self.timeline.control(
            {
                "session_id": "session",
                "media_id": self.timeline.media_id,
                "client_id": "browser",
                "request_id": 1,
                "action": "prepare",
                "media_time_s": 0.0,
                "duration_s": 60.0,
            }
        )

    def test_no_packet_is_published_without_a_running_session(self):
        self.sessions.records = []
        self.assertIsNone(self.broadcaster.tick())
        self.sessions.records = [{"id": "session", "status": "stopped"}]
        # Mutation: publishing regardless of session status leaves a dead
        # timeline on screen as if it were live.
        self.assertIsNone(self.broadcaster.tick())
        self.assertEqual(self.publisher.snapshot()["sequence"], 0)

    def test_the_packet_is_the_timeline_snapshot_under_the_reserved_source(self):
        self.prepare()
        packet = self.broadcaster.tick()
        self.assertIsNotNone(packet)
        self.assertEqual(packet["type"], "media")
        self.assertEqual(packet["source"], "server")
        payload = packet["payload"]
        # These four fields are what `decoders.js` maps to `syncStatus`,
        # `playbackState`, `revision` and `mediaId`; the gate compares them with
        # the controller's own snapshot field by field.
        self.assertEqual(payload["sync_status"], "observed")
        self.assertEqual(payload["playback_state"], "paused")
        self.assertEqual(payload["revision"], self.timeline.revision)
        self.assertEqual(payload["media_id"], self.timeline.media_id)
        self.assertEqual(payload["media_time_s"], 0.0)

    def test_timestamps_do_not_regress_within_the_stream(self):
        self.prepare()
        first = self.broadcaster.tick()
        self.now += 0.25
        second = self.broadcaster.tick()
        self.assertEqual(first["timestamp"], 0.0)
        self.assertAlmostEqual(second["timestamp"], 0.25, places=6)
        # Mutation: publishing the raw clock value instead of elapsed seconds
        # makes the publisher refuse the second packet (timestamp regression).
        self.assertEqual(second["sequence"], first["sequence"] + 1)

    def test_a_publisher_refusal_is_counted_and_not_raised(self):
        self.prepare()
        self.publisher.end("session")
        self.assertIsNone(self.broadcaster.tick())
        # Mutation: letting the ValueError escape kills the broadcaster thread on
        # its first tick after a session closes.
        self.assertEqual(self.broadcaster.failures, 1)


class _StubSession:
    """The smallest session a producer can publish frames for."""

    kind = "stub"
    simulated = False
    window_seconds = 5.0

    def __init__(self):
        candidate = type("C", (), {"id": "A", "label": "a", "name": "a"})
        other = type("C", (), {"id": "B", "label": "b", "name": "b"})
        self.references = type("R", (), {"candidates": (candidate(), other())})()
        self.policy = type("P", (), {"to_dict": staticmethod(lambda: {})})()

    def run(self, emit, stop):
        raise AssertionError("this stub is never run")


class ProducerEchoTests(unittest.TestCase):
    """The attention and gain packets of one frame must carry one reference."""

    def setUp(self):
        self.calls = 0
        self.value = {
            "media_id": "m1",
            "media_revision": 7,
            "media_time_s": 12.5,
        }

    def reference(self):
        self.calls += 1
        return dict(self.value)

    def frame(self):
        return type(
            "F",
            (),
            {
                "timestamp": 1.0,
                "decision": "A",
                "correlation_a": 0.2,
                "correlation_b": 0.1,
                "gain_a_db": 0.0,
                "gain_b_db": -6.0,
                "quality": 1.0,
                "artifact": False,
                "reasons": (),
                "window_end": 1.0,
                "evidence_count": 1,
                "media_time_s": 0.0,
            },
        )()

    def test_one_frame_reads_the_timeline_once_and_stamps_both_packets(self):
        producer = AttentionProducer(_StubSession(), media_reference=self.reference)
        published = []
        producer._emit_frame(
            lambda kind, timestamp, source, payload: published.append((kind, payload)),
            self.frame(),
        )
        # Mutation: calling `self._media()` inside `_attention` and again inside
        # `_gain` is the bug this pins - a report landing between the two reads
        # makes `playbackGains` see two different times and fall back to neutral.
        self.assertEqual(self.calls, 1)
        attention = next(payload for kind, payload in published if kind == "attention")
        gain = next(payload for kind, payload in published if kind == "gain")
        self.assertEqual(attention["media_time_s"], gain["media_time_s"])
        self.assertEqual(attention["media_revision"], gain["media_revision"])
        self.assertEqual(attention["media_id"], gain["media_id"])

    def test_both_packets_carry_all_three_reference_fields(self):
        producer = AttentionProducer(_StubSession(), media_reference=self.reference)
        published = []
        producer._emit_frame(
            lambda kind, timestamp, source, payload: published.append((kind, payload)),
            self.frame(),
        )
        for kind, payload in published:
            if kind not in ("attention", "gain"):
                continue
            # Mutation: omitting `media_time_s` from one packet leaves the gate's
            # `|Δt| <= .75` clause with `undefined` and closes the gain path.
            self.assertEqual(
                {"media_id", "media_revision", "media_time_s"}.issubset(payload),
                True,
                f"{kind} payload keys were {sorted(payload)}",
            )

    def test_gain_never_exceeds_zero_db_even_if_a_frame_asks_for_amplification(self):
        producer = AttentionProducer(_StubSession(), media_reference=self.reference)
        frame = self.frame()
        frame.gain_a_db = 3.0
        frame.gain_b_db = 0.0
        published = []
        producer._emit_frame(
            lambda kind, timestamp, source, payload: published.append((kind, payload)),
            frame,
        )
        gain = next(payload for kind, payload in published if kind == "gain")
        # Mutation: publishing `frame.gain_a_db` unchanged lets a positive gain
        # through, and `playbackGains` then rejects the whole frame as neutral.
        self.assertEqual(gain["a_db"], 0.0)
        self.assertLessEqual(gain["b_db"], 0.0)

    def test_without_a_reference_no_media_field_is_published(self):
        producer = AttentionProducer(_StubSession())
        published = []
        producer._emit_frame(
            lambda kind, timestamp, source, payload: published.append((kind, payload)),
            self.frame(),
        )
        for kind, payload in published:
            if kind in ("attention", "gain"):
                # Mutation: defaulting to zeros would claim a playback position
                # this process never observed (decision D-02).
                self.assertNotIn("media_time_s", payload)


class TimelineRefusalTests(unittest.TestCase):
    """The transport refusals the controller's own validation depends on."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.wav"
        self.path.write_bytes(b"RIFF" + bytes(64))
        self.now = 200.0
        self.timeline = MediaTimeline(self.path, "Conversation", clock=lambda: self.now)
        self.timeline.bind("session")
        self.request = 0

    def command(self, action, position=0.0, client="browser"):
        self.request += 1
        return self.timeline.control(
            {
                "session_id": "session",
                "media_id": self.timeline.media_id,
                "client_id": client,
                "request_id": self.request,
                "action": action,
                "media_time_s": position,
                "duration_s": 600.0,
            }
        )

    def test_a_jump_beyond_the_elapsed_time_is_refused_and_is_sticky(self):
        self.command("prepare")
        self.now += 0.5
        self.command("playing")
        self.now += 0.5
        with self.assertRaises(ValueError):
            self.command("report", 5.0)
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")
        # A later command that the timeline *can* explain is still accepted - the
        # position never moved - but it must not repair the invalidity:
        # Mutation: setting `valid = True` on any accepted command lets a client
        # that seeks and then returns to the old position look synchronized again
        # without a fresh prepare, which is the failure D-02's sticky rule covers.
        self.command("report", 0.5)
        self.assertIsNone(self.timeline.media_reference())
        self.assertEqual(self.timeline.snapshot()["sync_status"], "desynchronized")

    def test_just_over_half_a_second_ahead_is_refused_while_playing(self):
        self.command("prepare")
        self.now += 0.25
        self.command("playing")
        self.now += 0.25
        # 0.25 s elapsed + 0.5 s allowance: 0.8 s is over. The recovery is the
        # documented one - stop, then prepare a new revision - because a refused
        # position leaves the timeline invalid on purpose.
        with self.assertRaises(ValueError):
            self.command("report", 0.8)
        self.command("stopped")
        self.now += 0.25
        self.command("prepare")
        self.now += 0.25
        self.command("playing")
        self.now += 0.25
        # 0.25 s elapsed + 0.5 s allowance: 0.7 s is under.
        self.assertEqual(self.command("report", 0.7)["media_time_s"], 0.7)

    def test_pausing_and_resuming_keeps_the_revision_and_the_identity(self):
        snapshot = self.command("prepare")
        revision = snapshot["revision"]
        self.now += 1.0
        self.command("playing")
        self.now += 1.0
        paused = self.command("paused", 1.0)
        self.assertEqual(paused["playback_state"], "paused")
        self.assertEqual(paused["revision"], revision)
        # Mutation: bumping the revision on pause makes the browser's playback
        # object stale, and `media.revision === playback.revision` goes false.
        self.now += 1.0
        resumed = self.command("playing", 1.0)
        self.assertEqual(resumed["revision"], revision)
        self.assertEqual(resumed["media_id"], self.timeline.media_id)

    def test_stopping_is_terminal_for_that_revision(self):
        prepared = self.command("prepare")
        self.now += 0.5
        stopped = self.command("stopped")
        self.assertEqual(stopped["sync_status"], "desynchronized")
        self.assertEqual(stopped["playback_state"], "stopped")
        # Mutation: keeping the revision on stop lets a client keep sending
        # reports it will never have acknowledged.
        self.assertEqual(stopped["revision"], prepared["revision"] + 1)
        with self.assertRaises(ValueError):
            self.command("report", 0.0)

    def test_another_client_cannot_take_over_a_prepared_timeline(self):
        self.command("prepare")
        self.now += 0.5
        # Mutation: dropping the client check lets two browsers drive one
        # timeline, and the acknowledged revision stops meaning one session.
        with self.assertRaises(ValueError):
            self.command("playing", 0.0, client="other")

    def test_the_descriptor_still_hides_the_path_with_a_reference_available(self):
        self.command("prepare")
        descriptor = self.timeline.descriptor()
        self.assertEqual(descriptor["url"], "/api/media/file")
        self.assertNotIn(str(self.path), json.dumps(self.timeline.snapshot()))


if __name__ == "__main__":
    unittest.main()
