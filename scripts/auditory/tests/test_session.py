"""Unit tests for one attention session, its sources and its producer adapter.

Every fixture is built inside the test's own temporary directory (see
``session_fixture``), so none of these tests needs a dataset, a model file or a
network. The degradation paths are the point: an invalid window, a dropped item,
stale evidence and an uncovered audio range each have to produce a named reason
rather than a confident decision, and the label-isolation test proves the
decoder's input cannot depend on ground truth.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np

from nova2026.auditory.config import CALIBRATED_MARGIN, MIN_MARGIN, AuditoryConfig
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.envelopes import EnvelopeVerificationError
from nova2026.auditory.producer import AttentionProducer
from nova2026.auditory.session import (
    DECISIONS,
    AttentionSession,
    RunPolicy,
    attenuation_db,
)
from nova2026.auditory.sources import (
    AUDIO_UNAVAILABLE,
    ReferenceEnvelopes,
    ReplaySource,
    trial_envelope_paths,
)
from nova2026.transport.protocol import validate_packet
from scripts.auditory.tests.session_fixture import build_fixture, relabelled
from scripts.auditory_ui.session import inspect

SECONDS = 20.0
"""Fixture length. Long enough for ten windows at the model's one-second step."""

SPEED = 8.0
"""Replay speed for unit tests: fast, but still slow enough that nothing drops."""


class SlowModel:
    """A decoder whose scoring is slow enough to overflow a two-item queue."""

    def __init__(self, model, delay=0.15):
        self._model = model
        self.delay = float(delay)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def score(self, window):
        import time

        time.sleep(self.delay)
        return self._model.score(window)


class BrokenModel(SlowModel):
    """A decoder whose scoring always fails, standing in for a numerical fault."""

    def score(self, window):
        raise ValueError("scoring failed on purpose")


class FixtureCase(unittest.TestCase):
    """Build one shared fixture per test class; nothing here touches the repo data."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls._temporary.name)
        cls.model, cls.trial, cls.paths = build_fixture(
            cls.directory, seconds=SECONDS
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def session(self, *, trial=None, model=None, speed=SPEED, policy=None, **kwargs):
        """A ready session over the fixture, with the documented relaxed policy."""

        references = ReferenceEnvelopes.load(self.paths, (model or self.model).config)
        source = ReplaySource(
            trial if trial is not None else self.trial, speed=speed, simulated=True
        )
        return AttentionSession(
            source=source,
            decoder=model or self.model,
            references=references,
            policy=policy or RunPolicy(check_channels=False, max_bad_channels=0),
            **kwargs,
        )

    def run_session(self, session):
        """Run to completion, collecting every frame the session publishes."""

        frames = []
        summary = session.run(frames.append, Event())
        return frames, summary


class PolicyTests(unittest.TestCase):
    """The run policy is explicit, recorded, and refuses to be implicit."""

    def test_channel_policy_has_no_default(self):
        with self.assertRaises(TypeError):
            RunPolicy()
        policy = RunPolicy(check_channels=False, max_bad_channels=2)
        self.assertFalse(policy.check_channels)
        self.assertEqual(policy.to_dict()["max_bad_channels"], 2)

    def test_kuleuven_policy_is_the_documented_relaxed_choice(self):
        policy = RunPolicy.kuleuven_replay()
        self.assertFalse(policy.check_channels)
        self.assertEqual(policy.max_bad_channels, 0)
        self.assertEqual(policy.to_dict()["check_channels"], False)

    def test_policy_rejects_values_that_would_change_behaviour_silently(self):
        with self.assertRaises(ValueError):
            RunPolicy(check_channels=False, max_bad_channels=-1)
        with self.assertRaises(ValueError):
            RunPolicy(check_channels=False, max_bad_channels=0, frame_seconds=0.0)
        with self.assertRaises(TypeError):
            RunPolicy(check_channels=False, max_bad_channels=0, exclude_channels="Cz")
        with self.assertRaises(TypeError):
            RunPolicy(check_channels="no", max_bad_channels=0)
        for bad in (0.0, -0.05, float("nan"), float("inf"), True):
            with self.assertRaises((ValueError, TypeError)):
                RunPolicy(check_channels=False, max_bad_channels=0, margin=bad)


class MarginPolicyTests(FixtureCase):
    """The margin is a policy field, and leaving it out must change nothing.

    Decision D-29 wires the calibrated 0.05 operating point into the demo. The
    risk that wiring creates is the silent one: a session that was never asked
    for a calibrated margin quietly getting one. These tests pin the default at
    :data:`MIN_MARGIN` and prove the session reads the *policy*, not a constant.
    """

    def test_to_dict_records_the_effective_margin(self):
        policy = RunPolicy(check_channels=False, max_bad_channels=0)
        self.assertEqual(policy.margin, MIN_MARGIN)
        self.assertEqual(policy.to_dict()["margin"], MIN_MARGIN)
        self.assertEqual(
            RunPolicy(
                check_channels=False, max_bad_channels=0, margin=CALIBRATED_MARGIN
            ).to_dict()["margin"],
            CALIBRATED_MARGIN,
        )

    def test_a_session_without_a_margin_keeps_the_documented_default(self):
        session = self.session()
        self.assertEqual(session.controller.margin, MIN_MARGIN)
        self.assertEqual(session.policy.to_dict()["margin"], MIN_MARGIN)
        # The default controller path is what D-29 wired, so assert the number
        # came from the policy: build one with a calibrated policy and read it.
        self.assertEqual(
            self.session(
                policy=RunPolicy(
                    check_channels=False, max_bad_channels=0, margin=CALIBRATED_MARGIN
                )
            ).policy.to_dict()["margin"],
            CALIBRATED_MARGIN,
        )
        frames, summary = self.run_session(session)
        self.assertTrue(frames)
        self.assertEqual(summary.policy["margin"], MIN_MARGIN)

    def test_the_policy_margin_reaches_the_controller(self):
        session = self.session(
            policy=RunPolicy(
                check_channels=False, max_bad_channels=0, margin=CALIBRATED_MARGIN
            )
        )
        self.assertEqual(session.controller.margin, CALIBRATED_MARGIN)
        frames, summary = self.run_session(session)
        self.assertTrue(frames)
        self.assertEqual(summary.policy["margin"], CALIBRATED_MARGIN)

    def test_an_explicit_controller_still_wins(self):
        controller = AttentionController(margin=0.25)
        session = self.session(
            policy=RunPolicy(
                check_channels=False, max_bad_channels=0, margin=CALIBRATED_MARGIN
            ),
            controller=controller,
        )
        self.assertIs(session.controller, controller)
        self.assertEqual(session.controller.margin, 0.25)

    def test_a_lower_margin_decides_more_frames(self):
        """The calibration's own claim, on controlled evidence.

        The same trial, the same windows, three margins and one fixed score
        difference: what the margin buys is the number of committed frames. This
        is a plumbing check, not an accuracy claim - a real trial's scores are
        not a constant, which is exactly why the operating point is calibrated on
        held-out data and not here.
        """

        class FixedGapModel:
            """A decoder whose two correlations always differ by the same amount."""

            def __init__(self, model, gap):
                self._model = model
                self.gap = float(gap)

            def __getattr__(self, name):
                return getattr(self._model, name)

            def score(self, window):
                return np.array([0.6, 0.6 - self.gap])

        # The fixed gap is 0.05, between the calibrated margin and the documented
        # default: it must commit at 0.05 and must not at 0.5. Half a margin is
        # the smallest gap that is unambiguously one side or the other.
        counts = {}
        for margin in (0.05, 0.5):
            frames, _ = self.run_session(
                self.session(
                    model=FixedGapModel(self.model, 0.05),
                    policy=RunPolicy(
                        check_channels=False, max_bad_channels=0, margin=margin
                    ),
                )
            )
            counts[margin] = sum(1 for frame in frames if frame.decision in ("A", "B"))
        self.assertGreater(counts[0.05], 0, f"0.05 must commit on a 0.05 gap; {counts}")
        self.assertEqual(counts[0.5], 0, f"0.5 must not commit on a 0.05 gap; {counts}")


class BadChannelEvidenceTests(FixtureCase):
    """A dead electrode must be visible even when the policy will not reject it.

    Plan section 3.17 item 6: with ``check_channels=False`` the monitor's
    ``reasons()`` is empty *by construction*, so a dry cap would read
    ``quality=1.0, artifact=false``. The census is carried instead, and the fix
    must not be a new rejection - the relaxed policy exists precisely so a faulty
    channel cannot stop the run.
    """

    @staticmethod
    def dead_channel_trial(trial, name: str):
        """The fixture with one electrode held at a constant value."""

        index = trial.channel_names.index(name)
        trial.eeg = trial.eeg.copy()
        trial.eeg[:, index] = 0.5
        return trial

    def test_a_dead_channel_is_named_and_marks_the_frame(self):
        trial = self.dead_channel_trial(self.trial, "F3")
        frames, summary = self.run_session(self.session(trial=trial))
        flagged = [frame for frame in frames if frame.bad_channels]
        self.assertTrue(flagged, "a constant electrode must appear in the census")
        for frame in flagged:
            self.assertEqual(frame.bad_channels, ("F3",))
            self.assertTrue(frame.artifact)
            self.assertEqual(frame.quality, 0.0)
        # The census counts *windows*; many frames share one window's verdict, so
        # the two counts are only equal once the frames are deduplicated by the
        # evidence they were published with.
        window_verdicts = {frame.evidence_count: frame for frame in frames}
        self.assertEqual(
            summary.bad_channel_census.get("F3"),
            sum(1 for frame in window_verdicts.values() if frame.bad_channels),
        )

    def test_the_relaxed_policy_still_does_not_reject_the_window(self):
        """The fix must not re-introduce the halt the relaxed policy avoids."""

        trial = self.dead_channel_trial(self.trial, "F3")
        frames, summary = self.run_session(self.session(trial=trial))
        self.assertIsNone(summary.failure)
        self.assertTrue(frames)
        self.assertLess(
            summary.invalid,
            summary.windows,
            f"rejections must be the exception, not the rule: {summary.invalid} "
            f"of {summary.windows}",
        )
        self.assertGreater(summary.scored, 0)
        for frame in frames:
            self.assertNotIn("flatline", frame.reasons)
        # And the census is reported, not merely counted. The flatline detector
        # cannot report before half a second of unchanged signal has been seen,
        # so the first flagged window must end at or after 0.5 s: this is the
        # boundary case, not "some channel was flagged somewhere".
        self.assertGreater(summary.bad_channel_census.get("F3", 0), 0)
        windows = sorted(
            {
                frame.window_end
                for frame in frames
                if frame.bad_channels and frame.window_end is not None
            }
        )
        self.assertTrue(windows)
        self.assertGreaterEqual(windows[0], 0.5)
        self.assertIn(
            "F3", {name for frame in frames for name in frame.bad_channels}
        )

    def test_a_clean_run_flags_nothing(self):
        frames, summary = self.run_session(self.session())
        self.assertEqual(summary.bad_channel_census, {})
        self.assertTrue(frames)
        judged = [frame for frame in frames if frame.quality is not None]
        self.assertTrue(judged, "at least one window must have been judged")
        for frame in judged:
            self.assertEqual(frame.bad_channels, ())
            self.assertFalse(frame.artifact)
            self.assertEqual(frame.quality, 1.0)


class EnvelopeTests(FixtureCase):
    """A session refuses to start without the exact envelope it needs."""

    def test_missing_envelope_names_the_file_it_cannot_find(self):
        missing = self.directory / "absent_candidate.npz"
        with self.assertRaises(EnvelopeVerificationError) as caught:
            ReferenceEnvelopes.load((self.paths[0], missing), self.model.config)
        self.assertIn("absent_candidate.npz", str(caught.exception))

    def test_envelope_that_no_longer_matches_its_audio_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, _, paths = build_fixture(directory, seconds=8.0)
            # Same file name, different audio: only the recorded hash can catch it.
            from scipy.io import wavfile

            wavfile.write(
                directory / "candidate_a.wav", 8000, np.zeros(8000, dtype=np.int16)
            )
            with self.assertRaises(EnvelopeVerificationError) as caught:
                ReferenceEnvelopes.load(paths, self.model.config)
            self.assertIn("SHA256", str(caught.exception))

    def test_envelope_for_a_different_feature_config_is_refused(self):
        with self.assertRaises(EnvelopeVerificationError) as caught:
            ReferenceEnvelopes.load(self.paths, AuditoryConfig(band=(2.0, 9.0)))
        self.assertIn("band", str(caught.exception))

    def test_trial_group_names_the_two_envelopes_in_candidate_order(self):
        paths = trial_envelope_paths(self.trial, self.directory)
        self.assertEqual([path.name for path in paths], ["candidate_a.npz", "candidate_b.npz"])
        self.trial.group = "candidate_a.wav"
        try:
            with self.assertRaises(EnvelopeVerificationError):
                trial_envelope_paths(self.trial, self.directory)
        finally:
            self.trial.group = "candidate_a.wav|candidate_b.wav"


class ReplaySourceTests(FixtureCase):
    """The source owns the clock and never invents a sample time."""

    def test_chunks_carry_the_trials_own_timestamps(self):
        source = ReplaySource(self.trial, speed=SPEED)
        times = []
        for _, samples, stamps in source.chunks(Event()):
            self.assertEqual(samples.shape[1], len(self.trial.channel_names))
            times.append(np.asarray(stamps))
        joined = np.concatenate(times)
        np.testing.assert_array_equal(joined, self.trial.timestamps)
        np.testing.assert_array_equal(source.start, self.trial.timestamps[0])

    def test_pacing_divides_wall_time_by_the_speed(self):
        sleeps = []
        ticks = iter([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        source = ReplaySource(
            self.trial,
            chunk_seconds=1.0,
            speed=2.0,
            sleep=sleeps.append,
            timer=lambda: next(ticks, 10.0),
        )
        list(source.chunks(Event()))
        self.assertTrue(sleeps)
        self.assertAlmostEqual(sleeps[0], 0.3, places=6)

    def test_stop_event_ends_the_source_promptly(self):
        stop = Event()
        stop.set()
        self.assertEqual(list(ReplaySource(self.trial, speed=SPEED).chunks(stop)), [])


class SessionTests(FixtureCase):
    """The decisions, the degradation reasons and the label isolation."""

    def test_warmup_reports_unavailable_not_uncertain(self):
        frames, _ = self.run_session(self.session())
        warm = [frame for frame in frames if frame.timestamp < 2.0]
        self.assertTrue(warm)
        for frame in warm:
            self.assertEqual(frame.decision, "unavailable")
            self.assertEqual(frame.reasons, ("warmup",))
            self.assertEqual((frame.gain_a_db, frame.gain_b_db), (0.0, 0.0))

    def test_a_clean_run_decides_and_attenuates_one_candidate(self):
        frames, summary = self.run_session(self.session())
        decided = [frame for frame in frames if frame.decision in ("A", "B")]
        self.assertTrue(decided, "the fixture's chain should produce decisions")
        self.assertGreater(summary.scored, 0)
        self.assertEqual(summary.evidence_gaps, 0)
        for frame in decided:
            self.assertLessEqual(frame.gain_a_db, 0.0)
            self.assertLessEqual(frame.gain_b_db, 0.0)
            ducked = frame.gain_a_db if frame.decision == "B" else frame.gain_b_db
            self.assertAlmostEqual(ducked, -6.0, places=6)
            self.assertIsNotNone(frame.correlation_a)
            self.assertIsNotNone(frame.correlation_b)

    def test_every_decision_word_is_one_of_the_four(self):
        frames, _ = self.run_session(self.session())
        self.assertTrue({frame.decision for frame in frames} <= set(DECISIONS))

    def test_dropped_window_becomes_an_explicit_evidence_gap(self):
        # max_age is generous on purpose: at replay speed the source clock runs
        # far ahead of the worker, and this test is about the gap path rather
        # than about staleness, which has its own test.
        session = self.session(
            model=SlowModel(self.model),
            speed=400.0,
            controller=AttentionController(max_age=60.0),
        )
        frames, summary = self.run_session(session)
        self.assertGreater(summary.evidence_gaps, 0, "the queue should have overflowed")
        gapped = [frame for frame in frames if "evidence_gap" in frame.reasons]
        self.assertTrue(gapped, "a dropped window must be visible in a frame")
        for frame in gapped:
            self.assertIn(frame.decision, ("uncertain", "unavailable"))

    def test_stale_evidence_becomes_unavailable(self):
        controller = AttentionController(max_age=0.05)
        frames, _ = self.run_session(self.session(controller=controller))
        stale = [frame for frame in frames if "evidence_stale" in frame.reasons]
        self.assertTrue(stale, "evidence older than max_age must be reported")
        for frame in stale:
            self.assertEqual(frame.decision, "unavailable")
            self.assertEqual((frame.gain_a_db, frame.gain_b_db), (0.0, 0.0))

    def test_window_outside_the_recorded_envelope_is_never_scored(self):
        session = self.session()
        session.source = ReplaySource(
            self.trial, speed=SPEED, simulated=True, audio_offset=SECONDS + 5.0
        )
        frames, summary = self.run_session(session)
        self.assertEqual(summary.scored, 0)
        self.assertEqual(summary.invalid, summary.windows)
        self.assertTrue(frames)
        for frame in frames:
            self.assertNotIn(frame.decision, ("A", "B"))
        self.assertTrue(
            any(AUDIO_UNAVAILABLE in frame.reasons for frame in frames),
            "an uncovered window must name the alignment failure",
        )

    def test_scoring_failure_is_counted_and_never_a_decision(self):
        session = self.session(model=BrokenModel(self.model))
        frames, summary = self.run_session(session)
        self.assertGreater(summary.failed, 0)
        self.assertEqual(summary.scored, 0)
        for frame in frames:
            self.assertNotIn(frame.decision, ("A", "B"))
        self.assertTrue(any("scoring_failed" in frame.reasons for frame in frames))

    def test_a_label_cannot_reach_the_decoder_input(self):
        first_frames, _ = self.run_session(self.session(trial=self.trial))
        flipped = relabelled(self.trial, 1)
        second_frames, _ = self.run_session(self.session(trial=flipped))
        self.assertTrue(first_frames)
        self.assertEqual(
            [frame.to_dict() for frame in first_frames],
            [frame.to_dict() for frame in second_frames],
            "ground truth must not influence any published value",
        )
        seen = {}

        class RecordingModel(SlowModel):
            def score(self, window):
                seen["labels"] = hasattr(window, "labels")
                seen["keys"] = sorted(vars(window))
                return self._model.score(window)

        self.run_session(self.session(model=RecordingModel(self.model, delay=0.0)))
        self.assertFalse(seen["labels"])
        # The decoder's input is the window, and this is everything on it. The
        # census travels with the window (it is what `signal_quality` is judged
        # from when the quality policy will not reject), and it is chain evidence
        # like `reasons` - not truth. The assertion is deliberately exhaustive:
        # a new attribute here has to be argued for, because one of them could
        # carry the label.
        self.assertEqual(
            seen["keys"],
            [
                "available_at", "bad_channels", "contract", "eeg", "envelopes",
                "reasons", "segment", "timestamps", "valid",
            ],
        )

    def test_the_chain_window_and_channels_come_from_the_model_contract(self):
        session = self.session()
        self.assertEqual(session.settings.window_seconds, self.model.contract["window_seconds"])
        self.assertEqual(session.settings.step_seconds, self.model.contract["step_seconds"])
        self.assertEqual(
            list(session.settings.eeg_channels), list(self.model.contract["eeg_channels"])
        )
        self.assertFalse(session.settings.check_channels)

    def test_a_source_without_the_decoders_channels_is_refused(self):
        stripped = relabelled(self.trial, 0)
        stripped.channel_names = ("F3",)
        stripped.eeg = stripped.eeg[:, :1]
        with self.assertRaises(ValueError) as caught:
            self.session(trial=stripped)
        self.assertIn("channels this source does not carry", str(caught.exception))


class AttenuationTests(unittest.TestCase):
    """Gains are attenuation only, by construction."""

    def test_attenuation_db_never_exceeds_zero(self):
        self.assertEqual(attenuation_db(1.0), 0.0)
        self.assertEqual(attenuation_db(2.5), 0.0)
        self.assertEqual(attenuation_db(float("nan")), 0.0)
        self.assertAlmostEqual(attenuation_db(10 ** (-6 / 20)), -6.0, places=6)
        self.assertTrue(math.isfinite(attenuation_db(0.0)))


class ProducerTests(FixtureCase):
    """The producer's packets are what the vendored frontend actually reads."""

    def packets_from(self, session, stopped=False):
        packets = []

        def publish(kind, timestamp, source, payload):
            packets.append(
                {
                    "version": 1,
                    "type": kind,
                    "timestamp": timestamp,
                    "sequence": len(packets) + 1,
                    "source": source,
                    "session_id": "unit-session",
                    "payload": payload,
                }
            )

        stop = Event()
        if stopped:
            stop.set()
        AttentionProducer(session).run(publish, stop)
        return packets

    def test_every_packet_is_valid_and_uses_the_frontend_field_names(self):
        packets = self.packets_from(self.session())
        self.assertTrue(packets)
        for packet in packets:
            validate_packet(packet)
        kinds = {packet["type"] for packet in packets}
        self.assertTrue(
            {"audio_sources", "attention", "gain", "signal_quality", "sync", "prediction"}
            <= kinds
        )
        self.assertNotIn("session", kinds)

        sources = [p for p in packets if p["type"] == "audio_sources"][0]["payload"]
        self.assertEqual([source["id"] for source in sources["sources"]], ["A", "B"])
        self.assertEqual(
            sources["policy"]["check_channels"], False, "the policy travels with the run"
        )

        for packet in packets:
            payload = packet["payload"]
            if packet["type"] == "attention":
                self.assertIn(payload["decision"], DECISIONS)
                for key in ("correlation_a", "correlation_b"):
                    self.assertIn(key, payload)
                    self.assertTrue(
                        payload[key] is None or math.isfinite(payload[key])
                    )
            elif packet["type"] == "gain":
                self.assertLessEqual(payload["a_db"], 0.0)
                self.assertLessEqual(payload["b_db"], 0.0)
            elif packet["type"] == "signal_quality":
                self.assertIn(payload["quality"], (None, 0.0, 1.0))
                self.assertIn(payload["artifact"], (None, False, True))
            elif packet["type"] == "sync":
                self.assertIsInstance(payload["status"], str)
                self.assertIsNone(payload["offset_ms"])
            elif packet["type"] == "prediction":
                self.assertIsInstance(payload["window_id"], str)
                self.assertIsInstance(payload["reasons"], list)
                self.assertIn("policy", payload["metadata"])
                self.assertIn(payload["status"], ("ok", "invalid", "unavailable", "error"))

    def test_a_decision_packet_never_carries_attended_for_an_uncertain_frame(self):
        packets = self.packets_from(self.session())
        for packet in packets:
            if packet["type"] != "attention":
                continue
            if packet["payload"]["decision"] in ("A", "B"):
                self.assertEqual(packet["payload"]["attended"], packet["payload"]["decision"])
            else:
                self.assertNotIn("attended", packet["payload"])

    def test_no_media_fields_without_a_media_timeline(self):
        packets = self.packets_from(self.session())
        for packet in packets:
            self.assertNotIn("media_id", packet["payload"])
            self.assertNotIn("media_time_s", packet["payload"])

    def test_media_fields_appear_only_when_a_timeline_provides_them(self):
        session = self.session()
        producer = AttentionProducer(
            session,
            media_reference=lambda: {
                "media_id": "trial-008",
                "media_revision": 1,
                "media_time_s": 12.5,
            },
        )
        packets = []

        def publish(kind, timestamp, source, payload):
            packets.append((kind, payload))

        producer.run(publish, Event())
        attention = [payload for kind, payload in packets if kind == "attention"]
        self.assertTrue(attention)
        self.assertEqual(attention[0]["media_id"], "trial-008")
        self.assertEqual(attention[0]["media_time_s"], 12.5)

    def test_a_simulated_fixture_says_so_and_a_measurement_does_not(self):
        fixture = self.packets_from(self.session())
        self.assertTrue(all(packet["payload"]["simulated"] for packet in fixture))
        real = self.session()
        real.source.simulated = False
        measured = self.packets_from(real)
        self.assertTrue(all(packet["payload"]["simulated"] is False for packet in measured))
        self.assertFalse(AttentionProducer(real).simulated)

    def test_a_failed_session_is_raised_so_the_transport_records_error(self):
        class FailedSession:
            simulated = False
            policy = RunPolicy(check_channels=False, max_bad_channels=0)
            kind = "stub"
            window_seconds = 5.0
            references = None

            def run(self, emit, stop):
                from nova2026.auditory.session import SessionSummary

                return SessionSummary(
                    kind="stub",
                    simulated=False,
                    policy=self.policy.to_dict(),
                    channel_count=2,
                    window_seconds=5.0,
                    step_seconds=1.0,
                    started=0.0,
                    ended=1.0,
                    wall_seconds=0.1,
                    failure={"code": "processing_failed", "timestamp": 1.0, "detail": "x"},
                )

        producer = AttentionProducer(FailedSession())
        producer.session.references = type(
            "References", (), {"candidates": ()}
        )()
        with self.assertRaises(RuntimeError) as caught:
            producer.run(lambda *args: None, Event())
        self.assertIn("processing_failed", str(caught.exception))


class EvidenceCommandTests(unittest.TestCase):
    """The evidence command's own checks are falsifiable, not decoration.

    ``inspect`` is what decides whether a run counts as evidence, so a check that
    cannot fail would be worse than no check. These tests feed it a stream that
    passes and then the same stream with one property broken.
    """

    def stream(self, *, gain_db=0.0, session_source="server", decision="A"):
        """A small but complete packet stream, in sequence order."""

        def packet(sequence, kind, payload, source="nova-aad", timestamp=0.0):
            return {
                "version": 1,
                "type": kind,
                "timestamp": timestamp,
                "sequence": sequence,
                "source": source,
                "session_id": "s",
                "payload": payload,
            }

        return [
            packet(1, "session", {"id": "s", "status": "running"}, session_source),
            packet(2, "audio_sources", {"sources": [{"id": "A"}, {"id": "B"}]}),
            packet(3, "sync", {"status": "unobserved", "offset_ms": None}),
            packet(4, "attention", {"decision": decision}, timestamp=1.0),
            packet(5, "gain", {"a_db": gain_db, "b_db": 0.0}, timestamp=1.0),
            packet(
                6,
                "signal_quality",
                {"quality": 1.0, "artifact": False},
                timestamp=1.0,
            ),
            packet(7, "prediction", {"window_id": "w1"}, timestamp=1.0),
            packet(8, "session", {"id": "s", "status": "stopped"}, session_source, 2.0),
        ]

    def lifecycle(self, packets, cursor=3):
        return {
            "snapshot": {
                "sequence": cursor,
                "session_id": "s",
                "packets": packets[:cursor],
            },
            "stopped": {"id": "s", "status": "stopped"},
        }

    def summary(self, **overrides):
        summary = {
            "policy": {"check_channels": False, "max_bad_channels": 0},
            "windows": 2,
            "scored": 1,
            "invalid": 1,
        }
        summary.update(overrides)
        return summary

    def verdict(self, checks, name):
        matches = [item for item in checks if item["name"] == name]
        self.assertEqual(len(matches), 1, f"no check named {name!r}")
        return matches[0]["ok"]

    def test_the_expected_stream_passes_every_check(self):
        packets = self.stream()
        checks = inspect(packets, self.lifecycle(packets), self.summary())
        failed = [item for item in checks if not item["ok"]]
        self.assertEqual(failed, [], f"unexpected failures: {failed}")
        self.assertGreater(len(checks), 8)

    def test_a_positive_gain_is_flagged(self):
        packets = self.stream(gain_db=3.0)
        checks = inspect(packets, self.lifecycle(packets), self.summary())
        self.assertFalse(self.verdict(checks, "gains are attenuation only (never above 0 dB)"))

    def test_a_forged_lifecycle_packet_is_flagged(self):
        packets = self.stream(session_source="nova-aad")
        checks = inspect(packets, self.lifecycle(packets), self.summary())
        self.assertFalse(
            self.verdict(
                checks,
                "the session lifecycle is published by the transport, not duplicated",
            )
        )

    def test_an_unrecorded_policy_is_flagged(self):
        packets = self.stream()
        checks = inspect(
            packets,
            self.lifecycle(packets),
            self.summary(policy={"check_channels": True, "max_bad_channels": 0}),
        )
        self.assertFalse(self.verdict(checks, "the run policy is recorded with the run"))

    def test_a_decision_word_outside_the_four_is_flagged(self):
        packets = self.stream(decision="maybe")
        checks = inspect(packets, self.lifecycle(packets), self.summary())
        self.assertFalse(
            self.verdict(checks, "decision is one of the four allowed words")
        )


if __name__ == "__main__":
    unittest.main()
