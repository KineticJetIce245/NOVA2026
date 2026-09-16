"""Integration tests: the attention producer wired into the real transport app.

These drive the ASGI app in process through ``fastapi.testclient`` - no port is
bound - and they build their own fixture (two candidate WAVs, their offline
envelopes and a fitted decoder) in a temporary directory, so nothing here depends
on the dataset or the models directory being present.

What is being tested is the seam the plan cares about: the transport's
``producer_factory`` is filled by a real session, a session whose envelope is
missing refuses to start instead of computing one, and the packets that come out
of ``/ws/live`` are the ones the vendored frontend decodes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from nova2026.auditory.envelopes import EnvelopeVerificationError
from nova2026.auditory.producer import AttentionProducer
from nova2026.auditory.session import DECISIONS, AttentionSession, RunPolicy
from nova2026.auditory.sources import ReferenceEnvelopes, ReplaySource
from nova2026.transport.protocol import validate_packet
from nova2026.transport.server import create_app
from scripts.auditory.tests.session_fixture import build_fixture

SECONDS = 20.0
TERMINAL = ("stopped", "error")
REQUIRED_TYPES = (
    "session",
    "audio_sources",
    "attention",
    "gain",
    "signal_quality",
    "sync",
    "prediction",
)


class ProducerIntegrationTests(unittest.TestCase):
    """One shared fixture, five ways of looking at the same seam."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls._temporary.name)
        cls.model, cls.trial, cls.paths = build_fixture(
            cls.directory, seconds=SECONDS
        )
        cls.static = cls.directory / "dist"
        cls.static.mkdir()
        (cls.static / "index.html").write_text(
            "<!doctype html><title>attune dashboard</title>", encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def factory(self, *, speed=1.0, simulated=True, paths=None, referenced=None):
        """A producer factory exactly like the one a demo would inject."""

        def build():
            references = ReferenceEnvelopes.load(
                paths if paths is not None else self.paths, self.model.config
            )
            if referenced is not None:
                referenced.append(references)
            source = ReplaySource(
                self.trial, speed=speed, simulated=simulated, kind="fixture_replay"
            )
            session = AttentionSession(
                source=source,
                decoder=self.model,
                references=references,
                policy=RunPolicy(check_channels=False, max_bad_channels=0),
            )
            return AttentionProducer(session)

        return build

    def test_without_a_factory_the_transport_still_refuses_to_invent_one(self):
        app = create_app()
        with TestClient(app) as client:
            response = client.post("/api/session/start")
        self.assertEqual(response.status_code, 409)
        self.assertIn("no producer factory configured", response.json()["detail"])

    def test_a_session_without_its_envelope_refuses_to_start(self):
        missing = self.directory / "missing_candidate.npz"
        factory = self.factory(paths=(self.paths[0], missing))
        with self.assertRaises(EnvelopeVerificationError) as caught:
            factory()
        self.assertIn("missing_candidate.npz", str(caught.exception))

        app = create_app(producer_factory=factory)
        with TestClient(app) as client:
            response = client.post("/api/session/start")
            # EnvelopeVerificationError is a RuntimeError, so the transport's
            # "refused" branch answers 409 and relays the message, which is what
            # makes the missing envelope nameable at the point of failure.
            self.assertEqual(response.status_code, 409)
            self.assertIn("missing_candidate.npz", response.json()["detail"])
            # No session was recorded as running behind the refusal.
            self.assertEqual(client.get("/api/sessions").json(), [])

    def test_the_live_socket_carries_the_packets_the_frontend_decodes(self):
        app = create_app(producer_factory=self.factory(speed=8.0), static_dir=self.static)
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health").json()["protocol_version"], 1)
            index = client.get("/")
            self.assertEqual(index.status_code, 200)
            self.assertIn("attune dashboard", index.text)

            started = client.post("/api/session/start").json()
            self.assertTrue(started["simulated"], "a fixture must be labelled as one")
            cursor = client.get("/api/state").json()

            packets = []
            with client.websocket_connect("/ws/live") as socket:
                while len(packets) < 400:
                    packet = socket.receive_json()
                    packets.append(packet)
                    if (
                        packet["type"] == "session"
                        and packet["payload"].get("status") in TERMINAL
                    ):
                        break

        self.assertTrue(packets)
        for packet in packets:
            validate_packet(packet)
        kinds = {packet["type"] for packet in packets}
        self.assertTrue(set(REQUIRED_TYPES) <= kinds, sorted(kinds))
        self.assertTrue(
            all(
                packet["source"] == "server"
                for packet in packets
                if packet["type"] == "session"
            ),
            "the producer must not forge lifecycle packets",
        )
        decisions = {
            packet["payload"]["decision"]
            for packet in packets
            if packet["type"] == "attention"
        }
        self.assertTrue(decisions <= set(DECISIONS), decisions)
        for packet in packets:
            if packet["type"] == "gain":
                self.assertLessEqual(packet["payload"]["a_db"], 0.0)
                self.assertLessEqual(packet["payload"]["b_db"], 0.0)

        # The snapshot is delivered before anything newer than its cursor.
        older = [p["sequence"] for p in packets if p["sequence"] <= cursor["sequence"]]
        newer = [p["sequence"] for p in packets if p["sequence"] > cursor["sequence"]]
        self.assertTrue(not older or not newer or max(older) < min(newer))
        self.assertTrue(
            {packet["type"] for packet in cursor["packets"]} <= kinds,
            "every stream in the snapshot must also arrive on the socket",
        )
        self.assertEqual([packet["sequence"] for packet in packets],
                         sorted({packet["sequence"] for packet in packets}))
        self.assertEqual(
            packets[-1]["payload"]["status"], "stopped", "the run must end explicitly"
        )

    def test_stop_joins_the_producer_and_closes_the_record(self):
        app = create_app(producer_factory=self.factory(speed=1.0))
        with TestClient(app) as client:
            started = client.post("/api/session/start").json()
            self.assertEqual(started["status"], "running")
            with client.websocket_connect("/ws/live") as socket:
                first = [socket.receive_json() for _ in range(3)]
                self.assertEqual(
                    [packet["type"] for packet in first],
                    ["session", "audio_sources", "sync"],
                    "the snapshot's streams arrive first, in sequence order",
                )
                stopped = client.post("/api/session/stop").json()
                self.assertEqual(stopped["status"], "stopped")
                for _ in range(200):
                    packet = socket.receive_json()
                    if (
                        packet["type"] == "session"
                        and packet["payload"].get("status") in TERMINAL
                    ):
                        break
                else:  # pragma: no cover - the loop always breaks in practice
                    self.fail("no terminal session packet arrived after stop")
            self.assertEqual(packet["payload"]["status"], "stopped")
            records = client.get("/api/sessions").json()
        self.assertEqual(records[-1]["status"], "stopped")
        self.assertEqual(len(records), 1)

    def test_a_measurement_is_not_labelled_as_simulated(self):
        app = create_app(producer_factory=self.factory(speed=8.0, simulated=False))
        with TestClient(app) as client:
            started = client.post("/api/session/start").json()
            self.assertFalse(started["simulated"])
            packets = []
            with client.websocket_connect("/ws/live") as socket:
                while len(packets) < 400:
                    packet = socket.receive_json()
                    packets.append(packet)
                    if (
                        packet["type"] == "session"
                        and packet["payload"].get("status") in TERMINAL
                    ):
                        break
        attention = [p for p in packets if p["type"] == "attention"]
        self.assertTrue(attention)
        self.assertTrue(all(p["payload"]["simulated"] is False for p in attention))


if __name__ == "__main__":
    unittest.main()
