"""Contract tests for delivery over HTTP and the live WebSocket.

No port is bound: the REST and WebSocket tests drive the ASGI app in-process
through ``fastapi.testclient`` (httpx), and the close-1013 paths are driven
through the real delivery loop with a socket stand-in, so the timing rules are
tested without a network and without waiting for a real timeout.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nova2026.transport.media import MediaTimeline
from nova2026.transport.protocol import validate_packet
from nova2026.transport.publisher import LaggedSubscriber, Publisher
from nova2026.transport.server import CLOSE_LAGGED, create_app, stream

RECONNECT_REASON = "Reconnect for latest snapshot"


class IdleProducer:
    """Keeps a session alive without publishing, so tests decide the packets."""

    simulated = False

    def run(self, publish, stop):
        stop.wait()


class RecordingSocket:
    """A WebSocket stand-in that records what the delivery loop sent and closed."""

    def __init__(self):
        self.sent = []
        self.closed = []
        self.first_send = asyncio.Event()

    async def send_json(self, packet):
        self.sent.append(packet)
        self.first_send.set()

    async def receive(self):
        await asyncio.Event().wait()  # never disconnects on its own

    async def close(self, code=1000, reason=None):
        self.closed.append((code, reason))


class StalledSocket(RecordingSocket):
    """A socket whose sends never complete, standing in for a stalled client."""

    async def send_json(self, packet):
        await asyncio.Event().wait()


class DeliveryLoopTests(unittest.TestCase):
    """The snapshot-then-increment rule and the two paths that close with 1013."""

    def test_snapshot_is_delivered_first_and_only_newer_events_follow(self):
        publisher = Publisher()
        publisher.begin("session")
        publisher.publish("attention", 0.0, "test-aad", "session", {"decision": "A"})
        socket = RecordingSocket()
        stopping = asyncio.Event()

        async def run():
            task = asyncio.create_task(
                stream(socket, publisher, stopping, send_timeout=1.0, poll_interval=0.01)
            )
            await asyncio.wait_for(socket.first_send.wait(), 1.0)
            publisher.publish("attention", 0.25, "test-aad", "session", {"decision": "B"})
            await asyncio.sleep(0.05)
            stopping.set()
            await asyncio.wait_for(task, 1.0)

        asyncio.run(run())
        self.assertEqual(
            [packet["payload"] for packet in socket.sent],
            [{"decision": "A"}, {"decision": "B"}],
        )
        self.assertEqual([packet["sequence"] for packet in socket.sent], [1, 2])
        self.assertEqual(socket.closed, [])

    def test_overrunning_the_event_ring_closes_with_1013(self):
        publisher = Publisher(event_limit=4)
        publisher.begin("session")
        publisher.publish("session", 0.0, "server", "session", {"status": "running"})
        socket = RecordingSocket()

        async def run():
            task = asyncio.create_task(
                stream(socket, publisher, asyncio.Event(), send_timeout=1.0, poll_interval=0.05)
            )
            # The snapshot send has happened; the loop is now polling the ring, so
            # publishing more than it holds is a deterministic overrun.
            await asyncio.wait_for(socket.first_send.wait(), 1.0)
            for index in range(6):
                publisher.publish("attention", index * 0.25, "test-aad", "session", {"i": index})
            await asyncio.wait_for(task, 2.0)

        asyncio.run(run())
        self.assertEqual(socket.closed, [(CLOSE_LAGGED, RECONNECT_REASON)])

    def test_a_send_timeout_closes_with_1013(self):
        publisher = Publisher()
        publisher.begin("session")
        publisher.publish("attention", 0.0, "test-aad", "session", {"decision": "A"})
        socket = StalledSocket()

        async def run():
            await stream(socket, publisher, asyncio.Event(), send_timeout=0.05, poll_interval=0.01)

        asyncio.run(asyncio.wait_for(run(), 5.0))
        self.assertEqual(socket.closed, [(CLOSE_LAGGED, RECONNECT_REASON)])

    def test_a_lagged_subscriber_closes_with_1013(self):
        class Overrun(Publisher):
            def events_after(self, cursor):
                raise LaggedSubscriber("forced overflow")

        publisher = Overrun()
        publisher.begin("session")
        publisher.publish("session", 0.0, "server", "session", {"status": "running"})
        socket = RecordingSocket()
        asyncio.run(
            asyncio.wait_for(
                stream(socket, publisher, asyncio.Event(), send_timeout=1.0, poll_interval=0.01),
                5.0,
            )
        )
        self.assertEqual(socket.closed, [(CLOSE_LAGGED, RECONNECT_REASON)])

    def test_a_client_disconnect_ends_the_stream_without_closing_1013(self):
        class Disconnecting(RecordingSocket):
            async def receive(self):
                return {"type": "websocket.disconnect"}

        publisher = Publisher()
        publisher.begin("session")
        publisher.publish("attention", 0.0, "test-aad", "session", {})
        socket = Disconnecting()
        asyncio.run(
            asyncio.wait_for(
                stream(socket, publisher, asyncio.Event(), send_timeout=1.0, poll_interval=0.01),
                5.0,
            )
        )
        self.assertEqual(socket.closed, [])


class TransportAppTests(unittest.TestCase):
    """The HTTP surface, the WebSocket contract and the SPA mount."""

    def test_health_state_and_session_commands(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            self.assertEqual(
                client.get("/api/health").json(), {"status": "ok", "protocol_version": 1}
            )
            state = client.get("/api/state").json()
            self.assertEqual(state["packets"], [])
            self.assertIsNone(state["session_id"])
            self.assertIsNone(client.post("/api/session/stop").json())
            record = client.post("/api/session/start").json()
            self.assertEqual(record["status"], "running")
            self.assertEqual(client.post("/api/session/start").json()["id"], record["id"])
            self.assertEqual(client.get(f"/api/sessions/{record['id']}").status_code, 200)
            self.assertEqual(client.get("/api/sessions/missing").status_code, 404)
            self.assertEqual(client.post("/api/session/stop").json()["status"], "stopped")
        self.assertTrue(app.state.sessions.closed)
        self.assertFalse(app.state.sessions.thread.is_alive())

    def test_websocket_delivers_the_snapshot_then_increments(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            session_id = client.post("/api/session/start").json()["id"]
            publisher = app.state.publisher
            publisher.publish("attention", 0.0, "test-aad", session_id, {"decision": "A"})
            publisher.publish("gain", 0.0, "test-gain", session_id, {"a_db": 0, "b_db": -6})
            expected = client.get("/api/state").json()["packets"]
            self.assertEqual(len(expected), 3)  # the start packet plus two streams

            with client.websocket_connect("/ws/live") as socket:
                snapshot = [socket.receive_json() for _ in expected]
                self.assertEqual(snapshot, expected)
                for packet in snapshot:
                    validate_packet(packet)

                socket.send_text("not a command")
                publisher.publish("attention", 0.25, "test-aad", session_id, {"decision": "B"})
                event = socket.receive_json()
                self.assertEqual(event["payload"], {"decision": "B"})
                self.assertEqual(event["sequence"], expected[-1]["sequence"] + 1)
                self.assertEqual(event["session_id"], session_id)
                validate_packet(event)

    def test_unknown_type_and_partial_payload_are_delivered_unchanged(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            session_id = client.post("/api/session/start").json()["id"]
            packet = app.state.publisher.publish(
                "attention", 0.0, "test-aad", session_id, {"decision": "uncertain", "extra": None}
            )
            with client.websocket_connect("/ws/live") as socket:
                first = socket.receive_json()
                self.assertEqual(first["type"], "session")
                self.assertEqual(socket.receive_json(), packet)

    def test_a_client_that_lags_is_closed_with_1013(self):
        class Overrun(Publisher):
            def events_after(self, cursor):
                raise LaggedSubscriber("forced overflow")

        with TestClient(create_app(IdleProducer, Overrun())) as client:
            client.post("/api/session/start")
            with client.websocket_connect("/ws/live") as socket:
                self.assertEqual(socket.receive_json()["type"], "session")
                with self.assertRaises(WebSocketDisconnect) as caught:
                    socket.receive_json()
                self.assertEqual(caught.exception.code, CLOSE_LAGGED)

    def test_a_disconnect_leaves_no_subscriber_behind(self):
        app = create_app(IdleProducer)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/live") as socket:
                socket.send_bytes(b"ignored")
            self.assertEqual(len(app.state.sockets), 0)

    def test_a_failing_factory_is_a_clean_http_error(self):
        def failure():
            raise ValueError("private path")

        with TestClient(create_app(failure)) as client:
            response = client.post("/api/session/start")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private", response.text)
            self.assertEqual(client.get("/api/health").status_code, 200)

    def test_a_missing_producer_factory_is_an_explicit_start_error(self):
        with TestClient(create_app()) as client:
            response = client.post("/api/session/start")
            self.assertEqual(response.status_code, 409)
            self.assertIn("producer factory", response.json()["detail"])
            self.assertEqual(client.get("/api/state").json()["packets"], [])

    def test_the_spa_is_served_from_disk_and_api_routes_keep_precedence(self):
        dist = Path(tempfile.mkdtemp(prefix="nova-dist-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(dist, ignore_errors=True))
        (dist / "index.html").write_text("<html>attune</html>", encoding="utf-8")
        (dist / "assets").mkdir()
        (dist / "assets" / "app.js").write_text("export {};", encoding="utf-8")

        with TestClient(create_app(IdleProducer, static_dir=dist)) as client:
            self.assertEqual(client.get("/").text, "<html>attune</html>")
            self.assertEqual(client.get("/assets/app.js").text, "export {};")
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/api/state").json()["packets"], [])

        self.assertEqual(create_app(IdleProducer).state.static_dir, None)
        with self.assertRaises(ValueError) as caught:
            create_app(IdleProducer, static_dir=dist / "missing")
        self.assertIn("static_dir", str(caught.exception))

    def test_media_endpoints_serve_the_asset_and_require_a_running_session(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "sample.wav"
        path.write_bytes(b"RIFF" + bytes(64))
        timeline = MediaTimeline(path, "Conversation")

        with TestClient(create_app(IdleProducer, media=timeline)) as client:
            descriptor = client.get("/api/media").json()
            self.assertNotIn(str(path), json.dumps(descriptor))
            response = client.get(descriptor["url"], headers={"Range": "bytes=0-3"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"RIFF")
            self.assertEqual(response.headers["cache-control"], "no-store")

            body = dict(
                session_id="none",
                media_id=descriptor["media_id"],
                client_id="browser",
                request_id=1,
                action="prepare",
                media_time_s=0,
                duration_s=90,
            )
            self.assertEqual(client.post("/api/media/control", json=body).status_code, 409)

            session_id = client.post("/api/session/start").json()["id"]
            body["session_id"] = session_id
            acknowledgement = client.post("/api/media/control", json=body).json()
            self.assertEqual(acknowledgement["sync_status"], "observed")
            self.assertEqual(client.get("/api/state").json()["session_id"], session_id)

            body["request_id"] = 2
            body["action"] = "report"
            body["media_time_s"] = 40
            rejected = client.post("/api/media/control", json=body)
            self.assertEqual(rejected.status_code, 409)
            self.assertNotIn("traceback", rejected.text.lower())

        with TestClient(create_app(IdleProducer)) as client:
            self.assertIsNone(client.get("/api/media").json())
            self.assertEqual(client.get("/api/media/file").status_code, 404)


if __name__ == "__main__":
    unittest.main()
