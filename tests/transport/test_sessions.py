"""Contract tests for session command serialization and the producer lifecycle."""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from nova2026.transport.publisher import Publisher
from nova2026.transport.sessions import RESERVED_SOURCE, Sessions


class IdleProducer:
    """A producer that publishes nothing and only waits for cancellation."""

    simulated = False

    def run(self, publish, stop):
        stop.wait()


class BurstProducer:
    """A deterministic producer: one fixed burst, then it is done."""

    simulated = True

    def __init__(self, count=3):
        self.count = count

    def run(self, publish, stop):
        for index in range(self.count):
            publish("scripted", index * 0.25, "test-producer", {"index": index})


class LifecycleTests(unittest.TestCase):
    """Start and stop are serialized, idempotent commands."""

    def test_commands_are_idempotent_and_restart_creates_a_new_session(self):
        sessions = Sessions(Publisher(), IdleProducer)
        try:
            self.assertIsNone(sessions.stop())
            first = sessions.start()
            self.assertEqual(first["status"], "running")
            self.assertFalse(first["simulated"])
            self.assertEqual(sessions.start()["id"], first["id"])
            self.assertEqual(sessions.stop()["status"], "stopped")
            self.assertFalse(sessions.thread.is_alive())
            self.assertEqual(sessions.stop()["status"], "stopped")
            self.assertNotEqual(sessions.start()["id"], first["id"])
        finally:
            sessions.close()
        self.assertFalse(sessions.thread.is_alive())
        with self.assertRaises(RuntimeError):
            sessions.start()

    def test_concurrent_start_starts_exactly_one_worker(self):
        sessions = Sessions(Publisher(), IdleProducer)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                records = list(pool.map(lambda _: sessions.start(), range(8)))
            self.assertEqual(len({record["id"] for record in records}), 1)
        finally:
            sessions.close()

    def test_start_without_a_producer_factory_is_explicit(self):
        sessions = Sessions(Publisher())
        try:
            with self.assertRaises(RuntimeError) as caught:
                sessions.start()
            self.assertIn("producer factory", str(caught.exception))
        finally:
            sessions.close()

    def test_close_refuses_further_starts(self):
        sessions = Sessions(Publisher(), IdleProducer)
        sessions.start()
        worker = sessions.thread
        sessions.close()
        self.assertTrue(sessions.closed)
        self.assertFalse(worker.is_alive())
        with self.assertRaises(RuntimeError):
            sessions.start()

    def test_stop_deadline_reports_a_worker_that_ignores_cancellation(self):
        release = Event()
        entered = Event()

        class Blocked:
            def run(self, publish, stop):
                entered.set()
                release.wait(2)

        sessions = Sessions(Publisher(), Blocked, stop_timeout=0.01)
        sessions.start()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                sessions.stop()
        finally:
            release.set()
            sessions.thread.join(1)
            sessions.close()

    def test_commands_must_be_usable(self):
        with self.assertRaises(ValueError):
            Sessions(Publisher(), producer_factory="not callable")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Sessions(Publisher(), IdleProducer, stop_timeout=0)


class ProducerBoundaryTests(unittest.TestCase):
    """What a producer may publish, and what a finished session refuses."""

    def test_published_packets_carry_the_running_session(self):
        publisher = Publisher()
        sessions = Sessions(publisher, BurstProducer)
        try:
            record = sessions.start()
            sessions.thread.join(3)
            packets = publisher.events_after(0)
        finally:
            sessions.close()

        self.assertEqual(record["simulated"], True)
        self.assertEqual(packets[0]["type"], "session")
        self.assertEqual(packets[0]["source"], RESERVED_SOURCE)
        self.assertEqual(packets[0]["payload"]["status"], "running")
        scripted = [packet for packet in packets if packet["type"] == "scripted"]
        self.assertEqual([packet["sequence"] for packet in scripted], [2, 3, 4])
        self.assertTrue(all(packet["session_id"] == record["id"] for packet in packets))
        self.assertEqual(packets[-1]["payload"]["status"], "stopped")

    def test_terminal_session_packet_carries_a_session_relative_time(self):
        publisher = Publisher()
        sessions = Sessions(publisher, BurstProducer)
        try:
            sessions.start()
            sessions.thread.join(3)
            terminal = publisher.events_after(0)[-1]
        finally:
            sessions.close()
        self.assertEqual(terminal["type"], "session")
        self.assertGreaterEqual(terminal["timestamp"], 0)
        self.assertLess(terminal["timestamp"], 5)

    def test_a_stopped_session_can_publish_nothing_further(self):
        publisher = Publisher()
        sessions = Sessions(publisher, BurstProducer)
        try:
            record = sessions.start()
            sessions.thread.join(3)
            self.assertEqual(sessions.list()[-1]["status"], "stopped")
            with self.assertRaises(ValueError):
                publisher.publish("late", 10, "test-producer", record["id"], {})
        finally:
            sessions.close()

    def test_producer_failure_is_reported_without_its_details(self):
        class Failing:
            def run(self, publish, stop):
                raise ValueError("private diagnostic details")

        publisher = Publisher()
        sessions = Sessions(publisher, Failing)
        try:
            sessions.start()
            sessions.thread.join(3)
            record = sessions.list()[-1]
            packets = publisher.events_after(0)
        finally:
            sessions.close()

        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"], "producer_failed")
        self.assertNotIn("private", str(record))
        self.assertEqual(packets[-1]["payload"]["status"], "error")
        self.assertEqual(packets[-1]["payload"]["error"], "producer_failed")

    def test_the_server_source_is_reserved(self):
        class Forger:
            def run(self, publish, stop):
                try:
                    publish("session", 0, RESERVED_SOURCE, {"status": "running"})
                except ValueError as error:
                    publish("rejected", 0, "test-producer", {"message": str(error)})

        publisher = Publisher()
        sessions = Sessions(publisher, Forger)
        try:
            sessions.start()
            sessions.thread.join(3)
            packets = publisher.events_after(0)
        finally:
            sessions.close()

        rejected = [packet for packet in packets if packet["type"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("reserved", rejected[0]["payload"]["message"])
        lifecycle = [packet for packet in packets if packet["source"] == RESERVED_SOURCE]
        self.assertEqual([packet["type"] for packet in lifecycle], ["session", "session"])


if __name__ == "__main__":
    unittest.main()
