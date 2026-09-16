"""Contract tests for publication, the snapshot and the bounded event ring."""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from nova2026.transport.protocol import MAX_SAFE_INTEGER
from nova2026.transport.publisher import (
    DEFAULT_EVENT_LIMIT,
    DEFAULT_STATE_LIMIT,
    LaggedSubscriber,
    Publisher,
)


class SnapshotTests(unittest.TestCase):
    """The snapshot is the latest value per stream, and nothing more."""

    def test_snapshot_shape_and_one_latest_packet_per_stream(self):
        publisher = Publisher()
        publisher.begin("one")
        for index in range(5):
            publisher.publish("attention", index, "aad", "one", {"value": index})
        publisher.publish("gain", 0, "gain-source", "one", {"a_db": -6})

        snapshot = publisher.snapshot()
        self.assertEqual(set(snapshot), {"sequence", "session_id", "packets"})
        self.assertEqual(snapshot["session_id"], "one")
        self.assertEqual(snapshot["sequence"], 6)
        # Ordered by sequence: the attention stream's latest packet is 5, the gain
        # packet is 6.
        self.assertEqual([packet["type"] for packet in snapshot["packets"]], ["attention", "gain"])
        attention = next(p for p in snapshot["packets"] if p["type"] == "attention")
        self.assertEqual(attention["payload"], {"value": 4})
        self.assertEqual(attention["sequence"], 5)

    def test_snapshot_and_events_are_defensive_copies(self):
        publisher = Publisher()
        publisher.begin("one")
        publisher.publish("attention", 0, "aad", "one", {"value": 1})
        snapshot = publisher.snapshot()
        snapshot["packets"][0]["payload"]["value"] = -1
        snapshot["packets"][0]["sequence"] = -1
        self.assertEqual(publisher.snapshot()["packets"][0]["payload"]["value"], 1)
        events = publisher.events_after(0)
        events[0]["payload"]["value"] = -1
        self.assertEqual(publisher.events_after(0)[0]["payload"]["value"], 1)

    def test_starting_a_session_clears_latest_streams_but_keeps_the_ring(self):
        publisher = Publisher()
        publisher.begin("one")
        publisher.publish("attention", 0, "aad", "one", {"value": 1})
        publisher.end("one")
        publisher.begin("two")

        snapshot = publisher.snapshot()
        self.assertEqual(snapshot["session_id"], "two")
        self.assertEqual(snapshot["packets"], [])
        # A connected client must still observe the session boundary event.
        self.assertEqual([packet["type"] for packet in publisher.events_after(0)], ["attention"])

    def test_sequence_is_process_wide_and_does_not_reset_between_sessions(self):
        publisher = Publisher()
        publisher.begin("one")
        self.assertEqual(publisher.publish("x", 0, "src", "one", {})["sequence"], 1)
        publisher.end("one")
        publisher.begin("two")
        self.assertEqual(publisher.publish("x", 0, "src", "two", {})["sequence"], 2)

    def test_default_bounds_are_the_documented_ones(self):
        self.assertEqual(DEFAULT_EVENT_LIMIT, 256)
        self.assertEqual(DEFAULT_STATE_LIMIT, 128)
        publisher = Publisher()
        self.assertEqual(publisher.events.maxlen, 256)
        self.assertEqual(publisher.state_limit, 128)

    def test_128_stream_bound_evicts_least_recently_updated(self):
        publisher = Publisher()
        publisher.begin("one")
        for index in range(130):
            publisher.publish(f"kind-{index}", 0, "src", "one", {})
        packets = publisher.snapshot()["packets"]
        self.assertEqual(len(packets), 128)
        self.assertNotIn("kind-0", [packet["type"] for packet in packets])
        self.assertNotIn("kind-1", [packet["type"] for packet in packets])
        self.assertIn("kind-129", [packet["type"] for packet in packets])

    def test_updating_a_stream_refreshes_its_eviction_position(self):
        publisher = Publisher(state_limit=2)
        publisher.begin("one")
        publisher.publish("first", 0, "src", "one", {})
        publisher.publish("second", 0, "src", "one", {})
        publisher.publish("first", 1, "src", "one", {})
        publisher.publish("third", 0, "src", "one", {})
        self.assertEqual(
            [packet["type"] for packet in publisher.snapshot()["packets"]], ["first", "third"]
        )


class PublicationTests(unittest.TestCase):
    """What publication accepts, what it refuses, and in which order."""

    def test_inactive_sessions_cannot_publish(self):
        publisher = Publisher()
        with self.assertRaises(ValueError):
            publisher.publish("x", 0, "src", "one", {})  # before begin
        publisher.begin("one")
        with self.assertRaises(ValueError):
            publisher.publish("x", 0, "src", "other", {})
        publisher.end("one")
        with self.assertRaises(ValueError):
            publisher.publish("x", 0, "src", "one", {})
        publisher.end("other")  # ignoring another session is not an error
        self.assertTrue(publisher.events_after(0) == [])

    def test_timestamp_may_not_regress_within_a_stream(self):
        publisher = Publisher()
        publisher.begin("one")
        publisher.publish("attention", 2, "aad", "one", {})
        with self.assertRaises(ValueError):
            publisher.publish("attention", 1, "aad", "one", {})
        # Equal timestamps are allowed, and other streams have their own timebase.
        publisher.publish("attention", 2, "aad", "one", {})
        publisher.publish("gain", 0, "gain", "one", {})

    def test_rejected_publication_leaves_the_publisher_unchanged(self):
        publisher = Publisher()
        publisher.begin("one")
        publisher.publish("x", 2, "src", "one", {})
        before = publisher.snapshot()
        for args in (("x", 1, "src", "one", {}), ("x", 3, "src", "old", {})):
            with self.assertRaises(ValueError):
                publisher.publish(*args)
        self.assertEqual(publisher.sequence, 1)
        self.assertEqual(publisher.snapshot(), before)

    def test_bounds_must_be_positive_integers(self):
        for kwargs in (
            {"event_limit": 0},
            {"state_limit": 0},
            {"event_limit": True},
            {"state_limit": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Publisher(**kwargs)

    def test_begin_requires_a_usable_session_identifier(self):
        publisher = Publisher()
        for session_id in ("", None, 4, "x" * 129):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                publisher.begin(session_id)  # type: ignore[arg-type]

    def test_cursor_must_be_a_nonnegative_integer(self):
        publisher = Publisher()
        for cursor in (-1, True, 1.5, "0"):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                publisher.events_after(cursor)  # type: ignore[arg-type]

    def test_concurrent_publication_assigns_each_sequence_once(self):
        publisher = Publisher()
        publisher.begin("one")

        def publish_one(index):
            return publisher.publish("x", 0, f"src-{index % 4}", "one", {"i": index})

        with ThreadPoolExecutor(max_workers=4) as pool:
            packets = list(pool.map(publish_one, range(100)))
        self.assertEqual(sorted(packet["sequence"] for packet in packets), list(range(1, 101)))
        self.assertLessEqual(publisher.sequence, MAX_SAFE_INTEGER)


class EventRingTests(unittest.TestCase):
    """The ring bounds memory and reports overruns instead of hiding them."""

    def test_ring_keeps_the_newest_events(self):
        publisher = Publisher(event_limit=3)
        publisher.begin("one")
        for index in range(5):
            publisher.publish("x", index, "src", "one", {"i": index})
        # The ring holds sequences 3..5, so cursor 2 is the oldest readable one.
        self.assertEqual([packet["sequence"] for packet in publisher.events_after(2)], [3, 4, 5])
        self.assertEqual([packet["sequence"] for packet in publisher.events_after(4)], [5])
        self.assertEqual(publisher.events_after(5), [])

    def test_lagged_subscriber_is_raised_rather_than_returning_a_gap(self):
        publisher = Publisher(event_limit=3)
        publisher.begin("one")
        for index in range(5):
            publisher.publish("x", index, "src", "one", {"i": index})
        for cursor in (0, 1):
            with self.subTest(cursor=cursor), self.assertRaises(LaggedSubscriber):
                publisher.events_after(cursor)


if __name__ == "__main__":
    unittest.main()
