"""Contract tests for the version-1 packet envelope.

These pin the rules the vendored frontend re-checks on its own side: accepting
anything here that the client rejects would mean a frame counted as delivered and
then silently dropped in the browser.
"""

from __future__ import annotations

import json
import unittest

from nova2026.transport.protocol import (
    MAX_JSON_DEPTH,
    MAX_PACKET_BYTES,
    MAX_TEXT,
    VERSION,
    make_packet,
    validate_packet,
)


def envelope(**overrides):
    """A minimal valid packet with selected envelope fields replaced."""

    packet = make_packet("attention", 0, 1, "source", "session", {})
    packet.update(overrides)
    return packet


class EnvelopeTests(unittest.TestCase):
    """The envelope fields, their types and their bounds."""

    def test_packet_creation_preserves_unknown_fields_and_detaches(self):
        payload = {"future": [None, {"x": 1}]}
        packet = make_packet("future", 1.5, 7, "source", "session", payload)
        self.assertEqual(packet["version"], VERSION)
        self.assertEqual(packet["timestamp"], 1.5)
        self.assertEqual(packet["sequence"], 7)
        self.assertEqual(packet["payload"], payload)

        # The published packet is a copy: later producer mutation cannot rewrite it.
        payload["future"].append("changed")
        self.assertEqual(len(packet["payload"]["future"]), 2)

        # Unknown envelope fields survive a re-validation, and so does the type.
        packet["extension"] = {"kept": True}
        again = validate_packet(packet)
        self.assertTrue(again["extension"]["kept"])
        self.assertEqual(again["type"], "future")

    def test_malformed_envelopes_are_rejected(self):
        cases = [
            ("version", 2),
            ("version", True),
            ("version", "1"),
            ("version", None),
            ("sequence", -1),
            ("sequence", True),
            ("sequence", 2**53),
            ("sequence", 1.0),
            ("sequence", "1"),
            ("timestamp", float("nan")),
            ("timestamp", float("inf")),
            ("timestamp", float("-inf")),
            ("timestamp", -1),
            ("timestamp", True),
            ("timestamp", None),
            ("source", ""),
            ("session_id", None),
            ("type", 4),
            ("payload", []),
            ("payload", None),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_packet(envelope(**{field: value}))

        for raw in (None, [], "not json", {}, 4):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_packet(raw)

    def test_text_fields_name_their_limit(self):
        for field in ("type", "source", "session_id"):
            with self.subTest(field=field):
                bounded = validate_packet(envelope(**{field: "x" * MAX_TEXT}))
                self.assertEqual(len(bounded[field]), MAX_TEXT)
                with self.assertRaises(ValueError) as caught:
                    validate_packet(envelope(**{field: "x" * (MAX_TEXT + 1)}))
                self.assertIn(field, str(caught.exception))

    def test_oversized_timestamp_is_refused(self):
        with self.assertRaises(ValueError):
            make_packet("x", 10**400, 1, "source", "session", {})


class PayloadTests(unittest.TestCase):
    """Payload values are not interpreted, but they must be JSON."""

    def test_non_finite_and_non_json_values_are_rejected(self):
        payloads = [
            {"x": float("nan")},
            {"x": float("inf")},
            {"x": float("-inf")},
            {"x": object()},
            {1: "not a JSON key"},
            {"x": (1, 2)},
            {"x": {1, 2}},
            {"x": b"bytes"},
            {"nested": [{"deep": object()}]},
        ]
        for case, payload in enumerate(payloads):
            with self.subTest(case=case), self.assertRaises(ValueError):
                make_packet("future", 0, 1, "source", "session", payload)
        with self.assertRaises(ValueError):
            make_packet("future", 0, 1, "source", "session", [])  # type: ignore[arg-type]

    def test_null_and_partial_payload_values_are_kept(self):
        payload = {"value": None, "nested": {"present": None}}
        self.assertEqual(make_packet("future", 0, 1, "s", "id", payload)["payload"], payload)

    def test_nesting_depth_is_bounded_including_the_envelope(self):
        def nested(levels):
            payload = {}
            for _ in range(levels):
                payload = {"nested": payload}
            return payload

        # The envelope root counts as level 0, so the payload may hold 31 nested
        # objects; one more reaches level 33. Being one level stricter than the
        # client is the safe direction.
        self.assertEqual(
            make_packet("x", 0, 1, "s", "id", nested(MAX_JSON_DEPTH - 1))["type"], "x"
        )
        with self.assertRaises(ValueError) as caught:
            make_packet("x", 0, 1, "s", "id", nested(MAX_JSON_DEPTH))
        self.assertIn("deep", str(caught.exception))

    def test_encoded_size_is_bounded(self):
        under = {"blob": "x" * (MAX_PACKET_BYTES - 512)}
        packet = make_packet("x", 0, 1, "s", "id", under)
        self.assertLessEqual(len(json.dumps(packet, allow_nan=False).encode()), MAX_PACKET_BYTES)
        with self.assertRaises(ValueError) as caught:
            make_packet("x", 0, 1, "s", "id", {"blob": "x" * (MAX_PACKET_BYTES + 1)})
        self.assertIn("limit", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
