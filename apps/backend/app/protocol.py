"""Packet envelope construction, version 1 — a test double.

This module is NOT the transport implementation. It exists so the vendored
frontend tests in ``apps/attune-ui/tests/`` can generate real version-1 packets
from Python across the language boundary: three of those tests spawn an
interpreter and feed the resulting JSON into the browser-side validators.

``src/nova2026/transport`` is the implementation of record and replaces this
fixture when it lands. Nothing in the demo imports this module; if the two ever
disagree, the frontend's validator is what decides, and the fixture is wrong.
"""

from __future__ import annotations

import math

# Envelope limits the client enforces (frontend/src/protocol.js).
MAX_TEXT = 128
MAX_JSON_DEPTH = 32
MAX_PACKET_BYTES = 256 * 1024
VERSION = 1


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string.")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{field} must be at most {MAX_TEXT} characters.")
    return value


def _depth(value, level: int = 0) -> int:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Payload keys must be strings.")
        return max([level] + [_depth(item, level + 1) for item in value.values()])
    if isinstance(value, (list, tuple)):
        return max([level] + [_depth(item, level + 1) for item in value])
    return level


def _finite(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Payload numbers must be finite.")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def make_packet(
    packet_type: str,
    timestamp: float,
    sequence: int,
    session_id: str,
    source: str,
    payload: dict,
) -> dict:
    """Build one version-1 packet, rejecting what the client would drop.

    The field order matches ``backend/app/protocol.py`` in the teammate's
    repository so a fixture packet is byte-comparable with a real one.
    """

    _text(packet_type, "type")
    _text(session_id, "session_id")
    _text(source, "source")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a nonnegative integer.")
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be a finite nonnegative number.")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object.")
    _finite(payload)
    if _depth(payload) > MAX_JSON_DEPTH:
        raise ValueError("payload nests deeper than the client accepts.")

    packet = {
        "version": VERSION,
        "type": packet_type,
        "timestamp": float(timestamp),
        "sequence": int(sequence),
        "source": source,
        "session_id": session_id,
        "payload": payload,
    }
    encoded = len(__import__("json").dumps(packet))
    if encoded > MAX_PACKET_BYTES:
        raise ValueError(f"packet of {encoded} bytes exceeds the client's limit.")
    return packet


def validate_packet(packet: dict) -> None:
    """Re-check a packet the way the client does; raise on anything it drops."""

    if not isinstance(packet, dict):
        raise ValueError("A packet must be an object.")
    if packet.get("version") != VERSION:
        raise ValueError("The client accepts version 1 only.")
    for field in ("type", "source", "session_id"):
        _text(packet.get(field), field)
    sequence = packet.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a nonnegative integer.")
    timestamp = packet.get("timestamp")
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be a finite nonnegative number.")
    if not isinstance(packet.get("payload"), dict):
        raise ValueError("payload must be an object.")


def session_packet(status: str, timestamp: float, sequence: int, session_id: str) -> dict:
    """The lifecycle packet the client reads to decide whether a session runs."""

    if status not in ("running", "stopped", "error"):
        raise ValueError("Unknown session status.")
    return make_packet("session", timestamp, sequence, session_id, "server", {"status": status})


def failure_packet(reason: str, timestamp: float, sequence: int, session_id: str, source: str) -> dict:
    """A producer failure carries a code, never a raw exception message."""

    return make_packet("error", timestamp, sequence, session_id, source, {"reason": reason})
