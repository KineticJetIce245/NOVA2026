"""Version-1 packet envelope: construction, validation and the size limit.

Every frame this process sends over ``/ws/live`` is one of these packets, and the
vendored frontend re-validates every field before it renders anything
(``apps/attune-ui/src/protocol.js``). The transport therefore has to be the
stricter of the two sides: whatever this module accepts must also survive the
client's validator, or the frame is dropped *after* the server counted it as
delivered. Validation is exhaustive rather than convenient for that reason -
non-finite numbers, non-string object keys, nesting deeper than
:data:`MAX_JSON_DEPTH` and encoded packets above :data:`MAX_PACKET_BYTES` are
refused instead of coerced.

Two fields carry meaning beyond their type:

* ``timestamp`` is nonnegative **session-relative** seconds, never wall-clock
  time. The transport has no authority over when a sample was captured, so it
  must not invent a timestamp either.
* ``sequence`` is assigned by the publisher, is a JavaScript-safe integer, and is
  the only ordering the stream has.

The packet is payload-agnostic: unknown types and extra envelope or payload fields
are preserved, and a validated packet is detached from the producer's objects, so
later mutation cannot rewrite what was already published. Values must be plain
JSON types - convert ``numpy`` scalars with ``float()``/``int()`` at the producer
boundary, because the strict type check is what keeps this module and the
browser's validator from disagreeing.
"""

from __future__ import annotations

import json
import math
from typing import Any

VERSION = 1
"""The only protocol version this module builds or accepts."""

MAX_TEXT = 128
"""Maximum length of the ``type``, ``source`` and ``session_id`` strings."""

MAX_JSON_DEPTH = 32
"""Maximum JSON nesting depth of a whole packet, the root envelope included."""

MAX_PACKET_BYTES = 256 * 1024
"""Maximum encoded packet size in bytes (256 KiB)."""

MAX_SAFE_INTEGER = 2**53 - 1
"""Largest integer JavaScript represents exactly; bounds sequence and timestamp."""


def _check_json_value(value: Any, depth: int = 0) -> None:
    """Reject anything that is not a finite, string-keyed JSON value.

    Args:
        value: Candidate value, walked recursively.
        depth: Nesting level of ``value`` below the packet root.

    Raises:
        ValueError: If ``value`` nests deeper than :data:`MAX_JSON_DEPTH`, holds a
            non-finite number, is keyed by a non-string, or is of a type JSON
            cannot represent.
    """

    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"packet nests deeper than {MAX_JSON_DEPTH} levels.")
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("packet numbers must be finite; NaN and Infinity are not JSON.")
        return
    if isinstance(value, list):
        for child in value:
            _check_json_value(child, depth + 1)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("packet object keys must be strings.")
            _check_json_value(child, depth + 1)
        return
    raise ValueError(
        "packet values must be JSON: null, boolean, string, finite number, list or object."
    )


def validate_packet(packet: object) -> dict[str, Any]:
    """Validate one version-1 packet and return a detached copy of it.

    Args:
        packet: Candidate packet, typically just decoded from JSON.

    Returns:
        A JSON round-tripped copy, detached from every object the caller passed in.

    Raises:
        ValueError: If an envelope field, the payload type, a JSON value, the
            nesting depth or the encoded size violates the version-1 contract. The
            message names the field that failed.
    """

    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object.")
    if type(packet.get("version")) is not int or packet["version"] != VERSION:
        raise ValueError(f"packet version must be the integer {VERSION}.")
    for field in ("type", "source", "session_id"):
        value = packet.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"packet {field} must be a nonempty string.")
        if len(value) > MAX_TEXT:
            raise ValueError(f"packet {field} must be at most {MAX_TEXT} characters.")
    sequence = packet.get("sequence")
    if type(sequence) is not int or not 0 <= sequence <= MAX_SAFE_INTEGER:
        raise ValueError(f"packet sequence must be an integer in [0, {MAX_SAFE_INTEGER}].")
    timestamp = packet.get("timestamp")
    if type(timestamp) not in (int, float) or not 0 <= timestamp <= MAX_SAFE_INTEGER:
        raise ValueError(
            "packet timestamp must be a finite, nonnegative number of session seconds."
        )
    if not isinstance(packet.get("payload"), dict):
        raise ValueError("packet payload must be a JSON object.")
    _check_json_value(packet)
    try:
        encoded = json.dumps(packet, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("packet must contain only finite JSON values.") from error
    if len(encoded.encode()) > MAX_PACKET_BYTES:
        raise ValueError(f"packet exceeds the {MAX_PACKET_BYTES}-byte limit.")
    # Detach mutable producer objects: the decoded copy is what gets published.
    return json.loads(encoded)


def make_packet(
    kind: str,
    timestamp: float,
    sequence: int,
    source: str,
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build one version-1 packet, validated before it can be published.

    The positional order is the order of the transport this was ported from, so
    that callers and tests written against it keep working unchanged.

    Args:
        kind: Packet type, a nonempty string of at most :data:`MAX_TEXT`.
        timestamp: Session-relative seconds, finite and nonnegative.
        sequence: Publisher-assigned, JavaScript-safe nonnegative integer.
        source: Producing stream, a nonempty string of at most :data:`MAX_TEXT`.
        session_id: Session the packet belongs to.
        payload: JSON object; its semantics are not interpreted here.

    Returns:
        The validated, detached packet.

    Raises:
        ValueError: If any field violates the version-1 contract.
    """

    return validate_packet(
        {
            "version": VERSION,
            "type": kind,
            "timestamp": timestamp,
            "sequence": sequence,
            "source": source,
            "session_id": session_id,
            "payload": payload,
        }
    )
