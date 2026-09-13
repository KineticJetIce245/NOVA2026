"""Versioned, payload-agnostic transport. Times use session-relative seconds."""
import json
import math

MAX_PACKET_BYTES = 262144


def _json_value(value, depth=0):
    if depth > 32:
        raise ValueError('JSON nesting too deep')
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for child in value:
            _json_value(child, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for child in value.values():
            _json_value(child, depth + 1)
        return
    raise ValueError('packet must contain finite JSON values with string keys')


def validate_packet(packet):
    if not isinstance(packet, dict):
        raise ValueError('packet must be an object')
    if type(packet.get('version')) is not int or packet['version'] != 1:
        raise ValueError('unsupported version')
    for key in ('type', 'source', 'session_id'):
        if not isinstance(packet.get(key), str) or not 0 < len(packet[key]) <= 128:
            raise ValueError('invalid ' + key)
    if type(packet.get('sequence')) is not int or not 0 <= packet['sequence'] <= 9007199254740991:
        raise ValueError('invalid sequence')
    timestamp = packet.get('timestamp')
    if type(timestamp) not in (int, float) or not 0 <= timestamp <= 9007199254740991:
        raise ValueError('invalid timestamp')
    if not isinstance(packet.get('payload'), dict):
        raise ValueError('payload must be an object')
    _json_value(packet)
    try:
        encoded = json.dumps(packet, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('packet must contain finite JSON values') from exc
    if len(encoded.encode()) > MAX_PACKET_BYTES:
        raise ValueError('packet too large')
    return json.loads(encoded)  # Detach mutable producer objects.


def make_packet(kind, timestamp, sequence, source, session_id, payload):
    return validate_packet(dict(version=1, type=kind, timestamp=timestamp,
                                sequence=sequence, source=source, session_id=session_id, payload=payload))
