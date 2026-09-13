"""Deterministic mock results — a test double, not a data source.

Three vendored frontend tests spawn an interpreter and feed these packets into
the browser-side validators, so this module exists to make the language-boundary
test real rather than to demonstrate anything. The values are fixed illustrations
and every payload says ``simulated: true``; a consumer that displays them as
measurements would be wrong, and the frontend says so on screen.

``src/nova2026/transport`` replaces this when the real producer lands.
"""

from __future__ import annotations

# The session identifier these results are published under. It matches the value
# the vendored tests pass to ``make_packet`` as their own session id: the client
# drops every stream when a packet arrives under a different session id, so a
# fixture that disagreed would have its lifecycle packet erase its own results.
SESSION_ID = "test"

# The order is part of the fixture: a frontend test asserts the decoded stream
# order, and the client keeps streams in latest-update order.
TYPES = (
    "audio_sources",
    "attention",
    "vigilance",
    "signal_quality",
    "sync",
    "eeg_display",
    "gain",
    "prediction",
)


def _audio_sources(index):
    return {
        "sources": [
            {"id": "A", "label": "Candidate A", "input_type": "development_fixture", "reference": None},
            {"id": "B", "label": "Candidate B", "input_type": "development_fixture", "reference": None},
        ],
        "simulated": True,
    }


def _attention(index):
    attended = "A" if (index // 16) % 2 == 0 else "B"
    first, second = (0.42, 0.19) if attended == "A" else (0.19, 0.42)
    return {
        "decision": attended,
        "attended": attended,
        "correlation_a": first,
        "correlation_b": second,
        "simulated": True,
    }


def _vigilance(index):
    # An explicitly simulated illustration, not a measured vigilance score.
    return {"score": 0.5, "metric": "vigilance", "lapse_score": 0.5, "simulated": True}


def _signal_quality(index):
    # Unavailable is null, never a measured zero.
    return {"quality": None, "artifact": None, "simulated": True}


def _sync(index):
    return {
        "status": "unknown",
        "offset_ms": None,
        "drift_warning": None,
        "timeline": "session_relative",
        "fixed_latency_ms": None,
        "simulated": True,
    }


def _eeg_display(index):
    samples = [round(0.4 * ((position + index) % 8 - 4) / 4, 3) for position in range(32)]
    return {
        "sample_rate": 32.0,
        "channels": ["display_only"],
        "samples": [samples],
        "simulated": True,
    }


def _gain(index):
    attended = "A" if (index // 16) % 2 == 0 else "B"
    return {
        "a_db": 0.0 if attended == "A" else -6.0,
        "b_db": -6.0 if attended == "A" else 0.0,
        "simulated": True,
    }


def _prediction(index):
    return {
        "status": "ok",
        "provider_id": "mock",
        "provider_name": "Mock producer",
        "provider_version": "1",
        "task": "transport_demo",
        "window_id": "window-0",
        "window_start": 0.0,
        "window_end": 1.0,
        "outputs": [
            {"name": "development_value", "value": 1, "semantic_type": "development", "label": "Illustration"},
        ],
        "reasons": [],
        "metadata": {"mock": True, "development_only": True},
        "simulated": True,
    }


BUILDERS = {
    "audio_sources": _audio_sources,
    "attention": _attention,
    "vigilance": _vigilance,
    "signal_quality": _signal_quality,
    "sync": _sync,
    "eeg_display": _eeg_display,
    "gain": _gain,
    "prediction": _prediction,
}


def results(index, *, types=TYPES):
    """Yield ``(type, timestamp, source, payload)`` rows for one sample index.

    Every row of one index shares its timestamp, so a consumer cannot rely on
    sub-sample ordering. Timestamps start at one second rather than zero: a
    frontend test injects a lifecycle packet at ``timestamp: 1`` after consuming
    one sample, and the client rejects a packet whose timestamp regresses below
    a stream's high-water mark. A producer that started at zero would have its
    own lifecycle packet dropped, which is a property of the fixture rather than
    of the transport.
    """

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a nonnegative integer.")
    timestamp = 1.0 + index * 0.25
    wanted = tuple(types)
    unknown = [name for name in wanted if name not in BUILDERS]
    if unknown:
        raise ValueError(f"Unknown result types: {unknown}.")
    for name in wanted:
        yield name, timestamp, SESSION_ID, BUILDERS[name](index)
