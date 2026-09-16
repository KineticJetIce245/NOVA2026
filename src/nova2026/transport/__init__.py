"""The attention transport: a version-1 packet protocol behind one same-origin app.

The modules are layered so that each one can be read - and tested - on its own:

* :mod:`nova2026.transport.protocol` - envelope construction and validation.
* :mod:`nova2026.transport.publisher` - thread-safe publication, bounded ring and
  the latest-state snapshot.
* :mod:`nova2026.transport.sessions` - serialized commands and the producer thread.
* :mod:`nova2026.transport.media` - the observed browser playback timeline.
* :mod:`nova2026.transport.server` - the FastAPI app, the WebSocket and the SPA
  mount. It is **not** imported here: it is the only module that needs the
  ``transport`` optional dependency group, so importing this package stays cheap
  for callers that only build packets.

    from nova2026.transport.server import create_app

Nothing in this package computes a measurement: it moves version-1 packets from a
producer to a client without inventing a timestamp, a decision or a fallback.
"""

from .media import MediaTimeline
from .protocol import MAX_PACKET_BYTES, VERSION, make_packet, validate_packet
from .publisher import DEFAULT_EVENT_LIMIT, DEFAULT_STATE_LIMIT, LaggedSubscriber, Publisher
from .sessions import RESERVED_SOURCE, Producer, Sessions

__all__ = [
    "DEFAULT_EVENT_LIMIT",
    "DEFAULT_STATE_LIMIT",
    "LaggedSubscriber",
    "MAX_PACKET_BYTES",
    "MediaTimeline",
    "Producer",
    "Publisher",
    "RESERVED_SOURCE",
    "Sessions",
    "VERSION",
    "make_packet",
    "validate_packet",
]
