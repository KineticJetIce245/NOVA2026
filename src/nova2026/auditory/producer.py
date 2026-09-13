"""Adapt one :class:`~nova2026.auditory.session.AttentionSession` to the transport.

The transport's producer protocol is two things: ``run(publish, stop)`` and the
``simulated`` attribute that says whether this is a measurement. This module is
the only place where a session's :class:`SessionFrame` becomes packets, and every
payload here is written to the field names the vendored frontend actually reads
(``apps/attune-ui/src/decoders.js``), not to names that merely look right:

``audio_sources``
    Exactly two sources with ids ``A`` and ``B``; anything else decodes to an
    empty list and the focus card loses its labels.
``attention``
    ``decision`` in ``{A, B, uncertain, unavailable}`` plus the raw
    ``correlation_a``/``correlation_b``. The client prefers ``decision`` and
    falls back to ``attended``, so ``attended`` is only ever sent for a real A/B.
``gain``
    ``a_db``/``b_db``, attenuation only and never above 0 dB.
``signal_quality``
    ``quality`` and ``artifact``; ``null`` means "not judged yet", never zero.
``sync``
    ``status``, ``offset_ms``, ``drift_warning`` and the timeline it refers to.
    Without a media timeline this is ``unobserved``: reporting an offset nobody
    measured would be an invention, and the plan's rule is that the browser owns
    playback time (decision D-02).
``prediction``
    One packet per new piece of evidence, carrying the window bounds, the reason
    set and the run policy in its metadata. This is where ``evidence_gap``,
    ``warmup`` and ``evidence_stale`` become visible in the UI instead of being
    silently absent.

The transport already publishes the ``session`` lifecycle packets under the
reserved source ``server`` (``nova2026.transport.sessions``), and the frontend
decodes them from that source, so this producer deliberately publishes none:
duplicating them would fight the session record the transport owns.
"""

from __future__ import annotations

from typing import Any, Callable

PRODUCER_SOURCE = "nova-aad"
"""Packet source for this producer. ``server`` is reserved by the transport."""

SYNC_STATUS = "unobserved"
"""No browser has reported its clock yet, so no offset has been measured."""

SYNC_TIMELINE = "eeg_source_clock"
"""The timeline ``sync`` refers to until a media timeline is bound (step 9)."""

PREDICTION_STATUS = {
    "A": "ok",
    "B": "ok",
    "uncertain": "invalid",
    "unavailable": "unavailable",
}
"""Decision word to the frontend's ``prediction.status`` vocabulary."""

class AttentionProducer:
    """Run one session and publish its frames as version-1 packets.

    Args:
        session: The prepared session. Building it is where an envelope refusal
            surfaces, so a factory that raises here means the demo cannot start.
        source: Packet source name, recorded in every packet.
        media_reference: Optional callable returning
            ``{"media_id", "media_revision", "media_time_s"}`` for the packets
            that carry a media reference. ``None`` (the default) publishes no
            media fields at all, which is what makes the frontend treat the
            stream as having no media timeline rather than a fabricated one.
        provider_id: Identifier recorded in the ``prediction`` packets.
        provider_name: Human-readable provider name for the same packets.
        provider_version: Provider version string.

    Attributes:
        simulated: Whether this producer's data is a fixture. The transport reads
            it to label the session record; a producer without the attribute is
            recorded as a real measurement, so it is set explicitly from the
            session's source.
        summary: The :class:`SessionSummary` of the last run, or ``None``.
    """

    def __init__(
        self,
        session,
        *,
        source: str = PRODUCER_SOURCE,
        media_reference: Callable[[], dict[str, Any] | None] | None = None,
        provider_id: str = "nova2026.ridge",
        provider_name: str = "Ridge envelope decoder",
        provider_version: str = "1",
    ) -> None:
        if not isinstance(source, str) or not source or source == "server":
            raise ValueError("producer source must be a nonempty name other than 'server'.")
        self.session = session
        self.source = source
        self.media_reference = media_reference
        self.provider_id = str(provider_id)
        self.provider_name = str(provider_name)
        self.provider_version = str(provider_version)
        self.simulated = bool(session.simulated)
        self.summary = None
        self._last_evidence = 0
        self._last_reasons: tuple[str, ...] | None = None

    # ------------------------------------------------------------- producer API

    def run(self, publish, stop) -> None:
        """Publish this session until it ends or ``stop`` is set.

        Raises:
            RuntimeError: If the session ended because the chain refused to
                continue. The transport turns that into a session record with
                status ``error``, which is the explicit failure the plan requires;
                the last published frame already says ``unavailable``.
        """

        self._publish(publish, "audio_sources", 0.0, self._audio_sources())
        self._publish(publish, "sync", 0.0, self._sync(()))
        self.summary = self.session.run(
            lambda frame: self._emit_frame(publish, frame), stop
        )
        failure = self.summary.failure
        if failure is not None:
            raise RuntimeError(
                f"session ended with {failure['code']} at {failure['timestamp']:g}s"
            )

    # ------------------------------------------------------------------ packets

    def _publish(self, publish, kind: str, timestamp: float, payload: dict) -> None:
        """Publish one packet, tolerating a session that is already closing."""

        publish(kind, float(timestamp), self.source, payload)

    def _media(self) -> dict[str, Any]:
        """Media reference fields, or an empty mapping when there is no timeline."""

        if self.media_reference is None:
            return {}
        reference = self.media_reference()
        return dict(reference) if reference else {}

    def _audio_sources(self) -> dict[str, Any]:
        """The two candidate identities the focus card needs, and the run policy."""

        sources = [
            {
                "id": candidate.id,
                "label": candidate.label,
                "input_type": self.session.kind,
                "reference": candidate.name,
            }
            for candidate in self.session.references.candidates
        ]
        return {
            "sources": sources,
            "policy": self.session.policy.to_dict(),
            "simulated": self.simulated,
        }

    def _sync(self, reasons) -> dict[str, Any]:
        """The synchronization state; ``null`` fields mean "not measured"."""

        return {
            "status": SYNC_STATUS,
            "offset_ms": None,
            "drift_warning": None,
            "timeline": SYNC_TIMELINE,
            "fixed_latency_ms": None,
            "reasons": list(reasons),
            "simulated": self.simulated,
        }

    def _attention(self, frame) -> dict[str, Any]:
        """The decision, the two raw correlations, and the media reference."""

        payload: dict[str, Any] = {
            "decision": frame.decision,
            "correlation_a": frame.correlation_a,
            "correlation_b": frame.correlation_b,
            "reasons": list(frame.reasons),
            "simulated": self.simulated,
        }
        if frame.decision in ("A", "B"):
            # Only a real decision gets ``attended``: the client uses it as the
            # fallback for older producers and must never see a guess there.
            payload["attended"] = frame.decision
        return payload

    def _gain(self, frame) -> dict[str, Any]:
        """Per-candidate attenuation; never above 0 dB."""

        return {
            "a_db": min(0.0, frame.gain_a_db),
            "b_db": min(0.0, frame.gain_b_db),
            "simulated": self.simulated,
        }

    def _quality(self, frame) -> dict[str, Any]:
        """Signal quality; ``null`` until a window has been judged.

        ``bad_channels`` rides along because ``artifact`` alone cannot be audited:
        under the relaxed quality policy the chain's reason set is empty for a
        dead electrode, so the census is the only field that names it (plan
        section 3.17 item 6). It is evidence for display and for the run record,
        never a filter.
        """

        return {
            "quality": frame.quality,
            "artifact": frame.artifact,
            # A frame-like object from an older caller need not carry the census;
            # absence means "no channel was flagged", which is what an empty list
            # says. Reporting a missing attribute as a channel would be worse.
            "bad_channels": list(getattr(frame, "bad_channels", ()) or ()),
            "simulated": self.simulated,
        }

    def _prediction(self, frame) -> dict[str, Any]:
        """One evidence record: window bounds, reasons and the policy behind it."""

        window_end = frame.window_end
        window_start = (
            None if window_end is None else window_end - self.session.window_seconds
        )
        outputs = [
            {
                "name": "correlation_a",
                "value": frame.correlation_a,
                "semantic_type": "correlation",
                "label": "Candidate A",
            },
            {
                "name": "correlation_b",
                "value": frame.correlation_b,
                "semantic_type": "correlation",
                "label": "Candidate B",
            },
            {
                "name": "decision",
                "value": frame.decision,
                "semantic_type": "decision",
                "label": "Attended candidate",
            },
        ]
        return {
            "status": PREDICTION_STATUS[frame.decision],
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "task": "auditory_attention",
            "timestamp": frame.timestamp,
            "window_id": f"w{frame.evidence_count}",
            "window_start": window_start,
            "window_end": window_end,
            "outputs": outputs,
            "reasons": list(frame.reasons),
            "metadata": {
                "decision": frame.decision,
                "evidence_count": frame.evidence_count,
                "media_time_s": frame.media_time_s,
                "policy": self.session.policy.to_dict(),
                "source_kind": self.session.kind,
                "simulated": self.simulated,
            },
            "simulated": self.simulated,
        }

    def _emit_frame(self, publish, frame) -> None:
        """Publish one frame: state every frame, evidence and sync when they change.

        The media reference is resolved **once** per frame and stamped onto both
        the attention and the gain packet. That is not tidiness: the frontend's
        ``mediaFocusReady`` compares the attention packet's position with the
        playback clock, and ``playbackGains`` additionally demands that the gain
        packet's position equals the attention packet's to the bit. Two separate
        reads of a timeline the controller is still reporting into could differ by
        one 250 ms step, and the gain path would then fall back to neutral while
        every field still looked correct - the exact failure mode decision D-02 is
        about.
        """

        media = self._media()
        attention = self._attention(frame)
        attention.update(media)
        gain = self._gain(frame)
        gain.update(media)
        self._publish(publish, "attention", frame.timestamp, attention)
        self._publish(publish, "gain", frame.timestamp, gain)
        self._publish(publish, "signal_quality", frame.timestamp, self._quality(frame))
        if frame.evidence_count != self._last_evidence:
            self._last_evidence = frame.evidence_count
            # A window that produced no evidence still gets a record, so the
            # reason is visible instead of the previous decision staying on screen.
            self._publish(
                publish,
                "prediction",
                frame.timestamp,
                self._prediction(frame),
            )
        if frame.reasons != self._last_reasons:
            self._last_reasons = frame.reasons
            self._publish(publish, "sync", frame.timestamp, self._sync(frame.reasons))

__all__ = [
    "PREDICTION_STATUS",
    "PRODUCER_SOURCE",
    "SYNC_STATUS",
    "SYNC_TIMELINE",
    "AttentionProducer",
]
