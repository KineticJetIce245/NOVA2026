"""Run one attention session through the real transport and report the packets.

This is the step-8 evidence command. It starts the actual FastAPI app on a
loopback port, starts a session through ``POST /api/session/start``, subscribes to
``/ws/live`` with a real WebSocket client, and then asserts what arrived:

* every packet passes this repository's own version-1 validator,
* the REST snapshot is delivered before any packet newer than its cursor,
* the packet types the vendored frontend decodes are all present,
* ``decision`` is one of the four allowed words and gains never exceed 0 dB,
* the ``session`` lifecycle comes from the transport (source ``server``) and the
  stop packet ends the stream,
* the producer thread stops, and the session record says so.

The label is used **only** after the fact, to report which way the published
decisions went; it never enters the session, and the numbers this prints are a
plumbing result, not an accuracy claim.

    python -B -m scripts.auditory_ui.session --synthetic --out results/auditory_session_synthetic.json
    python -B -m scripts.auditory_ui.session --trial datasets/AAD-KULeuven/converted/S1/trial_008.npz \\
        --model models/auditory_kuleuven.npz --speed 1 --out results/auditory_session_trial008.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/auditory_ui/session.py`
    sys.path.insert(0, str(REPO))

from nova2026.auditory.config import AuditoryConfig  # noqa: E402
from nova2026.auditory.controller import AttentionController  # noqa: E402
from nova2026.auditory.data import AuditoryTrial, load_trial  # noqa: E402
from nova2026.auditory.decoder import RidgeDecoder  # noqa: E402
from nova2026.auditory.producer import AttentionProducer  # noqa: E402
from nova2026.auditory.render import (  # noqa: E402
    DEFAULT_CROSSMIX_WEIGHT,
    DEFAULT_PRESENTATION,
    PRESENTATION_MODES,
    presentation_mode,
    render_stereo,
)
from nova2026.auditory.session import AttentionSession, RunPolicy  # noqa: E402
from nova2026.auditory.sources import (  # noqa: E402
    ReferenceEnvelopes,
    ReplaySource,
    trial_envelope_paths,
)
from nova2026.auditory.timing import TimestampedAudio  # noqa: E402
from nova2026.transport.media import MediaTimeline  # noqa: E402
from nova2026.transport.protocol import validate_packet  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.auditory.envelopes import convert_audio, save_envelope  # noqa: E402
from scripts.auditory.synthetic import synthetic_trial  # noqa: E402
from scripts.auditory.train import train  # noqa: E402
from scripts.auditory_ui.media import MEDIA_TICK_SECONDS, SimulatedMediaClient  # noqa: E402

DECISIONS = ("A", "B", "uncertain", "unavailable")
REQUIRED_TYPES = (
    "session",
    "audio_sources",
    "attention",
    "gain",
    "signal_quality",
    "sync",
    "prediction",
)
TERMINAL_STATUSES = ("stopped", "error")


def free_port() -> int:
    """An unused loopback port. The server binds this exact port, host below."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def clip_trial(trial: AuditoryTrial, seconds: float | None) -> AuditoryTrial:
    """Trim a trial to its first ``seconds``, keeping every array consistent."""

    if seconds is None or seconds >= trial.timestamps[-1] - trial.timestamps[0]:
        return trial
    stop = float(trial.timestamps[0]) + float(seconds)
    count = int(np.searchsorted(trial.timestamps, stop, side="right"))
    if count < 2:
        raise ValueError("--seconds leaves too few samples for a session.")
    audio_count = min(len(trial.audio), int(round(seconds * trial.audio_rate)))
    return AuditoryTrial(
        trial.eeg[:count],
        trial.timestamps[:count],
        trial.audio[:audio_count],
        trial.audio_rate,
        trial.labels[:count],
        trial.subject,
        trial.trial_id,
        trial.channel_names,
        trial.reference,
        trial.upstream_processing,
        trial.group,
    )


def fixture_trial(directory: Path, seconds: float, config: AuditoryConfig):
    """Write a synthetic trial's two candidates as WAVs plus offline envelopes.

    The fixture exists so a session can be exercised without a dataset, and it
    keeps the plan's rule intact: the envelopes are generated *before* the session
    starts, stored, and then verified at load (source hash included), never
    computed at run time.

    Returns:
        ``(trial, envelope_paths)`` - the trial to replay and the two verified
        envelope files.
    """

    directory.mkdir(parents=True, exist_ok=True)
    trial = synthetic_trial("session", seconds=int(seconds))
    names = []
    from scipy.io import wavfile

    for index, identity in enumerate(("a", "b")):
        target = directory / f"candidate_{identity}.wav"
        samples = np.clip(trial.audio[:, index], -1.0, 1.0)
        wavfile.write(target, round(trial.audio_rate), (samples * 32767).astype(np.int16))
        names.append(target.name)
        envelope, timestamps, metadata = convert_audio(target, config)
        save_envelope(directory / f"candidate_{identity}.npz", envelope, timestamps, metadata)
    # The trial's group names its two candidates, exactly as a converted KU
    # Leuven trial does, so the envelope lookup is the same code path.
    trial.group = "|".join(names)
    return trial, (directory / "candidate_a.npz", directory / "candidate_b.npz")


def build_session(args, trial, model, envelope_paths):
    """Assemble the session the producer will run, or refuse to."""

    references = ReferenceEnvelopes.load(envelope_paths, model.config)
    source = ReplaySource(
        trial,
        speed=args.speed,
        simulated=args.synthetic,
        kind="synthetic_replay" if args.synthetic else "kuleuven_replay",
    )
    policy = RunPolicy(
        check_channels=False if not args.strict_policy else True,
        max_bad_channels=0,
        warmup_seconds=args.warmup,
        frame_seconds=args.frame_seconds,
    )
    controller = AttentionController(margin=args.margin)
    session = AttentionSession(
        source=source,
        decoder=model,
        references=references,
        policy=policy,
        controller=controller,
    )
    return session


async def collect(app, port: int, args, packets: list, media_client=None) -> dict:
    """Drive the running server: REST snapshot, start, WebSocket, stop.

    With a ``media_client`` the media path is driven too, and by a **separate
    concurrent client** rather than inline: the browser is not the EEG, so its
    control requests must interleave with the packet stream instead of being
    inserted between two packets by the driver's own scheduling. The collector
    sets ``media_stop`` when the terminal lifecycle packet arrives, which is what
    ends playback and any pending report.
    """

    import httpx
    import websockets

    base = f"http://127.0.0.1:{port}"
    media_stop = asyncio.Event()
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        for _ in range(200):
            try:
                if (await client.get("/api/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                await asyncio.sleep(0.05)
        else:
            raise RuntimeError("the server never became ready on loopback")

        health = (await client.get("/api/health")).json()
        static = None
        if args.static_dir is not None:
            response = await client.get("/")
            static = {
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "bytes": len(response.content),
            }
        started = (await client.post("/api/session/start")).json()
        # The snapshot and the cursor are read together, which is what the
        # delivery loop promises: snapshot first, then only newer events.
        snapshot = (await client.get("/api/state")).json()

        media_task = None
        media_state: dict = {}
        lifecycle = {
            "health": health,
            "static": static,
            "started": started,
            "snapshot": snapshot,
            "terminal": None,
            "stopped": None,
            "sessions": None,
            "media_stop": media_stop,
            "media_state": media_state,
        }

        async def watch() -> None:
            """Drain the socket until the session's terminal packet arrives.

            Two orderings are recorded per packet besides the packet itself: the
            receive order (so a later step can line the stream up with the packet
            file byte for byte) and the *client's* playback clock at that instant.
            The latter is the only honest source for the gate's ``|Δt|``: the
            browser measures its own position, so a report that arrives while the
            packet is in flight must not be compared against the position the
            packet carried as if they were simultaneous.
            """

            terminal = None
            order = 0
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/ws/live", max_size=None, open_timeout=10
            ) as socket:
                while True:
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=args.idle_timeout)
                    except asyncio.TimeoutError:
                        break
                    packet = json.loads(raw)
                    packets.append(packet)
                    order += 1
                    if packet.get("type") == "gain" and media_client is not None:
                        snapshot = media_state.get("snapshot")
                        media_state.setdefault("at_gain_packet", []).append(
                            {
                                "order": order,
                                "timestamp": packet.get("timestamp"),
                                "payload": packet.get("payload", {}),
                                "client": snapshot() if callable(snapshot) else {},
                            }
                        )
                    if is_terminal(packet):
                        terminal = packet
                        break
            lifecycle["terminal"] = terminal

        tasks = [asyncio.create_task(watch())]
        if media_client is not None:
            media_task = asyncio.create_task(media_client(port, lifecycle))
            tasks.append(media_task)
        try:
            await tasks[0]
        finally:
            media_stop.set()
            if media_task is not None:
                media_state.update(await media_task)
            for task in tasks[1:]:
                task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)

        stopped = (await client.post("/api/session/stop")).json()
        sessions = (await client.get("/api/sessions")).json()

    lifecycle["stopped"] = stopped
    lifecycle["sessions"] = sessions
    return lifecycle


def is_terminal(packet: dict) -> bool:
    """Whether a packet is the transport's own end-of-session record."""

    return (
        packet.get("source") == "server"
        and packet.get("type") == "session"
        and packet.get("payload", {}).get("status") in TERMINAL_STATUSES
    )


def check(name: str, ok: bool, detail: str = "") -> dict:
    """One assertion record; the exit code is derived from these."""

    return {"name": name, "ok": bool(ok), "detail": detail}


def render_media(trial: AuditoryTrial, args) -> tuple[MediaTimeline, dict]:
    """Render the trial's two candidates as the stereo file the browser plays.

    The file is the trial's own two channels, cut to the EEG's length - the
    recording is what the session replays, and the browser's ``currentTime`` is
    session position, so a longer asset would drift out of the window the
    frontend enforces. Per section 3.3/3.15 the left channel is candidate A and
    the right is candidate B; ``--presentation crossmix`` renders the interim
    "same mixture, differently weighted" route instead, and the mode is recorded
    in the report either way.

    Returns:
        The bound-less timeline over the rendered file, and the render report.
    """

    mode = presentation_mode(args.presentation)
    report = render_stereo(
        trial.audio,
        args.media_out,
        sample_rate=round(trial.audio_rate),
        presentation=mode,
        crossmix_weight=args.crossmix_weight,
    )
    report["candidate_seconds"] = round(float(len(trial.audio)) / trial.audio_rate, 3)
    report["eeg_seconds"] = round(float(trial.timestamps[-1] - trial.timestamps[0]), 3)
    return MediaTimeline(args.media_out, args.media_title), report


async def run_media_client(port: int, args, media: MediaTimeline, lifecycle: dict, media_report: dict) -> dict:
    """Play the rendered timeline over REST from a client that behaves like the browser.

    The duration is the *rendered file's* own length, not the source recording's:
    the browser reads ``element.duration`` from the element it loaded, and the
    transport refuses a command whose position exceeds the duration it was told,
    so the two must be the same number.
    """

    descriptor = media.descriptor()
    client = SimulatedMediaClient(
        f"http://127.0.0.1:{port}",
        client_id="simulated-browser",
        duration_s=media_report["seconds"],
        clip_seconds=args.seconds,
        tick_seconds=args.media_tick,
    )
    client.configure(lifecycle["started"]["id"], descriptor["media_id"])
    # The playback clock has to be readable by the packet watcher at every
    # instant, because a gain packet is judged against the position the browser
    # held when that packet arrived - not against the position it carries. This
    # is the same construction `mediaController.js` exports as its snapshot.
    lifecycle["media_state"]["snapshot"] = client.playback_snapshot
    result = await client.play(lifecycle["media_stop"])
    return {
        "descriptor": descriptor,
        "duration_s": client.duration_s,
        "client": result.to_dict(),
    }


def timing_evidence(packets: list, media_state: dict, exchange_log: list) -> dict:
    """The `|Δt|` and drift numbers this step is asked to record.

    ``|Δt|`` is measured per gain packet between the media position the packet
    carries and the position the client would have reported when that packet
    arrived - the browser's own clock, sampled live, which is the quantity the
    frontend's 0.75 s clause compares. Sampling it at arrival rather than pairing
    frame *k* with report *k* afterwards is what keeps the number honest: the
    client's report cadence and the producer's frame cadence are independent, and
    their phase offset can reach a full step.
    """

    gains = [packet for packet in packets if packet.get("type") == "gain"]
    frames = [packet for packet in packets if packet.get("type") == "attention"]
    acknowledged = [entry for entry in exchange_log if entry.get("status") == 200]
    arrivals = media_state.get("at_gain_packet") or []
    deltas = []
    for index, packet in enumerate(gains):
        arrival = arrivals[index] if index < len(arrivals) else None
        client_time = ((arrival or {}).get("client") or {}).get("time")
        value = packet.get("payload", {}).get("media_time_s")
        if isinstance(value, (int, float)) and isinstance(client_time, (int, float)):
            deltas.append(abs(float(value) - float(client_time)))
    positions = [
        packet["payload"]["media_time_s"]
        for packet in frames
        if isinstance(packet.get("payload", {}).get("media_time_s"), (int, float))
    ]
    steps = np.diff(positions) if len(positions) > 1 else np.zeros(0)
    latest = frames[-1]["payload"] if frames else {}
    return {
        "gain_packets": len(gains),
        "attention_packets": len(frames),
        "gain_packets_carrying_a_media_reference": sum(
            1 for packet in gains if "media_time_s" in packet.get("payload", {})
        ),
        "delta_seconds": {
            "count": len(deltas),
            "max": max(deltas) if deltas else None,
            "mean": float(np.mean(deltas)) if deltas else None,
            "min": min(deltas) if deltas else None,
            "within_075": all(value <= 0.75 for value in deltas) if deltas else None,
        },
        "media_time_step_seconds": {
            "count": int(steps.size),
            "max": float(steps.max()) if steps.size else None,
            "mean": float(steps.mean()) if steps.size else None,
        },
        "frame_positions_first_last": (
            [positions[0], positions[-1]] if positions else None
        ),
        "last_frame": {
            "decision": latest.get("decision"),
            "media_id": latest.get("media_id"),
            "media_revision": latest.get("media_revision"),
            "media_time_s": latest.get("media_time_s"),
            "a_db": (packets[-1].get("payload", {}).get("a_db") if packets else None),
        },
        "transport_timeline": media_state.get("timeline"),
        "round_trip_seconds": {
            "count": len(acknowledged),
            "max": max((entry["round_trip_seconds"] for entry in acknowledged), default=None),
            "mean": (
                float(np.mean([entry["round_trip_seconds"] for entry in acknowledged]))
                if acknowledged
                else None
            ),
        },
        "envelope_coverage_seconds": media_state.get("envelope_coverage_seconds"),
        "audio_anchor_seconds": media_state.get("audio_anchor_seconds"),
        "block_timing": {
            "available": False,
            "why": (
                "TimestampedAudio.diagnostics() describes DAC block timestamps from a real "
                "audio device; it is not on this replay path, which is fed from the "
                "converted trial, so reporting drift here would be an invention. "
                "It belongs to the live/device path (steps 11 and 12)."
            ),
        },
        "replay_clock": media_state.get("replay_clock"),
    }


def media_assertions(media_state: dict, timing: dict) -> list:
    """The media-specific assertions, alongside the session's own checks."""

    exchanges = media_state.get("client", {}).get("exchanges", [])
    actions = [entry["action"] for entry in exchanges]
    acknowledged = [entry for entry in exchanges if entry.get("status") == 200]
    descriptor = media_state.get("descriptor") or {}
    revisions = {entry.get("acknowledged_revision") for entry in acknowledged}
    return [
        check(
            "the media descriptor names an id, a kind and the file route",
            bool(descriptor.get("media_id"))
            and descriptor.get("kind") == "audio"
            and descriptor.get("url") == "/api/media/file",
            json.dumps(descriptor),
        ),
        check(
            "the handshake ran prepare then playing before any report",
            actions[:2] == ["prepare", "playing"],
            ", ".join(actions[:4]),
        ),
        check(
            "every acknowledgement carried the observed sync status and an integer revision",
            all(
                entry.get("sync_status") == "observed"
                and isinstance(entry.get("acknowledged_revision"), int)
                for entry in acknowledged
            )
            and all(entry.get("status") == 200 for entry in exchanges),
            f"{len(acknowledged)}/{len(exchanges)} acknowledged, revisions={sorted(revisions)}",
        ),
        check(
            "prepare bumped the revision and every later command kept it",
            len(revisions) == 1 and acknowledged and acknowledged[0]["action"] == "prepare",
            f"revision={sorted(revisions)}",
        ),
        check(
            "reports advanced the timeline monotonically",
            all(
                later["media_time_s"] >= earlier["media_time_s"]
                for earlier, later in zip(acknowledged, acknowledged[1:])
            ),
            f"{len(acknowledged)} commands",
        ),
        check(
            "the attention and gain packets carried a media reference",
            timing["gain_packets_carrying_a_media_reference"] == timing["gain_packets"]
            and timing["gain_packets"] > 0,
            f"{timing['gain_packets_carrying_a_media_reference']}/{timing['gain_packets']} gain packets",
        ),
        check(
            "|Δt| between the echoed position and the client's own time stayed under 0.75 s",
            bool(timing["delta_seconds"]["within_075"]),
            json.dumps(timing["delta_seconds"]),
        ),
        check(
            "the media packet stream reached the client with the observed timeline",
            media_state.get("media_packets", 0) > 0
            and media_state.get("timeline", {}).get("sync_status") in ("observed", "desynchronized"),
            f"{media_state.get('media_packets', 0)} media packets, "
            f"last sync_status={media_state.get('timeline', {}).get('sync_status')}",
        ),
        check(
            "the reference envelopes cover the played range and the anchor is a real number",
            bool(media_state.get("envelope_coverage_seconds"))
            and isinstance(media_state.get("audio_anchor_seconds"), (int, float)),
            json.dumps(
                {
                    "coverage": media_state.get("envelope_coverage_seconds"),
                    "anchor": media_state.get("audio_anchor_seconds"),
                }
            ),
        ),
    ]


def inspect(packets: list, lifecycle: dict, summary: dict) -> list:
    """Every assertion this command makes, as records rather than bare asserts."""

    checks = []
    by_type: dict[str, int] = {}
    sequences = []
    for packet in packets:
        by_type[packet["type"]] = by_type.get(packet["type"], 0) + 1
        sequences.append(packet["sequence"])

    try:
        for packet in packets:
            validate_packet(packet)
        checks.append(check("every packet passes the version-1 validator", True, f"{len(packets)} packets"))
    except ValueError as error:
        checks.append(check("every packet passes the version-1 validator", False, str(error)))

    missing = [name for name in REQUIRED_TYPES if name not in by_type]
    checks.append(
        check(
            "all packet types the frontend decodes are present",
            not missing,
            "missing: " + ", ".join(missing) if missing else ", ".join(sorted(by_type)),
        )
    )

    snapshot = lifecycle["snapshot"]
    cursor = snapshot["sequence"]
    older = [p["sequence"] for p in packets if p["sequence"] <= cursor]
    newer = [p["sequence"] for p in packets if p["sequence"] > cursor]
    snapshot_first = not older or not newer or max(older) < min(newer)
    checks.append(
        check(
            "the snapshot is delivered before newer events",
            snapshot_first,
            f"cursor={cursor}, snapshot packets={len(older)}, later={len(newer)}",
        )
    )
    checks.append(
        check(
            "the snapshot's streams all arrive",
            set(range(1, cursor + 1)) <= set(sequences) or cursor == 0,
            f"cursor={cursor}, received {len(set(sequences))} distinct sequences",
        )
    )
    checks.append(
        check(
            "sequence is strictly increasing across the stream",
            all(b > a for a, b in zip(sequences, sequences[1:])),
            f"{len(sequences)} packets",
        )
    )

    decisions = {p["payload"].get("decision") for p in packets if p["type"] == "attention"}
    checks.append(
        check(
            "decision is one of the four allowed words",
            decisions and decisions <= set(DECISIONS),
            ", ".join(sorted(str(word) for word in decisions)),
        )
    )
    gains = [
        p["payload"][key]
        for p in packets
        if p["type"] == "gain"
        for key in ("a_db", "b_db")
    ]
    checks.append(
        check(
            "gains are attenuation only (never above 0 dB)",
            bool(gains) and all(value is None or value <= 1e-9 for value in gains),
            f"{len(gains)} gain values, max {max(gains) if gains else None}",
        )
    )
    sources = [
        p["payload"].get("sources") for p in packets if p["type"] == "audio_sources"
    ]
    checks.append(
        check(
            "audio_sources carries exactly the two candidates A and B",
            bool(sources)
            and all(
                [s["id"] for s in value] == ["A", "B"] for value in sources if value
            ),
            json.dumps(sources[-1]) if sources else "absent",
        )
    )
    session_packets = [p for p in packets if p["type"] == "session"]
    checks.append(
        check(
            "the session lifecycle is published by the transport, not duplicated",
            bool(session_packets)
            and all(p["source"] == "server" for p in session_packets)
            and session_packets[0]["payload"]["status"] == "running"
            and is_terminal(session_packets[-1]),
            f"{len(session_packets)} lifecycle packets, last "
            f"{session_packets[-1]['payload']['status'] if session_packets else 'none'}",
        )
    )
    checks.append(
        check(
            "the stream ends with the terminal session packet",
            not packets or is_terminal(packets[-1]),
            packets[-1]["type"] if packets else "no packets",
        )
    )
    checks.append(
        check(
            "the producer stopped and the session record agrees",
            lifecycle["stopped"] is not None
            and lifecycle["stopped"]["status"] in TERMINAL_STATUSES,
            json.dumps(lifecycle["stopped"]),
        )
    )
    checks.append(
        check(
            "no packet was published after the stop record",
            lifecycle["stopped"] is not None
            and lifecycle["stopped"].get("status") == "stopped",
            f"session status {lifecycle['stopped'].get('status') if lifecycle['stopped'] else None}",
        )
    )
    checks.append(
        check(
            "the run policy is recorded with the run",
            summary["policy"]["check_channels"] is False
            and summary["policy"]["max_bad_channels"] == 0,
            json.dumps(summary["policy"]),
        )
    )
    checks.append(
        check(
            "the enabled chain judged windows",
            summary["windows"] > 0 and (summary["scored"] + summary["invalid"]) == summary["windows"],
            f"windows={summary['windows']} scored={summary['scored']} invalid={summary['invalid']}",
        )
    )
    return checks


def evaluate_gate(packets: list, media_state: dict, record_path: Path, args):
    """Run the frontend's own ``mediaFocusReady`` over this run, in Node.

    The gate is a JavaScript function over a client state object, so the only
    faithful evaluation is to execute it: :mod:`scripts.auditory_ui.gate_evidence`
    replays the recorded packets through the vendored frontend's validator and
    state machine and calls the function from ``mediaAudio.js``. A writer's
    absence is not a failure of this run - it is reported as "not evaluated"
    rather than as a pass - but a falsified gate is, and it is what the caller's
    exit code then reflects.
    """

    script = Path(args.gate_script)
    if not script.is_file():
        return None
    stream_path = Path(args.stream_out)
    try:
        completed = subprocess.run(
            [
                "node",
                str(script),
                str(stream_path),
                str(record_path),
                str(args.gate_out),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"gate evidence could not run: {error}")
        return {"gate_open": False, "clauses": [], "error": str(error)}
    if completed.returncode != 0 and not Path(args.gate_out).is_file():
        print(f"gate evidence exited {completed.returncode}: {completed.stderr.strip()[:400]}")
        return {"gate_open": False, "clauses": [], "error": completed.stderr.strip()[:400]}
    return json.loads(Path(args.gate_out).read_text())


def plumbing(packets: list, trial: AuditoryTrial) -> dict:
    """Post-hoc comparison of published decisions with the trial's own label.

    This runs in the report, after the stream is closed, on purpose: the label is
    the one thing that must never reach the decision path, and the only way to
    demonstrate that is for the code that reads it to run afterwards.
    """

    frames = [
        (p["timestamp"], p["payload"]["decision"])
        for p in packets
        if p["type"] == "attention"
    ]
    if not frames or len(trial.labels) == 0:
        return {"label": None, "decided": 0, "agree": 0, "note": "no decisions"}
    index = np.clip(
        np.searchsorted(trial.timestamps, np.array([f[0] for f in frames]), side="right") - 1,
        0,
        len(trial.labels) - 1,
    )
    truth = trial.labels[index]
    words = {"A": 0, "B": 1}
    decided = [(truth[i], words[word]) for i, (_, word) in enumerate(frames) if word in words]
    agree = sum(1 for label, choice in decided if label == choice)
    return {
        "label": sorted({int(value) for value in truth.tolist() if value in (0, 1)}),
        "decided": len(decided),
        "agree": agree,
        "note": "plumbing only: a window-level agreement count is not an accuracy claim",
    }


def excerpt(packets: list) -> dict:
    """A bounded view of the stream: ends, decision changes, and type counts."""

    by_type: dict[str, int] = {}
    for packet in packets:
        by_type[packet["type"]] = by_type.get(packet["type"], 0) + 1
    transitions = []
    previous = None
    for packet in packets:
        if packet["type"] != "attention":
            continue
        decision = packet["payload"]["decision"]
        if decision != previous:
            transitions.append(packet)
            previous = decision
    return {
        "total": len(packets),
        "by_type": by_type,
        "first": packets[:3],
        "last": packets[-2:],
        "transitions": transitions[:40],
        "transition_count": len(transitions),
    }


async def main_async(args) -> int:
    config = AuditoryConfig()
    if args.synthetic:
        trial, envelope_paths = fixture_trial(
            Path(args.fixture_dir), args.seconds, config
        )
        # Training, validation and replay are three different fixtures: the
        # repository's own split guard refuses a reused trial, and a hold-out
        # split is the habit worth keeping even for a plumbing fixture.
        model, _ = train(
            [synthetic_trial("session-training", seconds=int(args.seconds))],
            [synthetic_trial("session-validation", seconds=int(args.seconds))],
            alphas=(100.0,),
        )
        trial = clip_trial(trial, args.seconds)
    else:
        model = RidgeDecoder.load(args.model)
        trial = clip_trial(load_trial(args.trial), args.seconds)
        envelope_paths = trial_envelope_paths(trial, args.envelope_dir)

    session = build_session(args, trial, model, envelope_paths)
    producer = AttentionProducer(session)

    # The media path is opt-in: without ``--media-out`` this run is exactly the
    # step-8 evidence command, with no audio file, no timeline and no media
    # reference in any packet (the frontend then holds neutral gain, correctly).
    media_report: dict | None = None
    media = None
    if args.media_out:
        media, media_report = render_media(trial, args)
        producer.media_reference = media.media_reference
        media_report["title"] = args.media_title
        media_report["media_id"] = media.media_id
        media_report["descriptor"] = media.descriptor()

    app = create_app(
        producer_factory=lambda: producer, media=media, static_dir=args.static_dir
    )

    def media_client(port: int, lifecycle: dict):
        """Bind this run's media objects to the generic client coroutine."""

        return run_media_client(port, args, media, lifecycle, media_report)

    port = free_port()
    if args.host not in LOOPBACK_HOSTS:
        raise SystemExit(f"the demo binds loopback only; refusing host {args.host!r}")
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=port,
            log_level="warning",
            ws="websockets",
            lifespan="on",
        )
    )
    thread = threading.Thread(target=server.run, name="nova-uvicorn", daemon=True)
    thread.start()

    packets: list = []
    begun = time.time()
    try:
        lifecycle = await collect(
            app, port, args, packets, media_client if media is not None else None
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    summary = producer.summary.to_dict() if producer.summary else {}
    checks = inspect(packets, lifecycle, summary)
    media_state = lifecycle.get("media_state") or {}
    timing = None
    gate = None
    if media is not None:
        timeline = media.snapshot()
        # The watcher's live playback-clock getter is not evidence and not
        # JSON-serialisable; what it produced is already in `at_gain_packet`.
        media_state.pop("snapshot", None)
        media_state.update(
            {
                "timeline": timeline,
                "media_packets": sum(1 for packet in packets if packet["type"] == "media"),
                "envelope_coverage_seconds": session.references.coverage(),
                "audio_anchor_seconds": float(getattr(session.source, "audio_start", 0.0)),
                "replay_clock": {
                    "kind": "deterministic_virtual",
                    "speed": float(getattr(session.source, "speed", 1.0)),
                },
            }
        )
        timing = timing_evidence(
            packets, media_state, media_state.get("client", {}).get("exchanges", [])
        )
        checks.extend(media_assertions(media_state, timing))
    report = {
        "kind": "auditory session run record; plumbing evidence, not an accuracy claim",
        "started_at": datetime.fromtimestamp(begun, timezone.utc).isoformat(),
        "seconds": round(time.time() - begun, 2),
        "command": sys.argv,
        "trial": {
            "subject": trial.subject,
            "trial_id": trial.trial_id,
            "group": trial.group,
            "samples": int(len(trial.eeg)),
            "source_seconds": round(float(trial.timestamps[-1] - trial.timestamps[0]), 3),
            "audio_seconds": round(len(trial.audio) / trial.audio_rate, 3),
        },
        "model": {
            "path": args.model if not args.synthetic else "trained from fixtures",
            "channels": list(model.contract["eeg_channels"]),
            "window_seconds": session.window_seconds,
            "step_seconds": session.step_seconds,
        },
        "envelopes": {
            "paths": [str(path) for path in envelope_paths],
            "covered_media_seconds": session.references.coverage(),
            "candidates": [
                {"id": candidate.id, "label": candidate.label} for candidate in session.references.candidates
            ],
        },
        "server": {
            "host": args.host,
            "port": port,
            "static_dir": str(args.static_dir) if args.static_dir else None,
            "health": lifecycle["health"],
            "index": lifecycle["static"],
        },
        "session": summary,
        "packets": excerpt(packets),
        "assertions": checks,
        "plumbing": plumbing(packets, trial),
    }
    media_record = None
    if media is not None:
        report["media"] = media_report
        report["timing"] = timing
        media_record = {
            "kind": "step-9 media timeline run record; the browser's own handshake, replayed",
            "started_at": report["started_at"],
            "command": sys.argv,
            "seconds": report["seconds"],
            "render": media_report,
            "media": media_state,
            "timing": timing,
            "session": summary,
            "assertions": checks,
            "packet_types": report["packets"]["by_type"],
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False))
    stream_path = Path(args.stream_out)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.write_text("\n".join(json.dumps(packet) for packet in packets))
    if media_record is not None:
        media_path = Path(args.media_record)
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_text(json.dumps(media_record, indent=2, sort_keys=False))
        print(f"media record: {media_path}")
        gate = evaluate_gate(packets, media_state, media_path, args)
        if gate is not None:
            failed = [item for item in checks if not item["ok"]]
            gate_failed = [clause for clause in gate["clauses"] if not clause["ok"]]
            checks.append(
                check(
                    "the frontend's own gain gate reported itself satisfiable",
                    bool(gate["gate_open"]) and not gate_failed,
                    f"gate_open={gate['gate_open']} at {gate['gate_opened_at_seconds']}s; "
                    f"{len(gate_failed)} clause(s) false",
                )
            )
            report["assertions"] = checks
            out.write_text(json.dumps(report, indent=2, sort_keys=False))
            print(f"gate evidence: {args.gate_out}")

    print(f"session summary: {json.dumps(summary)}")
    print(f"packets: {json.dumps(report['packets']['by_type'])}")
    for item in checks:
        print(f"  [{'PASS' if item['ok'] else 'FAIL'}] {item['name']} -- {item['detail']}")
    if timing is not None:
        print(f"timing: {json.dumps(timing['delta_seconds'])}")
        print(
            "handshake: "
            + json.dumps(
                [
                    {
                        "action": entry["action"],
                        "media_time_s": entry["media_time_s"],
                        "revision": entry["acknowledged_revision"],
                        "sync_status": entry["sync_status"],
                    }
                    for entry in media_state.get("client", {}).get("exchanges", [])[:4]
                ]
            )
        )
    if gate is not None:
        for clause in gate["clauses"]:
            print(f"  [{'PASS' if clause['ok'] else 'FAIL'}] gate: {clause['name']} -- {clause['detail']}")
        print(f"gate open: {gate['gate_open']}; attune gains that differ from neutral: "
              f"{gate.get('playback_gains_applied')} of {gate.get('sampled_gain_packets')}")
        if gate.get("first_applied_gain"):
            print(f"first applied gain: {json.dumps(gate['first_applied_gain'])}")
    print(f"plumbing: {json.dumps(report['plumbing'])}")
    print(f"report: {out}")
    print(f"packet stream: {stream_path}")
    failed = [item for item in checks if not item["ok"]]
    print(f"{len(checks) - len(failed)}/{len(checks)} assertions held")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", help="converted trial .npz to replay")
    parser.add_argument("--synthetic", action="store_true", help="use a generated fixture")
    parser.add_argument("--model", default="models/auditory_kuleuven.npz")
    parser.add_argument("--envelope-dir", default="datasets/audio")
    parser.add_argument("--fixture-dir", default="output/auditory_ui/fixture")
    parser.add_argument("--static-dir", default=None, help="built frontend to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seconds", type=float, default=None, help="replay only this much")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed; 1.0 is real time")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument(
        "--strict-policy",
        action="store_true",
        help="use the chain default quality policy instead of the documented relaxed one",
    )
    parser.add_argument("--out", default="results/auditory_session.json")
    parser.add_argument("--stream-out", default="output/auditory_ui/packets.jsonl")
    parser.add_argument(
        "--media-out",
        default=None,
        help="render this trial's two candidates to a stereo WAV and serve it",
    )
    parser.add_argument(
        "--media-title", default="KU Leuven trial", help="title in the media descriptor"
    )
    parser.add_argument(
        "--presentation",
        default=DEFAULT_PRESENTATION,
        choices=PRESENTATION_MODES,
        help="presentation route; dichotic is L=A, R=B (plan sections 3.3/3.15)",
    )
    parser.add_argument(
        "--crossmix-weight",
        type=float,
        default=DEFAULT_CROSSMIX_WEIGHT,
        help="level of the other candidate per ear in --presentation crossmix",
    )
    parser.add_argument(
        "--media-tick",
        type=float,
        default=MEDIA_TICK_SECONDS,
        help="seconds between media reports, as MediaPlayback.js ticks",
    )
    parser.add_argument(
        "--media-record",
        default="results/auditory_media.json",
        help="where the media evidence record goes",
    )
    parser.add_argument(
        "--gate-script", default="scripts/auditory_ui/gate_evidence.mjs"
    )
    parser.add_argument(
        "--gate-out", default="output/auditory_ui/gate_evidence.json"
    )
    args = parser.parse_args(argv)
    if bool(args.trial) == bool(args.synthetic):
        parser.error("choose exactly one of --trial or --synthetic")
    if args.synthetic and args.seconds is None:
        args.seconds = 24.0
    if args.static_dir is None:
        candidate = REPO / "apps" / "attune-ui" / "dist"
        args.static_dir = str(candidate) if candidate.is_dir() else None
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
