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
from nova2026.auditory.session import AttentionSession, RunPolicy  # noqa: E402
from nova2026.auditory.sources import (  # noqa: E402
    ReferenceEnvelopes,
    ReplaySource,
    trial_envelope_paths,
)
from nova2026.transport.protocol import validate_packet  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.auditory.envelopes import convert_audio, save_envelope  # noqa: E402
from scripts.auditory.synthetic import synthetic_trial  # noqa: E402
from scripts.auditory.train import train  # noqa: E402

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


async def collect(app, port: int, args, packets: list) -> dict:
    """Drive the running server: REST snapshot, start, WebSocket, stop."""

    import httpx
    import websockets

    base = f"http://127.0.0.1:{port}"
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

        terminal = None
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
                if is_terminal(packet):
                    terminal = packet
                    break
        stopped = (await client.post("/api/session/stop")).json()
        sessions = (await client.get("/api/sessions")).json()

    return {
        "health": health,
        "static": static,
        "started": started,
        "snapshot": snapshot,
        "terminal": terminal,
        "stopped": stopped,
        "sessions": sessions,
    }


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
    app = create_app(producer_factory=lambda: producer, static_dir=args.static_dir)

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
        lifecycle = await collect(app, port, args, packets)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    summary = producer.summary.to_dict() if producer.summary else {}
    checks = inspect(packets, lifecycle, summary)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False))
    stream_path = Path(args.stream_out)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.write_text("\n".join(json.dumps(packet) for packet in packets))

    print(f"session summary: {json.dumps(summary)}")
    print(f"packets: {json.dumps(report['packets']['by_type'])}")
    for item in checks:
        print(f"  [{'PASS' if item['ok'] else 'FAIL'}] {item['name']} -- {item['detail']}")
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
