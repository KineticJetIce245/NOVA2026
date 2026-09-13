"""One command that runs the whole auditory demo, and says what it is doing.

    python -B -m scripts.auditory_ui.demo

That is the entire procedure. The command

1. selects a real KU Leuven trial and the fitted decoder (or says which of them
   is missing and stops),
2. loads and **verifies** the two reference envelopes before anything starts -
   the plan's rule 6: a missing envelope fails at start-up, never silently at
   run time,
3. renders the trial's two candidates to a stereo WAV (left = candidate A,
   right = candidate B, per plan sections 3.3/3.15) and serves it from the same
   origin,
4. starts the transport on a loopback port with the built frontend from
   ``apps/attune-ui/dist`` mounted at ``/`` (section 3.6: the default serving
   path is the build, not the Vite dev server),
5. starts the session through ``POST /api/session/start`` and prints the exact
   URL to open,
6. shuts down cleanly on Ctrl-C, and writes a run record either way.

**What is running while it runs.** The session replays the trial's EEG at 1x
through the real chain, the real producer and the real transport. The browser
position that the gain gate demands (decision D-02) is supplied by
:class:`~scripts.auditory_ui.media.SimulatedMediaClient`, which speaks
``mediaController.js``'s protocol exactly - because the acceptance run has no
browser, and a backend that invented a playback position would be violating the
one rule the frontend checks field by field. A human can watch the same run at
the printed URL; ``--browser`` additionally waits after the replay so the page
stays up.

**The operating point.** ``--margin`` defaults to
:data:`~nova2026.auditory.config.CALIBRATED_MARGIN` (0.05), the point measured on
held-out stories in ``results/aad_margin_calibration_*.md``. The documented
default :data:`~nova2026.auditory.config.MIN_MARGIN` (0.5) is one flag away
(``--margin 0.5``) and is **never** changed by this script; the effective value
is printed, put in the run record and carried in every ``audio_sources`` and
``prediction`` packet. Window length is deliberately not a flag: the session
reads it from the decoder's contract, and a longer window would need a model
fitted at that length (decision D-29).

**Safety.** Loopback only - any other ``--host`` is refused. The client never
supplies a path: the media file is server-side configuration and
``/api/media/file`` serves exactly that one file. Nothing here logs participant
data; the record holds timings, packet counts, policy and decision words.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/auditory_ui/demo.py`
    sys.path.insert(0, str(REPO))

from nova2026.auditory.config import (  # noqa: E402
    CALIBRATED_MARGIN,
    MIN_MARGIN,
    AuditoryConfig,
)
from nova2026.auditory.data import load_trial  # noqa: E402
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
from nova2026.transport.media import MediaTimeline  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.auditory_ui.media import MEDIA_TICK_SECONDS, SimulatedMediaClient  # noqa: E402
from scripts.auditory_ui.session import clip_trial, fixture_trial, free_port  # noqa: E402

DEFAULT_TRIAL = "datasets/AAD-KULeuven/converted/S1/trial_008.npz"
"""The trial steps 8 and 9 used, kept so a run can be compared with theirs."""

DEFAULT_MODEL = "models/auditory_kuleuven.npz"
"""64-channel model, the plan's fallback contract (section 3.11 item 4)."""

DEFAULT_MEDIA_OUT = "output/auditory_ui/demo_stereo.wav"
"""Rendered stereo asset; under ``output/``, which is git-ignored."""

TERMINAL_STATUSES = ("stopped", "error")


def log(message: str, lines: list[str]) -> None:
    """Print one timestamped line and keep it for the run record."""

    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    lines.append(line)


def build(app_state: dict, args):
    """Build the session, the producer and the media timeline for one run.

    Called twice by design: once to prove start-up can succeed (before the URL is
    printed) and once by the transport's own factory when
    ``POST /api/session/start`` arrives, so the session a viewer sees is started
    through the real REST path rather than behind its back.
    """

    trial, model, envelope_paths = app_state["trial"], app_state["model"], app_state["envelopes"]
    references = ReferenceEnvelopes.load(envelope_paths, model.config)
    source = ReplaySource(
        trial,
        speed=args.speed,
        simulated=args.synthetic,
        kind="synthetic_replay" if args.synthetic else "kuleuven_replay",
    )
    policy = RunPolicy(
        check_channels=False,
        max_bad_channels=0,
        margin=float(args.margin),
        warmup_seconds=args.warmup,
        frame_seconds=args.frame_seconds,
    )
    session = AttentionSession(
        source=source, decoder=model, references=references, policy=policy
    )
    producer = AttentionProducer(session)
    if app_state.get("media") is not None:
        producer.media_reference = app_state["media"].media_reference
    return session, producer


def prepare_media(trial, args) -> tuple[MediaTimeline, dict]:
    """Render the trial's candidates to the stereo file the browser plays."""

    out = Path(args.media_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = render_stereo(
        trial.audio,
        out,
        sample_rate=round(trial.audio_rate),
        presentation=presentation_mode(args.presentation),
        crossmix_weight=args.crossmix_weight,
    )
    report["candidate_seconds"] = round(float(len(trial.audio)) / trial.audio_rate, 3)
    report["eeg_seconds"] = round(float(trial.timestamps[-1] - trial.timestamps[0]), 3)
    timeline = MediaTimeline(out, args.media_title)
    report["media_id"] = timeline.media_id
    report["descriptor"] = timeline.descriptor()
    return timeline, report


async def capture(port: int, packets: list, stop: asyncio.Event, client, arrivals: list) -> None:
    """Subscribe to ``/ws/live`` and keep every packet, as the browser would.

    The packets are the demo's primary evidence: they are what the frontend
    decodes, what the gate is evaluated over, and what the frame-level metrics are
    counted from. Recording them here - on the wire, in arrival order - is the
    only way all three can be about the same stream.

    Each ``gain`` packet also gets the **client's own playback snapshot** at the
    instant it arrived, taken from the client rather than re-derived afterwards,
    because that is the quantity the frontend's 0.75 s clause compares.
    """

    import websockets

    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/ws/live", max_size=None, open_timeout=10
        ) as socket:
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                packet = json.loads(raw)
                packets.append(packet)
                if packet.get("type") == "gain" and client is not None:
                    arrivals.append(
                        {"order": len(packets), "timestamp": packet.get("timestamp"),
                         "payload": packet.get("payload"), "client": client.playback_snapshot()}
                    )
    except (OSError, asyncio.CancelledError):  # pragma: no cover - shutdown race
        return


async def drive(port: int, args, session, media_report: dict, stop: asyncio.Event) -> dict:
    """Start the transport session, run the client, and watch it end.

    Three things happen at once, and they are separate on purpose: the simulated
    browser reports its playback position over REST, the packet listener records
    the stream over the WebSocket, and a polling loop watches the session record
    for its terminal status. The polling loop is what ends an unattended demo when
    the replay is over.
    """

    import httpx

    base = f"http://127.0.0.1:{port}"
    packets: list = []
    packets_arrival: list = []
    state: dict = {
        "started": None, "media": None, "final": None,
        "packets": packets, "arrivals": packets_arrival,
    }
    client = SimulatedMediaClient(
        base,
        client_id="demo-runner",
        duration_s=media_report["seconds"],
        tick_seconds=args.media_tick,
    )
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as http:
        for _ in range(200):
            try:
                if (await http.get("/api/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                await asyncio.sleep(0.05)
        else:
            raise RuntimeError("the transport never became ready on loopback")

        listener = asyncio.create_task(capture(port, packets, stop, client, packets_arrival))
        state["started"] = (await http.post("/api/session/start")).json()
        client.configure(state["started"]["id"], media_report["media_id"])
        client.state["prepared"] = False
        await asyncio.sleep(0)
        # The simulated client reports from *its own* clock, so playback starts
        # when the session does rather than when a human presses play.
        playback = asyncio.create_task(client.play(stop))
        try:
            while not stop.is_set():
                await asyncio.sleep(args.poll)
                records = (await http.get("/api/sessions")).json()
                last = records[-1] if records else None
                if last is not None and last["status"] in TERMINAL_STATUSES:
                    state["final"] = last
                    break
        finally:
            stop.set()
            state["media"] = (await playback).to_dict()
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener
    return state


async def main_async(args) -> int:
    """The whole run, from fixture selection to the record on disk.

    It is the coroutine ``main`` hands to ``asyncio.run``, so everything that
    needs the loop - the drive, the optional browser, the bounded stay-open -
    happens inside it. The synchronous work before that (loading, verifying,
    rendering) runs on the same thread and returns its own exit code instead of
    raising: a demo that cannot be faithful says which file is missing.
    """

    lines: list[str] = []
    began = time.time()
    config = AuditoryConfig()
    if args.synthetic:
        from scripts.auditory.synthetic import synthetic_trial
        from scripts.auditory.train import train

        trial, envelope_paths = fixture_trial(
            Path(args.fixture_dir), args.seconds or 24.0, config
        )
        model, _ = train(
            [synthetic_trial("demo-training", seconds=int(args.seconds or 24.0))],
            [synthetic_trial("demo-validation", seconds=int(args.seconds or 24.0))],
            alphas=(100.0,),
        )
        trial = clip_trial(trial, args.seconds)
    else:
        trial_path, model_path = Path(args.trial), Path(args.model)
        for path, what, flag in (
            (trial_path, "trial", "--trial"),
            (model_path, "model", "--model"),
        ):
            if not path.is_file():
                print(f"{what} not found: {path} (pass {flag}, or --synthetic)", file=sys.stderr)
                return 2
        model = RidgeDecoder.load(model_path)
        trial = clip_trial(load_trial(trial_path), args.seconds)
        envelope_paths = trial_envelope_paths(trial, args.envelope_dir)

    # Verified before a port exists: rule 6 says a missing or mismatched envelope
    # fails at start-up, naming the file.
    references = ReferenceEnvelopes.load(envelope_paths, model.config)
    log(
        f"trial {trial.subject}/{trial.trial_id}: {len(trial.eeg)} samples "
        f"({trial.timestamps[-1] - trial.timestamps[0]:.1f}s), "
        f"{len(trial.channel_names)} channels",
        lines,
    )
    log(
        f"decoder {args.model if not args.synthetic else 'trained from fixtures'}: "
        f"window {model.contract['window_seconds']}s, "
        f"step {model.contract['step_seconds']}s, "
        f"channels {len(model.contract['eeg_channels'])}",
        lines,
    )
    log(
        f"envelopes verified: {envelope_paths[0].name}, {envelope_paths[1].name} "
        f"(coverage {references.coverage()})",
        lines,
    )
    log(
        f"operating point: margin={float(args.margin):g} "
        f"({'calibrated default' if abs(float(args.margin) - CALIBRATED_MARGIN) < 1e-12 else 'explicit'}), "
        f"policy check_channels=False (section 3.17 item 1)",
        lines,
    )

    media, media_report = prepare_media(trial, args)
    log(
        f"rendered {args.presentation} stereo {media_report['seconds']}s -> {args.media_out} "
        f"(left sha256 {media_report['sha256']['left'][:12]}...)",
        lines,
    )

    app_state = {
        "trial": trial,
        "model": model,
        "envelopes": envelope_paths,
        "media": media,
        "holder": {},
    }

    def factory():
        session, producer = build(app_state, args)
        app_state["holder"]["producer"] = producer
        app_state["holder"]["session"] = session
        return producer

    static_dir = Path(args.static_dir) if args.static_dir else None
    if static_dir is not None and not static_dir.is_dir():
        print(f"static dir not found: {static_dir}", file=sys.stderr)
        return 2
    app = create_app(producer_factory=factory, media=media, static_dir=static_dir)

    port = free_port()
    if args.host not in LOOPBACK_HOSTS:
        print(f"loopback only; refusing host {args.host!r}", file=sys.stderr)
        return 2
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            app, host=args.host, port=port, log_level=args.log_level, ws="websockets"
        )
    )
    thread = threading.Thread(target=server.run, name="nova-uvicorn", daemon=True)
    thread.start()

    url = f"http://{args.host}:{port}/"
    stop = asyncio.Event()
    loop_holder: dict = {}

    def request_stop(*_ignored):
        """Ctrl-C (or SIGTERM) ends the replay promptly and cleanly."""

        print("\n[demo] stop requested; ending the session and shutting down", flush=True)
        holder = loop_holder.get("loop")
        if holder is not None:
            holder.call_soon_threadsafe(stop.set)
        else:
            stop.set()

    for name in ("SIGINT", "SIGTERM"):
        handle = getattr(signal, name, None)
        if handle is not None:
            try:
                signal.signal(handle, request_stop)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    async def runner() -> dict:
        open_task = None
        if args.open_browser:
            open_task = asyncio.create_task(asyncio.to_thread(open_browser, url, lines))
        state = await drive(port, args, None, media_report, stop)
        if open_task is not None:
            await open_task
        if args.browser or args.serve_seconds > 0:
            # The page keeps being served after the replay, so a human can look
            # at the final state instead of a closed port. `--browser` waits for
            # Ctrl-C; `--serve-seconds` bounds the wait so the same behaviour can
            # be verified without a person in the loop.
            deadline = None if args.browser else time.monotonic() + args.serve_seconds
            log(
                f"replay finished; still serving {url} "
                f"({'until Ctrl-C' if deadline is None else f'for {args.serve_seconds:g}s'})",
                lines,
            )
            while not stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.25)
        return state

    loop_holder["loop"] = asyncio.get_running_loop()
    state: dict = {}
    exit_code = 0
    try:
        state = await runner()
    except KeyboardInterrupt:  # pragma: no cover - the handler usually catches it
        stop.set()
    except (RuntimeError, OSError, ValueError) as error:
        log(f"the run failed: {type(error).__name__}: {error}", lines)
        exit_code = 1
    finally:
        # Stop the session through the transport's own endpoint, then ask the
        # server thread to finish. The stop call is what tells the transport the
        # run is over; `should_exit` only ends the listener.
        try:
            import httpx

            with httpx.Client(base_url=f"http://{args.host}:{port}", timeout=5.0) as http:
                http.post("/api/session/stop")
        except Exception:  # pragma: no cover - the server may already be gone
            pass
        server.should_exit = True
        thread.join(timeout=15)
        log(
            f"transport stopped (server thread alive: {thread.is_alive()})",
            lines,
        )

    producer = app_state["holder"].get("producer")
    session = app_state["holder"].get("session")
    summary = producer.summary.to_dict() if producer is not None and producer.summary else {}
    packets = state.get("packets") or []
    log(f"{len(packets)} packets recorded from /ws/live", lines)
    evidence = gather_evidence(args, packets, state, media_report, lines)
    metrics = replay_metrics(trial, summary, args)
    log(
        "replay metrics: "
        + json.dumps(
            {
                "window_accuracy": metrics.get("window_accuracy"),
                "windows_decided": metrics.get("windows_decided"),
                "balanced_accuracy_decided": metrics.get("balanced_accuracy_decided"),
                "window_null_majority": metrics.get("window_null_majority"),
                "false_selection_changes": (metrics.get("selection") or {}).get(
                    "false_selection_changes"
                ),
            }
        ),
        lines,
    )
    return finish(
        args, trial, media_report, state, summary, metrics, evidence, lines, began, url, exit_code
    )


def gather_evidence(args, packets, state, media_report, lines) -> dict:
    """Write the packet stream and let the frontend's own code judge it.

    Two Node scripts, both of which execute the vendored frontend rather than a
    copy of it: ``decode_packets.mjs`` replays the stream through
    ``protocol.js``/``state.js``/``decoders.js`` and renders the dashboard, and
    ``gate_evidence.mjs`` calls ``mediaFocusReady`` and ``playbackGains`` from
    ``mediaAudio.js``. There is no browser in this repository's path, so this is
    the strongest rendering evidence available - and it is reported as what it is
    rather than as a screenshot.
    """

    if not packets:
        return {"available": False, "why": "no packets were recorded"}
    stream = Path(args.stream_out)
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.write_text("\n".join(json.dumps(packet) for packet in packets))
    record = {
        "media": {
            "descriptor": media_report["descriptor"],
            "duration_s": media_report["seconds"],
            "client": state.get("media") or {},
            # One entry per gain packet, in stream order, each carrying the
            # playback snapshot the client held when that packet arrived. Taken
            # live by the client itself: pairing frame k with report k afterwards
            # would be off by however long the first report happened to take.
            "at_gain_packet": state.get("arrivals") or [],
            "timeline": None,
        },
        "render": media_report,
    }
    media_record = Path(args.media_record)
    media_record.parent.mkdir(parents=True, exist_ok=True)
    media_record.write_text(json.dumps(record, indent=2))
    return {
        "available": True,
        "stream": str(stream),
        "media_record": str(media_record),
        "packets": len(packets),
        "decode": run_node(
            args.decode_script,
            [stream, args.render_out],
            lines,
        ),
        "gate": run_node(
            args.gate_script,
            [stream, media_record, args.gate_out],
            lines,
        ),
    }


def run_node(script: str, argv: list, lines: list) -> dict:
    """Run one evidence script and read the JSON it wrote, or say why not."""

    path = Path(script)
    if not path.is_file():
        return {"available": False, "why": f"script not found: {path}"}
    command = ["node", str(path)] + [str(item) for item in argv]
    try:
        completed = subprocess.run(
            command, cwd=str(REPO), capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        log(f"{path.name} could not run: {type(error).__name__}: {error}", lines)
        return {"available": False, "why": str(error)}
    out = Path(argv[-1])
    result = {"available": out.is_file(), "exit_code": completed.returncode, "command": command}
    if out.is_file():
        if out.suffix == ".json":
            try:
                result["report"] = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:  # pragma: no cover - a broken writer
                result["why"] = f"unreadable output: {error}"
        else:
            # A renderer writes an artifact, not a report: it prints its verdicts
            # as a JSON summary *and* a line per check on stdout. Keeping that text
            # is the difference between "the dashboard was written" and "here is
            # what the frontend's own checks said about the stream it was written
            # from", so the first complete JSON value on stdout is read out of it.
            try:
                result["report"] = first_json(completed.stdout)
            except ValueError:
                result["why"] = "the script printed no JSON summary"
                result["stdout_tail"] = completed.stdout.strip()[-800:]
    else:
        result["why"] = completed.stderr.strip()[-400:] or "no output file"
    return result


def first_json(text: str) -> dict:
    """The first JSON object in ``text``, ignoring whatever surrounds it.

    Raises:
        ValueError: If no complete JSON object can be decoded from the start of
            any ``{`` in the text.
    """

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("no JSON object on stdout")


def replay_metrics(trial, summary: dict, args) -> dict:
    """Window- and frame-level agreement with the trial's own label.

    This runs **after** the stream is closed, on purpose: the label is the one
    thing that must never reach the decision path, and the only way to show that
    is for the code reading it to run afterwards. Two honesties are stated in the
    result rather than left to the reader:

    * it is a **replay of KU Leuven**, one trial of one subject, so it is not a
      live-participant accuracy and not a generalisation claim;
    * the null is the **majority class within this trial**, measured here, not
      0.50 - across the dataset the majority rate by time is 66.1% (plan section 1
      fact 13).

    The window-level count uses the session's own ``decision_stream``: one entry
    per window the controller committed on, taken at the instant it committed.
    Recomputing the controller here would be a second implementation of it.
    """

    import numpy as np

    from nova2026.auditory.evaluation import selection_metrics

    stream = summary.get("decision_stream") or []
    times = np.asarray(trial.timestamps, dtype=float)
    labels = np.asarray(trial.labels)
    result: dict = {
        "kind": "KU Leuven replay, one converted trial; plumbing evidence, not a live-participant claim",
        "margin": float(args.margin),
        "window_seconds": summary.get("window_seconds"),
        "windows_scored": summary.get("scored"),
        "frames": summary.get("frames"),
        "decision_stream_length": len(stream),
    }
    label_here = sorted({int(value) for value in labels.tolist() if value in (0, 1)})
    result["trial_label"] = label_here
    if not stream or not label_here:
        result["note"] = "no committed window to score"
        return result

    decisions = np.asarray([0 if word == "A" else 1 for _, word in stream])
    moments = np.asarray([float(second) for second, _ in stream])
    truth = labels[np.clip(np.searchsorted(times, moments, side="right") - 1, 0, len(labels) - 1)]
    agree = decisions == truth
    result.update(
        {
            "window_accuracy": float(agree.mean()),
            "windows_decided": int(len(decisions)),
            "window_null_majority": float(
                max((truth == value).mean() for value in (0, 1))
            ),
            "balanced_accuracy_decided": float(
                np.mean(
                    [
                        (agree[truth == value].mean() if (truth == value).any() else 0.0)
                        for value in (0, 1)
                    ]
                )
            ),
            "recall_decided": {
                "A": float(agree[truth == 0].mean()) if (truth == 0).any() else None,
                "B": float(agree[truth == 1].mean()) if (truth == 1).any() else None,
            },
            "first_decision_seconds": float(moments[0]),
            "last_decision_seconds": float(moments[-1]),
        }
    )
    # The dataset-wide null, by time: the plan's section 1 fact 13 measures the
    # majority class at 66.1% of all labelled seconds, so 0.50 is the wrong null
    # for any accuracy claim about this dataset. It is quoted, not re-derived.
    result["dataset_null_majority_by_time"] = 0.661
    result["selection"] = selection_metrics(
        moments,
        decisions.astype(float),
        truth,
        end_time=float(moments[-1] + float(summary.get("step_seconds") or 1.0)),
    )
    result["note"] = (
        "window-level agreement over the committed windows of ONE converted KU "
        "Leuven trial, replayed at 1x; an abstention is not counted as an error "
        "and is reported as uncovered time, not as a miss. Not a live-participant "
        "accuracy and not a cross-subject claim."
    )
    return result


def finish(
    args, trial, media_report, state, summary, metrics, evidence, lines, began, url, exit_code
) -> int:
    """Write the run record and print the numbers a reviewer needs."""

    final = state.get("final") or {}
    status = final.get("status")
    log(f"session ended with status {status!r}", lines)
    log(
        "decisions: " + json.dumps(summary.get("decisions", {})),
        lines,
    )
    log(
        "effective run policy: " + json.dumps(summary.get("policy", {})),
        lines,
    )
    census = summary.get("bad_channel_census") or {}
    log(
        f"bad-channel census: {json.dumps(census) if census else 'no channel was flagged'}",
        lines,
    )
    if status == "error":
        log(f"the session reported an error: {json.dumps(summary.get('failure'))}", lines)
        exit_code = 1
    record = {
        "kind": "step-10 one-command demo run record; a KU Leuven replay, not a live participant",
        "started_at": datetime.fromtimestamp(began, timezone.utc).isoformat(),
        "seconds": round(time.time() - began, 2),
        "command": sys.argv,
        "url": url,
        "trial": {
            "subject": trial.subject,
            "trial_id": trial.trial_id,
            "group": trial.group,
            "samples": int(len(trial.eeg)),
            "seconds": round(float(trial.timestamps[-1] - trial.timestamps[0]), 3),
            "channels": list(trial.channel_names),
        },
        "operating_point": {
            "margin": float(args.margin),
            "margin_source": (
                "calibrated (results/aad_margin_calibration_*.md, decision D-29)"
                if abs(float(args.margin) - CALIBRATED_MARGIN) < 1e-12
                else "explicit --margin"
            ),
            "documented_default_margin": MIN_MARGIN,
            "margin_overridden": abs(float(args.margin) - MIN_MARGIN) > 1e-12,
            "window_seconds": summary.get("window_seconds"),
            "window_seconds_source": "decoder contract; not a policy field (decision D-29)",
            "presentation": args.presentation,
            "speed": float(args.speed),
            "clip_seconds": args.seconds,
        },
        "media": media_report,
        "static_dir": args.static_dir,
        "session": summary,
        "media_client": state.get("media"),
        "session_record": final,
        "replay_metrics": metrics,
        "frontend_evidence": evidence,
        "packet_stream": {
            "path": evidence.get("stream"),
            "count": evidence.get("packets"),
            "by_type": packet_census(state.get("packets") or []),
        },
        "log": lines,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=False))
    print(f"run record: {out}")
    print(f"open: {url}")
    print(f"effective margin: {float(args.margin):g} (documented default {MIN_MARGIN})")
    print(
        "replay metrics: "
        + json.dumps(
            {
                "window_accuracy": metrics.get("window_accuracy"),
                "windows_decided": metrics.get("windows_decided"),
                "coverage": (metrics.get("selection") or {}).get("coverage"),
                "false_selection_changes": (metrics.get("selection") or {}).get(
                    "false_selection_changes"
                ),
            }
        )
    )
    gate = ((evidence.get("gate") or {}).get("report") or {})
    if gate:
        print(
            f"gain gate: open={gate.get('gate_open')} at {gate.get('gate_opened_at_seconds')}s, "
            f"attenuated gain frames {gate.get('playback_gains_applied')} "
            f"of {gate.get('sampled_gain_packets')}"
        )
    decode = ((evidence.get("decode") or {}).get("report") or {})
    if decode:
        print(f"frontend: rejected={decode.get('rejects')} over {decode.get('packets')} packets")
    return exit_code


def packet_census(packets: list) -> dict:
    """``{type: count}`` for the recorded stream, for the record's summary."""

    census: dict[str, int] = {}
    for packet in packets:
        kind = str(packet.get("type"))
        census[kind] = census.get(kind, 0) + 1
    return census


def open_browser(url: str, lines: list) -> None:
    """Ask the operating system's own browser to open the URL, if it has one.

    This is not a dependency and not a download: it calls the default handler the
    machine already has. A machine without one reports the URL instead of failing
    the run - the URL is the deliverable, the window is a convenience.
    """

    import webbrowser

    try:
        opened = webbrowser.open(url)
    except Exception as error:  # pragma: no cover - headless machines
        log(f"could not open a browser ({type(error).__name__}); open {url}", lines)
        return
    log(f"browser open: {opened} ({url})", lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", default=DEFAULT_TRIAL, help="converted trial .npz")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="fitted decoder .npz")
    parser.add_argument("--envelope-dir", default="datasets/audio")
    parser.add_argument("--synthetic", action="store_true", help="use a generated fixture")
    parser.add_argument("--fixture-dir", default="output/auditory_ui/fixture")
    parser.add_argument(
        "--margin",
        type=float,
        default=CALIBRATED_MARGIN,
        help=(
            f"controller commit margin; default {CALIBRATED_MARGIN} is the point "
            f"calibrated on held-out stories (decision D-29). The documented "
            f"policy default is {MIN_MARGIN}."
        ),
    )
    parser.add_argument("--seconds", type=float, default=None, help="replay only this much")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed; 1.0 is real time")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--poll", type=float, default=0.5, help="seconds between status polls")
    parser.add_argument("--media-tick", type=float, default=MEDIA_TICK_SECONDS)
    parser.add_argument("--media-out", default=DEFAULT_MEDIA_OUT)
    parser.add_argument("--media-title", default="KU Leuven trial")
    parser.add_argument(
        "--presentation",
        default=DEFAULT_PRESENTATION,
        choices=PRESENTATION_MODES,
        help="presentation route; dichotic is L=A, R=B (plan sections 3.3/3.15)",
    )
    parser.add_argument("--crossmix-weight", type=float, default=DEFAULT_CROSSMIX_WEIGHT)
    parser.add_argument("--static-dir", default=str(REPO / "apps" / "attune-ui" / "dist"))
    parser.add_argument("--host", default="127.0.0.1", help="loopback only")
    parser.add_argument("--log-level", default="warning")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="ask the OS to open the URL (its default browser, if it has one)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="keep the page served after the replay until Ctrl-C",
    )
    parser.add_argument(
        "--serve-seconds",
        type=float,
        default=0.0,
        help="keep the page served this long after the replay, then exit 0",
    )
    parser.add_argument(
        "--out", default="results/demo_run.json", help="where the run record goes"
    )
    parser.add_argument(
        "--stream-out",
        default="output/auditory_ui/demo_packets.jsonl",
        help="where the recorded packet stream goes (git-ignored; the record cites it)",
    )
    parser.add_argument(
        "--render-out",
        default="output/auditory_ui/demo_dashboard.html",
        help="where the frontend-rendered dashboard goes",
    )
    parser.add_argument(
        "--media-record",
        default="output/auditory_ui/demo_media.json",
        help="handshake record the gain gate is evaluated over",
    )
    parser.add_argument("--gate-out", default="output/auditory_ui/demo_gate.json")
    parser.add_argument("--decode-script", default="scripts/auditory_ui/decode_packets.mjs")
    parser.add_argument("--gate-script", default="scripts/auditory_ui/gate_evidence.mjs")
    args = parser.parse_args(argv)
    if args.static_dir is not None and not Path(args.static_dir).is_dir():
        print(
            f"warning: no built frontend at {args.static_dir}; serving /api and /ws only",
            file=sys.stderr,
        )
    return _exit(_execute(args))


def _execute(args) -> int:
    """Run the session on its own event loop and return the command's exit code.

    The seam is here rather than inside the run so ``main`` hands ``asyncio.run``
    exactly what it wants - one coroutine, always - and a refusal that happens
    before the loop (a missing trial, a non-loopback host) comes back as an
    ordinary exit code instead of a ``SystemExit`` a caller would have to trap.
    A failure inside the run is reported with its type rather than vanishing into
    a bare ``1``: whoever reads the log needs to know what happened.
    """

    try:
        return _exit(asyncio.run(main_async(args)))
    except (RuntimeError, OSError, ValueError) as error:
        print(f"demo failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _exit(value) -> int:
    """``main`` always returns a number, whatever the coroutine handed back."""

    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
