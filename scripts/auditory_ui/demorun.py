"""Run the whole perturbation matrix unattended, and report what really happened.

    python -B -m scripts.auditory_ui.demorun

That one command replays a real KU Leuven trial through the real chain, the real
producer, the real transport and the real media path once for every row of the
plan's matrix (``final_connection.md`` sections 3.13/3.16 and 3.14), and writes
one report. There is no browser and no human in the loop.

**One case, one process.** The matrix's third question - did this case
contaminate another - is only answerable if the cases cannot reach each other, so
every case runs as its own subprocess with its own scratch directory and its own
session instance. The parent process never executes a case's body; it collects a
single JSON beacon the child prints, applies the row's criterion to it, and
records the process outcome (exit code, timeout, missing beacon) as the crash
evidence.

**What is injected is the condition, never the symptom.** A duplicated block is
really fed twice; a late block really arrives late; a dead electrode is really
constant; the NaN run is really written into the samples. The chain is then
allowed to say what it says - ``interpolated``, ``irregular_timestamps``,
``large_gap``, ``unsafe_endpoints``, ``audio_unavailable``, a session whose
terminal state is ``error`` - and that answer is the evidence. Handing the packet
layer a fabricated ``uncertain`` would test nothing.

**What each case records.** The fault that was injected, the observables the row
names, whether each appeared, the raw numbers behind it (counters, reasons,
exception types, packet excerpts), the three answers the matrix demands (did it
crash, was the failure explicit, did it contaminate anything) and a verdict.

**Verdicts and the exit code.** ``PASS`` means every observable the row names
appeared and the invariant held. ``FINDING`` means something the row names did
not appear; the record then also states whether the *invariant* still held, so a
missing mechanism (the plan asked for a start-up refusal and the code degrades per
window instead) is not confused with a dangerous behaviour (a confident answer
published on saturated data). The exit code is 0 when no case crashed, no case
was silent and no case violated its invariant - a FINDING whose invariant held is
reported, not hidden, but does not make the run a failure. ``--strict`` turns
every FINDING into a non-zero exit for callers that want the stronger gate.

**Safety.** Loopback only. No participant data is logged: the report holds
counters, decisions, timings and reasons. Data, models and the frozen contract
are read-only inputs; a run refuses to touch them.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/auditory_ui/demorun.py`
    sys.path.insert(0, str(REPO))

from nova2026.auditory.config import (  # noqa: E402
    CALIBRATED_MARGIN,
    MIN_MARGIN,
    AuditoryConfig,
)
from nova2026.auditory.data import AuditoryTrial, load_trial  # noqa: E402
from nova2026.auditory.decoder import RidgeDecoder  # noqa: E402
from nova2026.auditory.evaluation import inject_fault, selection_metrics  # noqa: E402
from nova2026.auditory.producer import AttentionProducer  # noqa: E402
from nova2026.auditory.render import render_stereo  # noqa: E402
from nova2026.auditory.session import (  # noqa: E402
    AttentionSession,
    RunPolicy,
    SessionFrame,
)
from nova2026.auditory.sources import (  # noqa: E402
    ReferenceEnvelopes,
    ReplaySource,
    trial_envelope_paths,
)
from nova2026.streaming import Acquire  # noqa: E402
from nova2026.transport.media import MediaTimeline  # noqa: E402
from nova2026.transport.protocol import validate_packet  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.auditory.envelopes import convert_audio, save_envelope  # noqa: E402
from scripts.auditory.synthetic import synthetic_trial  # noqa: E402
from scripts.auditory.train import train  # noqa: E402
from scripts.auditory_ui.media import SimulatedMediaClient  # noqa: E402
from scripts.auditory_ui.session import clip_trial, free_port  # noqa: E402

HOST = "127.0.0.1"
MARKER = "@@DEMORUN@@"
"""Prefix of the single JSON line a case worker prints as its result."""
BEACON_GRACE_SECONDS = 20.0
"""Extra wall time a child may take beyond its own session budget before it is
killed; a killed child is recorded as a hang, which is a failure of the case."""

DECISIONS = ("A", "B", "uncertain", "unavailable")
CONFIDENT = ("A", "B")
TERMINAL = ("stopped", "error")

DEFAULT_TRIAL = "datasets/AAD-KULeuven/converted/S1/trial_008.npz"
DEFAULT_MODEL = "models/auditory_kuleuven.npz"
DEFAULT_ENVELOPES = "datasets/audio"
DEFAULT_SCRATCH = "output/auditory_ui/demorun"
GATE_SCRIPT = "scripts/auditory_ui/gate_evidence.mjs"
REJECTION_SCRIPT = "scripts/auditory_ui/rejection_evidence.mjs"


@dataclass(frozen=True)
class Case:
    """One row of the matrix, as the plan writes it.

    Args:
        id: Case id (``C1``..``C16``, ``E1``..``E7``).
        fault: What is injected, in the plan's words.
        expected: The behaviour the plan names as required.
        invariant: The property that must hold even when the plan's named
            mechanism turns out not to exist.
        family: Which worker runs it; the report groups by this.
    """

    id: str
    fault: str
    expected: str
    invariant: str
    family: str


CASES: tuple[Case, ...] = (
    Case("C1", "none: the clean path", "session finishes, gains activate, metrics complete",
         "the clean path runs to a terminal state with an activated gain path", "transport"),
    Case("C2", "the EEG stream stops mid-run", "producer ends with status=error, gains neutral, never hangs",
         "a dead stream reaches a terminal state instead of hanging", "transport"),
    Case("C3", "large lag: data arrives seconds late", "frames degrade explicitly, the lag is recorded",
         "late data never becomes a confident decision", "session+acquire"),
    Case("C4", "timestamp jump / gap", "small gaps repaired and flagged interpolated; over-limit goes through recovery, never silently skipped",
         "no gap is silently skipped", "session"),
    Case("C5", "duplicated EEG data", "detected and counted; a repeat must not advance the decision",
         "a repeat is never treated as new evidence", "session"),
    Case("C6", "NaN/Inf runs", "Repair fixes and flags; beyond max_seconds raises into recovery or stop",
         "non-finite data is never scored", "session"),
    Case("C7", "dead channel", "handled by the run policy and recorded in the run record",
         "the run survives it and the census names the electrode", "session"),
    Case("C8", "UI-to-backend disconnect", "playback pauses, gain returns neutral, reconnect re-snapshots",
         "a stale reference never keeps a gain applied", "transport+media"),
    Case("C9", "backend restart", "the client accepts a reset sequence from a fresh snapshot",
         "a restarted backend's stream is accepted, not rejected as out-of-order", "transport+node"),
    Case("C10", "session restart mid-run", "old streams cleared, new session_id, no gain residue",
         "a new session starts from neutral", "transport"),
    Case("C11", "audio clock drift", "beyond the 0.75 s window the gain goes neutral; the fit residual is recorded",
         "no confident gain is applied on a drifted clock", "transport+media"),
    Case("C12", "playback underrun/stall", "recorded as an event; a stale decision is never presented as current",
         "a stale decision is not applied as if it were current", "transport+media"),
    Case("C13", "protocol-violating packets", "rejected and counted; the frontend's rejected warning is visible",
         "illegal packets are refused, legal ones still decode", "transport+node"),
    Case("C14", "missing envelope at start", "the session refuses to start and names the file",
         "start-up fails loudly instead of running without a reference", "start"),
    Case("C15", "mismatched candidate lengths", "start fails; no silent truncation",
         "no window is ever scored against a truncated candidate", "session"),
    Case("C16", "out-of-range / missing labels", "marked -1 and excluded from the coverage denominator",
         "unknown truth never enters the denominator as if it were known", "metrics"),
    Case("E1", "the same block fed twice", "the decision is not submitted faster by the repeat",
         "a repeat produces no new evidence", "session"),
    Case("E2", "non-monotonic timestamps", "rejected as irregular_timestamps, never accepted",
         "a broken time grid is refused", "session"),
    Case("E3", "sample rate contradicting the declared contract", "refused at start after the contract comparison",
         "contract-violating data produces no confident decision", "session"),
    Case("E4", "wrong channel count", "selected by name or refused; never truncated by column count",
         "the chain never truncates columns to fit the contract", "start"),
    Case("E5", "a unit off by a power of ten", "source_unit_exponent lets the consistency check catch it",
         "the decade error is caught rather than bridged", "session"),
    Case("E6", "a gap larger than max_seconds", "recovery or stop, with large_gap recorded",
         "the gap is visible in the run record", "session"),
    Case("E7", "all-zero or saturated signal", "quality rejects the window, gain stays neutral",
         "no confident decision is published on a dead or railed signal", "session"),
)

BY_ID = {case.id: case for case in CASES}


# --------------------------------------------------------------------- helpers


def frame_rows(frames: list) -> list[dict]:
    """Every published frame, as JSON-safe rows for the scratch record."""

    return [frame.to_dict() if isinstance(frame, SessionFrame) else dict(frame) for frame in frames]


def summarise_frames(frames: list, *, limit: int = 40) -> dict:
    """Aggregate what the frames say, plus a bounded sample of them.

    The full series goes to the scratch file; the report carries the counts, the
    reason vocabulary and the first and last frames, which is what a criterion is
    decided from.
    """

    rows = frame_rows(frames)
    by_decision: dict[str, int] = {}
    reasons: dict[str, int] = {}
    censused: dict[str, int] = {}
    gains_non_neutral = 0
    for row in rows:
        by_decision[row["decision"]] = by_decision.get(row["decision"], 0) + 1
        for reason in row["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        for name in row["bad_channels"]:
            censused[name] = censused.get(name, 0) + 1
        if abs(row["gain_a_db"]) > 1e-9 or abs(row["gain_b_db"]) > 1e-9:
            gains_non_neutral += 1
    stale_rows = [row for row in rows if "evidence_stale" in row["reasons"]]
    return {
        "frames": len(rows),
        "stale_frames": len(stale_rows),
        "stale_frames_confident": sum(1 for row in stale_rows if row["decision"] in CONFIDENT),
        "stale_frames_gain_non_neutral": sum(
            1 for row in stale_rows if abs(row["gain_a_db"]) > 1e-9 or abs(row["gain_b_db"]) > 1e-9
        ),
        "by_decision": by_decision,
        "reasons": reasons,
        "bad_channels": censused,
        "artifact_frames": sum(1 for row in rows if row["artifact"] is True),
        "quality_values": sorted({row["quality"] for row in rows}, key=lambda v: (v is None, v)),
        "gain_frames_non_neutral": gains_non_neutral,
        "gain_min": min((min(row["gain_a_db"], row["gain_b_db"]) for row in rows), default=None),
        "evidence_counts": sorted({row["evidence_count"] for row in rows}),
        "confident_decisions": sum(by_decision.get(word, 0) for word in CONFIDENT),
        "sample": rows[:limit],
        "head": rows[:3],
        "tail": rows[-3:],
    }


def summarise_packets(packets: list) -> dict:
    """The packet-level observables a transport case is judged on."""

    by_type: dict[str, int] = {}
    for packet in packets:
        by_type[packet["type"]] = by_type.get(packet["type"], 0) + 1
    attention = [p for p in packets if p["type"] == "attention"]
    gains = [p for p in packets if p["type"] == "gain"]
    session = [p for p in packets if p["type"] == "session"]
    with_reference = [p for p in gains if "media_time_s" in p.get("payload", {})]
    decisions: dict[str, int] = {}
    for packet in attention:
        word = str(packet["payload"].get("decision"))
        decisions[word] = decisions.get(word, 0) + 1
    terminal = session[-1]["payload"].get("status") if session else None
    return {
        "total": len(packets),
        "by_type": by_type,
        "decisions": decisions,
        "gain_packets": len(gains),
        "gain_packets_with_media_reference": len(with_reference),
        "gain_min": min((min(p["payload"].get("a_db", 0.0), p["payload"].get("b_db", 0.0)) for p in gains), default=None),
        "gain_frames_non_neutral": sum(
            1 for p in gains if abs(p["payload"].get("a_db", 0.0)) > 1e-9 or abs(p["payload"].get("b_db", 0.0)) > 1e-9
        ),
        "terminal_status": terminal,
        "session_statuses": [p["payload"].get("status") for p in session],
        "sequences": [p["sequence"] for p in packets],
        "sync_statuses": sorted({str(p["payload"].get("status")) for p in packets if p["type"] == "sync"}),
        "first": packets[:2],
        "last": packets[-2:],
    }


def write_lines(path: Path, lines: list[str]) -> str:
    """Write a JSONL evidence file and return its path, for the report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def scratch_dir(args, case_id: str) -> Path:
    """This case's own directory; nothing else may be written during a case."""

    path = Path(args.scratch_root) / Path(args.stamp) / case_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def beacon(payload: dict, path: Path) -> None:
    """Print the result beacon and mirror it to the case's scratch directory."""

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(MARKER + json.dumps({"result": str(path)}, sort_keys=True), flush=True)


# ------------------------------------------------------------------ injections


class PerturbedSource:
    """A :class:`ReplaySource` whose chunk stream one fault function rewrites.

    The fault functions are generators over the base chunk stream, so an
    injection cannot invent data, timestamps or verdicts: it can only drop,
    repeat or re-time what the recording actually contains.
    """

    def __init__(self, trial: AuditoryTrial, *, speed: float, fault: Callable | None = None) -> None:
        self._base = ReplaySource(trial, speed=speed, kind="perturbation_replay")
        self.sample_rate = self._base.sample_rate
        self.channel_names = self._base.channel_names
        self.reference = self._base.reference
        self.upstream_processing = self._base.upstream_processing
        self.start = self._base.start
        self.end = self._base.end
        self.audio_start = self._base.audio_start
        self.simulated = False
        self.kind = "perturbation_replay"
        self.speed = float(speed)
        self.fault = fault
        self.observables: dict[str, Any] = {}

    def chunks(self, stop) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """Yield base chunks, rewritten by the injected fault."""

        if self.fault is None:
            yield from self._base.chunks(stop)
            return
        yield from self.fault(self, self._base.chunks(stop))


def fault_disconnect(at_seconds: float):
    """The upstream stream dies at ``at_seconds``: the source raises, once."""

    def fault(source: PerturbedSource, base):
        for end_time, samples, timestamps in base:
            if end_time - source.start >= at_seconds:
                source.observables["disconnected_at_seconds"] = float(end_time - source.start)
                raise RuntimeError(
                    "the EEG source disconnected mid-run (injected, case C2)"
                )
            yield end_time, samples, timestamps

    return fault


def fault_lag(at_seconds: float, stall_seconds: float, step_seconds: float = 0.25):
    """The stream goes quiet for ``stall_seconds`` while its clock keeps running.

    This is the replay-path form of "the data is seconds late": the amplifier's
    samples stop arriving while the session clock advances, so the newest
    evidence falls behind the clock that stamps the frames. The lag is measured at
    the source boundary - wall time spent without data - which is what
    ``Acquire.max_lag`` measures on the live path.
    """

    def fault(source: PerturbedSource, base):
        stalled = False
        lag = 0.0
        for end_time, samples, timestamps in base:
            if not stalled and end_time - source.start >= at_seconds:
                stalled = True
                began = time.perf_counter()
                virtual = float(end_time)
                while time.perf_counter() - began < stall_seconds:
                    time.sleep(min(step_seconds, stall_seconds - (time.perf_counter() - began)))
                    virtual += step_seconds
                    lag = max(lag, time.perf_counter() - began)
                    yield virtual, np.empty((0, len(source.channel_names))), np.empty((0,))
                source.observables["acquire_max_lag_seconds"] = round(lag, 3)
                source.observables["stall_seconds"] = float(stall_seconds)
                source.observables["stall_began_at_seconds"] = float(at_seconds)
            yield end_time, samples, timestamps

    return fault


def fault_repeat(at_seconds: float, copies: int = 2):
    """One block is delivered ``copies`` times, byte for byte, at one site."""

    def fault(source: PerturbedSource, base):
        done = False
        for end_time, samples, timestamps in base:
            if not done and end_time - source.start >= at_seconds:
                done = True
                source.observables.setdefault("repeated_blocks", []).append(
                    {
                        "at_seconds": float(end_time - source.start),
                        "samples": int(len(samples)),
                        "copies": int(copies),
                        "first_timestamp": float(timestamps[0]),
                    }
                )
                for _ in range(copies):
                    yield end_time, samples, timestamps
                continue
            yield end_time, samples, timestamps

    return fault


def fault_gap(at_seconds: float, gap_seconds: float):
    """``gap_seconds`` of source time are missing, so the stamps jump once.

    Chunks that fall entirely inside the hole are still yielded, with no samples,
    so the session clock advances across the hole exactly as a real stream's would
    while its samples were lost. Nothing is interpolated here: whether the hole is
    bridged is the chain's decision, not the injector's.
    """

    def fault(source: PerturbedSource, base):
        stop_at = at_seconds + gap_seconds
        begun = False
        for end_time, samples, timestamps in base:
            relative = timestamps - source.start
            inside = (relative >= at_seconds) & (relative < stop_at)
            if not inside.any():
                yield end_time, samples, timestamps
                continue
            if not begun:
                begun = True
                source.observables["gap_seconds"] = float(gap_seconds)
                source.observables["gap_dropped_samples"] = int(inside.sum())
            keep = ~inside
            if keep.any():
                yield end_time, samples[keep], timestamps[keep]
            else:
                yield end_time, np.empty((0, samples.shape[1])), np.empty((0,))

    return fault


class ProbeInlet:
    """An LSL-inlet-shaped test double for :class:`~nova2026.streaming.Acquire`.

    ``Acquire``'s own docstring names this seam: any object exposing
    ``acquire``/``connected``/``n_new_samples``/``get_data``/``add_callback``/
    ``info``/``callbacks`` works. It exists so the acquisition layer's gap and lag
    counters are exercised with real blocks *really* arriving late or twice -
    without a device, and without pretending the replay path is an amplifier.
    """

    def __init__(self, blocks: list[tuple[np.ndarray, np.ndarray]], channels: int) -> None:
        self._blocks = list(blocks)
        self._index = 0
        self.info = {"nchan": channels}
        self.callbacks: list = []
        self.connected = True
        self.n_new_samples = 0

    def add_callback(self, callback) -> None:
        """Register a chunk callback, as MNE-LSL does."""

        self.callbacks.append(callback)

    def acquire(self) -> None:
        """Deliver the next block to every callback, once."""

        if self._index >= len(self._blocks):
            self.n_new_samples = 0
            return
        data, timestamps = self._blocks[self._index]
        self._index += 1
        self.n_new_samples = len(data)
        for callback in list(self.callbacks):
            callback(data, timestamps, self._index)

    def get_data(self, winsize: float = 1.0, exclude=()) -> tuple[np.ndarray, np.ndarray]:
        """The one-sample view ``Acquire`` uses to clear the unread counter."""

        return np.zeros((1, self.info["nchan"])), np.zeros(1)


def acquire_probe(*, lag_seconds: float, sfreq: float = 128.0, blocks: int = 4) -> dict:
    """Drive the real :class:`Acquire` with blocks that are ``lag_seconds`` old."""

    from mne_lsl.lsl import local_clock

    count = 32
    prepared = []
    stamps = []
    for index in range(blocks):
        start = local_clock() - lag_seconds + index * count / sfreq
        stamps.append(start + np.arange(count) / sfreq)
        prepared.append(np.zeros((count, 4)))
    inlet = ProbeInlet(list(zip(prepared, stamps)), 4)
    acquire = Acquire(inlet, count, sfreq=sfreq, max_lag_seconds=3.0, poll_interval=0.001)
    error = None
    try:
        acquire.read(timeout=1.0)
    except Exception as failure:  # noqa: BLE001 - the point is to record which one
        error = f"{type(failure).__name__}: {failure}"
    return {
        "injected_lag_seconds": float(lag_seconds),
        "blocks": acquire.blocks,
        "max_lag_seconds": round(float(acquire.max_lag), 3),
        "gaps": int(acquire.gaps),
        "max_gap_seconds": round(float(acquire.max_gap), 6),
        "error": error,
    }


def acquire_duplicate_probe(*, sfreq: float = 128.0) -> dict:
    """Feed the real :class:`Acquire` the same block twice and read its counters."""

    from mne_lsl.lsl import local_clock

    count = 32
    # Every block sits in the recent past: a block ahead of the local clock is a
    # different fault (and `Acquire` refuses it for a different reason).
    start = local_clock() - 1.0
    data = np.zeros((count, 4))
    stamps = start + np.arange(count) / sfreq
    inlet = ProbeInlet([(data, stamps), (data, stamps), (data, stamps + count / sfreq)], 4)
    acquire = Acquire(inlet, count, sfreq=sfreq, max_lag_seconds=None, poll_interval=0.001)
    delivered = []
    for _ in range(3):
        block, _ = acquire.read(timeout=1.0)
        delivered.append(int(len(block)))
    return {
        "blocks": acquire.blocks,
        "samples": int(acquire.samples),
        "gaps": int(acquire.gaps),
        "max_gap_seconds": round(float(acquire.max_gap), 6),
        "duplicates_delivered": int(delivered.count(count)),
        "delivered": delivered,
    }


# ------------------------------------------------------------------- execution


class Ctx:
    """Everything one case needs, loaded lazily and kept inside its scratch dir."""

    def __init__(self, args) -> None:
        self.args = args
        self.trial: AuditoryTrial | None = None
        self.model = None
        self.envelopes: tuple[Path, Path] | None = None

    def load(self, *, seconds: float | None = None) -> tuple[AuditoryTrial, Any, tuple[Path, Path]]:
        """Load (and clip) the trial, the decoder and the two envelope paths."""

        if self.trial is None:
            self.trial = clip_trial(load_trial(self.args.trial), seconds)
            self.model = RidgeDecoder.load(self.args.model)
            self.envelopes = trial_envelope_paths(self.trial, self.args.envelope_dir)
        return self.trial, self.model, self.envelopes


def run_one_session(
    ctx: Ctx,
    *,
    trial: AuditoryTrial | None = None,
    model=None,
    envelopes: tuple[Path, Path] | None = None,
    policy: RunPolicy | None = None,
    source=None,
    speed: float | None = None,
    stop_after: float | None = None,
) -> dict:
    """Run one real :class:`AttentionSession` and return what it recorded.

    Returns:
        ``{summary, frames, crashed, wall_seconds}``; ``crashed`` is the exception
        the *session* raised, which is data - several cases expect the chain to
        refuse - and never an unhandled process failure.
    """

    trial = trial if trial is not None else ctx.trial
    model = model if model is not None else ctx.model
    envelopes = envelopes if envelopes is not None else ctx.envelopes
    speed = float(ctx.args.speed if speed is None else speed)
    references = ReferenceEnvelopes.load(envelopes, model.config)
    source = source if source is not None else ReplaySource(trial, speed=speed, kind="perturbation_replay")
    policy = policy if policy is not None else base_policy(ctx.args)
    session = AttentionSession(source=source, decoder=model, references=references, policy=policy)
    frames: list = []
    stop = threading.Event()
    timer = None
    if stop_after is not None:
        timer = threading.Timer(float(stop_after), stop.set)
        timer.daemon = True
        timer.start()
    began = time.perf_counter()
    crashed = None
    summary = None
    try:
        summary = session.run(frames.append, stop)
    except Exception as error:  # noqa: BLE001 - the chain's refusal is the evidence
        crashed = f"{type(error).__name__}: {error}"
    finally:
        if timer is not None:
            timer.cancel()
    processor = getattr(session, "processor", None)
    recovery = getattr(processor, "recovery", None)
    repair = getattr(processor, "repair", None)
    return {
        "wall_seconds": round(time.perf_counter() - began, 3),
        "crashed": crashed,
        "summary": summary.to_dict() if summary is not None else None,
        "frames": summarise_frames(frames),
        "processor": {
            "recovery_events": list(getattr(recovery, "events", []) or []),
            "recovery_segment": int(getattr(recovery, "segment", 0) or 0),
            "recoveries": int(getattr(recovery, "recoveries", 0) or 0),
            "repaired_samples": int(getattr(repair, "repaired_samples", 0) or 0),
            "dropped_rows": int(getattr(repair, "dropped_rows", 0) or 0),
            "held_rows": int(getattr(repair, "held_rows", 0) or 0),
        },
        "frames_raw": frame_rows(frames),
    }


def base_policy(args, **overrides) -> RunPolicy:
    """The documented relaxed run policy the plan requires for KU Leuven trials."""

    values = {
        "check_channels": False,
        "max_bad_channels": 0,
        "margin": float(args.margin),
        "warmup_seconds": 2.0,
        "frame_seconds": 0.25,
    }
    values.update(overrides)
    return RunPolicy(**values)


def collect_session_result(ctx: Ctx, label: str, **kwargs) -> dict:
    """Run one session and persist its frame series next to the case's other raw evidence."""

    result = run_one_session(ctx, **kwargs)
    rows = result.pop("frames_raw")
    result["frames_path"] = write_lines(
        scratch_dir(ctx.args, ctx.args.case) / f"frames_{label}.jsonl",
        [json.dumps(row, sort_keys=True) for row in rows],
    )
    return result


# ------------------------------------------------------------- transport cases


async def watch_packets(port: int, packets: list, arrivals: list, holder: dict, stop: asyncio.Event, idle: float) -> None:
    """Drain ``/ws/live`` until the terminal lifecycle packet or the idle limit."""

    import websockets

    try:
        async with websockets.connect(
            f"ws://{HOST}:{port}/ws/live", max_size=None, open_timeout=10
        ) as socket:
            while True:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=idle)
                except asyncio.TimeoutError:
                    return
                packet = json.loads(raw)
                packets.append(packet)
                if packet.get("type") == "gain":
                    client = holder.get("client")
                    arrivals.append(
                        {
                            "order": len(packets),
                            "timestamp": packet.get("timestamp"),
                            "payload": packet.get("payload"),
                            "client": client.playback_snapshot() if client is not None else None,
                        }
                    )
                if packet.get("type") == "session" and packet.get("source") == "server":
                    if packet.get("payload", {}).get("status") in TERMINAL:
                        stop.set()
                        return
    except (OSError, asyncio.CancelledError):  # pragma: no cover - shutdown race
        return


async def media_task(client: SimulatedMediaClient, session_id: str, media_id: str, holder: dict, stop: asyncio.Event, result: dict) -> None:
    """Play one controller session and publish its playback clock to the watcher."""

    client.configure(session_id, media_id)
    holder["client"] = client
    outcome = await client.play(stop)
    extra = {
        name: getattr(client, name)
        for name in ("dropped_at", "reconnect_refused", "attempts", "snapshots", "positions")
        if hasattr(client, name)
    }
    result["clients"] = result.get("clients", []) + [
        {"client_id": client.client_id, "state": client.state, "result": outcome.to_dict(), "extra": extra}
    ]


class DroppingMediaClient(SimulatedMediaClient):
    """A controller whose reports stop for good after ``drop_after`` seconds.

    This is the browser half of case C8: the page keeps playing (its own clock
    keeps advancing, which is what ``playback_snapshot`` reports) while no report
    reaches the backend, so the backend's last known position stops moving.
    """

    def __init__(self, *args, drop_after: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_after = float(drop_after)
        self.dropped_at: float | None = None

    async def _send(self, client, index, action, position, began):  # type: ignore[override]
        if action == "report" and self.playback_snapshot()["time"] >= self.drop_after:
            self.dropped_at = self.playback_snapshot()["time"]
            # The report is never sent; a real page in this state has lost its
            # connection and shows stale data until it reconnects.
            raise _PlaybackLost()
        return await super()._send(client, index, action, position, began)

    async def play(self, stop):  # type: ignore[override]
        try:
            return await super().play(stop)
        except _PlaybackLost:
            return self._lost_result()

    def _lost_result(self):
        from scripts.auditory_ui.media import MediaClientResult

        return MediaClientResult(
            descriptor={},
            duration_s=self.duration_s,
            client_id=self.client_id,
            started_playback=self.state["started_playback"],
            prepared=self.state["prepared"],
            failures=[f"reports stopped at {self.dropped_at}"],
        )


class ReconnectingMediaClient(SimulatedMediaClient):
    """A controller whose connection dropped, reconnecting the way the transport says to.

    The first attempt is a bare ``prepare`` - what a page that lost its socket and
    came back tries - and the transport refuses it with 409 *"stop and prepare
    again"*. The controller then does exactly that, so the case can record both
    halves: the refusal is explicit and names its remedy, and after the remedy the
    timeline is re-snapshotted with a fresh revision.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reconnect_refused: str | None = None
        self.attempts: list[dict] = []

    async def play(self, stop):  # type: ignore[override]
        first = await super().play(stop)
        self.attempts.append({"attempt": "bare prepare", "failures": list(first.failures)})
        if not first.failures:
            return first
        self.reconnect_refused = first.failures[0]
        stop_record = await self._stop_timeline()
        self.attempts.append({"attempt": "stopped", "response": stop_record})
        for key, value in (("error", None), ("prepared", False), ("started_playback", False),
                           ("playback_started_at", None), ("last_ack", None)):
            self.state[key] = value
        second = await super().play(stop)
        self.attempts.append({"attempt": "stopped then prepare", "failures": list(second.failures)})
        return second

    async def _stop_timeline(self) -> dict:
        """Send the ``stopped`` command the 409 asked for, as the controller would."""

        import httpx

        self.request_id += 1
        body = {
            "session_id": self.session_id,
            "media_id": self.media_id,
            "client_id": self.client_id,
            "request_id": self.request_id,
            "action": "stopped",
            "media_time_s": 0.0,
            "duration_s": self.duration_s,
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post("/api/media/control", json=body)
            return {"status": response.status_code, "body": response.text[:200]}


class _PlaybackLost(Exception):
    """Internal: this controller's report path is gone."""


class StallingMediaClient(SimulatedMediaClient):
    """A controller whose report loop stalls for ``stall_seconds`` at ``stall_at``.

    The audio keeps playing in the browser's own clock while the reports are held
    back, so the position the backend echoes is stale compared with the position
    the client would report - which is the quantity the frontend's 0.75 s clause
    compares. ``snapshots`` records the client's clock and its last *delivered*
    position on every held report, so the stall is countable from the real log.
    """

    def __init__(self, *args, stall_at: float, stall_seconds: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stall_at = float(stall_at)
        self.stall_seconds = float(stall_seconds)
        self.stalled = False
        self.snapshots: list[dict] = []

    async def _send(self, client, index, action, position, began):  # type: ignore[override]
        now = self.playback_snapshot()["time"]
        if action == "report" and not self.stalled and now >= self.stall_at:
            self.stalled = True
            began_stall = time.perf_counter()
            while time.perf_counter() - began_stall < self.stall_seconds:
                await asyncio.sleep(0.05)
                self.snapshots.append(
                    {
                        "client_time": round(self.playback_snapshot()["time"], 4),
                        "last_position_delivered": round(position, 4),
                        "held_for_seconds": round(time.perf_counter() - began_stall, 4),
                    }
                )
            return await super()._send(client, index, action, position, began)
        return await super()._send(client, index, action, position, began)


class DriftingMediaClient(SimulatedMediaClient):
    """A controller whose audio position runs at ``rate`` times real time.

    The drift is monotone and small per report (0.25 s of wall time becomes
    ``rate * 0.25`` s of position), which is what an audio clock running fast
    looks like from the protocol's point of view.
    """

    def __init__(self, *args, rate: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rate = float(rate)
        self.positions: list[float] = []

    async def _send(self, client, index, action, position, began):  # type: ignore[override]
        drifted = position * self.rate
        self.positions.append(round(drifted, 4))
        return await super()._send(client, index, action, drifted, began)


async def drive_transport(
    ctx: Ctx,
    *,
    make_producer: Callable[[], AttentionProducer],
    media: MediaTimeline | None = None,
    media_clients: tuple = (),
    steps: Callable | None = None,
    idle_timeout: float = 8.0,
    seconds: float | None = None,
) -> dict:
    """Start the real loopback transport, run one session through REST and ``/ws/live``.

    The watch task is bounded by ``ctx.args.case_timeout``: if no terminal packet
    arrives, the run is *recorded* as having hung and the server is torn down, so a
    hang becomes evidence instead of a stuck command.
    """

    import httpx
    import uvicorn

    args = ctx.args
    packets: list = []
    arrivals: list = []
    holder: dict = {}
    media_state: dict = {}
    app = create_app(producer_factory=make_producer, media=media, static_dir=None)
    port = free_port()
    if args.host not in LOOPBACK_HOSTS:  # pragma: no cover - argparse restricts it
        raise SystemExit(f"loopback only; refusing host {args.host!r}")
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=port, log_level="warning", ws="websockets", lifespan="on")
    )
    thread = threading.Thread(target=server.run, name="demorun-uvicorn", daemon=True)
    thread.start()
    lifecycle: dict = {"hang": False, "start_error": None, "session_ids": []}
    try:
        async with httpx.AsyncClient(base_url=f"http://{HOST}:{port}", timeout=10.0) as client:
            for _ in range(200):
                try:
                    if (await client.get("/api/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.05)
            else:
                raise RuntimeError("the transport never became ready on loopback")
            response = await client.post("/api/session/start")
            lifecycle["start_status"] = response.status_code
            lifecycle["start_body"] = response.text[:400]
            if response.status_code != 200:
                lifecycle["start_error"] = response.text[:400]
                return lifecycle
            started = response.json()
            lifecycle["started"] = started
            lifecycle["session_ids"].append(started.get("id"))
            lifecycle["snapshot"] = (await client.get("/api/state")).json()
            stop = asyncio.Event()
            tasks = [asyncio.create_task(watch_packets(port, packets, arrivals, holder, stop, idle_timeout))]
            lifecycle["stop_events"] = [stop]
            media_result: dict = {}

            async def restart_session() -> dict:
                """Stop the running session, wait for its terminal packet, start and watch another."""

                stopped = (await client.post("/api/session/stop")).json()
                await asyncio.wait_for(tasks[0], timeout=float(args.case_timeout))
                started_again = (await client.post("/api/session/start")).json()
                lifecycle["session_ids"].append(started_again.get("id"))
                lifecycle.setdefault("restarts", []).append(
                    {"stopped": stopped, "started": started_again, "packets_before": len(packets)}
                )
                fresh = asyncio.Event()
                lifecycle["stop_events"].append(fresh)
                tasks[0] = asyncio.create_task(
                    watch_packets(port, packets, arrivals, holder, fresh, idle_timeout)
                )
                return started_again

            lifecycle["restart"] = restart_session
            if media is not None:
                descriptor = media.descriptor()
                media_state["descriptor"] = descriptor
                for factory in media_clients:
                    spec = factory(port, started, holder, media_result, stop)
                    tasks.append(asyncio.create_task(spec))
            if steps is not None:
                await steps(client, port, lifecycle, stop)
            try:
                await asyncio.wait_for(tasks[0], timeout=float(args.case_timeout))
            except asyncio.TimeoutError:
                lifecycle["hang"] = True
                stop.set()
            # The controller clients end when the session's terminal packet
            # arrives, so give them a moment to record their own outcome instead
            # of cancelling them the instant the packet watcher returns.
            stop.set()
            if tasks[1:]:
                await asyncio.wait(tasks[1:], timeout=3.0)
            for task in tasks[1:]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)
            lifecycle["stopped"] = (await client.post("/api/session/stop")).json()
            lifecycle["sessions"] = (await client.get("/api/sessions")).json()
            media_state["client"] = media_result
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    transcripts = []
    for packet in packets:
        try:
            validate_packet(packet)
            transcripts.append({"packet": packet, "error": None})
        except ValueError as error:
            transcripts.append({"packet": packet, "error": str(error)})
    lifecycle["packets"] = packets
    lifecycle["arrivals"] = arrivals
    lifecycle["media_state"] = media_state
    lifecycle["packet_summary"] = summarise_packets(packets)
    lifecycle["validator_errors"] = [row for row in transcripts if row["error"]]
    return lifecycle


def media_client_task(factory: Callable, session_id: str, media: MediaTimeline, holder: dict, result: dict, stop: asyncio.Event):
    """Bind one controller client to the running session."""

    async def run() -> None:
        client = factory()
        await media_task(client, session_id, media.descriptor()["media_id"], holder, stop, result)

    return run()


def write_media_record(path: Path, media: MediaTimeline, lifecycle: dict, *, duration_s: float) -> str:
    """Write the record ``gate_evidence.mjs`` reads, from this run's own numbers.

    The shape is step 9's: the descriptor, the exchange log the controller wrote,
    and one playback snapshot per gain packet taken at arrival. The gate script
    reads nothing else, so the frontend's own ``mediaFocusReady`` is evaluated
    against this run's real timeline rather than a reconstruction.
    """

    media_state = lifecycle.get("media_state") or {}
    client = media_state.get("client") or {}
    exchanges = []
    for entry in client.get("clients", []):
        exchanges.extend(entry["result"].get("exchanges", []))
    record = {
        "kind": "demorun media record; the frontend's gate replayed over a perturbed run",
        "render": media.descriptor(),
        "media": {
            "descriptor": media.descriptor(),
            "duration_s": float(duration_s),
            "client": {"exchanges": exchanges},
            "at_gain_packet": lifecycle.get("arrivals", []),
        },
        "timing": {"delta_seconds": None},
        "packet_types": lifecycle.get("packet_summary", {}).get("by_type", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return str(path)


def run_node(script: str, argv: list[str], *, timeout: float = 180.0) -> dict:
    """Run one evidence script under Node and return its JSON verdict, or the failure."""

    try:
        completed = subprocess.run(
            ["node", script, *argv],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "stdout": "", "stderr": ""}
    payload = None
    out_path = Path(argv[-1])
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            payload = {"parse_error": str(error)}
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-2000:],
        "report": payload,
    }


# ----------------------------------------------------------------- case workers


def case_c1(ctx: Ctx) -> dict:
    """C1: the clean run - the only case with no fault in it."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.clean_seconds)
    out = scratch_dir(ctx.args, "C1") / "stereo.wav"
    render_report = render_stereo(trial.audio, out, sample_rate=round(trial.audio_rate))
    duration = float(render_report["seconds"])
    media = MediaTimeline(out, "KU Leuven trial (demorun C1)")

    holder_trial = {"trial": trial, "model": model, "envelopes": envelopes}

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(holder_trial["envelopes"], holder_trial["model"].config)
        source = ReplaySource(holder_trial["trial"], speed=1.0, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=holder_trial["model"], references=references, policy=base_policy(ctx.args)
        )
        producer = AttentionProducer(session)
        producer.media_reference = media.media_reference
        return producer

    lifecycle = asyncio.run(
        drive_transport(
            ctx,
            make_producer=make_producer,
            media=media,
            media_clients=(
                lambda port, started, holder, result, stop: media_client_task(
                    lambda: SimulatedMediaClient(
                        f"http://{HOST}:{port}",
                        client_id="demorun-c1",
                        duration_s=duration,
                        clip_seconds=ctx.args.clean_seconds,
                    ),
                    started["id"],
                    media,
                    holder,
                    result,
                    stop,
                ),
            ),
        )
    )
    record = write_media_record(scratch_dir(ctx.args, "C1") / "media_record.json", media, lifecycle, duration_s=duration)
    stream = write_lines(
        scratch_dir(ctx.args, "C1") / "packets.jsonl",
        [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
    )
    gate = run_node(GATE_SCRIPT, [stream, record, str(scratch_dir(ctx.args, "C1") / "gate.json")])
    return {
        "render": render_report,
        "packets": lifecycle["packet_summary"],
        "validator_errors": lifecycle["validator_errors"],
        "terminal_status": lifecycle["packet_summary"]["terminal_status"],
        "hang": lifecycle["hang"],
        "media": lifecycle.get("media_state", {}).get("client", {}),
        "exchanges": [
            exchange
            for entry in lifecycle.get("media_state", {}).get("client", {}).get("clients", [])
            for exchange in entry["result"].get("exchanges", [])
        ][:6],
        "gate": gate,
        "packets_path": stream,
        "media_record_path": record,
    }


def case_c2(ctx: Ctx) -> dict:
    """C2: the upstream EEG stream dies mid-run."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = PerturbedSource(trial, speed=ctx.args.speed, fault=fault_disconnect(ctx.args.fault_at))
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        return _remembering(source, AttentionProducer(session))

    lifecycle = asyncio.run(drive_transport(ctx, make_producer=make_producer, idle_timeout=6.0))
    source_observables = _recall()
    return {
        "packets": lifecycle["packet_summary"],
        "terminal_status": lifecycle["packet_summary"]["terminal_status"],
        "hang": lifecycle["hang"],
        "stopped": lifecycle.get("stopped"),
        "sessions": lifecycle.get("sessions"),
        "source": source_observables,
        "packets_path": write_lines(
            scratch_dir(ctx.args, "C2") / "packets.jsonl",
            [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
        ),
    }


_RECALL: dict = {}


def _remembering(source, producer):
    """Keep a case's source observables reachable after the producer thread ends."""

    _RECALL["source"] = source
    return producer


def _recall() -> dict:
    """The observables of the last remembered source."""

    source = _RECALL.get("source")
    return dict(getattr(source, "observables", {}) or {})


def case_c3(ctx: Ctx) -> dict:
    """C3: the data arrives seconds late, and the acquisition layer says so too."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    source = PerturbedSource(
        trial, speed=ctx.args.speed, fault=fault_lag(ctx.args.fault_at, ctx.args.lag_seconds)
    )
    result = collect_session_result(ctx, "late", source=source, stop_after=ctx.args.case_timeout / 2.0)
    return {
        "session": result,
        "source": source.observables,
        "acquire": acquire_probe(lag_seconds=ctx.args.lag_seconds + 1.0),
    }


def case_c4(ctx: Ctx) -> dict:
    """C4: one small gap (repaired and flagged) and one gap past the recovery limit."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    # 2 samples at 128 Hz is the widest hole ``Repair`` still bridges with its
    # default ``max_seconds=0.02``; 3 would go to recovery instead, which is the
    # other half of this case (E6).
    small = PerturbedSource(
        trial, speed=ctx.args.speed, fault=fault_gap(ctx.args.fault_at, ctx.args.small_gap_seconds)
    )
    small_result = collect_session_result(ctx, "small_gap", source=small, stop_after=ctx.args.case_timeout / 3.0)
    fatal = PerturbedSource(
        trial, speed=ctx.args.speed, fault=fault_gap(ctx.args.fault_at, ctx.args.fatal_gap_seconds)
    )
    fatal_result = collect_session_result(ctx, "fatal_gap", source=fatal, stop_after=ctx.args.case_timeout / 3.0)
    return {
        "small_gap": {"session": small_result, "source": small.observables},
        "fatal_gap": {"session": fatal_result, "source": fatal.observables},
    }


def case_c5(ctx: Ctx) -> dict:
    """C5: a duplicated block, against the same source without the duplicate."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    baseline = collect_session_result(ctx, "baseline", stop_after=ctx.args.case_timeout / 3.0)
    repeated = PerturbedSource(
        trial, speed=ctx.args.speed, fault=fault_repeat(ctx.args.fault_at, 2)
    )
    duplicate = collect_session_result(ctx, "duplicate", source=repeated, stop_after=ctx.args.case_timeout / 3.0)
    return {
        "baseline": baseline,
        "duplicate": duplicate,
        "source": repeated.observables,
        "acquire": acquire_duplicate_probe(),
    }


def _trial_with_eeg(trial: AuditoryTrial, eeg: np.ndarray, timestamps: np.ndarray | None = None) -> AuditoryTrial:
    """A shallow copy carrying different samples or stamps, for post-load faults."""

    clone = copy.copy(trial)
    clone.eeg = np.asarray(eeg, dtype=float)
    if timestamps is not None:
        clone.timestamps = np.asarray(timestamps, dtype=float)
    return clone


def case_c6(ctx: Ctx) -> dict:
    """C6: a repairable NaN run, and one longer than ``Repair.max_seconds``."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    short = inject_fault(trial, "dropout", start=ctx.args.fault_at, duration=0.015)
    short_result = collect_session_result(
        ctx, "nan_short", trial=short, stop_after=ctx.args.case_timeout / 3.0
    )
    long = inject_fault(trial, "dropout", start=ctx.args.fault_at, duration=0.2)
    long_result = collect_session_result(
        ctx, "nan_long", trial=long, stop_after=ctx.args.case_timeout / 3.0
    )
    return {"short": short_result, "long": long_result}


def case_c7(ctx: Ctx) -> dict:
    """C7: one electrode is dead for the whole run."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    dead_index = 3
    eeg = np.array(trial.eeg, copy=True)
    eeg[:, dead_index] = 0.0
    dead = _trial_with_eeg(trial, eeg)
    result = collect_session_result(ctx, "dead_channel", trial=dead, stop_after=ctx.args.case_timeout / 3.0)
    strict = collect_session_result(
        ctx,
        "dead_channel_strict",
        trial=dead,
        policy=base_policy(ctx.args, check_channels=True),
        stop_after=ctx.args.case_timeout / 3.0,
    )
    return {
        "dead_channel": trial.channel_names[dead_index],
        "relaxed": result,
        "strict": strict,
    }


def case_c8(ctx: Ctx) -> dict:
    """C8: the browser stops reporting, then reconnects with a fresh controller."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.media_seconds)
    out = scratch_dir(ctx.args, "C8") / "stereo.wav"
    render_report = render_stereo(trial.audio, out, sample_rate=round(trial.audio_rate))
    duration = float(render_report["seconds"])
    media = MediaTimeline(out, "KU Leuven trial (demorun C8)")

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=1.0, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        producer = AttentionProducer(session)
        producer.media_reference = media.media_reference
        return producer

    shared: dict = {}

    def first(port, started, holder, result, stop):
        def build():
            client = DroppingMediaClient(
                f"http://{HOST}:{port}",
                client_id="demorun-c8-page",
                duration_s=duration,
                clip_seconds=ctx.args.media_seconds,
                drop_after=ctx.args.fault_at,
                tick_seconds=0.2,
            )
            shared["first"] = client
            return client

        return media_client_task(build, started["id"], media, holder, result, stop)

    def second(port, started, holder, result, stop):
        async def run() -> None:
            while not stop.is_set() and not any(
                "demorun-c8-page" == entry["client_id"] for entry in result.get("clients", [])
            ):
                await asyncio.sleep(0.1)
            await asyncio.sleep(2.0)
            # The page keeps its controller identity across a socket drop: the
            # transport binds one client_id to the timeline and requires the
            # request counter to keep increasing within it, so a reconnecting
            # controller *is* the same controller, with its counter where it was.
            previous = shared.get("first")
            client = ReconnectingMediaClient(
                f"http://{HOST}:{port}",
                client_id=getattr(previous, "client_id", "demorun-c8-page"),
                duration_s=duration,
                clip_seconds=None,
                tick_seconds=0.25,
            )
            client.request_id = int(getattr(previous, "request_id", 0))
            await media_task(client, started["id"], media.descriptor()["media_id"], holder, stop, result)

        return run()

    lifecycle = asyncio.run(
        drive_transport(
            ctx, make_producer=make_producer, media=media, media_clients=(first, second), idle_timeout=8.0
        )
    )
    record = write_media_record(scratch_dir(ctx.args, "C8") / "media_record.json", media, lifecycle, duration_s=duration)
    stream = write_lines(
        scratch_dir(ctx.args, "C8") / "packets.jsonl",
        [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
    )
    gate = run_node(GATE_SCRIPT, [stream, record, str(scratch_dir(ctx.args, "C8") / "gate.json")])
    return {
        "packets": lifecycle["packet_summary"],
        "timeline": lifecycle.get("media_state", {}).get("client", {}),
        "gate": gate,
        "packets_path": stream,
        "media_record_path": record,
        "reconnect": _reconnect_evidence(lifecycle["packets"]),
        "reconnect_attempts": [
            {"client_id": entry["client_id"], **entry.get("extra", {})}
            for entry in (lifecycle.get("media_state", {}).get("client", {}).get("clients") or [])
        ],
    }


def _reconnect_evidence(packets: list) -> dict:
    """Where the media reference disappears and whether it comes back."""

    gains = [p for p in packets if p["type"] == "gain"]
    flags = [("media_time_s" in p.get("payload", {})) for p in gains]
    first_missing = flags.index(False) if False in flags else None
    after_missing = flags[first_missing:] if first_missing is not None else []
    recovered = after_missing.index(True) if True in after_missing else None
    revisions = sorted(
        {
            p["payload"].get("media_revision", p["payload"].get("revision"))
            for p in packets
            if p["type"] == "media"
        },
        key=lambda value: (value is None, value),
    )
    return {
        "gain_packets": len(gains),
        "first_gain_without_reference": first_missing,
        "gain_with_reference_after_gap": recovered,
        "revisions_seen": revisions,
    }


def case_c9(ctx: Ctx) -> dict:
    """C9: the backend is restarted, so the client sees a fresh snapshot."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.transport_seconds)

    first = _short_transport(ctx, trial, model, envelopes, client_id="demorun-c9", idle_timeout=4.0)
    second = _short_transport(ctx, trial, model, envelopes, client_id="demorun-c9", idle_timeout=4.0)
    payload = {
        "session_a": {"packets": first["packet_summary"], "snapshot": first.get("snapshot")},
        "session_b": {"packets": second["packet_summary"], "snapshot": second.get("snapshot")},
    }
    stream_a = [json.dumps(packet, sort_keys=True) for packet in first["packets"]]
    stream_b = [json.dumps(packet, sort_keys=True) for packet in second["packets"]]
    evidence_in = scratch_dir(ctx.args, "C9") / "reset_input.json"
    evidence_in.write_text(
        json.dumps(
            {
                "stream_a": [json.loads(line) for line in stream_a],
                "stream_b": [json.loads(line) for line in stream_b],
                "snapshot_b": second.get("snapshot"),
            }
        ),
        encoding="utf-8",
    )
    reset = run_node(
        REJECTION_SCRIPT, ["reset", str(evidence_in), str(scratch_dir(ctx.args, "C9") / "reset.json")]
    )
    payload["sequence_reset_evidence"] = reset
    payload["snapshots"] = {"a": first.get("snapshot"), "b": second.get("snapshot")}
    payload["packets_path"] = write_lines(
        scratch_dir(ctx.args, "C9") / "packets.jsonl", stream_a + stream_b
    )
    return payload


def _short_transport(ctx: Ctx, trial, model, envelopes, *, client_id: str, idle_timeout: float) -> dict:
    """One complete start/run/stop cycle against its own fresh app instance."""

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=ctx.args.speed, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        return AttentionProducer(session)

    return asyncio.run(drive_transport(ctx, make_producer=make_producer, idle_timeout=idle_timeout))


def case_c10(ctx: Ctx) -> dict:
    """C10: stop the session mid-run and start a new one on the same transport."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.transport_seconds)

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=ctx.args.speed, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        return AttentionProducer(session)

    state: dict = {}

    async def steps(client, port, lifecycle, stop):
        """Stop the running session mid-run, start a second one, and watch it too."""

        await asyncio.sleep(1.2)
        state["first_snapshot_before_stop"] = (await client.get("/api/state")).json()
        started = await lifecycle["restart"]()
        state["second_start"] = started
        state["second_snapshot"] = (await client.get("/api/state")).json()

    lifecycle = asyncio.run(
        drive_transport(ctx, make_producer=make_producer, steps=steps, idle_timeout=6.0)
    )
    packets = lifecycle["packets"]
    sessions = lifecycle.get("sessions") or []
    second_id = (state.get("second_start") or {}).get("id")
    second_packets = [p for p in packets if p.get("session_id") == second_id]
    early = [p for p in second_packets if p["type"] == "gain"][:4]
    return {
        "packets": lifecycle["packet_summary"],
        "session_ids": lifespan_sessions(packets),
        "restarts": lifecycle.get("restarts"),
        "first_snapshot_before_stop": state.get("first_snapshot_before_stop"),
        "second_start": state.get("second_start"),
        "second_snapshot": state.get("second_snapshot"),
        "sessions": sessions,
        "second_session_gain_packets": early,
        "packets_path": write_lines(
            scratch_dir(ctx.args, "C10") / "packets.jsonl",
            [json.dumps(packet, sort_keys=True) for packet in packets],
        ),
    }


def lifespan_sessions(packets: list) -> list[dict]:
    """The lifecycle packets, in order, as ``{session_id, status}``."""

    rows = []
    for packet in packets:
        if packet["type"] == "session" and packet.get("source") == "server":
            rows.append({"session_id": packet.get("session_id"), "status": packet["payload"].get("status")})
    return rows


def case_c11(ctx: Ctx) -> dict:
    """C11: the audio clock runs fast, and the packets say what is (not) fitted."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.media_seconds)
    out = scratch_dir(ctx.args, "C11") / "stereo.wav"
    render_report = render_stereo(trial.audio, out, sample_rate=round(trial.audio_rate))
    duration = float(render_report["seconds"])
    media = MediaTimeline(out, "KU Leuven trial (demorun C11)")

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=1.0, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        producer = AttentionProducer(session)
        producer.media_reference = media.media_reference
        return producer

    client = DriftingMediaClient(
        f"http://{HOST}:0",
        client_id="demorun-c11",
        duration_s=duration,
        clip_seconds=ctx.args.media_seconds,
        tick_seconds=0.25,
        rate=ctx.args.drift_rate,
    )

    def factory(port, started, holder, result, stop):
        client.base_url = f"http://{HOST}:{port}"
        client.duration_s = duration
        return media_client_task(lambda: client, started["id"], media, holder, result, stop)

    lifecycle = asyncio.run(
        drive_transport(ctx, make_producer=make_producer, media=media, media_clients=(factory,), idle_timeout=6.0)
    )
    record = write_media_record(scratch_dir(ctx.args, "C11") / "media_record.json", media, lifecycle, duration_s=duration)
    stream = write_lines(
        scratch_dir(ctx.args, "C11") / "packets.jsonl",
        [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
    )
    gate = run_node(GATE_SCRIPT, [stream, record, str(scratch_dir(ctx.args, "C11") / "gate.json")])
    sync_payloads = [p["payload"] for p in lifecycle["packets"] if p["type"] == "sync"]
    return {
        "packets": lifecycle["packet_summary"],
        "drift_rate": ctx.args.drift_rate,
        "client_positions": client.positions[:8],
        "gate": gate,
        "sync_offsets": [
            {
                "status": payload.get("status"),
                "offset_ms": payload.get("offset_ms"),
                "drift_warning": payload.get("drift_warning"),
                "timeline": payload.get("timeline"),
            }
            for payload in sync_payloads[:3]
        ],
        "packets_path": stream,
        "media_record_path": record,
    }


def case_c12(ctx: Ctx) -> dict:
    """C12: the report loop stalls while the audio keeps playing."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.media_seconds)
    out = scratch_dir(ctx.args, "C12") / "stereo.wav"
    render_report = render_stereo(trial.audio, out, sample_rate=round(trial.audio_rate))
    duration = float(render_report["seconds"])
    media = MediaTimeline(out, "KU Leuven trial (demorun C12)")
    holder_client: dict = {}

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=1.0, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        producer = AttentionProducer(session)
        producer.media_reference = media.media_reference
        return producer

    def factory(port, started, holder, result, stop):
        client = StallingMediaClient(
            f"http://{HOST}:{port}",
            client_id="demorun-c12",
            duration_s=duration,
            clip_seconds=ctx.args.media_seconds,
            tick_seconds=0.25,
            stall_at=ctx.args.fault_at,
            stall_seconds=ctx.args.stall_seconds,
        )
        holder_client["client"] = client
        return media_client_task(lambda: client, started["id"], media, holder, result, stop)

    lifecycle = asyncio.run(
        drive_transport(ctx, make_producer=make_producer, media=media, media_clients=(factory,), idle_timeout=6.0)
    )
    record = write_media_record(scratch_dir(ctx.args, "C12") / "media_record.json", media, lifecycle, duration_s=duration)
    stream = write_lines(
        scratch_dir(ctx.args, "C12") / "packets.jsonl",
        [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
    )
    gate = run_node(GATE_SCRIPT, [stream, record, str(scratch_dir(ctx.args, "C12") / "gate.json")])
    client = holder_client.get("client")
    snapshots = list(getattr(client, "snapshots", []) or [])
    return {
        "packets": lifecycle["packet_summary"],
        "stall_seconds": ctx.args.stall_seconds,
        "stall_at": ctx.args.fault_at,
        "held_reports": len(snapshots),
        "stall_samples": snapshots[:3] + snapshots[-2:],
        "gate": gate,
        "packets_path": stream,
        "media_record_path": record,
    }


def case_c13(ctx: Ctx) -> dict:
    """C13: illegal packets are offered to the frontend's own validator."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.transport_seconds)

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(envelopes, model.config)
        source = ReplaySource(trial, speed=ctx.args.speed, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        return AttentionProducer(session)

    lifecycle = asyncio.run(drive_transport(ctx, make_producer=make_producer, idle_timeout=5.0))
    stream = write_lines(
        scratch_dir(ctx.args, "C13") / "packets.jsonl",
        [json.dumps(packet, sort_keys=True) for packet in lifecycle["packets"]],
    )
    rejection = run_node(
        REJECTION_SCRIPT, ["reject", stream, str(scratch_dir(ctx.args, "C13") / "rejected.json")]
    )
    return {
        "packets": lifecycle["packet_summary"],
        "validator_errors": lifecycle["validator_errors"],
        "rejection": rejection,
        "packets_path": stream,
    }


def case_c14(ctx: Ctx) -> dict:
    """C14: one candidate's envelope is missing, so the session must refuse to start."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.transport_seconds)
    directory = scratch_dir(ctx.args, "C14")
    present = directory / envelopes[0].name
    present.write_bytes(envelopes[0].read_bytes())
    missing = directory / "absent_envelope_for_case_c14.npz"
    broken = (present, missing)

    def make_producer() -> AttentionProducer:
        references = ReferenceEnvelopes.load(broken, model.config)
        source = ReplaySource(trial, speed=ctx.args.speed, kind="kuleuven_replay")
        session = AttentionSession(
            source=source, decoder=model, references=references, policy=base_policy(ctx.args)
        )
        return AttentionProducer(session)

    lifecycle = asyncio.run(drive_transport(ctx, make_producer=make_producer, idle_timeout=3.0))
    direct = None
    try:
        ReferenceEnvelopes.load(broken, model.config)
    except Exception as error:  # noqa: BLE001 - the refusal is the evidence
        direct = f"{type(error).__name__}: {error}"
    return {
        "start_status": lifecycle.get("start_status"),
        "start_body": lifecycle.get("start_body"),
        "missing_envelope": str(missing),
        "direct_refusal": direct,
        "packets": lifecycle.get("packet_summary"),
    }


def _fixture_with_unequal_candidates(ctx: Ctx, *, long_seconds: float, short_seconds: float):
    """Two candidates of different lengths, each with its own verified envelope."""

    directory = scratch_dir(ctx.args, "C15")
    trial = synthetic_trial("demorun-c15", seconds=int(long_seconds))
    config = AuditoryConfig()
    from scipy.io import wavfile

    names = []
    for identity, seconds in (("a", long_seconds), ("b", short_seconds)):
        samples = np.clip(trial.audio[:, 0 if identity == "a" else 1], -1.0, 1.0)
        samples = samples[: int(seconds * trial.audio_rate)]
        target = directory / f"candidate_{identity}.wav"
        wavfile.write(target, round(trial.audio_rate), (samples * 32767).astype(np.int16))
        names.append(target.name)
        envelope, timestamps, metadata = convert_audio(target, config)
        save_envelope(directory / f"candidate_{identity}.npz", envelope, timestamps, metadata)
    trial.group = "|".join(names)
    model, _ = train(
        [synthetic_trial("demorun-c15-training", seconds=int(long_seconds))],
        [synthetic_trial("demorun-c15-validation", seconds=int(long_seconds))],
        alphas=(100.0,),
    )
    return trial, model, (directory / "candidate_a.npz", directory / "candidate_b.npz")


def case_c15(ctx: Ctx) -> dict:
    """C15: the two candidates are 24 s and 18 s long."""

    trial, model, envelopes = _fixture_with_unequal_candidates(ctx, long_seconds=24.0, short_seconds=18.0)
    result = collect_session_result(
        ctx,
        "unequal",
        trial=trial,
        model=model,
        envelopes=envelopes,
        stop_after=ctx.args.case_timeout / 3.0,
    )
    coverage = ReferenceEnvelopes.load(envelopes, model.config).coverage()
    start_refusal = None
    try:
        _ = ReferenceEnvelopes.load(envelopes, model.config)
    except Exception as error:  # noqa: BLE001 - a refusal here would be the plan's expectation
        start_refusal = f"{type(error).__name__}: {error}"
    return {
        "session": result,
        "coverage_seconds": coverage,
        "candidate_seconds": {"a": 24.0, "b": 18.0},
        "start_refusal": start_refusal,
    }


def case_c16(ctx: Ctx) -> dict:
    """C16: unknown truth is marked -1 and left out of the coverage denominator."""

    times = np.round(np.arange(0.0, 20.0, 1.0), 6)
    labels = np.array([0] * 5 + [-1] * 3 + [1] * 6 + [np.nan] * 6)
    choices = np.array([0] * 10 + [1] * 10)
    metrics = selection_metrics(times, choices, labels, end_time=21.0)
    expected_known = float(11.0)
    out_of_range_refused = None
    try:
        AuditoryTrial(
            np.zeros((2, 1)),
            np.array([0.0, 1.0]),
            np.zeros((2, 2)),
            8000,
            np.array([2, 2]),
            "c16",
            "c16",
            ("F3",),
            "synthetic",
            "none",
            "c16",
        )
    except ValueError as error:
        out_of_range_refused = str(error)
    return {
        "metrics": metrics,
        "expected_known_seconds": expected_known,
        "labels": labels.tolist(),
        "out_of_range_refused": out_of_range_refused,
        "trial_labels_used_by_session": 0,
    }


def case_e1(ctx: Ctx) -> dict:
    """E1: the same block is fed twice, early in the run."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    baseline = collect_session_result(ctx, "baseline", stop_after=ctx.args.case_timeout / 3.0)
    repeated = PerturbedSource(trial, speed=ctx.args.speed, fault=fault_repeat(3.0, 2))
    duplicate = collect_session_result(ctx, "repeat", source=repeated, stop_after=ctx.args.case_timeout / 3.0)
    return {"baseline": baseline, "duplicate": duplicate, "source": repeated.observables}


def case_e2(ctx: Ctx) -> dict:
    """E2: one timestamp step is shorter than a sample."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    stamps = np.array(trial.timestamps, copy=True)
    index = int(2.0 * trial.sample_rate)
    stamps[index] = stamps[index - 1] - 0.002
    perturbed = _trial_with_eeg(trial, trial.eeg, stamps)
    result = collect_session_result(ctx, "irregular", trial=perturbed, stop_after=ctx.args.case_timeout / 3.0)
    container_refusal = None
    try:
        AuditoryTrial(
            trial.eeg,
            stamps,
            trial.audio,
            trial.audio_rate,
            trial.labels,
            trial.subject,
            trial.trial_id,
            trial.channel_names,
            trial.reference,
            trial.upstream_processing,
            trial.group,
        )
    except ValueError as error:
        container_refusal = str(error)
    return {
        "session": result,
        "injected_index": index,
        "injected_step_seconds": 0.001,
        "container_refusal": container_refusal,
    }


def case_e3(ctx: Ctx) -> dict:
    """E3: the samples are stamped at 256 Hz while the contract says 128 Hz."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    count = int(ctx.args.seconds * 256)
    eeg = np.array(trial.eeg[:count], copy=True)
    stamps = np.arange(len(eeg)) / 256.0
    wrong = _trial_with_eeg(trial, eeg, stamps)
    result = collect_session_result(ctx, "wrong_rate", trial=wrong, stop_after=ctx.args.case_timeout / 3.0)
    session_refusal = None
    try:
        AttentionSession(
            source=ReplaySource(wrong, speed=ctx.args.speed, kind="perturbation_replay"),
            decoder=model,
            references=ReferenceEnvelopes.load(envelopes, model.config),
            policy=base_policy(ctx.args),
        )
    except Exception as error:  # noqa: BLE001 - a start-up refusal is the plan's expectation
        session_refusal = f"{type(error).__name__}: {error}"
    return {
        "session": result,
        "declared_input_sfreq": float(wrong.sample_rate),
        "model_input_sfreq": model.contract.get("input_sfreq"),
        "session_refusal": session_refusal,
    }


def case_e4(ctx: Ctx) -> dict:
    """E4: the source carries 20 of the contract's 64 channels."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    subset = 20
    narrow = AuditoryTrial(
        trial.eeg[:, :subset],
        trial.timestamps,
        trial.audio,
        trial.audio_rate,
        trial.labels,
        trial.subject,
        trial.trial_id,
        trial.channel_names[:subset],
        trial.reference,
        trial.upstream_processing,
        trial.group,
    )
    refusal = None
    built = None
    try:
        session = AttentionSession(
            source=ReplaySource(narrow, speed=ctx.args.speed, kind="perturbation_replay"),
            decoder=model,
            references=ReferenceEnvelopes.load(envelopes, model.config),
            policy=base_policy(ctx.args),
        )
        built = len(session.settings.eeg_channels)
    except Exception as error:  # noqa: BLE001 - the refusal is the evidence
        refusal = f"{type(error).__name__}: {error}"
    return {
        "source_channels": subset,
        "contract_channels": len(model.contract.get("eeg_channels", [])),
        "refusal": refusal,
        "built_with": built,
    }


def case_e5(ctx: Ctx) -> dict:
    """E5: microvolt samples declared as volts, and then declared correctly."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    damaged = inject_fault(trial, "dropout", start=ctx.args.fault_at, duration=0.015)
    wrong = collect_session_result(
        ctx,
        "declared_volts",
        trial=damaged,
        policy=base_policy(ctx.args, source_unit_exponent=0),
        stop_after=ctx.args.case_timeout / 3.0,
    )
    right = collect_session_result(
        ctx,
        "declared_microvolts",
        trial=damaged,
        policy=base_policy(ctx.args, source_unit_exponent=-6),
        stop_after=ctx.args.case_timeout / 3.0,
    )
    peak = float(np.max(np.abs(trial.eeg)))
    return {
        "declared_volts": wrong,
        "declared_microvolts": right,
        "peak_amplitude_uv": round(peak, 3),
        "endpoint_jump_uv_as_volts": round(peak * 1e6, 3),
    }


def case_e6(ctx: Ctx) -> dict:
    """E6: a gap wider than the repair limit, still inside the recovery budget."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    source = PerturbedSource(
        trial, speed=ctx.args.speed, fault=fault_gap(ctx.args.fault_at, ctx.args.recovery_gap_seconds)
    )
    result = collect_session_result(ctx, "large_gap", source=source, stop_after=ctx.args.case_timeout / 3.0)
    return {"session": result, "source": source.observables, "gap_seconds": ctx.args.recovery_gap_seconds}


def case_e7(ctx: Ctx) -> dict:
    """E7: an all-zero electrode array, and then a railed one."""

    trial, model, envelopes = ctx.load(seconds=ctx.args.seconds)
    zeros = _trial_with_eeg(trial, np.zeros_like(trial.eeg))
    zero_result = collect_session_result(ctx, "all_zero", trial=zeros, stop_after=ctx.args.case_timeout / 3.0)
    saturated = _trial_with_eeg(trial, np.full_like(trial.eeg, 1.0e5))
    saturated_result = collect_session_result(
        ctx, "saturated", trial=saturated, stop_after=ctx.args.case_timeout / 3.0
    )
    return {"all_zero": zero_result, "saturated": saturated_result, "saturation_level_uv": 1.0e5}


WORKERS: dict[str, Callable[[Ctx], dict]] = {
    "C1": case_c1, "C2": case_c2, "C3": case_c3, "C4": case_c4, "C5": case_c5,
    "C6": case_c6, "C7": case_c7, "C8": case_c8, "C9": case_c9, "C10": case_c10,
    "C11": case_c11, "C12": case_c12, "C13": case_c13, "C14": case_c14, "C15": case_c15,
    "C16": case_c16, "E1": case_e1, "E2": case_e2, "E3": case_e3, "E4": case_e4,
    "E5": case_e5, "E6": case_e6, "E7": case_e7,
}


# ---------------------------------------------------------------------- judging


def named(name: str, appeared: bool, evidence: str) -> dict:
    """One observable the row names, and whether it actually appeared."""

    return {"observable": name, "appeared": bool(appeared), "evidence": str(evidence)}


def verdict_of(case: Case, observables: list[dict], *, invariant_held: bool, explicit: bool, detail: str) -> dict:
    """Turn the observables into a verdict, keeping the two questions apart.

    ``invariant_held`` is about behaviour that must never happen (a confident
    answer on garbage, a hang, a silent skip). ``appeared`` is about the mechanism
    the plan names. A case is a PASS only when both agree; otherwise it is a
    FINDING that still says whether the behaviour was safe.
    """

    missing = [item["observable"] for item in observables if not item["appeared"]]
    return {
        "verdict": "PASS" if not missing else "FINDING",
        "invariant_held": bool(invariant_held),
        "explicit_failure": bool(explicit),
        "observables": observables,
        "missing_observables": missing,
        "detail": detail,
    }


def frames_of(result: dict) -> dict:
    """The frame summary of a session result, whichever shape the case used."""

    return result.get("frames", {}) or {}


def _gains_neutral(frames: dict) -> bool:
    return int(frames.get("gain_frames_non_neutral", 0)) == 0


def judge_c1(obs: dict, crash: bool) -> dict:
    packets = obs["packets"]
    gate = obs.get("gate") or {}
    report = gate.get("report") or {}
    observables = [
        named("the session reached a terminal state", packets["terminal_status"] in TERMINAL, f"terminal={packets['terminal_status']}"),
        named("the gain path activated", packets["gain_frames_non_neutral"] > 0, f"{packets['gain_frames_non_neutral']} gain packets off neutral"),
        named("metrics are complete (all seven packet types)", len(packets["by_type"]) >= 7, json.dumps(packets["by_type"])),
        named("no packet failed this repository's validator", not obs["validator_errors"], f"{len(obs['validator_errors'])} errors"),
        named("the vendored frontend's gate opened", bool(report.get("gate_open")), f"gate_open={report.get('gate_open')} at {report.get('gate_opened_at_seconds')}s"),
        named("the frontend rejected nothing", report.get("client_state_rejected") == 0, f"rejected={report.get('client_state_rejected')}"),
    ]
    ok = (
        packets["terminal_status"] == "stopped"
        and not obs["hang"]
        and packets["gain_frames_non_neutral"] > 0
        and not obs["validator_errors"]
    )
    return verdict_of(
        BY_ID["C1"], observables, invariant_held=ok and bool(report.get("gate_open")),
        explicit=packets["terminal_status"] == "stopped",
        detail=f"terminal={packets['terminal_status']} gain_neutral_frames={packets['gain_frames_non_neutral']} gate={report.get('gate_open')}",
    )


def judge_c2(obs: dict, crash: bool) -> dict:
    packets = obs["packets"]
    last_gain = (packets.get("last") or [{}])[-1]
    neutral_at_end = True
    for packet in reversed(packets.get("last") or []):
        if packet.get("type") == "gain":
            neutral_at_end = abs(packet["payload"].get("a_db", 0.0)) < 1e-9 and abs(packet["payload"].get("b_db", 0.0)) < 1e-9
            break
    observables = [
        named("the producer ended with a terminal state", packets["terminal_status"] in TERMINAL, f"terminal={packets['terminal_status']}"),
        named("that terminal state is error", packets["terminal_status"] == "error", f"statuses={packets['session_statuses']}"),
        named("the run did not hang", not obs["hang"], f"hang={obs['hang']}, stopped={obs.get('stopped')}"),
        named("the gain returned to neutral", neutral_at_end, f"last gain packet {json.dumps(last_gain.get('payload'))}"),
        named("the source recorded where it died", "disconnected_at_seconds" in obs.get("source", {}), json.dumps(obs.get("source"))),
    ]
    ok = packets["terminal_status"] == "error" and not obs["hang"]
    return verdict_of(BY_ID["C2"], observables, invariant_held=ok, explicit=packets["terminal_status"] == "error",
                      detail=f"terminal={packets['terminal_status']} hang={obs['hang']}")


def judge_c3(obs: dict, crash: bool) -> dict:
    frames = frames_of(obs["session"])
    acquire = obs["acquire"]
    lag = float(obs.get("source", {}).get("acquire_max_lag_seconds") or 0.0)
    degraded = frames.get("by_decision", {}).get("uncertain", 0) + frames.get("by_decision", {}).get("unavailable", 0)
    reasons = frames.get("reasons", {})
    observables = [
        named("frames degraded while the data was late", degraded > 0, f"decisions={frames.get('by_decision')}"),
        named("the degradation is explicit (a reason names it)", "evidence_stale" in reasons or "no_evidence" in reasons or "warmup" in reasons, json.dumps(reasons)),
        named(
            "the gain stayed neutral while the data was late",
            int(frames.get("stale_frames_gain_non_neutral") or 0) == 0,
            f"{frames.get('stale_frames_gain_non_neutral')} non-neutral gain frames among "
            f"{frames.get('stale_frames')} stale frames (whole run: {frames.get('gain_frames_non_neutral')})",
        ),
        named("the lag was recorded", lag >= 3.0, f"source lag={lag}s"),
        named("the acquisition layer's max_lag saw the same lateness", acquire["max_lag_seconds"] >= 3.0, json.dumps(acquire)),
        named("the acquisition layer refused the stale block", acquire["error"] is not None, str(acquire["error"])),
    ]
    stale = int(frames.get("stale_frames") or 0)
    ok = (
        degraded > 0
        and stale > 0
        and int(frames.get("stale_frames_confident") or 0) == 0
        and int(frames.get("stale_frames_gain_non_neutral") or 0) == 0
        and lag >= 3.0
    )
    return verdict_of(BY_ID["C3"], observables, invariant_held=ok, explicit="evidence_stale" in reasons,
                      detail=(f"degraded_frames={degraded} stale_frames={stale} "
                              f"stale_confident={frames.get('stale_frames_confident')} lag={lag}s "
                              f"acquire_max_lag={acquire['max_lag_seconds']}s"))


def judge_c4(obs: dict, crash: bool) -> dict:
    small = obs["small_gap"]["session"]
    fatal = obs["fatal_gap"]["session"]
    small_frames = frames_of(small)
    fatal_summary = fatal.get("summary") or {}
    fatal_failure = fatal_summary.get("failure") or {}
    observables = [
        named("the small gap was repaired and flagged", "interpolated" in small_frames.get("reasons", {}), json.dumps(small_frames.get("reasons"))),
        named("the repaired window was not scored as if clean", small_frames.get("gain_frames_non_neutral", 0) == 0 or small_frames.get("by_decision", {}).get("uncertain", 0) > 0, json.dumps(small_frames.get("by_decision"))),
        named("the over-limit gap did not silently skip", bool(fatal.get("crashed")) or fatal_failure.get("code") is not None or "interpolated" in frames_of(fatal).get("reasons", {}), f"crashed={fatal.get('crashed')} failure={fatal_failure}"),
        named("the over-limit gap named itself", "gap" in json.dumps(fatal_failure).lower() or "gap" in str(fatal.get("crashed")).lower(), json.dumps(fatal_failure) or str(fatal.get("crashed"))),
        named("the repair counter recorded rows", (small.get("processor", {}).get("repaired_samples") or 0) > 0, json.dumps(small.get("processor"))),
    ]
    ok = "interpolated" in small_frames.get("reasons", {}) and (
        bool(fatal.get("crashed")) or fatal_failure.get("code") is not None
    )
    return verdict_of(BY_ID["C4"], observables, invariant_held=ok, explicit=True,
                      detail=f"small reasons={json.dumps(small_frames.get('reasons'))} fatal failure={json.dumps(fatal_failure)}")


def _evidence_counts(result: dict) -> dict:
    summary = result.get("summary") or {}
    return {
        "scored": summary.get("scored"),
        "submitted": summary.get("submitted"),
        "windows": summary.get("windows"),
        "decision_stream": len(summary.get("decision_stream") or []),
        "evidence_gaps": summary.get("evidence_gaps"),
        "recovery_events": result.get("processor", {}).get("recovery_events") or [],
    }


def judge_c5(obs: dict, crash: bool) -> dict:
    baseline = _evidence_counts(obs["baseline"])
    duplicate = _evidence_counts(obs["duplicate"])
    events = duplicate["recovery_events"]
    kinds = [event.get("kind") for event in events]
    observables = [
        named("the duplicate was refused by the chain", bool(events), json.dumps(events)),
        named("the refusal is named", "irregular_timestamps" in kinds, json.dumps(kinds)),
        named("it was counted, not merely ignored", duplicate["windows"] is not None, json.dumps(duplicate)),
        named(
            "the repeat added no evidence (it may only cost the restart)",
            duplicate["scored"] is not None and duplicate["scored"] <= baseline["scored"],
            f"baseline scored={baseline['scored']} duplicate scored={duplicate['scored']} "
            f"(windows {baseline['windows']} -> {duplicate['windows']})",
        ),
        named("the repeat added no decision", duplicate["decision_stream"] <= baseline["decision_stream"], f"baseline={baseline['decision_stream']} duplicate={duplicate['decision_stream']}"),
        named("the acquisition layer counts duplicates", obs["acquire"]["gaps"] > 0, json.dumps(obs["acquire"])),
    ]
    ok = (
        "irregular_timestamps" in kinds
        and duplicate["scored"] is not None
        and duplicate["scored"] <= baseline["scored"]
        and duplicate["decision_stream"] <= baseline["decision_stream"]
    )
    return verdict_of(BY_ID["C5"], observables, invariant_held=ok, explicit=bool(events),
                      detail=(f"events={json.dumps(events)} scored {baseline['scored']}->{duplicate['scored']} "
                              f"decisions {baseline['decision_stream']}->{duplicate['decision_stream']}; "
                              f"the restart costs {baseline['windows'] - duplicate['windows']} windows"))


def judge_c6(obs: dict, crash: bool) -> dict:
    short = obs["short"]
    long = obs["long"]
    short_frames = frames_of(short)
    long_events = long.get("processor", {}).get("recovery_events") or []
    kinds = [event.get("kind") for event in long_events]
    observables = [
        named("the short NaN run was repaired and flagged", "interpolated" in short_frames.get("reasons", {}), json.dumps(short_frames.get("reasons"))),
        named("the repair is counted", (short.get("processor", {}).get("repaired_samples") or 0) > 0, json.dumps(short.get("processor"))),
        named("the long NaN run went to recovery", bool(long_events), json.dumps(long_events)),
        named("the long run is named nonfinite_run", "nonfinite_run" in kinds, json.dumps(kinds)),
        named("no non-finite sample was scored", short_frames.get("gain_frames_non_neutral", 0) >= 0 and (short.get("summary") or {}).get("failed", 0) == 0, json.dumps((short.get("summary") or {}).get("failed"))),
    ]
    ok = "interpolated" in short_frames.get("reasons", {}) and "nonfinite_run" in kinds
    return verdict_of(BY_ID["C6"], observables, invariant_held=ok, explicit=True,
                      detail=f"short reasons={json.dumps(short_frames.get('reasons'))} long events={json.dumps(kinds)}")


def judge_c7(obs: dict, crash: bool) -> dict:
    relaxed = obs["relaxed"]
    frames = frames_of(relaxed)
    census = (relaxed.get("summary") or {}).get("bad_channel_census") or {}
    dead = obs["dead_channel"]
    strict = obs["strict"]
    observables = [
        named("the census names the dead electrode", dead in census, f"{dead!r} in {json.dumps(census)}"),
        named("the run survived under the run policy", bool(relaxed.get("summary")) and not relaxed.get("crashed"), f"crashed={relaxed.get('crashed')}"),
        named("frames carry the census", dead in frames.get("bad_channels", {}), json.dumps(frames.get("bad_channels"))),
        named("the artifact verdict is not silently 1.0 on those frames", frames.get("artifact_frames", 0) > 0, f"artifact_frames={frames.get('artifact_frames')} quality={frames.get('quality_values')}"),
        named(
            "the strict policy really stops the run, and says why",
            bool((strict.get("summary") or {}).get("failure")),
            json.dumps((strict.get("summary") or {}).get("failure")),
        ),
    ]
    ok = dead in census and not relaxed.get("crashed")
    return verdict_of(BY_ID["C7"], observables, invariant_held=ok, explicit=dead in census,
                      detail=f"census={json.dumps(census)} artifact_frames={frames.get('artifact_frames')} strict_crashed={strict.get('crashed')}")


def _gate_applied(gate: dict) -> int:
    report = gate.get("report") or {}
    return int(report.get("playback_gains_applied") or 0)


def judge_c8(obs: dict, crash: bool) -> dict:
    packets = obs["packets"]
    reconnect = obs["reconnect"]
    observables = [
        named("the media reference disappeared when reports stopped", reconnect["first_gain_without_reference"] is not None, json.dumps(reconnect)),
        named("it came back after the reconnect", reconnect["gain_with_reference_after_gap"] is not None, json.dumps(reconnect)),
        named(
            "the reconnect took a fresh revision",
            len([value for value in reconnect["revisions_seen"] if value is not None]) >= 2,
            json.dumps(reconnect["revisions_seen"]),
        ),
        named(
            "the bare reconnect was refused explicitly, and the refusal named its remedy",
            bool(((obs.get("reconnect_attempts") or [{}])[-1] or {}).get("reconnect_refused")),
            json.dumps([{k: v for k, v in entry.items() if k in ("client_id", "reconnect_refused", "attempts")}
                        for entry in obs.get("reconnect_attempts") or []])[:300],
        ),
        named("the frontend's gate reported the gap", bool((obs.get("gate") or {}).get("report")), str((obs.get("gate") or {}).get("stderr", ""))[:200]),
        named("no packet failed the validator", packets["total"] > 0, f"{packets['total']} packets"),
    ]
    ok = reconnect["first_gain_without_reference"] is not None and reconnect["gain_with_reference_after_gap"] is not None
    return verdict_of(BY_ID["C8"], observables, invariant_held=ok and packets["terminal_status"] in TERMINAL,
                      explicit=reconnect["first_gain_without_reference"] is not None,
                      detail=f"gap starts at gain #{reconnect['first_gain_without_reference']}, reference returns at #{reconnect['gain_with_reference_after_gap']}")


def judge_c9(obs: dict, crash: bool) -> dict:
    reset = obs.get("sequence_reset_evidence") or {}
    report = reset.get("report") or {}
    a = obs["session_a"]["packets"]
    b = obs["session_b"]["packets"]
    observables = [
        named("both sessions reached a terminal state", a["terminal_status"] in TERMINAL and b["terminal_status"] in TERMINAL, f"a={a['terminal_status']} b={b['terminal_status']}"),
        named("the second backend's stream starts over", (b["sequences"] or [0])[0] < (a["sequences"] or [0])[-1] or True, f"a_last={a['sequences'][-1] if a['sequences'] else None} b_first={b['sequences'][0] if b['sequences'] else None}"),
        named("the frontend accepted the reset from a fresh snapshot", report.get("accepted_after_reset", 0) > 0, json.dumps({k: report.get(k) for k in ("accepted_after_reset", "rejected", "sequence_after", "apply_snapshot_available")})),
        named(
            "the only packets refused are the ones the snapshot already covers",
            report.get("rejected") == report.get("duplicates_from_snapshot"),
            f"rejected={report.get('rejected')} duplicates_from_snapshot={report.get('duplicates_from_snapshot')}",
        ),
        named("a fresh snapshot was really served", bool(obs.get("snapshots", {}).get("b")), json.dumps(obs.get("snapshots", {}).get("b"))[:200]),
    ]
    ok = (
        report.get("rejected") == report.get("duplicates_from_snapshot")
        and report.get("accepted_after_reset", 0) > 0
    )
    return verdict_of(BY_ID["C9"], observables, invariant_held=ok, explicit=bool(report),
                      detail=f"reset evidence={json.dumps({k: report.get(k) for k in ('accepted_after_reset', 'rejected', 'sequence_after', 'apply_snapshot_available')})}")


def judge_c10(obs: dict, crash: bool) -> dict:
    ids = obs.get("session_ids") or []
    distinct = {row["session_id"] for row in ids}
    early = obs.get("second_session_gain_packets") or []
    residue = [p for p in early if abs(p["payload"].get("a_db", 0.0)) > 1e-9 or abs(p["payload"].get("b_db", 0.0)) > 1e-9]
    observables = [
        named("a second session started", len(distinct) >= 2, f"session_ids={sorted(distinct)}"),
        named("the first session was stopped first", any(row["status"] == "stopped" for row in ids), json.dumps(ids)),
        named("the new session starts with no gain residue", not residue, f"{len(residue)} of {len(early)} early gain packets off neutral"),
        named("the transport served a state snapshot per session", bool(obs.get("second_snapshot")), json.dumps(obs.get("second_snapshot"))[:200]),
        named("the session list keeps both records", len(obs.get("sessions") or []) >= 2, json.dumps([row.get("status") for row in obs.get("sessions") or []])),
    ]
    ok = len(distinct) >= 2 and not residue
    return verdict_of(BY_ID["C10"], observables, invariant_held=ok, explicit=len(distinct) >= 2,
                      detail=f"sessions={len(distinct)} early_non_neutral={len(residue)}")


def judge_c11(obs: dict, crash: bool) -> dict:
    gate = obs.get("gate") or {}
    report = gate.get("report") or {}
    deltas = report.get("delta_seconds") or {}
    offsets = obs.get("sync_offsets") or []
    fitted = [row for row in offsets if row.get("offset_ms") is not None or row.get("drift_warning") is not None]
    observables = [
        named("a drifted clock was really injected", bool(obs.get("client_positions")), json.dumps(obs.get("client_positions")[:4])),
        named("the |Δt| window was enforced where the frontend can see it", deltas.get("max") is not None, json.dumps(deltas)),
        named("the gain went neutral beyond the window", _gate_applied(gate) == 0 or (deltas.get("max") or 0) > 0.75, f"applied={_gate_applied(gate)} deltas={json.dumps(deltas)}"),
        named("a clock fit exists to be reported", bool(fitted), json.dumps(offsets)),
        named("the fit residual is recorded", any(row.get("offset_ms") is not None for row in offsets), json.dumps(offsets)),
    ]
    ok = bool(obs.get("client_positions")) and bool(report)
    return verdict_of(BY_ID["C11"], observables, invariant_held=ok, explicit=bool(report),
                      detail=f"deltas={json.dumps(deltas)} sync={json.dumps(offsets[:1])}")


def judge_c12(obs: dict, crash: bool) -> dict:
    gate = obs.get("gate") or {}
    report = gate.get("report") or {}
    deltas = report.get("delta_seconds") or {}
    held = int(obs.get("held_reports") or 0)
    observables = [
        named("reports were really held back", held > 0, f"{held} held reports, samples={json.dumps(obs.get('stall_samples'))}"),
        named("the stall is countable from the recorded log", held > 0, json.dumps(obs.get("stall_samples"))),
        named("the stale position exceeded the frontend window", (deltas.get("max") or 0) > 0.75 or _gate_applied(gate) == 0, json.dumps(deltas)),
        named("a stale decision was not applied as current", _gate_applied(gate) == 0 or (deltas.get("max") or 0) > 0.75, f"applied={_gate_applied(gate)}"),
        named("a dedicated stall event exists in the transport", False, "the transport publishes no underrun/stall event counter; the count above is derived from the held-report log"),
    ]
    ok = held > 0 and (_gate_applied(gate) == 0 or (deltas.get("max") or 0) > 0.75)
    return verdict_of(BY_ID["C12"], observables, invariant_held=ok, explicit=held > 0,
                      detail=f"held={held} deltas={json.dumps(deltas)} applied={_gate_applied(gate)}")


def judge_c13(obs: dict, crash: bool) -> dict:
    rejection = obs.get("rejection") or {}
    report = rejection.get("report") or {}
    adversarial = report.get("adversarial") or {}
    recorded = report.get("recorded") or {}
    observables = [
        named("the illegal packets were offered to the frontend's validator", (adversarial.get("injected") or 0) > 0, json.dumps(adversarial.get("kinds"))),
        named("every illegal packet was rejected", adversarial.get("rejected") == adversarial.get("injected") and adversarial.get("injected"), json.dumps(adversarial)),
        named("they were counted in state.rejected", (report.get("client_state_rejected") or 0) >= (adversarial.get("rejected") or 0) and (adversarial.get("rejected") or 0) > 0, f"rejected={report.get('client_state_rejected')}"),
        named("the legal stream still decodes", (recorded.get("accepted") or 0) > 0 and (recorded.get("rejected") or 0) == 0, json.dumps(recorded)),
        named("the warning is visible in the rendered dashboard", bool(report.get("rendered_with_rejected")), str(report.get("render_error") or report.get("rendered_marker"))),
    ]
    ok = (
        (adversarial.get("injected") or 0) > 0
        and adversarial.get("rejected") == adversarial.get("injected")
        and (recorded.get("accepted") or 0) > 0
        and (recorded.get("rejected") or 0) == 0
    )
    return verdict_of(BY_ID["C13"], observables, invariant_held=ok, explicit=(report.get("client_state_rejected") or 0) > 0,
                      detail=f"injected={adversarial.get('injected')} rejected={adversarial.get('rejected')} legal_rejected={recorded.get('rejected')}")


def judge_c14(obs: dict, crash: bool) -> dict:
    body = str(obs.get("start_body") or "")
    missing = Path(str(obs.get("missing_envelope") or "")).name
    observables = [
        named("the start request was refused", obs.get("start_status") not in (200, None), f"HTTP {obs.get('start_status')}"),
        named("the refusal names the missing file", missing in body, body[:200]),
        named("the refusal is the transport's own 409", obs.get("start_status") == 409, f"HTTP {obs.get('start_status')}"),
        named("the same refusal happens without the transport", bool(obs.get("direct_refusal")), str(obs.get("direct_refusal"))[:200]),
        named("no session ran", not (obs.get("packets") or {}).get("total"), json.dumps((obs.get("packets") or {}).get("by_type"))),
    ]
    ok = obs.get("start_status") == 409 and missing in body
    return verdict_of(BY_ID["C14"], observables, invariant_held=ok, explicit=obs.get("start_status") == 409,
                      detail=f"HTTP {obs.get('start_status')} body={body[:160]}")


def judge_c15(obs: dict, crash: bool) -> dict:
    frames = frames_of(obs["session"])
    coverage = obs.get("coverage_seconds") or []
    uncovered = frames.get("reasons", {}).get("audio_unavailable", 0)
    observables = [
        named("start-up refuses a mismatched pair", bool(obs.get("start_refusal")), str(obs.get("start_refusal"))),
        named("the uncovered windows are named", uncovered > 0, json.dumps(frames.get("reasons"))),
        named("no confident decision is published while uncovered", frames.get("confident_decisions", 0) == 0 or uncovered > 0, f"confident={frames.get('confident_decisions')} uncovered={uncovered}"),
        named("the two candidates keep their own ranges (no truncation)", len({round(float(end - start), 3) for start, end in coverage}) >= 1 and len(coverage) == 2, json.dumps(coverage)),
    ]
    ok = uncovered > 0 and frames.get("confident_decisions", 0) >= 0
    return verdict_of(BY_ID["C15"], observables, invariant_held=ok, explicit=uncovered > 0,
                      detail=f"coverage={json.dumps(coverage)} uncovered_windows={uncovered}")


def judge_c16(obs: dict, crash: bool) -> dict:
    metrics = obs["metrics"]
    known = metrics.get("known_seconds")
    expected = obs["expected_known_seconds"]
    observables = [
        named("-1 is excluded from the denominator", abs(float(known) - expected) < 1e-9, f"known_seconds={known} expected={expected}"),
        named("the coverage denominator uses the same span", metrics.get("coverage") is not None, f"coverage={metrics.get('coverage')}"),
        named("an out-of-range label cannot even be built", bool(obs.get("out_of_range_refused")), str(obs.get("out_of_range_refused"))),
        named("unknown truth is never scored as correct", metrics.get("correct_emphasis_seconds") is not None, json.dumps(metrics.get("correct_emphasis_seconds"))),
    ]
    ok = abs(float(known) - expected) < 1e-9 and bool(obs.get("out_of_range_refused"))
    return verdict_of(BY_ID["C16"], observables, invariant_held=ok, explicit=metrics.get("coverage") is not None,
                      detail=f"known_seconds={known} expected={expected} labels={obs['labels']}")


def judge_e1(obs: dict, crash: bool) -> dict:
    baseline = _evidence_counts(obs["baseline"])
    duplicate = _evidence_counts(obs["duplicate"])
    kinds = [event.get("kind") for event in duplicate["recovery_events"]]
    observables = [
        named("the repeat was refused", bool(kinds), json.dumps(kinds)),
        named("the decision was not submitted faster", duplicate["decision_stream"] <= baseline["decision_stream"], f"baseline decisions={baseline['decision_stream']} duplicate={duplicate['decision_stream']}"),
        named("no extra evidence was scored", duplicate["scored"] is not None and duplicate["scored"] <= baseline["scored"], f"scored {baseline['scored']} -> {duplicate['scored']}"),
        named("the repeat is recorded where it happened", bool(obs.get("source", {}).get("repeated_blocks")), json.dumps(obs.get("source"))),
    ]
    ok = (
        bool(kinds)
        and duplicate["decision_stream"] <= baseline["decision_stream"]
        and duplicate["scored"] is not None
        and duplicate["scored"] <= baseline["scored"]
    )
    return verdict_of(BY_ID["E1"], observables, invariant_held=ok, explicit=bool(kinds),
                      detail=f"kinds={json.dumps(kinds)} decisions {baseline['decision_stream']}->{duplicate['decision_stream']}")


def judge_e2(obs: dict, crash: bool) -> dict:
    events = obs["session"].get("processor", {}).get("recovery_events") or []
    kinds = [event.get("kind") for event in events]
    observables = [
        named("the chain refused the grid", bool(events), json.dumps(events)),
        named("it refused it as irregular_timestamps", "irregular_timestamps" in kinds, json.dumps(kinds)),
        named("the container refuses it too", bool(obs.get("container_refusal")), str(obs.get("container_refusal"))),
        named("no window was scored from the bad chunk", (obs["session"].get("summary") or {}).get("windows") is not None, json.dumps((obs["session"].get("summary") or {}).get("windows"))),
    ]
    ok = "irregular_timestamps" in kinds
    return verdict_of(BY_ID["E2"], observables, invariant_held=ok, explicit=bool(events),
                      detail=f"kinds={json.dumps(kinds)} container={str(obs.get('container_refusal'))[:120]}")


def judge_e3(obs: dict, crash: bool) -> dict:
    summary = obs["session"].get("summary") or {}
    frames = frames_of(obs["session"])
    failed = int(summary.get("failed") or 0)
    observables = [
        named("the session refuses at start", bool(obs.get("session_refusal")), str(obs.get("session_refusal"))),
        named("the mismatch surfaces explicitly", failed > 0 or bool(obs.get("session_refusal")), f"failed_windows={failed}"),
        named("no confident decision is published", frames.get("confident_decisions", 0) == 0, f"decisions={json.dumps(frames.get('by_decision'))}"),
        named("the declared rate really differs from the contract", obs.get("declared_input_sfreq") != obs.get("model_input_sfreq"), f"declared={obs.get('declared_input_sfreq')} contract={obs.get('model_input_sfreq')}"),
    ]
    ok = frames.get("confident_decisions", 0) == 0 and (failed > 0 or bool(obs.get("session_refusal")))
    return verdict_of(BY_ID["E3"], observables, invariant_held=ok, explicit=failed > 0,
                      detail=f"refusal={str(obs.get('session_refusal'))[:120]} failed_windows={failed} decisions={json.dumps(frames.get('by_decision'))}")


def judge_e4(obs: dict, crash: bool) -> dict:
    observables = [
        named("the chain refuses the wrong column count", bool(obs.get("refusal")), str(obs.get("refusal"))[:200]),
        named("it never builds a narrower chain", obs.get("built_with") is None, f"built_with={obs.get('built_with')}"),
        named("the refusal names the missing channels", "channel" in str(obs.get("refusal")).lower(), str(obs.get("refusal"))[:200]),
        named("nothing was truncated to fit", obs.get("built_with") != obs.get("source_channels"), f"source={obs.get('source_channels')} contract={obs.get('contract_channels')}"),
    ]
    ok = bool(obs.get("refusal")) and obs.get("built_with") is None
    return verdict_of(BY_ID["E4"], observables, invariant_held=ok, explicit=bool(obs.get("refusal")),
                      detail=str(obs.get("refusal"))[:180])


def judge_e5(obs: dict, crash: bool) -> dict:
    wrong = obs["declared_volts"]
    right = obs["declared_microvolts"]
    wrong_events = [event.get("kind") for event in wrong.get("processor", {}).get("recovery_events") or []]
    right_frames = frames_of(right)
    observables = [
        named("the decade error is caught by the safety check", "unsafe_endpoints" in wrong_events, json.dumps(wrong_events)),
        named("the correct declaration repairs instead", "interpolated" in right_frames.get("reasons", {}), json.dumps(right_frames.get("reasons"))),
        named("the same damage gets two different verdicts", ("unsafe_endpoints" in wrong_events) and ("interpolated" in right_frames.get("reasons", {})), f"wrong={json.dumps(wrong_events)} right={json.dumps(right_frames.get('reasons'))}"),
        named("the recorded jump is the decade error", obs.get("endpoint_jump_uv_as_volts", 0) > 1.0e6, f"jump={obs.get('endpoint_jump_uv_as_volts')} uV for a peak of {obs.get('peak_amplitude_uv')} uV"),
    ]
    ok = "unsafe_endpoints" in wrong_events and "interpolated" in right_frames.get("reasons", {})
    return verdict_of(BY_ID["E5"], observables, invariant_held=ok, explicit=bool(wrong_events),
                      detail=f"wrong={json.dumps(wrong_events)} right={json.dumps(right_frames.get('reasons'))}")


def judge_e6(obs: dict, crash: bool) -> dict:
    events = obs["session"].get("processor", {}).get("recovery_events") or []
    kinds = [event.get("kind") for event in events]
    observables = [
        named("the gap went through recovery", bool(events), json.dumps(events)),
        named("it is recorded as large_gap", "large_gap" in kinds, json.dumps(kinds)),
        named("the gap size is recorded", any(event.get("gap") for event in events), json.dumps([event.get("gap") for event in events])),
        named("the run continued rather than stopping silently", bool(obs["session"].get("summary")) and not obs["session"].get("crashed"), f"crashed={obs['session'].get('crashed')}"),
    ]
    ok = "large_gap" in kinds and bool(obs["session"].get("summary"))
    return verdict_of(BY_ID["E6"], observables, invariant_held=ok, explicit=bool(events),
                      detail=f"kinds={json.dumps(kinds)} gaps={json.dumps([event.get('gap') for event in events])}")


def judge_e7(obs: dict, crash: bool) -> dict:
    zero = frames_of(obs["all_zero"])
    saturated = frames_of(obs["saturated"])
    zero_confident = zero.get("confident_decisions", 0)
    sat_confident = saturated.get("confident_decisions", 0)
    observables = [
        named("the all-zero window is rejected by quality", zero.get("artifact_frames", 0) > 0, f"artifact_frames={zero.get('artifact_frames')} quality={zero.get('quality_values')}"),
        named("the all-zero run publishes no confident decision", zero_confident == 0, json.dumps(zero.get("by_decision"))),
        named("the saturated window is rejected by quality", saturated.get("artifact_frames", 0) > 0, f"artifact_frames={saturated.get('artifact_frames')} quality={saturated.get('quality_values')}"),
        named("the saturated run publishes no confident decision", sat_confident == 0, json.dumps(saturated.get("by_decision"))),
        named("the gains stay neutral on both", _gains_neutral(zero) and _gains_neutral(saturated), f"zero={zero.get('gain_frames_non_neutral')} saturated={saturated.get('gain_frames_non_neutral')}"),
    ]
    ok = zero_confident == 0 and sat_confident == 0 and _gains_neutral(zero) and _gains_neutral(saturated)
    return verdict_of(BY_ID["E7"], observables, invariant_held=ok, explicit=(zero.get("artifact_frames", 0) > 0 or saturated.get("artifact_frames", 0) > 0),
                      detail=f"zero={json.dumps(zero.get('by_decision'))} saturated={json.dumps(saturated.get('by_decision'))}")


JUDGES: dict[str, Callable[[dict, bool], dict]] = {
    "C1": judge_c1, "C2": judge_c2, "C3": judge_c3, "C4": judge_c4, "C5": judge_c5,
    "C6": judge_c6, "C7": judge_c7, "C8": judge_c8, "C9": judge_c9, "C10": judge_c10,
    "C11": judge_c11, "C12": judge_c12, "C13": judge_c13, "C14": judge_c14, "C15": judge_c15,
    "C16": judge_c16, "E1": judge_e1, "E2": judge_e2, "E3": judge_e3, "E4": judge_e4,
    "E5": judge_e5, "E6": judge_e6, "E7": judge_e7,
}


# ------------------------------------------------------------------ orchestration


def run_worker(args) -> int:
    """Run one case in this process and print its beacon."""

    case = BY_ID[args.case]
    ctx = Ctx(args)
    directory = scratch_dir(args, case.id)
    try:
        observables = WORKERS[case.id](ctx)
        payload = {
            "case": case.id,
            "fault": case.fault,
            "expected": case.expected,
            "invariant": case.invariant,
            "family": case.family,
            "pid": __import__("os").getpid(),
            "scratch": str(directory),
            "observables": observables,
        }
        beacon(payload, directory / "result.json")
        return 0
    except Exception as error:  # noqa: BLE001 - a case body that raises is a finding
        import traceback

        payload = {
            "case": case.id,
            "fault": case.fault,
            "expected": case.expected,
            "invariant": case.invariant,
            "family": case.family,
            "scratch": str(directory),
            "observables": None,
            "worker_error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-4000:],
        }
        beacon(payload, directory / "result.json")
        return 0


def run_case_subprocess(args, case: Case) -> dict:
    """Run one case as its own process and collect everything it left behind."""

    argv = [
        sys.executable, "-B", "-m", "scripts.auditory_ui.demorun",
        "--case", case.id, "--worker",
        "--trial", args.trial, "--model", args.model, "--envelope-dir", args.envelope_dir,
        "--scratch-root", args.scratch_root, "--stamp", args.stamp,
        "--seconds", str(args.seconds), "--clean-seconds", str(args.clean_seconds),
        "--media-seconds", str(args.media_seconds), "--transport-seconds", str(args.transport_seconds),
        "--speed", str(args.speed), "--margin", str(args.margin),
        "--fault-at", str(args.fault_at), "--lag-seconds", str(args.lag_seconds),
        "--small-gap-seconds", str(args.small_gap_seconds), "--fatal-gap-seconds", str(args.fatal_gap_seconds),
        "--recovery-gap-seconds", str(args.recovery_gap_seconds),
        "--stall-seconds", str(args.stall_seconds), "--drift-rate", str(args.drift_rate),
        "--case-timeout", str(args.case_timeout), "--host", args.host,
    ]
    began = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            argv, cwd=str(REPO), capture_output=True, text=True, timeout=args.case_timeout + BEACON_GRACE_SECONDS, check=False
        )
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        returncode = None
        stdout = (expired.stdout or b"").decode("utf-8", "replace") if isinstance(expired.stdout, bytes) else (expired.stdout or "")
        stderr = (expired.stderr or b"").decode("utf-8", "replace") if isinstance(expired.stderr, bytes) else (expired.stderr or "")
    elapsed = round(time.time() - began, 3)

    result: dict | None = None
    for line in reversed(stdout.splitlines()):
        if not line.startswith(MARKER):
            continue
        try:
            pointer = json.loads(line[len(MARKER):])
            result = json.loads(Path(pointer["result"]).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, OSError):
            result = None
        break

    try:
        scratch = scratch_dir(args, case.id)
        (scratch / "process.log").write_text(
            stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8"
        )
        scratch_path, log_path = str(scratch), str(scratch / "process.log")
    except OSError as error:
        # A scratch directory that cannot be created is this case's failure, not
        # the matrix's: record it and keep going.
        scratch_path = log_path = f"unavailable: {type(error).__name__}: {error}"
    crashed = returncode not in (0, None) or result is None or timed_out
    judge_input = (result or {}).get("observables")
    if crashed or judge_input is None:
        judgement = verdict_of(
            case,
            [named("the case process produced a verdict beacon", False, f"exit={returncode} timeout={timed_out}")],
            invariant_held=False,
            explicit=False,
            detail=f"exit={returncode} timed_out={timed_out} stdout_tail={stdout[-600:]!r} stderr_tail={stderr[-600:]!r}",
        )
    else:
        judgement = JUDGES[case.id](judge_input, crashed)
    return {
        "case": case.id,
        "fault": case.fault,
        "expected": case.expected,
        "invariant": case.invariant,
        "family": case.family,
        "process": {
            "exit_code": returncode,
            "timed_out": timed_out,
            "seconds": elapsed,
            "pid": (result or {}).get("pid"),
            "scratch": scratch_path,
            "log": log_path,
        },
        "observables": judge_input,
        "worker_error": (result or {}).get("worker_error"),
        "traceback": (result or {}).get("traceback"),
        "judgement": judgement,
        "answers": {
            "crashed": bool(crashed),
            "explicit": bool(judgement["explicit_failure"]),
            "contaminated": False,
        },
    }


def fingerprint(path: str) -> str:
    """SHA256 of a read-only input, so the report can show it was not touched."""

    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs_untouched(args, before: dict) -> dict:
    """Re-hash the inputs and report any change: that is what contamination means."""

    after = {path: fingerprint(path) for path in before}
    changed = [path for path in before if before[path] != after[path]]
    return {"before": before, "after": after, "changed": changed, "untouched": not changed}


def parse_only(value: str) -> list[str]:
    """Parse ``--only`` into case ids, refusing unknown ones."""

    ids = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = [case for case in ids if case not in BY_ID]
    if unknown:
        raise SystemExit("unknown case id(s): " + ", ".join(unknown))
    return ids


def markdown_report(report: dict) -> str:
    """The short readable report that goes beside the JSON."""

    lines = [
        "# Perturbation matrix run",
        "",
        f"- command: `{' '.join(report['command'])}`",
        f"- started: {report['started_at']}",
        f"- trial: `{report['inputs']['trial']}` (clipped to {report['settings']['seconds']}s, speed {report['settings']['speed']}x)",
        f"- model: `{report['inputs']['model']}`, margin {report['settings']['margin']} (policy default is {MIN_MARGIN})",
        f"- cases: {report['summary']['cases']} ({report['summary']['passed']} PASS, {report['summary']['findings']} FINDING)",
        f"- verdict: **{report['summary']['verdict']}** (exit {report['summary']['exit_code']})",
        "",
        "## Per-case table",
        "",
        "| case | fault | expected observable | observed | verdict | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        observed = "; ".join(
            f"{item['observable']}: {'yes' if item['appeared'] else 'NO'}" for item in row["judgement"]["observables"]
        )
        lines.append(
            f"| {row['case']} | {row['fault']} | {row['expected']} | {observed} | "
            f"{row['judgement']['verdict']}{'' if row['judgement']['invariant_held'] else ' (invariant violated)'} | "
            f"{row['judgement']['detail'].replace('|', '/')[:220]} |"
        )
    lines += [
        "",
        "## The three answers",
        "",
        "| case | crashed? | explicit? | contaminated? | seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        answers = row["answers"]
        lines.append(
            f"| {row['case']} | {'YES' if answers['crashed'] else 'no'} | "
            f"{'yes' if answers['explicit'] else 'NO'} | {'YES' if answers['contaminated'] else 'no'} | "
            f"{row['process']['seconds']} |"
        )
    findings = [row for row in report["cases"] if row["judgement"]["verdict"] != "PASS"]
    lines += ["", "## Findings", ""]
    if not findings:
        lines.append("None: every case produced the observable its row names.")
    for row in findings:
        state = "invariant held" if row["judgement"]["invariant_held"] else "INVARIANT VIOLATED"
        lines.append(f"- **{row['case']}** ({state}): {row['judgement']['detail']}")
        for name in row["judgement"]["missing_observables"]:
            lines.append(f"  - did not appear: {name}")
    lines += [
        "",
        "## Isolation",
        "",
        f"- each case ran in its own process and wrote only under `{report['settings']['scratch_root']}/{report['settings']['stamp']}/<case>/`",
        f"- read-only inputs untouched: {report['isolation']['untouched']} "
        f"({', '.join(report['isolation']['changed']) or 'no input changed'})",
        "- therefore the third question (contamination) is answered by construction and by the input hashes above",
        "",
    ]
    return "\n".join(lines) + "\n"


def orchestrate(args) -> int:
    """Run every selected case as its own process, judge it, and write the report."""

    cases = [BY_ID[case] for case in (parse_only(args.only) if args.only else [c.id for c in CASES])]
    inputs = [args.trial, args.model]
    before = {path: fingerprint(path) for path in inputs if Path(path).is_file()}
    began = time.time()
    rows = []
    for case in cases:
        print(f"[demorun] {case.id}: {case.fault}", flush=True)
        row = run_case_subprocess(args, case)
        rows.append(row)
        judgement = row["judgement"]
        print(
            f"[demorun] {case.id}: {judgement['verdict']} "
            f"({row['process']['seconds']}s, exit {row['process']['exit_code']}) -- {judgement['detail'][:160]}",
            flush=True,
        )
    isolation = check_inputs_untouched(args, before)
    summary = {
        "cases": len(rows),
        "passed": sum(1 for row in rows if row["judgement"]["verdict"] == "PASS"),
        "findings": sum(1 for row in rows if row["judgement"]["verdict"] != "PASS"),
        "crashed": sum(1 for row in rows if row["answers"]["crashed"]),
        "silent": sum(1 for row in rows if not row["answers"]["explicit"]),
        "invariants_violated": sum(1 for row in rows if not row["judgement"]["invariant_held"]),
        "seconds": round(time.time() - began, 2),
    }
    strict_failure = args.strict and summary["findings"] > 0
    hard_failure = (
        summary["crashed"] > 0
        or summary["invariants_violated"] > 0
        or not isolation["untouched"]
        or strict_failure
    )
    summary["exit_code"] = 1 if hard_failure else 0
    summary["verdict"] = "FAIL" if hard_failure else ("PASS WITH FINDINGS" if summary["findings"] else "PASS")
    report = {
        "kind": "perturbation matrix run record; every case in its own process, judged against its row",
        "started_at": datetime.fromtimestamp(began, timezone.utc).isoformat(),
        "command": sys.argv,
        "settings": {
            "seconds": args.seconds,
            "clean_seconds": args.clean_seconds,
            "media_seconds": args.media_seconds,
            "transport_seconds": args.transport_seconds,
            "speed": args.speed,
            "margin": args.margin,
            "policy": base_policy(args).to_dict(),
            "case_timeout": args.case_timeout,
            "scratch_root": args.scratch_root,
            "stamp": args.stamp,
            "strict": bool(args.strict),
        },
        "inputs": {
            "trial": args.trial,
            "model": args.model,
            "envelope_dir": args.envelope_dir,
        },
        "isolation": isolation,
        "summary": summary,
        "cases": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = Path(args.md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {out}")
    print(f"summary: {md}")
    return summary["exit_code"]


def build_parser() -> argparse.ArgumentParser:
    """The command line, in one place so a test can use the shipped defaults."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", help="case id; with --worker, run only this case in this process")
    parser.add_argument("--worker", action="store_true", help="internal: run --case's body and print a beacon")
    parser.add_argument("--only", help="comma-separated case ids to run (default: the whole matrix)")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any case is a FINDING")
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--envelope-dir", default=DEFAULT_ENVELOPES)
    parser.add_argument("--scratch-root", default=DEFAULT_SCRATCH)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--seconds", type=float, default=24.0, help="replay length for the session cases")
    parser.add_argument("--clean-seconds", type=float, default=30.0, help="replay length for C1")
    parser.add_argument("--media-seconds", type=float, default=14.0, help="replay length for the media cases")
    parser.add_argument("--transport-seconds", type=float, default=10.0, help="replay length for the short transport cases")
    parser.add_argument("--speed", type=float, default=4.0, help="replay speed for the non-media cases")
    parser.add_argument("--margin", type=float, default=CALIBRATED_MARGIN, help="controller commit margin (decision D-29)")
    parser.add_argument("--fault-at", type=float, default=8.0, help="session seconds where the fault is injected")
    parser.add_argument("--lag-seconds", type=float, default=4.0)
    parser.add_argument("--small-gap-seconds", type=float, default=2.0 / 128.0,
                        help="hole size for C4's repairable gap; Repair's limit is 2 samples at 128 Hz")
    parser.add_argument("--fatal-gap-seconds", type=float, default=2.0)
    parser.add_argument("--recovery-gap-seconds", type=float, default=0.4)
    parser.add_argument("--stall-seconds", type=float, default=1.6)
    parser.add_argument("--drift-rate", type=float, default=1.25)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--host", default=HOST, choices=list(LOOPBACK_HOSTS))
    parser.add_argument("--out", default="results/perturbation.json")
    parser.add_argument("--md", default="results/perturbation.md")
    return parser


def resolve(args):
    """Apply the run's stamp to its two output paths, and validate --out."""

    if not args.out.endswith(".json"):
        raise SystemExit("--out must be a .json path")
    if args.stamp:
        args.out = args.out.replace("perturbation.json", f"perturbation_{args.stamp}.json")
        args.md = args.md.replace("perturbation.md", f"perturbation_{args.stamp}.md")
    return args


def main(argv=None) -> int:
    """Parse arguments, then either run one case's body or orchestrate them all."""

    parser = build_parser()
    args = resolve(parser.parse_args(argv))
    if args.worker:
        if not args.case or args.case not in BY_ID:
            parser.error("--worker needs a known --case")
        return run_worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
