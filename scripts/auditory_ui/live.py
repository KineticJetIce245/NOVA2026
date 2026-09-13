"""Command 2 of the two-command live demo: run the session the browser watches.

    .venv/Scripts/python.exe -B -m scripts.auditory_ui.live --sfreq 500   # Windows
    .venv/bin/python        -B -m scripts.auditory_ui.live --sfreq 500   # macOS / Linux

Both interpreters are named on purpose: a virtual environment keeps its Python
at ``.venv/Scripts/python.exe`` on Windows and at ``.venv/bin/python`` on macOS
and Linux, and the two layouts cannot be carried across. The launchers in
``scripts/getlive/live_demo.{ps1,sh}`` detect which one exists, and refuse with
the command to build it when neither does.

One command, and everything the demo needs is inside it: it acquires **live EEG
from the real eego outlet** through the repository's own live path (the same
``Acquire``, ``TimeBase``, channel contract and preprocessing chain
``scripts/getlive`` accepts hardware with), decodes with the **20-channel**
model fitted for exactly this headset, plays the two candidate audio files with
per-ear volume in the browser, serves the interface from the same origin, and
prints the URL. It writes a run record either way.

**It refuses to start when the model and the source disagree on a
processing-deciding key.** ``RidgeDecoder``'s gate keys - ``eeg_channels``,
``input_sfreq``, ``output_sfreq``, ``units``, ``bandpass``, ``filter_order``,
``resample_quality``, ``stage`` - are compared with the chain's own contract
*before* a port is opened. A difference exits ``2`` and names the keys, instead
of failing every window for the rest of the session
(``scripts/getlive/README.md``, "Which contract keys stop a run"). The two
provenance keys (``input_reference``, ``upstream_processing``) are recorded, not
gated, and the reason is printed: no run on this rig can equalise them.

**What this command cannot make true, and says so on every run:**

* the **audio-to-EEG loopback offset has never been measured**. 100 ms of
  misalignment costs 0.119-0.150 balanced accuracy on this decoder - five to
  seven times the cost of replacing 44 electrodes with 20 (plan sections
  3.11/3.17-4, ``results/aad_shift_sweep_*``). ``--require-calibration`` makes
  that a hard gate; by default the run warns, records
  ``audio_alignment.measured: false``, and still runs, because a demo that
  refuses to show anything until a cable has been plugged in is not a demo.
  The measurement itself is ``scripts/getlive/calibrate_loopback.py``.
* the **reference mismatch is real and is not repaired here**. The model was
  fitted on KU Leuven data (Cz derivation, 128 Hz); this rig records CPz at
  500 Hz (plan sections 3.11/3.17-4). For the live path the chain carries
  whatever reference the amplifier applied, so the decoder's learned spatial
  weights are being applied to a differently-measured signal. A decision from
  this command therefore measures that mismatch as much as it measures
  attention, and the record says so under ``trust``.
* the **quality policy is explicit, never inherited**. The chain's default
  (``check_channels=True, max_bad_channels=0``) halts a session once electrode
  faults persist past the recovery budget (section 3.17-1), which on a dry cap
  is every session. This command states ``--no-channel-check`` by default,
  records the whole policy, and lets ``--strict-policy`` or
  ``--exclude-channels``/``--max-bad-channels``/``--amplitude-limit-uv``
  replace it. What a relaxed policy costs is stated in the record
  (section 3.17-6: with the check off, ``signal_quality`` degenerates).
* **no participant data is logged.** The run record holds counters, the policy,
  the declared contract and decision words. No EEG sample, no packet payload
  and no audio content is written, and the console prints counts only.

Loopback only: any other ``--host`` is refused, and the client never supplies a
filesystem path - the media file is server-side configuration and
``/api/media/file`` serves exactly that one file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/auditory_ui/live.py`
    sys.path.insert(0, str(REPO))

from nova2026.auditory.config import (  # noqa: E402
    CALIBRATED_MARGIN,
    MIN_MARGIN,
    AuditoryConfig,
)
from nova2026.auditory.decoder import (  # noqa: E402
    GATE_KEYS as CONTRACT_GATE_KEYS,
    PROVENANCE_KEYS as CONTRACT_PROVENANCE_KEYS,
    RidgeDecoder,
)
from nova2026.auditory.producer import AttentionProducer  # noqa: E402
from nova2026.auditory.render import (  # noqa: E402
    DEFAULT_CROSSMIX_WEIGHT,
    DEFAULT_PRESENTATION,
    PRESENTATION_MODES,
    presentation_mode,
    render_stereo,
)
from nova2026.auditory.session import AttentionSession, RunPolicy  # noqa: E402
from nova2026.auditory.sources import ReferenceEnvelopes  # noqa: E402
from nova2026.auditory.streaming import AuditoryProcessor  # noqa: E402
from nova2026.transport.media import MediaTimeline  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.getlive.ant_source import AntStreamSource  # noqa: E402
from scripts.getlive.live_source import (  # noqa: E402
    DEFAULT_EXPECTED_CHANNELS,
    DEFAULT_MODEL,
    DEFAULT_SFREQ,
    REFUSAL,
    model_electrodes,
    unit_exponent,
)
from scripts.getlive.outlets import channel_facts, open_inlet, wait_for_outlet  # noqa: E402

DEFAULT_CANDIDATE_A = "tmp/antneurodata/audio_files/experiment/left_mono.wav"
DEFAULT_CANDIDATE_B = "tmp/antneurodata/audio_files/experiment/right_mono.wav"
"""The operator's own played stimulus: candidate A in the left ear, B in the right.

Read-only, and under ``tmp/``, which is git-ignored. Point ``--candidate-a`` /
``--candidate-b`` at any pair of mono WAVs; the envelopes must be the loudness
curves of *those* files, which is why both are explicit flags.
"""

DEFAULT_ENVELOPE_A = "datasets/audio/left_mono.npz"
DEFAULT_ENVELOPE_B = "datasets/audio/right_mono.npz"

DEFAULT_TITLE = "live"
TERMINAL_STATUSES = ("stopped", "error")

# The alignment budget the calibration has to meet, from plan section 3.17-4:
# the decoder's half-depth width is 104 ms and +-250 ms is chance.
ALIGNMENT_TOLERANCE_SECONDS = 0.030

# What a relaxed quality policy costs, in the record's own words.
RELAXED_POLICY_NOTE = (
    "check_channels=False is the documented run policy (plan section 3.17-1): the "
    "signal is unchanged, only the verdict is. It is what stops a dry cap from "
    "halting the session. The cost is stated rather than hidden - with the check "
    "off, QualityMonitor.reasons() returns nothing by policy, so signal_quality "
    "degenerates (section 3.17-6). Saturation and flatline are shown by the "
    "per-electrode statistics, and --strict-policy restores the chain default."
)

TRUST_NOTE = (
    "The model was fitted on KU Leuven recordings (Cz derivation, 128 Hz); this "
    "rig records CPz at 500 Hz and the chain is fed whatever reference the "
    "amplifier applied (plan sections 3.11/3.17-4). The two provenance keys are "
    "recorded, not gated, and nothing was equalised. A decision here therefore "
    "measures the rig mismatch as much as it measures attention. Do not quote "
    "the KU Leuven accuracy numbers as an expectation for this rig."
)


def log(message: str, lines: list[str]) -> None:
    """Print one timestamped line and keep it for the run record."""

    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    lines.append(line)


def free_port() -> int:
    """An unused loopback TCP port, so two runs never collide."""

    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --------------------------------------------------------------------- contract


class ContractRefused(RuntimeError):
    """The model and the live chain disagree on a key that decides processing."""

    def __init__(self, message: str, record: dict) -> None:
        super().__init__(message)
        self.record = record


def contract_gate(model, contract: dict, *, source: str = "live chain") -> dict:
    """Compare the chain's contract with the model's, and refuse a gated difference.

    This is the check ``RidgeDecoder.validate`` performs on every window, moved
    to start-up where it belongs. The two sides are compared verbatim on
    :data:`~nova2026.auditory.decoder.GATE_KEYS`; the provenance keys are
    recorded and reported but never gated, because no run on another rig can
    equalise them and copying the model's text across would be a false statement
    about this recording.

    Raises:
        ContractRefused: If any gate key differs, naming the keys.
    """

    record = dict(model.record_contract_difference(contract) or {})
    record["gate_keys"] = list(CONTRACT_GATE_KEYS)
    record["provenance_keys"] = list(CONTRACT_PROVENANCE_KEYS)
    record["model_contract"] = dict(model.contract or {})
    record["source_contract"] = dict(contract)
    record["source"] = str(source)
    gated = sorted(record.get("gated") or {})
    if gated:
        detail = "; ".join(
            f"{key}: chain={record['gated'][key].get('source')!r} "
            f"model={record['gated'][key].get('model')!r}"
            for key in gated
        )
        raise ContractRefused(
            f"the {source} contract disagrees with the model on "
            f"{len(gated)} processing-deciding key(s): {', '.join(gated)} ({detail}). "
            "Every window would be refused by the decoder, so this run refuses to "
            "start instead. Check --pre-resample, --source-units and --model; the "
            "gate keys are " + ", ".join(CONTRACT_GATE_KEYS) + ".",
            record,
        )
    return record


def contract_lines(record: dict) -> list[str]:
    """The contract assessment, as printed lines."""

    lines = [
        f"model contract check: {len(record['gate_keys'])} gate key(s) matched "
        f"({', '.join(record['gate_keys'])})",
        f"  recorded, not gated: {', '.join(record['provenance_keys'])} - "
        f"{sorted(record.get('provenance') or {}) or 'none'} differ on this rig",
    ]
    for key in sorted(record.get("provenance") or {}):
        values = record["provenance"][key]
        lines.append(
            f"  {key}: this rig declares {str(values.get('source'))[:120]!r} / the "
            f"model records {str(values.get('model'))[:120]!r}"
        )
    return lines


# ---------------------------------------------------------------------- policy


def display_channel(args) -> str | None:
    """The electrode the ``eeg_display`` tap carries for this run, or ``None``.

    ``--eeg-display-channel`` wins when given. Without it the tap stays off, so a
    run without the flag is the run it was before the traces existed. A caller that
    asks for the tap without naming an electrode this rig carries gets ``Cz`` when
    the contract lists it and the first electrode otherwise -- the same fallback
    ``scripts/auditory_ui/demo.py`` uses, and the producer records whichever name
    was chosen in the packet's ``channel_source`` so the panel never has to guess.
    """

    if not args.eeg_display_channel:
        return None
    raw = args.electrodes or ()
    if isinstance(raw, str):
        raw = tuple(part.strip() for part in raw.split(",") if part.strip())
    names = tuple(str(name) for name in raw)
    if args.eeg_display_channel in names:
        return args.eeg_display_channel
    if "Cz" in names:
        return "Cz"
    return names[0] if names else None


def quality_policy(args) -> dict:
    """The channel-quality policy in force, as the run record carries it.

    It is a declaration, not a measurement, and every number in it reaches both
    stages that judge an electrode - ``QualityMonitor``'s fault reasons and
    ``Repair``'s endpoint checks - so the quality verdict and the repair verdict
    cannot disagree about the same electrode.
    """

    excluded = tuple(part.strip() for part in (args.exclude_channels or "").split(",")
                     if part.strip())
    # The documented default is the RELAXED policy (section 3.17-1): the chain's
    # own default halts a dry cap mid-session, so it has to be asked for by name
    # rather than inherited. `--no-channel-check` says the default out loud.
    check = bool(args.strict_policy) and not args.no_channel_check
    return {
        "check_channels": bool(check),
        "max_bad_channels": int(args.max_bad_channels),
        "exclude_channels": list(excluded),
        "amplitude_limit_uv": (
            None if args.amplitude_limit_uv is None else float(args.amplitude_limit_uv)
        ),
        "amplitude_limit_source": (
            "chain default (500 uV) - not declared by this run"
            if args.amplitude_limit_uv is None
            else "declared on the command line"
        ),
        "saturation_limit_uv": (
            None if args.saturation_limit_uv is None else float(args.saturation_limit_uv)
        ),
        "saturation_limit_source": (
            "chain default (75 000 uV) - not declared by this run"
            if args.saturation_limit_uv is None
            else "declared on the command line"
        ),
        "margin": float(args.margin),
        "margin_source": (
            "calibrated default (results/aad_margin_calibration_*)"
            if math.isclose(float(args.margin), CALIBRATED_MARGIN)
            else "explicit on the command line"
        ),
        "documented_min_margin": float(MIN_MARGIN),
        "warmup_seconds": float(args.warmup),
        "frame_seconds": float(args.frame_seconds),
        "display_channel": display_channel(args),
        "note": RELAXED_POLICY_NOTE if not check else (
            "check_channels=True is the chain's own default, restored by "
            "--strict-policy: a faulting electrode that persists past the recovery "
            "budget stops the session."
        ),
    }


# ----------------------------------------------------------------------- media


def load_candidate(path: Path) -> tuple[int, np.ndarray]:
    """Read one mono candidate WAV as floats in [-1, 1).

    Scaling follows the file's own sample format, so an int16 and a float32 file
    of the same stimulus give the same waveform. A stereo file is refused: the
    two candidates are separate files by construction (plan section 3.3), and a
    stereo file silently read as "both" would break the ear-to-candidate
    mapping the gain path depends on.
    """

    from scipy.io import wavfile

    if not path.is_file():
        raise FileNotFoundError(
            f"candidate audio not found: {path}. Pass --candidate-a/--candidate-b."
        )
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError(f"{path.name} must be mono; it has {data.shape[1]} channels.")
    if data.dtype.kind == "i":
        scale = float(np.iinfo(data.dtype).max) + 1.0
    elif data.dtype.kind == "u":
        scale = float(np.iinfo(data.dtype).max) + 1.0
        data = data.astype(np.float64) - scale / 2.0
    elif data.dtype.kind == "f":
        scale = 1.0
    else:
        raise ValueError(f"{path.name} has unsupported sample type {data.dtype}.")
    return int(rate), np.asarray(data, dtype=np.float64) / scale


def build_stereo(
    left: np.ndarray, right: np.ndarray, *, rate: int, mismatch_seconds: float
) -> tuple[np.ndarray, dict]:
    """Stack the two candidates into the (samples, 2) array the renderer takes.

    Two candidates of different length are refused rather than silently
    truncated (plan case C15): a truncated candidate would be scored against an
    envelope that describes a longer file, so the decoder's own reference would
    be wrong for the tail of the run.
    """

    report = {
        "candidate_samples": [int(left.shape[0]), int(right.shape[0])],
        "candidate_seconds": [round(left.shape[0] / rate, 3), round(right.shape[0] / rate, 3)],
    }
    difference = abs(int(left.shape[0]) - int(right.shape[0])) / rate
    report["length_difference_seconds"] = round(difference, 6)
    if difference > float(mismatch_seconds):
        raise ValueError(
            f"the two candidates differ by {difference:.3f}s "
            f"({left.shape[0]} vs {right.shape[0]} samples at {rate} Hz), above the "
            f"{mismatch_seconds:.3f}s tolerance. Pass --max-length-mismatch to "
            "accept it, or use two files of the same length."
        )
    count = min(int(left.shape[0]), int(right.shape[0]))
    audio = np.column_stack([left[:count], right[:count]])
    report["published_samples"] = int(count)
    report["published_seconds"] = round(count / rate, 3)
    return audio, report


def prepare_media(args) -> tuple[MediaTimeline, dict]:
    """Render the two candidates to the stereo file the browser plays."""

    left_rate, left = load_candidate(Path(args.candidate_a))
    right_rate, right = load_candidate(Path(args.candidate_b))
    if left_rate != right_rate:
        raise ValueError(
            f"the candidates are at different rates ({left_rate} Hz vs "
            f"{right_rate} Hz); resample one of them first."
        )
    audio, report = build_stereo(
        left, right, rate=left_rate, mismatch_seconds=float(args.max_length_mismatch)
    )
    report["sample_rate"] = int(left_rate)
    report["candidates"] = {"A": str(args.candidate_a), "B": str(args.candidate_b)}
    out = Path(args.media_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.update(
        render_stereo(
            audio,
            out,
            sample_rate=int(left_rate),
            presentation=presentation_mode(args.presentation),
            crossmix_weight=float(args.crossmix_weight),
        )
    )
    timeline = MediaTimeline(out, args.media_title)
    report["media_id"] = timeline.media_id
    report["descriptor"] = timeline.descriptor()
    report["presentation"] = args.presentation
    return timeline, report


# ------------------------------------------------------------------- alignment


def media_alignment(packet: dict) -> float | None:
    """The session-clock time at which the browser's media reached position zero.

    The browser owns the playback position (decision D-02) and reports it in
    ``media`` packets; each packet carries the producer's own session time
    alongside the position it was told. Subtracting the two gives the anchor the
    envelopes need: *this* is the session time at which media position zero was
    playing. Returns ``None`` when the packet does not carry both numbers - the
    caller then keeps the declared anchor rather than inventing one.
    """

    if str(packet.get("type")) != "media":
        return None
    payload = packet.get("payload") or {}
    position = payload.get("media_time_s", payload.get("mediaTimeS"))
    stamp = packet.get("timestamp")
    if position is None or stamp is None:
        return None
    try:
        position = float(position)
        stamp = float(stamp)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(position) and math.isfinite(stamp)) or position < 0:
        return None
    return stamp - position


def apply_alignment(source, value: float | None, *, limit: float) -> dict:
    """Adopt a measured anchor only when it is inside the plausible window.

    A number that is not on the session clock - a monotonic reading, an epoch
    stamp - produces an anchor far outside any real playback delay, and adopting
    it would displace every envelope lookup for the whole run. The limit is what
    makes "we could not establish the anchor" a safe, recorded outcome rather
    than a silent corruption.
    """

    if value is None:
        return {"applied": False, "reason": "the transport published no media packet "
                                            "carrying both a position and a timestamp"}
    if not math.isfinite(value) or abs(value) > float(limit):
        return {
            "applied": False,
            "candidate_seconds": float(value),
            "limit_seconds": float(limit),
            "reason": "the derived anchor fell outside the plausible window, so it was "
                      "not adopted; the envelopes keep the declared anchor",
        }
    source.audio_start = float(value)
    return {"applied": True, "audio_start_seconds": float(value)}


def load_calibration(path: str | None) -> dict | None:
    """Read a loopback calibration profile written by ``calibrate_loopback``."""

    if not path:
        return None
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"calibration profile not found: {profile_path}. Run "
            "scripts.getlive.calibrate_loopback first, or drop --calibration."
        )
    return json.loads(profile_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- assembly


def build(app_state: dict, args):
    """Build the session and the producer: the factory the REST path calls.

    Called twice by design, exactly as the replay demo does it: once before the
    URL is printed, so a start-up that cannot succeed says so while the operator
    is still looking at the terminal, and once when ``POST /api/session/start``
    arrives.
    """

    source, references, model = (
        app_state["source"],
        app_state["references"],
        app_state["model"],
    )
    source.audio_start = float(app_state.get("audio_start", 0.0))
    declared = app_state.get("chain_rate")
    if declared is not None:
        # ``AttentionSession`` reads the chain settings off the SOURCE, so the
        # source must declare the rate the adapter actually delivers. The outlet
        # itself was pre-flighted at its own rate.
        source.sample_rate = float(declared)
    policy = RunPolicy(
        check_channels=bool(app_state["policy"]["check_channels"]),
        max_bad_channels=int(app_state["policy"]["max_bad_channels"]),
        margin=float(app_state["policy"]["margin"]),
        warmup_seconds=float(app_state["policy"]["warmup_seconds"]),
        frame_seconds=float(app_state["policy"]["frame_seconds"]),
        source_unit_exponent=int(app_state["source_unit_exponent"]),
        saturation_limit_uv=app_state["policy"]["saturation_limit_uv"],
        amplitude_limit_uv=app_state["policy"]["amplitude_limit_uv"],
        exclude_channels=tuple(app_state["policy"]["exclude_channels"]),
        display_channel=app_state["policy"]["display_channel"],
    )
    session = AttentionSession(
        source=source, decoder=model, references=references, policy=policy
    )
    producer = AttentionProducer(session)
    if app_state.get("media") is not None:
        producer.media_reference = app_state["media"].media_reference
    return session, producer


def chain_rate(args, model) -> float | None:
    """The rate the chain is actually fed, which it must also declare.

    ``RidgeDecoder.validate`` compares the chain's contract with the model's
    verbatim, so a 500 Hz outlet has to reach the chain at the rate the model's
    training trials were at. ``auto`` reads that rate off the model's contract -
    the only choice that cannot disagree with it - and the conversion happens in
    front of the chain, reported rather than silent.
    """

    if args.pre_resample == "off":
        return None
    if args.pre_resample == "auto":
        wanted = (model.contract or {}).get("input_sfreq")
        return None if wanted is None else float(wanted)
    return float(args.pre_resample)


def source_block(args, model, rate: float) -> dict:
    """What the run asserts about the amplifier, as the record keeps it."""

    declared = model.contract or {}
    return {
        "stream_name": args.stream_name,
        "source_id": args.source_id,
        "stream_type": args.stream_type,
        "sfreq_hz": float(args.sfreq),
        "source_units": args.source_units,
        "source_unit_exponent": unit_exponent(args.source_units),
        "electrodes": list(args.electrodes),
        "n_electrodes": len(args.electrodes),
        "timebase": args.timebase,
        "block_samples": int(args.block_samples),
        "pre_resample": args.pre_resample,
        "chain_input_hz": None if rate is None else float(rate),
        "model_input_hz": declared.get("input_sfreq"),
        "model_output_hz": declared.get("output_sfreq"),
        "window_seconds": declared.get("window_seconds"),
        "step_seconds": declared.get("step_seconds"),
        "reference": args.reference,
        "ground": args.ground,
        "upstream_processing": args.upstream,
        "assertions": (
            "--sfreq, --source-units, --reference, --ground and --upstream are "
            "operator assertions, not measurements: LSL advertises none of them "
            "(scripts/getlive/README.md). The package's pre-flight checks the rate "
            "and the units against what the outlet declares; gain, filters and "
            "impedance are the control software's and are not visible here."
        ),
    }


# ------------------------------------------------------------------ collection


async def capture(port: int, packets: list, stop: asyncio.Event, on_packet=None) -> None:
    """Subscribe to ``/ws/live`` and count packets, as the browser would.

    Only counters leave this function. The packets themselves are what the
    frontend draws, and they carry EEG display rows; nothing here writes them to
    the record, because the record must hold no participant data.
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
                if on_packet is not None:
                    on_packet(packet)
    except (OSError, asyncio.CancelledError):  # pragma: no cover - shutdown race
        return


def packet_census(packets: list) -> dict:
    """Counters and decision words only - never a payload."""

    types: dict[str, int] = {}
    words: dict[str, int] = {}
    for packet in packets:
        name = str(packet.get("type"))
        types[name] = types.get(name, 0) + 1
        payload = packet.get("payload") or {}
        for key in ("decision", "attention", "word"):
            value = payload.get(key)
            if isinstance(value, str):
                words[value] = words.get(value, 0) + 1
                break
    return {"packets": len(packets), "by_type": dict(sorted(types.items())),
            "decision_words": dict(sorted(words.items()))}


# ----------------------------------------------------------------------- run


async def drive(port: int, args, app_state: dict, stop: asyncio.Event, lines: list[str]) -> dict:
    """Start the session over REST, watch it, and adopt the audio anchor."""

    import httpx

    base = f"http://{args.host}:{port}"
    packets: list = []
    state: dict = {"started": None, "final": None, "alignment": {"applied": False,
                                                                "reason": "not observed"}}
    holder = app_state["holder"]

    def on_packet(packet: dict) -> None:
        if holder.get("aligned") or args.audio_anchor == "zero":
            return
        candidate = media_alignment(packet)
        if candidate is None:
            return
        outcome = apply_alignment(
            app_state["source"], candidate, limit=float(args.audio_anchor_limit)
        )
        outcome["candidate_seconds"] = None if candidate is None else round(candidate, 6)
        holder["aligned"] = bool(outcome.get("applied"))
        state["alignment"] = outcome
        log(
            f"audio anchor: {'adopted media position zero at session time ' + format(outcome['audio_start_seconds'], '.3f') + ' s' if outcome.get('applied') else 'kept the declared anchor - ' + str(outcome.get('reason'))}",
            lines,
        )

    async with httpx.AsyncClient(base_url=base, timeout=10.0) as http:
        for _ in range(200):
            with contextlib.suppress(Exception):
                if (await http.get("/api/health")).status_code == 200:
                    break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - the server thread starts before this
            raise RuntimeError("the transport never became ready on loopback")

        listener = asyncio.create_task(capture(port, packets, stop, on_packet))
        try:
            if args.auto_start:
                started = await http.post("/api/session/start")
                if started.status_code != 200:
                    raise RuntimeError(
                        f"POST /api/session/start answered {started.status_code}: "
                        f"{started.text.strip()!r}"
                    )
                state["started"] = started.json()
                log("session started; the page at the URL is live", lines)
            deadline = None if not args.seconds else time.monotonic() + float(args.seconds)
            while not stop.is_set():
                await asyncio.sleep(0.5)
                if deadline is not None and time.monotonic() >= deadline:
                    log(f"--seconds {args.seconds:g} elapsed; stopping", lines)
                    break
                if args.auto_start:
                    records = (await http.get("/api/sessions")).json()
                    last = records[-1] if records else None
                    if last is not None and last["status"] in TERMINAL_STATUSES:
                        state["final"] = last
                        log(f"session ended with status {last['status']!r}", lines)
                        break
        finally:
            stop.set()
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener
    state["packets"] = packets
    return state


def open_browser(url: str, lines: list[str]) -> None:
    """Open the interface once the server is listening, without blocking."""

    import webbrowser

    try:
        webbrowser.open(url)
        log(f"opened the default browser at {url}", lines)
    except Exception as error:  # noqa: BLE001 - a browser is a convenience, not a step
        log(f"could not open a browser ({error}); open {url} manually", lines)


def run_record(
    args, app_state, state, media_report, transport, contract, lines, began
) -> dict:
    """The record: what ran, under what policy, and what it cannot claim."""

    policy = app_state["policy"]
    census = packet_census(state.get("packets") or [])
    summary = {}
    producer = app_state["holder"].get("producer")
    if producer is not None and getattr(producer, "summary", None):
        with contextlib.suppress(Exception):
            summary = producer.summary.to_dict()
    calibration = app_state.get("calibration")
    return {
        "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "live amplifier run (no recording involved)",
        "seconds_served": round(time.time() - began, 3),
        "url": app_state.get("url"),
        "source": app_state["source_record"],
        "declared_by_the_outlet": app_state.get("declared"),
        "model": {
            "path": str(args.model),
            "contract": dict(getattr(app_state["model"], "contract", {}) or {}),
            "window_seconds": (getattr(app_state["model"], "contract", {}) or {}).get(
                "window_seconds"
            ),
        },
        "contract": contract,
        "quality_policy": policy,
        "media": media_report,
        "audio_alignment": {
            **dict(state.get("alignment") or {}),
            "mode": args.audio_anchor,
            "declared_audio_start_seconds": float(app_state.get("audio_start", 0.0)),
            "loopback_offset_seconds": (
                None if not calibration else calibration.get("offset_seconds")
            ),
            "loopback_measured": bool(calibration),
            "tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
            "cost_of_100ms": (
                "0.119 balanced accuracy audio-early / 0.150 audio-late, measured in "
                "results/aad_shift_sweep_* (plan sections 3.11/3.17-4) - five to "
                "seven times the 0.022 cost of the 44 electrodes the headset does "
                "not have"
            ),
            "note": (
                "The audio-to-EEG loopback offset is unmeasured unless "
                "loopback_measured is true. The anchor adopted here is derived from "
                "the browser's own reported playback position (decision D-02), which "
                "is the authority the gain gate uses, and it does NOT include the "
                "sound card's output latency or the acoustic path to the electrode. "
                "Only scripts.getlive.calibrate_loopback measures that."
            ),
        },
        "transport": transport,
        "session": {
            "started": state.get("started") is not None,
            "final_status": (state.get("final") or {}).get("status"),
            "summary": {
                key: summary.get(key)
                for key in ("windows", "decisions", "failed", "evidence_gaps",
                            "scoring_failures")
            },
        },
        "counters": census,
        "trust": {
            "reference_mismatch": TRUST_NOTE,
            "decision_is_not_a_probability": (
                "a decoder score is a correlation; the margin is the distance "
                "between the two candidates' scores, and --margin is where the "
                "system decides to stay quiet"
            ),
            "no_participant_data": (
                "this record holds counters, policy, declared contracts and decision "
                "words. No EEG sample, no packet payload and no audio content."
            ),
        },
        "console": lines,
    }


def markdown(record: dict) -> str:
    """The same record, as the page a reviewer reads first."""

    source = record["source"]
    policy = record["quality_policy"]
    alignment = record["audio_alignment"]
    transport = record.get("transport") or {}
    lines = [
        f"# Live run - {record['stamp']}",
        "",
        f"Ambient: `{source['stream_name']}`, {source['n_electrodes']} electrodes by "
        f"name at {source['sfreq_hz']:g} Hz, units {source['source_units']} "
        f"(exponent {source['source_unit_exponent']}), timebase {source['timebase']}, "
        f"chain fed at {source['chain_input_hz']} Hz.",
        "",
        "## What was checked before the port opened",
        "",
        f"- gate keys matched: {', '.join(record['contract']['gate_keys'])}",
        f"- recorded, not gated: "
        f"{', '.join(record['contract']['provenance_keys'])} "
        f"({sorted(record['contract'].get('provenance') or {}) or 'none'} differ on "
        "this rig)",
        f"- model: {record['model']['path']}, window "
        f"{record['model']['window_seconds']}s",
        "",
        "## The policy this run declared",
        "",
        f"- `check_channels={policy['check_channels']}`, "
        f"`max_bad_channels={policy['max_bad_channels']}`, "
        f"`exclude_channels={policy['exclude_channels'] or 'none'}`",
        f"- amplitude limit {policy['amplitude_limit_uv']} "
        f"({policy['amplitude_limit_source']}), saturation limit "
        f"{policy['saturation_limit_uv']} ({policy['saturation_limit_source']})",
        f"- margin {policy['margin']} ({policy['margin_source']}); the documented "
        f"default is {policy['documented_min_margin']}",
        f"- {policy['note']}",
        "",
        "## Alignment",
        "",
        f"- loopback calibration measured: {alignment['loopback_measured']}"
        + (
            f" (offset {alignment['loopback_offset_seconds']} s, tolerance "
            f"+-{alignment['tolerance_seconds']} s)"
            if alignment["loopback_measured"]
            else " - the offset is UNKNOWN on this run, see the note below"
        ),
        f"- anchor: {alignment.get('reason') or alignment.get('audio_start_seconds')}",
        f"- {alignment['note']}",
        f"- the cost of being 100 ms out: {alignment['cost_of_100ms']}",
        "",
        "## Transport",
        "",
        f"- blocks {transport.get('blocks')}, samples {transport.get('samples')}, "
        f"max lag {transport.get('max_lag_seconds')} s, gaps {transport.get('gaps')}",
        f"- source clock as delivered {transport.get('raw_rate_hz')} Hz; peak grid "
        f"deviation {transport.get('grid_deviation_peak_seconds')} s; re-locks "
        f"{transport.get('timebase_relocks')}",
        "",
        "## Counters (no participant data)",
        "",
        f"- packets {record['counters']['packets']}: "
        f"{json.dumps(record['counters']['by_type'])}",
        f"- decision words: {json.dumps(record['counters']['decision_words'])}",
        f"- session: {json.dumps(record['session'])}",
        "",
        "## What this proves, and what it does not",
        "",
        "Proves: an amplifier published EEG on this network, the repository's own "
        "live path acquired it, the 20-channel model's contract was checked against "
        "the chain before start-up, and the transport served decisions and gains to "
        "a browser from one origin.",
        "",
        f"Does not prove: that the decision is right. {record['trust']['reference_mismatch']}",
        "",
    ]
    return "\n".join(str(line) for line in lines)


def build_parser() -> argparse.ArgumentParser:
    """Build command 2's command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    amp = parser.add_argument_group("amplifier source")
    amp.add_argument("--stream-name", default=None,
                     help="LSL name to pin; default: the single EEG-like outlet")
    amp.add_argument("--source-id", default=None)
    amp.add_argument("--stream-type", default="EEG")
    amp.add_argument("--expected-channels", type=int, default=DEFAULT_EXPECTED_CHANNELS)
    amp.add_argument("--resolve-seconds", type=float, default=15.0,
                     help="how long to look for the amplifier before refusing")
    amp.add_argument("--connect-seconds", type=float, default=10.0)
    amp.add_argument("--bufsize", type=float, default=30.0)
    amp.add_argument("--block-samples", type=int, default=25)
    amp.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ,
                     help="declared rate; this rig measured 500 Hz (section 3.11)")
    amp.add_argument("--source-units", default="uV",
                     choices=("V", "mV", "uV", "nV"))
    amp.add_argument("--timebase", default="grid", choices=("grid", "stamps"))
    amp.add_argument("--reference", default="CPz (declared by the operator; the model records Cz)")
    amp.add_argument("--ground", default="AFz (declared by the operator)")
    amp.add_argument("--upstream", default=(
        "eego control software over LSL; gain and filters set in the software and "
        "not advertised over LSL - operator assertion, not a measurement"))
    amp.add_argument("--max-lag-seconds", type=float, default=3.0)
    amp.add_argument("--no-data-timeout", type=float, default=5.0)

    model = parser.add_argument_group("model and chain")
    model.add_argument("--model", default=DEFAULT_MODEL)
    model.add_argument("--electrodes", default=None,
                       help="comma-separated labels; default: the model's own list")
    model.add_argument("--eeg-display-channel", default=None,
                       help="publish the eeg_display packet for this electrode (e.g. Cz), "
                            "which is what puts the two EEG traces on the page. Omitted "
                            "publishes none: the tap and the packet exist only when a "
                            "channel is named")
    model.add_argument("--pre-resample", default="auto",
                       help="auto (default): deliver the model's input_sfreq; off: "
                            "deliver the source rate; or an explicit Hz")
    model.add_argument("--margin", type=float, default=CALIBRATED_MARGIN)
    model.add_argument("--warmup", type=float, default=2.0)
    model.add_argument("--frame-seconds", type=float, default=0.25)

    quality = parser.add_argument_group("channel-quality policy (declared, not inherited)")
    quality.add_argument("--exclude-channels", default="",
                         help="comma-separated electrodes known to be dead; they stay "
                              "in the data and are still counted")
    quality.add_argument("--max-bad-channels", type=int, default=0)
    quality.add_argument("--amplitude-limit-uv", type=float, default=None,
                         help="unset: the chain's own 500 uV default")
    quality.add_argument("--saturation-limit-uv", type=float, default=None,
                         help="unset: the chain's own 75 000 uV default")
    quality.add_argument("--no-channel-check", action="store_true",
                         help="the documented default: record faults, never reject a window")
    quality.add_argument("--strict-policy", action="store_true",
                         help="restore the chain's own default (check_channels=True)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--candidate-a", default=DEFAULT_CANDIDATE_A)
    audio.add_argument("--candidate-b", default=DEFAULT_CANDIDATE_B)
    audio.add_argument("--envelope-a", default=DEFAULT_ENVELOPE_A)
    audio.add_argument("--envelope-b", default=DEFAULT_ENVELOPE_B)
    audio.add_argument("--presentation", default=DEFAULT_PRESENTATION,
                       choices=PRESENTATION_MODES)
    audio.add_argument("--crossmix-weight", type=float, default=DEFAULT_CROSSMIX_WEIGHT)
    audio.add_argument("--media-out", default="output/auditory_ui/live_stereo.wav")
    audio.add_argument("--media-title", default=DEFAULT_TITLE)
    audio.add_argument("--max-length-mismatch", type=float, default=0.001)
    audio.add_argument("--audio-anchor", default="auto", choices=("auto", "zero"),
                       help="auto: derive the anchor from the browser's reported "
                            "playback position; zero: media position zero is session "
                            "time zero")
    audio.add_argument("--audio-anchor-limit", type=float, default=60.0)
    audio.add_argument("--calibration", default=None,
                       help="loopback calibration profile from calibrate_loopback")
    audio.add_argument("--require-calibration", action="store_true",
                       help="refuse to start unless a calibration profile was given")

    server = parser.add_argument_group("transport")
    server.add_argument("--host", default="127.0.0.1", help="loopback only")
    server.add_argument("--port", type=int, default=0)
    server.add_argument("--static-dir", default=str(REPO / "apps" / "attune-ui" / "dist"))
    server.add_argument("--log-level", default="warning")
    server.add_argument("--open-browser", action="store_true")
    server.add_argument("--auto-start", action="store_true", default=True)
    server.add_argument("--no-auto-start", dest="auto_start", action="store_false")
    server.add_argument("--seconds", type=float, default=None,
                        help="stop after this many seconds; default: until Ctrl+C")
    server.add_argument("--out", default=None,
                        help="run record path; default results/live_demo_<stamp>.json")
    return parser


def main(argv=None) -> int:
    """Everything except argument parsing, as one synchronous run."""

    args = build_parser().parse_args(argv)
    lines: list[str] = []
    began = time.time()

    try:
        args.electrodes = tuple(
            part.strip() for part in (args.electrodes or "").split(",") if part.strip()
        ) or model_electrodes(args.model)
        exponent = unit_exponent(args.source_units)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return REFUSAL

    if args.host not in LOOPBACK_HOSTS:
        print(f"loopback only; refusing host {args.host!r}", file=sys.stderr, flush=True)
        return REFUSAL
    static_dir = Path(args.static_dir) if args.static_dir else None
    if static_dir is not None and not static_dir.is_dir():
        print(f"static dir not found: {static_dir} (build apps/attune-ui)",
              file=sys.stderr, flush=True)
        return REFUSAL

    policy = quality_policy(args)
    model = RidgeDecoder.load(Path(args.model))
    rate = chain_rate(args, model)
    source_record = source_block(args, model, rate)

    log(f"live demo: {len(args.electrodes)} electrodes by name, {args.sfreq:g} Hz, "
        f"units {args.source_units} (exponent {exponent})", lines)
    log(f"model {args.model}: window {source_record['window_seconds']}s, "
        f"step {source_record['step_seconds']}s, input "
        f"{source_record['model_input_hz']} Hz -> output "
        f"{source_record['model_output_hz']} Hz", lines)
    log(f"policy: check_channels={policy['check_channels']}, "
        f"max_bad_channels={policy['max_bad_channels']}, "
        f"exclude={policy['exclude_channels'] or 'none'}, amplitude "
        f"{policy['amplitude_limit_uv']} ({policy['amplitude_limit_source']}), "
        f"margin {policy['margin']} ({policy['margin_source']})", lines)

    calibration = load_calibration(args.calibration)
    if args.require_calibration and calibration is None:
        print(
            "REFUSED: --require-calibration was given and no --calibration profile "
            "was supplied. The audio-to-EEG loopback offset is unmeasured, and 100 ms "
            "of misalignment costs 0.119-0.150 balanced accuracy. Run "
            "scripts.getlive.calibrate_loopback first.",
            file=sys.stderr, flush=True,
        )
        return REFUSAL
    if calibration is None:
        log("WARNING: the audio-to-EEG loopback offset is UNMEASURED. The anchor "
            "comes from the browser's reported playback position and excludes the "
            "sound card and the acoustic path. See "
            "documents/live_demo_two_commands.md, calibration.", lines)
    else:
        log(f"loopback calibration: offset "
            f"{calibration.get('offset_seconds')} s "
            f"(tolerance +-{ALIGNMENT_TOLERANCE_SECONDS} s)", lines)

    # ---- the amplifier, before anything is served -------------------------
    try:
        row = wait_for_outlet(
            name=args.stream_name,
            source_id=args.source_id,
            stream_type=args.stream_type,
            timeout=float(args.resolve_seconds),
            interval=1.0,
            expected_channels=None if args.stream_type else args.expected_channels,
        )
    except RuntimeError as error:
        print(f"\nREFUSED: {error}\n\nNothing was served, nothing was acquired. Run "
              "`python -B -m scripts.getlive.live_source` (command 1) to see what is "
              "on the network.", file=sys.stderr, flush=True)
        return REFUSAL
    log(f"resolved outlet name={row['name']!r} type={row['stype']!r} "
        f"channels={row['n_channels']} sfreq={row['sfreq']:g} source_id="
        f"{row['source_id']!r}", lines)

    stream = None
    try:
        stream = open_inlet(row, bufsize=float(args.bufsize),
                            connect_timeout=float(args.connect_seconds))
        facts = channel_facts(stream)
        declared = {
            "name": facts["name"], "type": facts["stype"],
            "source_id": facts["source_id"], "sfreq": facts["sfreq"],
            "channels": list(facts["channels"]), "types": list(facts["types"]),
            "units": None if facts["units"] is None else list(facts["units"]),
            "n_channels": facts["n_channels"],
        }
        log(f"the outlet declares: {facts['n_channels']} channels, "
            f"{facts['sfreq']:g} Hz, units "
            f"{sorted({str(unit) for unit in (facts['units'] or ())})}", lines)
        source = AntStreamSource(
            stream,
            channels=tuple(args.electrodes),
            sfreq=float(args.sfreq),
            reference=args.reference,
            upstream_processing=args.upstream,
            kind="eego_live",
            source_unit_exponent=int(exponent),
            block_samples=int(args.block_samples),
            timebase=args.timebase,
            pre_resample_sfreq=rate,
            max_lag_seconds=float(args.max_lag_seconds),
            no_data_timeout=float(args.no_data_timeout),
        )
    except (RuntimeError, ValueError, OSError) as error:
        print(f"\nREFUSED: the source failed pre-flight: {type(error).__name__}: "
              f"{error}\n\nIf the outlet declares positional labels, no units, or a "
              "montage that includes non-EEG leads, run command 1 in bridge mode "
              "(`--mode bridge`) and point this command at the bridge.",
              file=sys.stderr, flush=True)
        if stream is not None and getattr(stream, "connected", False):
            stream.disconnect()
        return REFUSAL

    # ---- references, media and the contract gate --------------------------
    try:
        references = ReferenceEnvelopes.load(
            [Path(args.envelope_a), Path(args.envelope_b)], model.config
        )
        log(f"envelopes verified: {Path(args.envelope_a).name}, "
            f"{Path(args.envelope_b).name} (coverage {references.coverage()})", lines)
        media, media_report = prepare_media(args)
        log(f"rendered {args.presentation} stereo {media_report['published_seconds']}s "
            f"-> {args.media_out}", lines)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"\nREFUSED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if getattr(stream, "connected", False):
            stream.disconnect()
        return REFUSAL

    app_state = {
        "source": source,
        "references": references,
        "model": model,
        "media": media,
        "holder": {},
        "policy": policy,
        "source_unit_exponent": int(exponent),
        "source_record": source_record,
        "declared": declared,
        "chain_rate": rate,
        "audio_start": 0.0,
        "calibration": calibration,
    }

    def factory():
        session, producer = build(app_state, args)
        app_state["holder"]["producer"] = producer
        app_state["holder"]["session"] = session
        return producer

    # The gate, before a port exists. `build` constructs the chain, so the keys
    # the chain declares and the keys the model records are both known here.
    try:
        probe_session, _ = build(app_state, args)
        contract = contract_gate(
            model,
            AuditoryProcessor(
                probe_session.settings, source_channels=probe_session.source_channels
            ).contract,
            source="live chain",
        )
        probe_session.stop() if hasattr(probe_session, "stop") else None
    except ContractRefused as error:
        record = {
            "kind": "refused before start",
            "source": source_record,
            "contract": error.record,
            "quality_policy": policy,
            "message": str(error),
        }
        out = Path(args.out) if args.out else (
            REPO / "results" / f"live_demo_refused_{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"\nREFUSED: {error}\n\nRecord: {out}", file=sys.stderr, flush=True)
        if getattr(stream, "connected", False):
            stream.disconnect()
        return REFUSAL
    except (RuntimeError, ValueError) as error:
        print(f"\nREFUSED: the session could not be built: {type(error).__name__}: "
              f"{error}", file=sys.stderr, flush=True)
        if getattr(stream, "connected", False):
            stream.disconnect()
        return REFUSAL

    for line in contract_lines(contract):
        log(line, lines)

    # ---- serve -------------------------------------------------------------
    app = create_app(producer_factory=factory, media=media, static_dir=static_dir)
    port = int(args.port) or free_port()
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=port, log_level=args.log_level,
                       ws="websockets")
    )
    thread = threading.Thread(target=server.run, name="nova-uvicorn", daemon=True)
    thread.start()

    url = f"http://{args.host}:{port}/"
    app_state["url"] = url
    stop = asyncio.Event()
    loop_holder: dict = {}

    def request_stop(*_ignored):
        print("\n[live] stop requested; ending the session and shutting down",
              flush=True)
        loop = loop_holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(stop.set)
        else:
            stop.set()

    for name in ("SIGINT", "SIGTERM"):
        handle = getattr(signal, name, None)
        if handle is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(handle, request_stop)

    print()
    log(f"OPEN THIS URL: {url}", lines)
    log("  press play in the page; the two candidates play one per ear and the "
        "unattended ear is attenuated", lines)
    log("  Ctrl+C here ends the run and writes the record", lines)
    print(flush=True)

    state: dict = {}
    exit_code = 0

    async def runner() -> dict:
        if args.open_browser:
            asyncio.create_task(asyncio.to_thread(open_browser, url, lines))
        return await drive(port, args, app_state, stop, lines)

    loop_holder["loop"] = asyncio.new_event_loop()
    asyncio.set_event_loop(loop_holder["loop"])
    try:
        state = loop_holder["loop"].run_until_complete(runner())
    except KeyboardInterrupt:  # pragma: no cover - the handler usually catches it
        stop.set()
    except (RuntimeError, OSError, ValueError) as error:
        log(f"the run failed: {type(error).__name__}: {error}", lines)
        exit_code = 1
    finally:
        try:
            import httpx

            with httpx.Client(base_url=f"http://{args.host}:{port}", timeout=5.0) as http:
                http.post("/api/session/stop")
        except Exception:  # pragma: no cover - the server may already be gone
            pass
        server.should_exit = True
        thread.join(timeout=15)
        loop_holder["loop"].close()
        with contextlib.suppress(Exception):
            source.close()
        if getattr(stream, "connected", False):
            with contextlib.suppress(Exception):
                stream.disconnect()
        log("transport stopped", lines)

    transport = {}
    with contextlib.suppress(Exception):
        transport = source.transport.to_dict()
    record = run_record(args, app_state, state, media_report, transport, contract,
                        lines, began)
    out = Path(args.out) if args.out else (
        REPO / "results" / f"live_demo_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    report = out.with_suffix(".md")
    report.write_text(markdown(record), encoding="utf-8")
    log(f"run record: {out}", lines)
    log(f"human-readable report: {report}", lines)
    census = record["counters"]
    log(f"counters: {census['packets']} packets {census['by_type']}; decision words "
        f"{census['decision_words']}", lines)
    print(flush=True)
    print(markdown(record), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
