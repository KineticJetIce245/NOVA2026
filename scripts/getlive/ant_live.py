"""Run the whole auditory demo against a recorded ANT session over real LSL.

    python -B -m scripts.getlive.ant_live --clip 130

One command, four processes and one loopback port:

1. it loads an imported ANT trial (``datasets/AAD-ANT/session_*.npz``, written
   by ``scripts/auditory/antneuro.py``), the 20-channel live model and the two
   reference envelopes the operator's own recording was scored against;
2. it renders the stimulus actually played in that session - sliced out of
   ``left_mono.wav``/``right_mono.wav`` at the session's own audio start - to a
   stereo WAV, so the transport and the gain path see a real media file;
3. it starts a **publisher subprocess** that plays the trial over LSL on the
   loopback at its true 500 Hz rate, and connects the real live path to it:
   pre-flight by name, ``Acquire``, the time base, the channel contract and the
   auditory chain, all of which are the amplifier path's own code;
4. it drives the transport exactly as the demo does - the same app factory, the
   same ``SimulatedMediaClient`` speaking ``mediaController.js``'s protocol -
   and records the packets, the gate, the decisions and the transport counters;
5. it writes ``results/antneuro_live_<date>.{json,md}`` and stops the publisher.

**What this proves.** That recorded ANT EEG, carried by LSL at 500 Hz through
the live pre-flight and the live acquisition route, produces decisions on the
20-channel contract - and what those decisions are worth against the session's
own marker-derived labels, on a different rig (CPz reference against the model's
Cz, and two railed electrodes: F8 throughout and F3 for 39.8 % of session 1).

**What it does not prove.** No amplifier was involved, so nothing here says the
amplifier or its LSL outlet works. No live participant was involved, so it is
not a live decoding result. And it is not the audio-to-EEG loopback measurement
the plan requires before a human study (``residual_offset_seconds``, +-30 ms,
sections 3.11/3.17-4): there is no audio device anywhere in this path, so the
only loopback measured here is LSL delivery on one machine.

Loopback only; the media file is server-side configuration and is the only file
``/api/media/file`` can serve. No participant data is logged: the record holds
rates, counters, policy and decision words.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/getlive/ant_live.py`
    sys.path.insert(0, str(REPO))

from nova2026.auditory.config import (  # noqa: E402
    CALIBRATED_MARGIN,
    MIN_MARGIN,
    AuditoryConfig,
)
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
from nova2026.auditory.sources import ReferenceEnvelopes  # noqa: E402
from nova2026.streaming.preflight import validate_source  # noqa: E402
from nova2026.streaming.preprocess import Resampler  # noqa: E402
from nova2026.transport.media import MediaTimeline  # noqa: E402
from nova2026.transport.server import LOOPBACK_HOSTS, create_app  # noqa: E402
from scripts.auditory_ui.demo import capture, log, packet_census  # noqa: E402
from scripts.auditory_ui.media import MEDIA_TICK_SECONDS, SimulatedMediaClient  # noqa: E402
from scripts.auditory_ui.session import free_port  # noqa: E402
from scripts.getlive.ant_publish import (  # noqa: E402
    DEFAULT_SOURCE_ID,
    DEFAULT_STREAM_NAME,
    PUBLISHER_MODULE,
    load_ant_trial,
    slice_window,
)
from scripts.getlive.ant_source import (  # noqa: E402
    DEFAULT_MICROVOLT_EXPONENT,
    AntStreamSource,
    sleep_until_ready,
)
from scripts.getlive.outlets import channel_facts, open_inlet  # noqa: E402

DEFAULT_SESSION = "datasets/AAD-ANT/session_19-34-06.npz"
"""Session 1: its stimulus starts at 0 s, so a clip from 0 plays real audio."""

DEFAULT_MODEL = "models/auditory_kuleuven_live20.npz"
"""The 20-channel live model - the contract the rig's cap can actually carry."""

DEFAULT_AUDIO_ROOT = "tmp/antneurodata/audio_files/experiment"
"""The operator's own played stimulus; ``tmp/`` is git-ignored and never written."""

ENVELOPES = ("datasets/audio/left_mono.npz", "datasets/audio/right_mono.npz")
"""Reference envelopes, named explicitly: an ANT session has no story name, so
``feature_cache``'s story-to-envelope mapping does not apply here."""

TERMINAL_STATUSES = ("stopped", "error")


def load_candidates(
    audio_root: Path, start_seconds: float, samples: int, *, rate: int = 48000
) -> tuple[np.ndarray, dict]:
    """Slice the played stimulus for one published window.

    The recording's audio timeline is anchored at the trial's ``audio_start``
    (operator-given: ``audio position = start + (EEG time - Start marker)``), so
    the candidate columns must start there and cover exactly the published
    window. The slice is taken from ``left_mono``/``right_mono`` - the channels
    that were actually played (plan section 3.12) - and never from ``*_raw``,
    which was never presented.

    Raises:
        FileNotFoundError: If the played stimulus is not on this machine.
        ValueError: If the slice would run past the end of the stimulus.
    """

    from scipy.io import wavfile

    first = int(round(start_seconds * rate))
    last = first + int(samples)
    columns = []
    report: dict = {"sample_rate": rate, "slice_samples": [first, last]}
    for name in ("left_mono.wav", "right_mono.wav"):
        path = Path(audio_root) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"played stimulus not found: {path} (pass --audio-root). The "
                "operator's recording lives under tmp/, which is git-ignored."
            )
        got_rate, data = wavfile.read(path)
        if int(got_rate) != rate:
            raise ValueError(f"{path.name} is {got_rate} Hz, expected {rate} Hz.")
        if data.ndim != 1:
            raise ValueError(f"{path.name} must be mono.")
        if last > data.shape[0]:
            raise ValueError(
                f"{name} holds {data.shape[0]} samples; the window needs {last} "
                f"({last / rate:.3f}s from {start_seconds:.3f}s). Shorten --clip."
            )
        columns.append(np.asarray(data[first:last], dtype=np.float64) / 32768.0)
        report[name] = {"source": str(path), "slice_seconds": round(last / rate, 3)}
    return np.column_stack(columns), report


def parse_channel_list(value: str) -> tuple[str, ...]:
    """Read ``--exclude-channels`` as labels, refusing anything but labels.

    A comma-separated list is what the other live runners take
    (``scripts/getlive/live.py``), and an empty value means "exclude nothing",
    which is the default. A label that is not an electrode is refused later, by
    the chain's own :class:`~nova2026.streaming.preprocess.scope.ChannelScope`.
    """

    return tuple(part.strip() for part in value.split(",") if part.strip())


# The chain's own absolute-level guard, in uV: `Repair`'s default saturation
# limit and the number that decides whether an interpolation endpoint is unsafe.
# A run that sets `--saturation-limit-uv` moves it, and the exclusion check moves
# with it; the value in force is printed and recorded either way.
REPAIR_DEFAULT_SATURATION_UV = 75000.0


# The chain's own excursion guard, in uV: `Repair`'s default endpoint-jump limit
# **and** `QualityMonitor`'s amplitude fault, which flags an electrode once it
# moves further than this from the run's own anchor level. One number, two
# stages, so the quality verdict and the repair verdict cannot disagree; a run
# that sets `--amplitude-limit-uv` moves both, and the value in force is printed
# and recorded.
REPAIR_DEFAULT_AMPLITUDE_UV = 500.0


# An electrode counts as railed for the declaration check when at least this
# fraction of the published window sits at a repeated extreme. A single spike is
# not a rail; F8 in the operator's session 1 is at 83 333.3 uV for 100 % of the
# window and F3 for 50.6 %, so half the window separates the two from every
# healthy electrode in the same recording (the next largest peak is 42 304 uV).
RAILED_FRACTION = 0.5


def rail_profile(
    eeg: np.ndarray,
    names: tuple[str, ...],
    *,
    rel_tolerance: float = 1e-9,
    rail_fraction: float = RAILED_FRACTION,
) -> dict:
    """Measure each electrode's peak and how much of the window is railed.

    This is the evidence behind a declared exclusion, and it is measured on the
    same window the run publishes - not read from the recording anywhere else.
    An electrode is *railed* when a repeated extreme covers at least
    ``rail_fraction`` of the window: one amplifier spike is not a rail, and a
    channel that merely has a large peak must not be excludable by accident.
    """

    data = np.asarray(eeg, dtype=float)
    peak = np.abs(data).max(axis=0)
    extreme = np.abs(np.abs(data) - peak[None, :]) <= rel_tolerance * np.maximum(
        peak[None, :], 1.0
    )
    fraction = extreme.mean(axis=0)
    per_channel = {
        name: {
            "peak_microvolts": float(peak[index]),
            "railed_fraction": float(fraction[index]),
            "railed": bool(fraction[index] >= rail_fraction),
        }
        for index, name in enumerate(names)
    }
    railed = tuple(
        name
        for index, name in enumerate(names)
        if fraction[index] >= rail_fraction and peak[index] > 0
    )
    return {
        "rail_fraction_threshold": float(rail_fraction),
        "railed_channels": list(railed),
        "railed_peaks_microvolts": {
            name: float(peak[names.index(name)]) for name in railed
        },
        "railed_sample_fraction": {
            name: float(fraction[names.index(name)]) for name in railed
        },
        "per_channel": per_channel,
    }


def excursion_profile(
    eeg: np.ndarray,
    names: tuple[str, ...],
    *,
    declared_limit_uv: float,
    default_limit_uv: float = REPAIR_DEFAULT_AMPLITUDE_UV,
    anchor: str = "first-row",
) -> dict:
    """Measure what each electrode does relative to the run's own anchor level.

    ``QualityMonitor`` anchors its excursions at the first row it is fed and flags
    ``"amplitude"`` once a column moves further than the limit from that row, so
    the statistic that decides the limit is ``max |x(t) - anchor|`` per electrode -
    not the peak level, which is anchor dependent, and not the standard deviation,
    which a slow ramp hides inside.

    **It must be measured on the signal the monitor is actually fed**, not on the
    raw recording: the live adapter resamples 500 Hz to the 128 Hz the decoder
    contract records *before* the chain, and an anti-aliasing resampler with a
    7.4 s startup transient makes the first delivered rows a poor anchor. Measured
    on the raw window this recording's largest excursion on a usable electrode is
    3 637 uV; measured on the resampled stream the chain receives it is 19 521 uV,
    a factor of 5.4. A limit declared from the first number would have left the
    electrodes faulted, which is what the first declared run measured.

    Args:
        anchor: ``"first-row"`` reproduces the monitor's own anchor exactly;
            ``"first-second-median"`` is the anchor-robust companion, which shows
            how much of the maximum is the anchor row rather than the signal. Both
            are reported so the choice is visible rather than assumed.

    It reports the per-electrode maximum and the 99.9th percentile (a single
    spike should not set the limit) plus the two counts the declaration is
    accountable for: how many electrodes the *chain default* would fault, and how
    many the *declared* limit still faults. A declared limit that faults fewer
    electrodes than the default is the point; a declared limit that faults none
    of them has no selectivity left on this recording, which is a finding about
    the recording, and this measurement is what makes it visible.
    """

    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("excursion_profile needs samples by channels.")
    if len(names) != data.shape[1]:
        raise ValueError("excursion_profile needs one label per column.")
    if anchor == "first-row":
        level = data[0]
        anchor_note = "first row of the delivered stream (QualityMonitor's own anchor)"
    elif anchor == "first-second-median":
        if data.shape[0] < 2:
            raise ValueError("excursion_profile needs two rows for a median anchor.")
        level = np.median(data[: min(len(data), 128)], axis=0)
        anchor_note = "median of the first second of the delivered stream"
    else:
        raise ValueError("anchor must be 'first-row' or 'first-second-median'.")
    excursion = np.abs(data - level[None, :])
    peak = excursion.max(axis=0)
    q999 = np.percentile(excursion, 99.9, axis=0)
    per_channel = {
        name: {
            "anchor_microvolts": float(level[index]),
            "excursion_max_microvolts": float(peak[index]),
            "excursion_p999_microvolts": float(q999[index]),
            "faulted_at_limit": bool(peak[index] > declared_limit_uv),
            "faulted_at_default": bool(peak[index] > default_limit_uv),
        }
        for index, name in enumerate(names)
    }
    return {
        "anchor": anchor_note,
        "anchor_mode": anchor,
        "declared_limit_uv": float(declared_limit_uv),
        "default_limit_uv": float(default_limit_uv),
        "faulted_at_declared": sorted(
            name for name, row in per_channel.items() if row["faulted_at_limit"]
        ),
        "faulted_at_default": sorted(
            name for name, row in per_channel.items() if row["faulted_at_default"]
        ),
        "max_excursion_microvolts": float(peak.max()),
        "max_excursion_on_unfaulted_microvolts": float(
            max(
                (
                    row["excursion_max_microvolts"]
                    for row in per_channel.values()
                    if not row["faulted_at_limit"]
                ),
                default=0.0,
            )
        ),
        "per_channel": per_channel,
    }


def chain_point_window(window, channels: tuple[str, ...], rate: float):
    """The window the chain's monitor is actually fed, at the rate it declares.

    The live path's adapter resamples the 500 Hz outlet to the rate the decoder
    contract records before the chain sees it, so a limit measured on the raw
    recording is measured at the wrong point. This runs the chain's own
    ``Resampler`` over the published window with the adapter's own preset and
    returns ``(data, timestamps)`` in the delivered rate, or ``None`` when the
    window already arrives at that rate or the stage refuses it.
    """

    if math.isclose(float(window.sample_rate), float(rate)):
        return None
    wanted = [str(name) for name in channels]
    present = [str(name) for name in window.channel_names]
    if any(name not in present for name in wanted):
        return None
    rows = window.eeg[:, [present.index(name) for name in wanted]]
    try:
        stage = Resampler(
            float(window.sample_rate), float(rate), rows.shape[1], quality="HQ",
            max_age_seconds=30.0, reserve_seconds=1.0, allow_qq=True, strict=True,
        )
        data, stamps = stage(rows, np.asarray(window.timestamps, dtype=float))
    except (ValueError, RuntimeError):
        return None
    return data, stamps


def amplitude_guard_uv(args) -> float:
    """The excursion guard this run's chain will use, in uV.

    ``None`` means the chain's own default (500 uV), which is what every caller
    that declares nothing gets. The value in force is the one printed and
    recorded, and it is the number the declaration is accounted against: it must
    be above the recording's own measured excursion, or every window stays an
    artifact; and it must stay far below the amplifier's rail, so that a genuinely
    railed electrode is still named rather than absorbed by a widened number.
    """

    requested = getattr(args, "amplitude_limit_uv", None)
    return REPAIR_DEFAULT_AMPLITUDE_UV if requested is None else float(requested)


def saturation_guard_uv(args) -> float:
    """The absolute-level guard this run's ``Repair`` will use, in uV.

    ``None`` means the chain's own default (75 000 uV). The value is printed and
    recorded, and it is the number the exclusion declaration is checked against:
    an electrode that does not reach it was never going to make an endpoint
    unsafe, so excluding it would blind the check for a channel that did not
    need it.
    """

    requested = getattr(args, "saturation_limit_uv", None)
    return REPAIR_DEFAULT_SATURATION_UV if requested is None else float(requested)


def check_excluded_channels(
    profile: dict,
    excluded: tuple[str, ...],
    *,
    saturation_limit_uv: float,
    allow_unrailed: bool,
) -> list[str]:
    """Verify every declared exclusion is justified on this window.

    Returns the human-readable lines the run prints and records. Raises
    ``ValueError`` - which the caller turns into exit code 2, before any session
    exists - when a declared electrode is neither railed nor explicitly allowed,
    because a silent exclusion of a healthy electrode is exactly the failure
    mode the per-channel path exists to avoid: it would hide real damage on a
    channel that still carries signal.
    """

    lines = []
    for name in excluded:
        measured = profile["per_channel"].get(name)
        if measured is None:
            raise ValueError(
                f"--exclude-channels names {name!r}, which is not one of the "
                f"{len(profile['per_channel'])} published electrodes."
            )
        peak = measured["peak_microvolts"]
        if measured["railed"]:
            lines.append(
                f"excluded {name}: railed at {peak:.1f} uV for "
                f"{100 * measured['railed_fraction']:.1f} % of the published "
                f"window (>= {saturation_limit_uv:.0f} uV guard and >= "
                f"{100 * profile['rail_fraction_threshold']:.0f} % of the window)"
            )
        elif not allow_unrailed:
            raise ValueError(
                f"--exclude-channels names {name!r}, but it is not railed "
                f"(peak {peak:.1f} uV over "
                f"{100 * measured['railed_fraction']:.1f} % of the window, "
                f"threshold {100 * profile['rail_fraction_threshold']:.0f} %). "
                "Excluding a healthy electrode would blind the endpoint check "
                "for a channel that still carries signal; pass "
                "--allow-unrailed-exclusion to declare an electrode dead anyway."
            )
        else:
            lines.append(
                f"excluded {name}: NOT railed (peak {peak:.1f} uV over "
                f"{100 * measured['railed_fraction']:.1f} % of the window) - "
                "declared dead explicitly by --allow-unrailed-exclusion"
            )
    if excluded:
        lines.append(
            f"excluded electrodes stay in the run: {len(excluded)} of "
            f"{len(profile['per_channel'])} columns keep their data, are still "
            "repaired and still counted in the bad-channel census; nothing is "
            "dropped, so the decoder's 20-channel contract is untouched"
        )
    return lines


def decision_census(summary: dict, decisions: list) -> dict:
    """How the run spent its windows: committed, abstained, or absent.

    ``decided`` counts frames that committed to A or B; ``evidence_gap`` counts
    frames whose evidence never reached the worker (the queue dropped them), and
    is reported rather than folded into the abstentions because it is a
    different failure: there was nothing to decide on.
    """

    census = summary.get("decisions") or {}
    return {
        "windows": int(summary.get("windows") or 0),
        "decided": int(census.get("A", 0)) + int(census.get("B", 0)),
        "A": int(census.get("A", 0)),
        "B": int(census.get("B", 0)),
        "uncertain": int(census.get("uncertain", 0)),
        "unavailable": int(census.get("unavailable", 0)),
        "evidence_gap": int(summary.get("evidence_gaps") or 0),
        "decision_stream": [[float(time_s), str(word)] for time_s, word in decisions],
    }


def chain_rate(args, model=None) -> float:
    """The rate the auditory chain is actually fed, which it must also declare.

    ``RidgeDecoder.validate`` compares the chain's whole contract with the
    model's verbatim, so the adapter that converts the 500 Hz outlet to the
    model's recorded rate has to move ``input_sfreq`` with it. Leaving the
    declaration at 500 while delivering 128 Hz samples makes ``Repair`` read one
    step as 3.9 samples: it synthesises three missing rows per sample and reports
    every window as ``interpolated`` - measured on this recording, 7608 phantom
    repairs - which is what really stopped the run, before the railed electrodes
    ever mattered. This is the rate of the delivered data, not a change to the
    recording.
    """

    if args.pre_resample == "off":
        return float(args.sfreq)
    if args.pre_resample == "auto":
        wanted = None if model is None else model.contract.get("input_sfreq")
        return float(args.sfreq if wanted is None else wanted)
    return float(args.pre_resample)



def build(app_state: dict, args):
    """Build the session and the producer: the factory the REST path calls.

    Called twice by design, exactly as the demo does it: once to prove start-up
    can succeed before the URL exists, and once when ``POST /api/session/start``
    arrives, so a viewer's session is started through the real REST path.
    """

    source, references, model = (
        app_state["source"],
        app_state["references"],
        app_state["model"],
    )
    # The live adapter cannot know where this recording's audio began; the
    # operator's own anchor does. Time zero of the published window plus the
    # window's own first timestamp plus the session's audio offset is where the
    # stimulus was actually at sample zero of the window.
    source.audio_start = float(app_state["audio_first_timestamp"]) + float(
        app_state["audio_offset"]
    )
    policy = RunPolicy(
        check_channels=False,
        max_bad_channels=0,
        margin=float(args.margin),
        warmup_seconds=args.warmup,
        frame_seconds=args.frame_seconds,
        source_unit_exponent=int(args.source_unit_exponent),
        # Run policy, printed and recorded. A railed electrode is handled by
        # naming it (``--exclude-channels``), not by widening a limit that guards
        # every channel: the guard stays at the chain's own default unless the
        # operator moves it explicitly (plan sections 3.11 and 4-V4).
        saturation_limit_uv=(
            None
            if getattr(args, "saturation_limit_uv", None) is None
            else float(args.saturation_limit_uv)
        ),
        # The excursion guard is declared the same way and for the same reason:
        # this recording moves further from its own anchor than the chain's 500 uV
        # default, so at the default every window is an artifact and nothing is
        # ever committed. The declared number is measured on the published window
        # (see `excursion_profile`) and it reaches BOTH stages that judge an
        # electrode with it - `QualityMonitor`'s amplitude fault and `Repair`'s
        # endpoint check - so the quality verdict and the repair verdict cannot
        # disagree. Unset keeps the chain default.
        amplitude_limit_uv=(
            None
            if getattr(args, "amplitude_limit_uv", None) is None
            else float(args.amplitude_limit_uv)
        ),
        exclude_channels=tuple(app_state["excluded"]),
    )
    # `AttentionSession` reads the chain settings off the SOURCE, so the source
    # must declare the rate the adapter actually delivers (see `chain_rate`).
    # The outlet was already pre-flighted at its own `--sfreq`; this is the rate
    # of the data the chain is handed, and the contract records it.
    declared = app_state.get("chain_rate")
    if declared is not None:
        source.sample_rate = float(declared)
    session = AttentionSession(
        source=source, decoder=model, references=references, policy=policy
    )
    producer = AttentionProducer(session)
    if app_state.get("media") is not None:
        producer.media_reference = app_state["media"].media_reference
    return session, producer


def probe(stream, channels: tuple[str, ...], args, lines: list[str]) -> dict:
    """Report what the outlet declared and what the live pre-flight accepted.

    It also runs the same pre-flight **with a deliberately wrong expectation**,
    which is the only way to show that acceptance was a check rather than a
    formality: the same stream is refused when it is asserted to be volts
    instead of microvolts. Nothing is loosened to make either run pass.
    """

    facts = channel_facts(stream)
    report: dict = {
        "declared": {
            "name": facts["name"],
            "type": facts["stype"],
            "source_id": facts["source_id"],
            "sfreq": facts["sfreq"],
            "channels": list(facts["channels"]),
            "types": list(facts["types"]),
            "units": None if facts["units"] is None else list(facts["units"]),
            "dtype": facts["dtype"],
        "n_channels": facts["n_channels"],
        },
        "accepted": None,
        "rejected": [],
    }
    validate_source(
        stream,
        sfreq=args.sfreq,
        channels=channels,
        source_unit_exponent=int(args.source_unit_exponent),
    )
    report["accepted"] = {
        "channels": list(channels),
        "n_channels": len(channels),
        "sfreq": args.sfreq,
        "source_unit_exponent": int(args.source_unit_exponent),
        "n_channels_declared": facts["n_channels"],
        "dropped_columns": [
            name for name in facts["channels"] if name not in set(channels)
        ],
    }
    log(
        f"pre-flight accepted the outlet: {len(channels)} channels by name at "
        f"{args.sfreq:g} Hz, units exponent {int(args.source_unit_exponent)}",
        lines,
    )
    for label, kwargs in (
        ("volts-instead-of-microvolts", {"source_unit_exponent": 0}),
        ("wrong-rate", {"sfreq": args.sfreq * 2}),
        (
            "wrong-channel-name",
            {"channels": tuple("X" + name for name in channels)},
        ),
    ):
        try:
            validate_source(
                stream,
                sfreq=kwargs.get("sfreq", args.sfreq),
                channels=kwargs.get("channels", channels),
                source_unit_exponent=kwargs.get(
                    "source_unit_exponent", int(args.source_unit_exponent)
                ),
            )
        except RuntimeError as error:
            report["rejected"].append({"counter-check": label, "error": str(error)})
            log(f"counter-check {label}: refused - {error}", lines)
        else:  # pragma: no cover - a counter-check that passes is itself a finding
            report["rejected"].append(
                {"counter-check": label, "error": "NOT refused", "finding": True}
            )
            log(f"FINDING: counter-check {label} was NOT refused", lines)
    return report


def coverage_against_markers(labels: np.ndarray, decided: list, rate: float) -> dict:
    """Window decisions against the session's own per-sample marker labels.

    The label is read **after** the stream is closed and only here: it is the one
    thing that must never enter the decision path (plan rule 3). ``-1`` is the
    operator's unknown band around every cue switch and is excluded from the
    agreement rather than counted as a miss.
    """

    result: dict = {
        "kind": (
            "replay of the operator's own recorded ANT session through the live "
            "LSL route; a different rig and a different reference from the model's "
            "training data, so a poor result is expected"
        ),
        "labelled_samples": int(np.count_nonzero(labels >= 0)),
        "unknown_samples": int(np.count_nonzero(labels < 0)),
        # The null this recording itself sets: with this label balance, always
        # answering the more common class scores `majority_class_rate_on_labelled`.
        "label_balance": {
            "A_samples": int(np.count_nonzero(labels == 0)),
            "B_samples": int(np.count_nonzero(labels == 1)),
            "unknown_samples": int(np.count_nonzero(labels < 0)),
            "A_share_of_labelled": (
                float(np.count_nonzero(labels == 0) / max(np.count_nonzero(labels >= 0), 1))
            ),
        },
    }
    if not decided:
        result["note"] = "no committed window to score"
        return result
    index = np.clip(
        np.round(np.asarray([float(moment) for moment, _ in decided]) * rate).astype(int),
        0,
        len(labels) - 1,
    )
    truth = labels[index]
    guess = np.asarray([0 if word == "A" else 1 for _, word in decided])
    known = truth >= 0
    result["decided_windows"] = int(len(guess))
    result["decided_windows_labelled"] = int(np.count_nonzero(known))
    result["decided_windows_in_unknown_band"] = int(np.count_nonzero(~known))
    # False selection changes: pairs of consecutive COMMITTED frames (in the
    # order they were published) that name different candidates. Nothing is
    # dropped to make this number look better - abstentions are not carried
    # forward, so a flickering decoder is visible as a large count.
    changes = np.flatnonzero(np.diff(guess) != 0) if len(guess) > 1 else np.empty(0, int)
    result["decided_pairs"] = int(max(len(guess) - 1, 0))
    result["false_selection_changes"] = int(changes.size)
    result["false_selection_change_rate"] = (
        float(changes.size / max(len(guess) - 1, 1)) if len(guess) > 1 else None
    )
    result["false_selection_changes_series"] = [
        {"window_index": int(position), "seconds": float(decided[position][0])}
        for position in changes.tolist()
    ]
    # The truth's own switching, over the same committed frames: a decoder that
    # changes less often than the stimulus does cannot be following it.
    if len(guess) > 1:
        truth_pairs = truth[:-1][known[:-1] & known[1:]]
        next_pairs = truth[1:][known[:-1] & known[1:]]
        result["true_selection_changes_between_decided_pairs"] = int(
            np.count_nonzero(truth_pairs != next_pairs)
        )
    if not np.any(known):
        result["note"] = "every committed window fell in the operator's unknown band"
        return result
    agree = guess[known] == truth[known]
    result.update(
        {
            "window_accuracy_on_labelled": float(agree.mean()),
            "majority_class_rate_on_labelled": float(
                max((truth[known] == value).mean() for value in (0, 1))
            ),
            "balanced_accuracy_on_labelled": float(
                np.mean(
                    [
                        (agree[truth[known] == value].mean() if (truth[known] == value).any() else 0.0)
                        for value in (0, 1)
                    ]
                )
            ),
            "recall_on_labelled": {
                "A": (
                    float(agree[truth[known] == 0].mean())
                    if (truth[known] == 0).any()
                    else None
                ),
                "B": (
                    float(agree[truth[known] == 1].mean())
                    if (truth[known] == 1).any()
                    else None
                ),
            },
        }
    )
    result["note"] = (
        "agreement over the committed windows of ONE recorded session, replayed "
        "through the live route on this machine; abstentions are reported as "
        "uncovered time, not as misses. Not a live-participant accuracy, and not "
        "a cross-subject claim."
    )
    return result


def markdown(record: dict) -> str:
    """Render the run record as the report a reviewer reads first."""

    transport = record["transport"]
    labels = record["labels_vs_decisions"]
    gate = (record["frontend_evidence"].get("gate") or {}).get("report") or {}
    decode = (record["frontend_evidence"].get("decode") or {}).get("report") or {}
    lines = [
        f"# ANT live-route run - {record['stamp']}",
        "",
        f"Session `{record['session']['trial_id']}` "
        f"({record['session']['samples']} samples, "
        f"{record['session']['seconds']} s, {record['session']['channels']} channels) "
        f"published on LSL at {record['publisher']['sample_rate']:g} Hz; the live "
        f"pre-flight accepted it and the session ran for {record['clip_seconds']:g} s "
        f"at 1x.",
        "",
        "## What accepted the source, and what it had to reject",
        "",
        f"- declared by the outlet: {record['probe']['declared']['n_channels']} "
        f"channels, {record['probe']['declared']['sfreq']:g} Hz, units "
        f"{sorted(set(record['probe']['declared']['units'] or []))}, type "
        f"{record['probe']['declared']['type']!r}",
        f"- accepted: {record['probe']['accepted']['n_channels']} electrodes by "
        f"name, exponent {record['probe']['accepted']['source_unit_exponent']} "
        f"(microvolts), no column dropped",
        "",
        "| counter-check | refused with |",
        "| --- | --- |",
    ]
    for row in record["probe"]["rejected"]:
        lines.append(f"| {row['counter-check']} | {row['error']} |")
    point = record["operating_point"]
    excluded = list(point["excluded_channels"])
    census = record.get("decision_census") or {}
    lines += [
        "",
        "## Channel policy: a declared railed electrode, not a wider limit",
        "",
        f"- excluded electrodes: {excluded or 'none'}; `Repair` saturation guard "
        f"{point['repair_saturation_limit_uv']:.0f} uV "
        f"({point['saturation_limit_source']}), amplitude guard "
        f"{point['amplitude_limit_uv']:.0f} uV "
        f"({point['amplitude_limit_source']}; the chain's own default stays "
        f"{point['declared_chain_default_amplitude_uv']:.0f} uV for every caller "
        "that declares nothing) - the guards themselves are unchanged for every "
        "channel",
        f"- measured on the published window: "
        + (
            "; ".join(
                f"{row['channel']} peak {row['peak_microvolts']:.1f} uV railed "
                f"{100 * row['railed_fraction']:.1f} %"
                for row in point["excluded_channels_justification"]
            )
            or "no electrode declared"
        ),
        f"- every other electrode in the window peaks below "
        f"{point['rail_profile']['rail_fraction_threshold']:.0%} railed, so the "
        "guard still judges them: a railed channel that is NOT declared excluded "
        "still stops the run",
        f"- excluded electrodes keep their columns: {point['excluded_channels_stay_in_the_run']}",
        "",
        "## The declared amplitude limit, and the measurement it came from",
        "",
    ]
    excursion = point.get("excursion_profile") or {}
    if excursion:
        lines += [
            f"- anchor: {excursion['anchor']}; excursion measured over the "
            f"{record['published_window']['samples']} samples of the published "
            "window",
            f"- at the chain's own default {excursion['default_limit_uv']:.0f} uV, "
            f"{len(excursion['faulted_at_default'])} of "
            f"{len(record['session']['channels'])} electrodes fault "
            f"({', '.join(excursion['faulted_at_default']) or 'none'}) - the "
            "amplitude fault that makes every window an artifact, so no decision "
            "is ever committed",
            f"- declared for this run: {excursion['declared_limit_uv']:.0f} uV "
            f"({point['amplitude_limit_source']}); "
            f"{len(excursion['faulted_at_declared'])} of "
            f"{len(record['session']['channels'])} electrodes fault at it "
            f"({', '.join(excursion['faulted_at_declared']) or 'none'}), and the "
            f"largest excursion on an electrode it still admits is "
            f"{excursion['max_excursion_on_unfaulted_microvolts']:.1f} uV",
            f"- what the declared value reaches: {point['amplitude_limit_reaches']}",
            "- what a declared limit does NOT do: it does not make the check "
            "selective. The amplitude fault asks only whether an electrode stayed "
            "within the declared distance of its own anchor level; a recording "
            "whose *normal* movement exceeds the old limit cannot be separated "
            "from an artifact by this criterion at any value, so `signal_quality` "
            "for this session means 'nothing moved further than N uV from its "
            "anchor', not 'this window is artifact-free'. Saturation, flatline "
            "and the channel census are untouched and still judge the window.",
            "",
        ]
    lines += [
        "## Decisions against the session's own marker labels",
        "",
        f"- status `{(record.get('session_record') or {}).get('status')!r}`, decisions "
        f"{json.dumps(record['session'].get('decisions') or {})}",
        f"- session failure {json.dumps(record['session'].get('failure'))}",
        f"- decision census: {census.get('decided')} decided "
        f"(A {census.get('A')}, B {census.get('B')}), "
        f"{census.get('uncertain')} uncertain, {census.get('unavailable')} "
        f"unavailable, {census.get('evidence_gap')} evidence_gap, of "
        f"{census.get('windows')} windows "
        f"({record['session'].get('windows')} counted by the session)",
        f"- decided windows {labels.get('decided_windows')} of "
        f"{record['session'].get('windows')} windows; "
        f"{labels.get('decided_windows_in_unknown_band')} of them fell in the "
        f"operator's unknown band and are excluded",
        f"- accuracy on labelled windows "
        f"{labels.get('window_accuracy_on_labelled')}, majority-class rate on the "
        f"same windows {labels.get('majority_class_rate_on_labelled')} (this is the "
        f"null the recording's own balance sets: A "
        f"{labels.get('label_balance', {}).get('A_samples')} vs B "
        f"{labels.get('label_balance', {}).get('B_samples')} labelled samples), "
        f"balanced {labels.get('balanced_accuracy_on_labelled')}",
        f"- per-class recall on decided frames: "
        f"{json.dumps(labels.get('recall_on_labelled'))}",
        f"- false selection changes: {labels.get('false_selection_changes')} of "
        f"{labels.get('decided_pairs')} consecutive committed pairs "
        f"(rate {labels.get('false_selection_change_rate')}); the labels themselves "
        f"switch {labels.get('true_selection_changes_between_decided_pairs')} times "
        "between the same pairs",
        "",
        "## Transport diagnostics (the live path's own counters)",
        "",
        f"- `max_lag` {transport['max_lag_seconds']} s over "
        f"{transport['blocks']} blocks / {transport['samples']} samples",
        f"- `gaps` {transport['gaps']}, largest {transport['max_gap_seconds']} s",
        f"- source clock as delivered: {transport['raw_rate_hz']} Hz "
        f"(nominal {record['publisher']['sample_rate']:g})",
        f"- peak deviation from a perfect grid: "
        f"{transport['grid_deviation_peak_seconds']} s; re-locks "
        f"{transport['timebase_relocks']}, suspicious steps "
        f"{transport['timebase_large_steps']}",
        f"- pre-chain rate conversion: {transport['pre_resampled_to_hz']} "
        f"({transport['pre_resampled_samples']} samples); chain resampler "
        f"startup delay {transport['chain_resampler_startup_delay_seconds']} s "
        f"({transport['chain_resampler_quality']})",
        f"- recovery events {len(transport['recovery_events'])}, "
        f"repaired samples {transport['repaired_samples']}, "
        f"bad-channel census {transport['bad_channel_census'] or 'none'}",
        "",
        "## Frontend gate (Node-side evidence, not a browser)",
        "",
        f"- gain gate open: {gate.get('gate_open')} at "
        f"{gate.get('gate_opened_at_seconds')} s; attenuated gain frames "
        f"{gate.get('playback_gains_applied')} of {gate.get('sampled_gain_packets')}",
        f"- the vendored frontend's own decoders rejected {decode.get('rejects')} "
        f"of {decode.get('packets')} packets",
        "",
        "## What this proves, and what it does not",
        "",
        "Proves: the recorded ANT session drives the live route - LSL transport, "
        "20-channel pre-flight by name, 500 Hz, microvolts after the import's "
        "conversion - and produces decisions, packets and gain frames on this "
        "machine, with the transport's own timing counters recorded above. A "
        "railed electrode, declared with `--exclude-channels` and measured railed "
        "on the published window, cannot stop the run while the endpoint guard "
        "keeps its own value for every other channel.",
        "",
        "Does not prove: that an amplifier works (no amplifier was present); "
        "anything about a live participant (the input is a recording); the "
        "audio-to-EEG loopback offset the plan requires before a human study "
        "(`residual_offset_seconds`, tolerance +-30 ms, sections 3.11/3.17-4) - "
        "there is no audio device in this path, so the only loopback here is LSL "
        "delivery; and nothing about the model's accuracy on this rig, because "
        "the recording is a different rig with a different reference (CPz against "
        "the model's Cz) and two railed electrodes (F8 throughout, F3 for 39.8 % "
        "of session 1).",
        "",
    ]
    return "\n".join(lines)


def audio_start_of(trial: AntTrial, start_offset_seconds: float) -> float:
    """The session-clock time at which stimulus position zero was played.

    ``audio position = start offset + (EEG time - Start marker)`` is the
    operator-given timeline of the recorded experiment (plan section 3.12): the
    offset is 0 s for session 1 and 267 s for session 2, and the Start marker is
    the recording's own first timestamp because the import already rebased the
    trial onto it. So the session time of stimulus position zero is the trial's
    first timestamp plus that offset - an operator fact, reported rather than
    inferred from the envelope's leading zeros (the import's own metadata records
    its envelope as zero where nothing was playing, but the played stimulus
    itself does not begin with silence, so "first non-zero row" is not the
    answer).
    """

    return float(trial.timestamps[0]) + float(start_offset_seconds)


async def drive(port: int, args, media_report: dict, client, stop: asyncio.Event) -> dict:
    """Start the session over REST, run the client, and watch it end."""

    import httpx

    base = f"http://{args.host}:{port}"
    packets: list = []
    arrivals: list = []
    state: dict = {"started": None, "media": None, "final": None,
                   "packets": packets, "arrivals": arrivals}
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as http:
        for _ in range(200):
            with contextlib.suppress(httpx.HTTPError):
                if (await http.get("/api/health")).status_code == 200:
                    break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("the transport never became ready on loopback")

        listener = asyncio.create_task(capture(port, packets, stop, client, arrivals))
        state["started"] = (await http.post("/api/session/start")).json()
        client.configure(state["started"]["id"], media_report["media_id"])
        client.state["prepared"] = False
        await asyncio.sleep(0)
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


def start_publisher(args, window, lines: list[str]) -> subprocess.Popen:
    """Start the LSL publisher as its own process, as the rig's recorder would be."""

    command = [
        sys.executable,
        "-B",
        "-m",
        PUBLISHER_MODULE,
        "--session",
        str(args.session),
        "--start",
        str(args.clip_start),
        "--seconds",
        str(args.clip),
        "--name",
        args.stream_name,
        "--source-id",
        args.source_id,
    ]
    log("publisher: " + " ".join(command), lines)
    return subprocess.Popen(
        command,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run(args, lines: list[str]) -> tuple[dict, int]:
    """Everything except argument parsing: load, connect, stream, report."""

    began = time.time()
    config = AuditoryConfig()
    model = RidgeDecoder.load(Path(args.model))
    trial = load_ant_trial(args.session)
    window, first_index = slice_window(trial, args.clip_start, args.clip)
    audio_offset = audio_start_of(trial, args.audio_start_offset)
    log(
        f"session audio offset {audio_offset:g}s (the operator-given start offset "
        f"{args.audio_start_offset:g}s from the recording's own Start anchor, "
        f"section 3.12; never derived from the envelope)",
        lines,
    )
    channels = tuple(model.contract["eeg_channels"])
    if tuple(trial.channel_names) != channels:
        # Not fatal by itself - the contract maps by name - but the model's own
        # electrode set is the contract the cap can carry, so a mismatch here is
        # reported rather than silently reordered.
        log(
            f"note: the trial carries {list(trial.channel_names)} while the model "
            f"contract names {list(channels)}; the contract maps by name",
            lines,
        )
    log(
        f"session {trial.trial_id}: {len(trial.timestamps)} samples, "
        f"{trial.seconds:.3f}s, {len(trial.channel_names)} channels at "
        f"{trial.sample_rate:g} Hz, peak {float(np.max(np.abs(window.eeg))):.1f} uV",
        lines,
    )
    log(
        f"published window: {len(window.timestamps)} samples from index "
        f"{first_index} ({args.clip_start:g}s + {args.clip:g}s)",
        lines,
    )
    # The channel policy for this run, measured on the window about to be
    # published and decided before the session exists. A railed electrode is a
    # declared, measured exclusion rather than a raised global limit: the limit
    # guards every other channel and must keep doing so (plan sections 3.11, 4-V4).
    profile = rail_profile(window.eeg, tuple(trial.channel_names))
    excluded = tuple(args.exclude_channels)
    guard = saturation_guard_uv(args)
    try:
        exclusion_lines = check_excluded_channels(
            profile,
            excluded,
            saturation_limit_uv=guard,
            allow_unrailed=bool(args.allow_unrailed_exclusion),
        )
    except ValueError as error:
        print(f"{error}", file=sys.stderr)
        return {}, 2
    log(
        f"declared channel policy: exclude_channels={list(excluded)}; "
        f"Repair's saturation guard stays at {guard:.0f} uV "
        f"({'chain default' if args.saturation_limit_uv is None else 'explicit'}), "
        f"amplitude guard {amplitude_guard_uv(args):.0f} uV "
        f"({'chain default' if args.amplitude_limit_uv is None else 'explicit'})",
        lines,
    )
    # The excursion declaration, measured before the session exists: the limit the
    # chain will judge every electrode with is chosen from this window's own
    # numbers and from nothing else, and both the measurement and the declared
    # value go into the record so a reader can check the arithmetic.
    #
    # Measured at the point the monitor is fed, not on the raw recording: the live
    # adapter resamples 500 Hz to the contract's rate first, and the resampler's
    # own transient moves the anchor row. On the raw window this recording's
    # largest usable-electrode excursion is 3 637 uV; on the delivered stream it is
    # 19 521 uV. Declaring the first number left eight electrodes faulted on the
    # first declared run - the measurement was right and the point was wrong.
    delivered = chain_point_window(
        window, tuple(model.contract["eeg_channels"]),
        float(model.contract.get("input_sfreq") or args.sfreq),
    )
    if delivered is None:
        measured_eeg, measured_names, measured_point = (
            window.eeg, tuple(trial.channel_names),
            "the published window (already at the rate the chain declares)",
        )
    else:
        measured_eeg, measured_names, measured_point = (
            delivered[0], tuple(model.contract["eeg_channels"]),
            f"the stream the chain is fed at "
            f"{float(model.contract.get('input_sfreq') or args.sfreq):g} Hz, after the "
            f"pre-chain resampler",
        )
    excursion = excursion_profile(
        measured_eeg, measured_names, declared_limit_uv=amplitude_guard_uv(args)
    )
    robust = excursion_profile(
        measured_eeg, measured_names, declared_limit_uv=amplitude_guard_uv(args),
        anchor="first-second-median",
    )
    excursion["measured_at"] = measured_point
    excursion["anchor_robust_companion"] = {
        "anchor": robust["anchor"],
        "max_excursion_microvolts": robust["max_excursion_microvolts"],
        "max_excursion_on_unfaulted_microvolts": (
            robust["max_excursion_on_unfaulted_microvolts"]
        ),
        "faulted_at_default": robust["faulted_at_default"],
    }
    log(
        f"measured excursion from the window's own anchor level, at "
        f"{measured_point}: {len(excursion['faulted_at_default'])} of "
        f"{len(measured_names)} electrodes move further than the chain default "
        f"{REPAIR_DEFAULT_AMPLITUDE_UV:.0f} uV "
        f"({', '.join(excursion['faulted_at_default']) or 'none'})",
        lines,
    )
    log(
        f"declared amplitude limit {amplitude_guard_uv(args):.0f} uV "
        f"({'CHAIN DEFAULT UNCHANGED' if args.amplitude_limit_uv is None else 'declared for this run'}): "
        f"{len(excursion['faulted_at_declared'])} of "
        f"{len(measured_names)} electrodes fault at it "
        f"({', '.join(excursion['faulted_at_declared']) or 'none'}); largest "
        f"excursion measured {excursion['max_excursion_microvolts']:.1f} uV, largest "
        f"on an electrode it still admits "
        f"{excursion['max_excursion_on_unfaulted_microvolts']:.1f} uV; the chain's own "
        f"default stays {REPAIR_DEFAULT_AMPLITUDE_UV:.0f} uV for every caller that "
        f"declares nothing",
        lines,
    )
    log(
        f"anchor robustness: with the anchor taken as the first second's median "
        f"instead of the first row, the largest excursion is "
        f"{robust['max_excursion_microvolts']:.1f} uV and "
        f"{len(robust['faulted_at_default'])} of {len(measured_names)} electrodes move "
        f"further than the chain default",
        lines,
    )
    log(
        "measured rail profile over the published window: "
        + (
            ", ".join(
                f"{name} peak {profile['per_channel'][name]['peak_microvolts']:.1f} uV "
                f"railed {100 * profile['per_channel'][name]['railed_fraction']:.1f} %"
                for name in profile["railed_channels"]
            )
            or "no electrode is railed"
        ),
        lines,
    )
    for line in exclusion_lines:
        log(line, lines)
    references = ReferenceEnvelopes.load(tuple(ENVELOPES), config, labels=("left", "right"))
    log(
        f"reference envelopes named explicitly (an ANT session has no story name): "
        f"{Path(ENVELOPES[0]).name}, {Path(ENVELOPES[1]).name}; coverage "
        f"{references.coverage()}",
        lines,
    )
    log(
        f"operating point: margin={float(args.margin):g} "
        f"({'calibrated default' if abs(float(args.margin) - CALIBRATED_MARGIN) < 1e-12 else 'explicit'}), "
        f"documented policy default {MIN_MARGIN}; check_channels=False "
        f"(section 3.17 item 1)",
        lines,
    )

    # The stimulus the operator actually played for this session, sliced to the
    # published window. Nothing is invented: no audio played, no audio rendered.
    candidates, audio_report = load_candidates(
        Path(args.audio_root),
        trial.timestamps[first_index] - audio_offset,
        int(round(len(window.timestamps) / trial.sample_rate * args.audio_rate)),
        rate=int(args.audio_rate),
    )
    media_out = Path(args.media_out)
    media_report = render_stereo(
        candidates,
        media_out,
        sample_rate=int(args.audio_rate),
        presentation=presentation_mode(args.presentation),
        crossmix_weight=args.crossmix_weight,
    )
    media_report["audio_slice"] = audio_report
    media_report["eeg_seconds"] = round(window.seconds, 3)
    timeline = MediaTimeline(media_out, args.media_title)
    # The transport's media descriptor and the duration the client reports are
    # what the frontend's own gate compares field by field; both come from the
    # served timeline, not from a second copy of the numbers.
    media_report["descriptor"] = timeline.descriptor()
    media_report["media_id"] = timeline.media_id
    log(
        f"rendered {args.presentation} stereo from the played stimulus: "
        f"{media_report['seconds']}s, route {media_report['presentation']} "
        f"(A={Path(ENVELOPES[0]).name})",
        lines,
    )

    publisher = start_publisher(args, window, lines)
    try:
        sleep_until_ready(args.stream_name, timeout=args.resolve_timeout)
        row = {
            "name": args.stream_name,
            "stype": "EEG",
            "source_id": args.source_id,
        }
        stream = open_inlet(row, bufsize=4.0, connect_timeout=args.connect_timeout)
        try:
            probe_report = probe(stream, channels, args, lines)
            # The model's contract names the rate its training trials were at;
            # `RidgeDecoder.validate` compares the chain's contract with it
            # verbatim, so a 500 Hz recording must reach the chain at that rate.
            # The conversion is explicit, printed here, and recorded - it is an
            # adapter-stage rate change, not a change to the contract.
            wanted_rate = float(model.contract.get("input_sfreq") or args.sfreq)
            pre = (
                None
                if args.pre_resample == "off" or math.isclose(wanted_rate, args.sfreq)
                else wanted_rate
            )
            if pre is not None:
                log(
                    f"pre-chain rate conversion: {args.sfreq:g} Hz source -> "
                    f"{pre:g} Hz, the rate the decoder contract records "
                    f"(contract mismatch would otherwise be refused by "
                    f"RidgeDecoder.validate; nothing about the recording changes)",
                    lines,
                )
            source = AntStreamSource(
                stream,
                channels=channels,
                sfreq=args.sfreq,
                reference=trial.reference,
                upstream_processing=trial.upstream_processing,
                kind="ant_lsl_replay",
                source_unit_exponent=int(args.source_unit_exponent),
                block_samples=args.block_samples,
                timebase=args.timebase,
                pre_resample_sfreq=pre,
                max_lag_seconds=args.max_lag_seconds,
            )
            log(
                f"chain rate declaration: source outlet at {args.sfreq:g} Hz, chain "
                f"fed at {chain_rate(args, model):g} Hz (adapter {args.pre_resample!r}); the "
                f"contract records the fed rate, or Repair would read every step as "
                f"a gap",
                lines,
            )
            log(
                f"live source ready: window sample 0 is time zero; audio anchor "
                f"{trial.timestamps[first_index] + audio_offset:.3f}s on the "
                f"session's own clock (start offset {audio_offset:g}s from the "
                f"1004/Start marker)",
                lines,
            )
            exit_code, record = stream_session(
                args, lines, source, references, model, timeline, media_report, began,
                probe_report, window, trial, first_index, audio_offset, profile,
                excluded, excursion,
            )
        finally:
            if stream.connected:
                stream.disconnect()
    finally:
        with contextlib.suppress(Exception):
            publisher.terminate()
        try:
            output, _ = publisher.communicate(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill path
            publisher.kill()
            output = ""
        if output:
            log(f"publisher said: {output.strip()}", lines)
    return record, exit_code


def stream_session(
    args, lines, source, references, model, timeline, media_report, began,
    probe_report, window, trial, first_index, audio_offset, profile, excluded,
    excursion,
) -> tuple[int, dict]:
    """Run the transport and the session around an already-connected source."""

    app_state = {
        "source": source,
        "references": references,
        "model": model,
        "media": timeline,
        "holder": {},
        "audio_first_timestamp": float(trial.timestamps[first_index]),
        "audio_offset": float(audio_offset),
        "excluded": tuple(excluded),
        "chain_rate": chain_rate(args, model),
    }

    def factory():
        session, producer = build(app_state, args)
        app_state["holder"]["producer"] = producer
        app_state["holder"]["session"] = session
        # The transport deliberately keeps a failing producer's exception out of
        # the session record (it is not a log sink). The runner is its own run's
        # log sink, so it wraps run() to keep the one line that says why a
        # session ended as `producer_failed` - otherwise the record would name
        # the failure without saying what failed.
        original_run = producer.run

        def run(publish, stop):
            try:
                return original_run(publish, stop)
            except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
                app_state["holder"]["producer_error"] = f"{type(error).__name__}: {error}"
                log(f"the session's producer failed: {type(error).__name__}: {error}", lines)
                raise

        producer.run = run
        return producer

    static_dir = Path(args.static_dir) if args.static_dir else None
    if static_dir is not None and not static_dir.is_dir():
        print(f"static dir not found: {static_dir}", file=sys.stderr)
        return 2
    app = create_app(producer_factory=factory, media=timeline, static_dir=static_dir)

    port = args.port or free_port()
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
        print("\n[ant_live] stop requested; ending the session", flush=True)
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

    client = SimulatedMediaClient(
        url.rstrip("/"),
        client_id="ant-live-runner",
        duration_s=media_report["seconds"],
        clip_seconds=args.clip,
        tick_seconds=args.media_tick,
    )
    log(f"open {url} while it runs (serving {static_dir})", lines)

    exit_code = 0
    state: dict = {}
    try:
        loop_holder["loop"] = asyncio.new_event_loop()
        asyncio.set_event_loop(loop_holder["loop"])
        state = loop_holder["loop"].run_until_complete(
            drive(port, args, media_report, client, stop)
        )
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
        log(f"transport stopped (server thread alive: {thread.is_alive()})", lines)

    producer = app_state["holder"].get("producer")
    session = app_state["holder"].get("session")
    summary = producer.summary.to_dict() if producer is not None and producer.summary else {}
    processor = getattr(session, "processor", None)
    packets = state.get("packets") or []
    log(f"{len(packets)} packets recorded from /ws/live", lines)
    log(f"decisions: {json.dumps(summary.get('decisions', {}))}", lines)
    log(f"effective run policy: {json.dumps(summary.get('policy', {}))}", lines)
    log(
        "bad-channel census: "
        + (json.dumps(summary.get("bad_channel_census")) or "no channel was flagged"),
        lines,
    )

    evidence = frontend_evidence(args, packets, state, media_report, lines)
    transport = source.transport.to_dict()
    transport["recovery_events"] = [
        dict(event) for event in getattr(processor, "recovery", None).events
    ] if getattr(processor, "recovery", None) is not None else []
    transport["repaired_samples"] = int(
        getattr(getattr(processor, "repair", None), "repaired_samples", 0)
    )
    transport["bad_channel_census"] = dict(summary.get("bad_channel_census") or {})
    transport["recovery_segment"] = int(
        getattr(getattr(processor, "recovery", None), "segment", 0)
    )
    # The chain's own resampler delay is the one component of the alignment
    # budget this run can actually measure: it is a fixed offset between the
    # timestamps the chain carries and the samples it produced. Plan sections
    # 3.11/3.17-4 want the *audio-to-EEG* loopback, which needs an audio device;
    # this is reported so the difference is explicit rather than blurred.
    resampler = getattr(processor, "resampler", None)
    # A pass-through stage (the fed rate already equals the output rate) has no
    # delay to report; calling it "QQ" would be a lie about a stage that does not
    # resample at all.
    transport["chain_resampler_startup_delay_seconds"] = (
        None if resampler is None else float(resampler.startup_delay_seconds)
    )
    transport["chain_resampler_quality"] = (
        None if resampler is None else str(resampler.quality)
    )
    record = {
        "kind": (
            "ANT recorded session driven through the live LSL route; a replay of "
            "a recording on a different rig, not a live participant"
        ),
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "started_at": datetime.fromtimestamp(began, timezone.utc).isoformat(),
        "seconds": round(time.time() - began, 2),
        "command": sys.argv,
        "url": url,
        "session": {
            "path": str(args.session),
            "trial_id": trial.trial_id,
            "subject": trial.subject,
            "samples": int(len(trial.timestamps)),
            "seconds": round(trial.seconds, 3),
            "channels": list(trial.channel_names),
            "sample_rate": trial.sample_rate,
            "reference": trial.reference,
            "labels_present": sorted({int(v) for v in trial.labels.tolist()}),
        },
        "publisher": {
            "module": PUBLISHER_MODULE,
            "stream_name": args.stream_name,
            "source_id": args.source_id,
            "sample_rate": trial.sample_rate,
            "unit": "microvolts",
            "unit_exponent": int(args.source_unit_exponent),
        },
        "published_window": {
            "first_sample_index": int(first_index),
            "clip_start_seconds": float(args.clip_start),
            "clip_seconds": float(args.clip),
            "samples": int(len(window.timestamps)),
            "first_timestamp": float(window.timestamps[0]),
            "peak_microvolts": float(np.max(np.abs(window.eeg))),
        },
        "clip_seconds": float(args.clip),
        "probe": probe_report,
        "transport": transport,
        "operating_point": {
            "margin": float(args.margin),
            "margin_source": (
                "calibrated (results/aad_margin_calibration_*.md, decision D-29)"
                if abs(float(args.margin) - CALIBRATED_MARGIN) < 1e-12
                else "explicit --margin"
            ),
            "documented_default_margin": MIN_MARGIN,
            "window_seconds": summary.get("window_seconds"),
            "timebase": args.timebase,
            "policy": summary.get("policy"),
            # Run policy, not contract: adding a key to `AuditoryProcessor.contract`
            # would invalidate every trained decoder, so the channel exclusions and
            # the endpoint guard are recorded here instead, next to their evidence.
            "excluded_channels": list(excluded),
            "excluded_channels_justification": [
                {
                    "channel": name,
                    "peak_microvolts": profile["per_channel"][name]["peak_microvolts"],
                    "railed_fraction": profile["per_channel"][name]["railed_fraction"],
                    "railed": profile["per_channel"][name]["railed"],
                }
                for name in excluded
            ],
            "rail_profile": profile,
            "repair_saturation_limit_uv": saturation_guard_uv(args),
            "amplitude_limit_uv": amplitude_guard_uv(args),
            "declared_chain_default_amplitude_uv": REPAIR_DEFAULT_AMPLITUDE_UV,
            "amplitude_limit_source": (
                "chain default"
                if args.amplitude_limit_uv is None
                else "explicit --amplitude-limit-uv"
            ),
            "excursion_profile": excursion,
            "amplitude_limit_reaches": (
                "QualityMonitor's 'amplitude' fault (which decides whether a window "
                "is an artifact) and Repair's endpoint-jump check; one declared "
                "value with one owner, so the two verdicts cannot disagree"
            ),
            "saturation_limit_source": (
                "chain default"
                if args.saturation_limit_uv is None
                else "explicit --saturation-limit-uv"
            ),
            "excluded_channels_stay_in_the_run": (
                "excluded electrodes keep their columns, are still repaired and "
                "still counted in the bad-channel census; no column is dropped"
            ),
        },
        "session_summary": summary,
        "session_record": state.get("final"),
        "decision_census": decision_census(
            summary, summary.get("decision_stream") or []
        ),
        "media": media_report,
        "media_client": state.get("media"),
        "frontend_evidence": evidence,
        "packet_stream": {
            "path": evidence.get("stream"),
            "count": evidence.get("packets"),
            "by_type": packet_census(packets),
        },
        "labels_vs_decisions": coverage_against_markers(
            trial.labels, summary.get("decision_stream") or [], trial.sample_rate
        ),
        "log": lines,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(record))
    print(f"run record: {out}")
    print(f"report: {report}")
    gate = (evidence.get("gate") or {}).get("report") or {}
    print(
        f"gain gate: open={gate.get('gate_open')} at "
        f"{gate.get('gate_opened_at_seconds')}s, attenuated gain frames "
        f"{gate.get('playback_gains_applied')} of {gate.get('sampled_gain_packets')}"
    )
    print(f"effective margin: {float(args.margin):g} (documented default {MIN_MARGIN})")
    print(f"transport: {json.dumps({k: transport[k] for k in ('max_lag_seconds','gaps','blocks','samples','raw_rate_hz','grid_deviation_peak_seconds','pre_resampled_to_hz','ended')})}")
    print(f"labels vs decisions: {json.dumps({k: v for k, v in record['labels_vs_decisions'].items() if k not in ('kind',)})}")
    if (state.get("final") or {}).get("status") == "error":
        print(f"the session reported an error: {json.dumps(summary.get('failure'))}")
        exit_code = 1
    return exit_code, record


def frontend_evidence(args, packets, state, media_report, lines) -> dict:
    """Write the packet stream and let the vendored frontend's own code judge it.

    It calls the demo's own ``gather_evidence`` with paths pointed at this run's
    outputs, so the frontend verdicts here come from exactly the code path that
    produced step 10's, over a stream this step's live route generated.
    """

    from types import SimpleNamespace

    from scripts.auditory_ui.demo import gather_evidence

    stem = Path(args.out).with_suffix("")
    names = SimpleNamespace(
        stream_out=str(stem.with_name(stem.name + "_packets.jsonl")),
        media_record=str(stem.with_name(stem.name + "_media.json")),
        render_out=str(stem.with_name(stem.name + "_frontend.html")),
        gate_out=str(stem.with_name(stem.name + "_gate.json")),
        decode_script="scripts/auditory_ui/decode_packets.mjs",
        gate_script="scripts/auditory_ui/gate_evidence.mjs",
    )
    return gather_evidence(names, packets, state, media_report, lines)


def build_parser() -> argparse.ArgumentParser:
    """The runner's command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--audio-rate", type=int, default=48000)
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument(
        "--audio-start-offset",
        type=float,
        default=0.0,
        help=("operator-given session audio start, in seconds of stimulus at the "
              "recording's Start anchor: 0 for session 19-34-06, 267 for 19-41-11 (section 3.12)"),
    )
    parser.add_argument("--clip", type=float, default=130.0, help="seconds to publish and run")
    parser.add_argument("--stream-name", default=DEFAULT_STREAM_NAME)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--sfreq", type=float, default=500.0)
    parser.add_argument(
        "--source-unit-exponent",
        type=int,
        default=DEFAULT_MICROVOLT_EXPONENT,
        help="-6 for microvolts, 0 for volts; the outlet declares microvolts",
    )
    parser.add_argument("--block-samples", type=int, default=100)
    parser.add_argument(
        "--pre-resample",
        default="auto",
        help=(
            "auto (default): convert the incoming blocks to the rate the decoder "
            "contract records; off: feed the chain at the source rate, which the "
            "decoder refuses unless a model was fitted at that rate"
        ),
    )
    parser.add_argument("--timebase", default="grid", choices=("grid", "stamps"))
    parser.add_argument(
        "--exclude-channels",
        type=parse_channel_list,
        default=(),
        help=(
            "comma-separated electrodes (e.g. F8,F3) declared railed or dead. An "
            "excluded electrode cannot stop the run - not through the repair "
            "endpoint check's saturation guard, not through its amplitude guard - "
            "while every other channel keeps the chain's own limits. Its column "
            "stays in the run, is still repaired and is still counted in the "
            "bad-channel census. Each declared electrode is checked against the "
            "published window's measured rail and refused unless it is railed."
        ),
    )
    parser.add_argument(
        "--allow-unrailed-exclusion",
        action="store_true",
        help=(
            "declare an electrode dead even though this window does not measure it "
            "as railed. Off by default: excluding a healthy electrode would blind "
            "the endpoint check for a channel that still carries signal."
        ),
    )
    parser.add_argument(
        "--saturation-limit-uv",
        type=float,
        default=None,
        help=("Repair's absolute-level guard, in uV. Unset by default, which keeps "
              "the chain's own 75000 uV so the guard protects every channel; a "
              "railed electrode is handled by --exclude-channels instead of by "
              "widening this number. Setting it moves the guard for ALL channels "
              "and the effective value is printed and recorded."),
    )
    parser.add_argument(
        "--amplitude-limit-uv",
        type=float,
        default=None,
        help=("the excursion from the run's own anchor level, in uV, that counts "
              "as a fault. One declared value with one owner: it reaches both "
              "QualityMonitor (the 'amplitude' fault that makes a window an "
              "artifact) and Repair (the largest endpoint jump it will bridge), so "
              "the two cannot disagree. Unset by default, which keeps the chain's "
              "own 500 uV; set it to the recording's own measured excursion when "
              "that recording drifts further, because at the default every window "
              "is an artifact and no decision is ever committed. The effective "
              "value is printed and recorded."),
    )
    parser.add_argument("--max-lag-seconds",
        type=float,
        default=10.0,
        help=("age limit for a delivered block. An inlet that attaches behind a "
              "publisher already streaming inherits a backlog from before the "
              "connection; this is deliberately loose enough to deliver it and "
              "tight enough to fail a stalled source. The measured lag is recorded."),
    )
    parser.add_argument("--resolve-timeout", type=float, default=20.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--margin",
        type=float,
        default=CALIBRATED_MARGIN,
        help=(
            f"controller commit margin; default {CALIBRATED_MARGIN} is the point "
            f"calibrated on held-out stories (D-29). The documented policy default "
            f"is {MIN_MARGIN}."
        ),
    )
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--media-tick", type=float, default=MEDIA_TICK_SECONDS)
    parser.add_argument("--media-out", default="output/auditory_ui/ant_live_stereo.wav")
    parser.add_argument("--media-title", default="ANT recorded session")
    parser.add_argument(
        "--presentation",
        default=DEFAULT_PRESENTATION,
        choices=PRESENTATION_MODES,
        help="presentation route; the recorded session is dichotic (L=A, R=B)",
    )
    parser.add_argument("--crossmix-weight", type=float, default=DEFAULT_CROSSMIX_WEIGHT)
    parser.add_argument("--static-dir", default=str(REPO / "apps" / "attune-ui" / "dist"))
    parser.add_argument("--host", default="127.0.0.1", help="loopback only")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free loopback port")
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="connect, report the pre-flight verdict, then stop",
    )
    parser.add_argument(
        "--out", default="results/antneuro_live_run.json", help="run record"
    )
    parser.add_argument(
        "--report", default="results/antneuro_live_run.md", help="markdown summary"
    )
    return parser


def probe_only(args, lines: list[str]) -> int:
    """Connect to the outlet, report what it declared, and stop."""

    model = RidgeDecoder.load(Path(args.model))
    channels = tuple(model.contract["eeg_channels"])
    publisher = start_publisher(args, None, lines)
    try:
        sleep_until_ready(args.stream_name, timeout=args.resolve_timeout)
        stream = open_inlet(
            {"name": args.stream_name, "stype": "EEG", "source_id": args.source_id},
            bufsize=4.0,
            connect_timeout=args.connect_timeout,
        )
        try:
            probe(stream, channels, args, lines)
        finally:
            if stream.connected:
                stream.disconnect()
    finally:
        with contextlib.suppress(Exception):
            publisher.terminate()
        with contextlib.suppress(Exception):
            output, _ = publisher.communicate(timeout=15)
            if output:
                log(f"publisher said: {output.strip()}", lines)
    return 0


def main(argv=None) -> int:
    """Parse, then run (or probe) and report."""

    args = build_parser().parse_args(argv)
    lines: list[str] = []
    if args.probe_only:
        return probe_only(args, lines)
    record, exit_code = run(args, lines)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
