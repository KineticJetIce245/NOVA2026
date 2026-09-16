"""Command 1 of the two-command live demo: make the amplifier's EEG reachable.

    .venv/Scripts/python.exe -B -m scripts.getlive.live_source

This is the acquisition side of ``documents/live_demo_two_commands.md``. It
answers one question and then gets out of the way: **is the eego publishing the
20 electrodes the live model was fitted on, at the rate and in the units this
run will assert?**

Three modes, and the default is the one a real session uses:

``--mode direct`` (default)
    Resolve the outlet, connect, read what it declares, and run the package's
    own pre-flight (:func:`nova2026.streaming.preflight.validate_source`)
    against the 20 electrodes **by name** at ``--sfreq`` and ``--source-units``.
    Then print the exact flags command 2 needs and exit ``0``. Nothing is
    acquired, no thread is started, no sample is consumed: connecting is
    metadata-only. This is the "the amplifier is publishing, and it publishes
    the right thing" gate.

``--mode bridge``
    The same resolution and the same check, and then it **republishes** exactly
    those 20 electrodes - in the model's own order, typed ``eeg``, with the
    declared unit - as a new outlet named ``--out-name`` (default ``NOVA_Live``).
    Use it when the eego declares positional labels, no units, an EOG lead
    classified as EEG, or extra electrodes: the package refuses such a source on
    purpose, and ``scripts/getlive/relay.py``'s docstring says why. It reuses
    that module's outlet construction and the library's ``TimeBase``; the only
    thing added here is **selection by electrode name** rather than by column
    index, which is what a cap montage needs. It runs until Ctrl+C.

``--mode replay``
    No amplifier: publish a recorded ANT session through the *same* outlet
    identity, using ``scripts.getlive.ant_publish`` as its own subprocess, so
    command 2 can be exercised end to end on this machine. This is the bench
    rehearsal, and it is what the automated tests use the shape of.

Everything it prints about the amplifier is a **declaration**, not a
measurement: LSL does not advertise gain, the amplifier's own filters, or the
impedance of a single electrode. ``scripts/getlive/README.md`` and
``documents/live_demo_two_commands.md`` both say so; the run records the strings
this command asserted so a later reader can tell assertion from measurement.

Exit codes: ``0`` the amplifier is publishing what the run needs, ``2`` refused
(nothing on the network, or a declared contract that does not match), ``1`` a
runtime error, ``130`` Ctrl+C out of ``--mode bridge``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/getlive/live_source.py`
    sys.path.insert(0, str(REPO))

from scripts.getlive.outlets import (  # noqa: E402
    channel_facts,
    describe_rows,
    find_outlets,
    open_inlet,
    select_outlet,
)

DEFAULT_MODEL = "models/auditory_kuleuven_live20.npz"
"""The model whose electrode list defines the contract. One source of truth."""

DEFAULT_SFREQ = 500.0
"""The rate this rig measured at (plan section 3.11): 500 Hz, 24 EEG electrodes."""

DEFAULT_EXPECTED_CHANNELS = 24
"""The cap on the rig. Used only to recognise the outlet before connecting."""

DEFAULT_STREAM_TYPE = "EEG"
DEFAULT_OUT_NAME = "NOVA_Live"
DEFAULT_OUT_SOURCE_ID = "nova-live-source"

UNIT_EXPONENTS = {"V": 0, "mV": -3, "uV": -6, "nV": -9}
"""The package's own unit convention (``preflight._UNIT_ALIASES``), by name."""

REFUSAL = 2
"""Exit code for "the amplifier is not there / is not what was asserted"."""


def unit_exponent(text: str) -> int:
    """Turn ``uV``/``V``/``mV``/``nV`` into the package's power of ten.

    The chain's convention is microvolts (plan section 3.11 item 6): a source
    already in uV gets exponent ``-6``, which makes ``_to_uv = 1.0``. The value
    is asserted by the operator and checked against what the outlet declares -
    it is never inferred from the data, because inferring it from amplitudes is
    exactly how a volts/microvolts mix-up passes unnoticed.
    """

    key = str(text).strip()
    for name, exponent in UNIT_EXPONENTS.items():
        if key.lower() == name.lower():
            return exponent
    raise ValueError(
        f"unknown unit {text!r}; use one of {', '.join(sorted(UNIT_EXPONENTS))}"
    )


def model_electrodes(model_path: str | Path) -> tuple[str, ...]:
    """The 20 electrode names the fitted model's contract requires, in order.

    Read from the model rather than typed into a command line: the two commands
    then agree on the contract by construction, and a model swap cannot leave a
    stale electrode list behind.
    """

    from nova2026.auditory.decoder import RidgeDecoder

    contract = RidgeDecoder.load(Path(model_path)).contract
    names = tuple(str(name) for name in contract.get("eeg_channels") or ())
    if not names:
        raise RuntimeError(f"{model_path} records no eeg_channels in its contract.")
    return names


def parse_electrodes(text: str | None, model_path: str | Path) -> tuple[str, ...]:
    """``--electrodes`` as a tuple, defaulting to the model's own list."""

    if text is None or not text.strip():
        return model_electrodes(model_path)
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise ValueError("--electrodes named no electrode.")
    if len(set(names)) != len(names):
        raise ValueError("--electrodes names the same electrode twice.")
    return names


def select_columns(
    declared: tuple[str, ...], wanted: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Source columns carrying ``wanted``, **by name**, plus what is missing.

    Selection is by name and never by position: the amplifier's column order is
    a declaration that has not been confirmed against this repository's offline
    records (``scripts/getlive/README.md``, "What is still unverified"), and a
    positional pick would silently decode the wrong electrodes. Extra columns the
    outlet publishes are dropped, never truncated by index (plan case E4).
    """

    index = {name: position for position, name in enumerate(declared)}
    missing = tuple(name for name in wanted if name not in index)
    columns = tuple(index[name] for name in wanted if name in index)
    return columns, missing


def plan_lines(facts: dict, electrodes: tuple[str, ...], args, stream_hint: str) -> list[str]:
    """What command 2 must be told, printed so it can be copied verbatim."""

    python = ".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python"
    flags = [f"--stream-name {facts['name']!r}"]
    if facts["source_id"]:
        flags.append(f"--source-id {facts['source_id']!r}")
    flags.append(f"--sfreq {float(args.sfreq):g}")
    flags.append(f"--source-units {args.source_units}")
    return [
        f"command 2 ({len(electrodes)} electrodes by name: "
        f"{', '.join(electrodes)}):",
        f"  {python} -B -m scripts.auditory_ui.live " + " ".join(flags),
        f"  ({stream_hint})",
    ]


def resolve_and_validate(args, lines: list[str]):
    """Resolve one EEG-like outlet and prove it satisfies this run's contract.

    Returns ``(stream, facts, row)``. Raises ``RuntimeError`` with a sentence an
    operator can act on - never a hang: resolution is bounded by
    ``--resolve-seconds``, and connecting is bounded by ``--connect-seconds``.
    """

    from nova2026.streaming.preflight import validate_source

    rows = find_outlets(timeout=float(args.resolve_seconds), interval=1.0)
    if not rows:
        raise RuntimeError(
            f"No LSL outlet appeared within {args.resolve_seconds:g}s. Check that "
            "the eego control software has 'Application options -> Network "
            "Operation -> Enable LSL EEG streaming' ticked and that the amplifier "
            "is connected, that this host is on the amplifier's network (LSL is "
            "multicast, and a firewall that blocks multicast blocks LSL), then run "
            "scripts.getlive.probe. Nothing was acquired."
        )
    try:
        row = select_outlet(
            rows,
            name=args.stream_name,
            source_id=args.source_id,
            stream_type=args.stream_type,
            expected_channels=None if args.stream_type else args.expected_channels,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}\nOutlets seen:\n{describe_rows(rows)}"
        ) from None

    print(f"resolved  : name={row['name']!r} type={row['stype']!r} "
          f"channels={row['n_channels']} sfreq={row['sfreq']:g} Hz "
          f"source_id={row['source_id']!r} host={row['hostname']!r}", flush=True)
    lines.append(f"resolved outlet {row['name']!r} ({row['n_channels']} channels)")

    stream = open_inlet(
        row,
        bufsize=float(args.bufsize),
        connect_timeout=float(args.connect_seconds),
    )
    facts = channel_facts(stream)
    print(
        f"declared  : {facts['n_channels']} channels, {facts['sfreq']:g} Hz, units "
        f"{sorted({str(unit) for unit in (facts['units'] or ())})}, types "
        f"{sorted(set(facts['types']))}, labels {list(facts['channels'])}",
        flush=True,
    )
    lines.append(
        f"declared {facts['n_channels']} channels at {facts['sfreq']:g} Hz, units "
        f"{sorted({str(unit) for unit in (facts['units'] or ())})}"
    )

    columns, missing = select_columns(facts["channels"], tuple(args.electrodes))
    if missing:
        raise RuntimeError(
            f"The outlet does not publish {len(missing)} of the "
            f"{len(args.electrodes)} electrodes the model contracted: "
            f"{', '.join(missing)}. It publishes {list(facts['channels'])}. Either "
            "the cap is not mounted / not connected, or the control software is "
            "streaming a different montage; compare with scripts.getlive.probe. "
            "Nothing was acquired."
        )
    print(f"selected  : columns {list(columns)} -> {list(args.electrodes)}", flush=True)

    # The package's own pre-flight, over the selected electrodes only. It checks
    # the declared types of exactly these columns, so an EOG lead elsewhere in
    # the montage cannot make an otherwise-correct 20-electrode run fail.
    validate_source(
        stream,
        sfreq=float(args.sfreq),
        channels=tuple(args.electrodes),
        source_unit_exponent=unit_exponent(args.source_units),
    )
    print(
        f"pre-flight: accepted {len(args.electrodes)} electrodes by name at "
        f"{float(args.sfreq):g} Hz, units {args.source_units} "
        f"(exponent {unit_exponent(args.source_units)})",
        flush=True,
    )
    lines.append("pre-flight accepted the outlet")

    # A counter-check, so "accepted" is visibly a check and not a formality: the
    # same stream is refused when it is asserted to be volts instead of the unit
    # that was just validated. Nothing is loosened to make either run pass.
    try:
        validate_source(
            stream,
            sfreq=float(args.sfreq),
            channels=tuple(args.electrodes),
            source_unit_exponent=0 if unit_exponent(args.source_units) != 0 else -6,
        )
    except RuntimeError as error:
        print(f"counter   : the same stream with the wrong unit was refused - "
              f"{error}", flush=True)
        lines.append("counter-check refused as expected")
    else:  # pragma: no cover - a counter-check that passes is itself a finding
        print("counter   : FINDING - the wrong unit declaration was NOT refused",
              flush=True)
        lines.append("FINDING: counter-check did not refuse")
    return stream, facts, row


def bridge(args, stream, lines: list[str]) -> int:
    """Republish the selected electrodes with declared metadata until Ctrl+C.

    The forwarding loop is deliberately the same shape as ``relay.py``'s - pull
    everything the source has, push it on with its own timestamps, optionally
    regrid through the library's ``TimeBase`` - with name-based column selection
    and contract order instead of ``--keep`` column indices.
    """

    from mne_lsl.lsl import StreamInlet, StreamOutlet

    from nova2026.streaming import GridPolicy, TimeBase
    from scripts.getlive.relay import REGRID_SOURCE_ID_SUFFIX, build_outlet_info

    info = stream.sinfo
    inlet = StreamInlet(info, max_buffered=10, processing_flags=["clocksync"])
    inlet.open_stream(timeout=float(args.connect_seconds))
    columns = np.asarray(
        select_columns(tuple(stream.ch_names), tuple(args.electrodes))[0], dtype=int
    )
    time_base = None
    try:
        sinfo = build_outlet_info(args, tuple(args.electrodes), float(args.sfreq), inlet.dtype)
        outlet = StreamOutlet(sinfo, chunk_size=int(args.chunk))
        print(
            f"publishing: name={args.out_name!r} type={args.out_type!r} "
            f"channels={len(args.electrodes)} sfreq={float(args.sfreq):g} Hz "
            f"units={args.units} source_id={sinfo.source_id!r}",
            flush=True,
        )
        if args.regrid:
            policy = GridPolicy.for_rate(float(args.sfreq))
            time_base = TimeBase(policy)
            print(
                f"regrid    : a counted grid at {float(args.sfreq):g} Hz from the "
                "library's time base; the source's stamps are read only to measure "
                "how far they drifted from that count",
                flush=True,
            )
        print("Ctrl+C to stop. Command 2 points at this outlet.\n", flush=True)
        forwarded = 0
        since = 0
        while True:
            data, stamps = inlet.pull_chunk(timeout=1.0)
            if stamps.size == 0:
                time.sleep(0.002)
                continue
            data = np.ascontiguousarray(np.asarray(data)[:, columns])
            stamps = np.asarray(stamps, dtype=float)
            if time_base is not None:
                stamps, _ = time_base.place(stamps)
            outlet.push_chunk(data, stamps)
            forwarded += int(stamps.size)
            since += int(stamps.size)
            if since >= int(float(args.sfreq)) * 5:
                line = f"forwarded {forwarded} samples"
                if time_base is not None:
                    state = time_base.state
                    line += (
                        f" ({len(state.relocks)} re-lock(s), "
                        f"{state.relocked_samples:.1f} sample(s) of drift absorbed)"
                    )
                print(line, flush=True)
                since = 0
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
        return 130
    finally:
        inlet.close_stream()
    return 0


def replay(args, lines: list[str]) -> int:
    """Publish a recorded ANT session instead of the amplifier, for rehearsal.

    It runs ``scripts.getlive.ant_publish`` as its own process - the module that
    already does this - rather than growing a second publisher here.
    """

    import subprocess

    from scripts.getlive.ant_publish import PUBLISHER_MODULE

    command = [
        sys.executable, "-B", "-m", PUBLISHER_MODULE,
        "--session", str(args.session),
        "--start", str(args.replay_start),
        "--seconds", str(args.replay_seconds),
        "--name", args.out_name,
        "--source-id", args.out_source_id,
    ]
    print("bench replay (no amplifier): " + " ".join(command), flush=True)
    print(
        "This publishes a RECORDING. It proves the two commands compose; it "
        "proves nothing about the amplifier, and no number from it may be "
        "quoted as a live result.",
        flush=True,
    )
    lines.append("bench replay mode: recorded ANT session, no amplifier")
    return subprocess.call(command, cwd=str(REPO))


def build_parser() -> argparse.ArgumentParser:
    """Build command 1's command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "bridge", "replay"),
        default="direct",
        help="direct (default): verify the amplifier and print command 2's flags; "
        "bridge: republish the 20 electrodes under --out-name; replay: publish a "
        "recorded session, for rehearsal without an amplifier",
    )
    source = parser.add_argument_group("amplifier outlet")
    source.add_argument("--stream-name", default=None,
                        help="LSL name to pin; default: the single EEG-like outlet")
    source.add_argument("--source-id", default=None, help="LSL source id to pin")
    source.add_argument("--stream-type", default=DEFAULT_STREAM_TYPE,
                        help="LSL stream type to pin; empty string disables the filter")
    source.add_argument("--expected-channels", type=int, default=DEFAULT_EXPECTED_CHANNELS,
                        help="channel count that makes an untyped outlet look like the cap")
    source.add_argument("--resolve-seconds", type=float, default=10.0,
                        help="how long to look for the outlet before refusing")
    source.add_argument("--connect-seconds", type=float, default=5.0)
    source.add_argument("--bufsize", type=float, default=30.0,
                        help="inlet buffer, seconds")

    contract = parser.add_argument_group("what the run asserts about the amplifier")
    contract.add_argument("--model", default=DEFAULT_MODEL,
                          help="model whose eeg_channels define the electrode list")
    contract.add_argument("--electrodes", default=None,
                          help="comma-separated labels; default: the model's own list")
    contract.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ,
                          help="declared rate in Hz; the rig measured 500 (section 3.11)")
    contract.add_argument("--source-units", default="uV", choices=sorted(UNIT_EXPONENTS),
                          help="unit the outlet carries, as the package names it")

    out = parser.add_argument_group("republished outlet (--mode bridge)")
    out.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    out.add_argument("--out-type", default="EEG")
    out.add_argument("--out-source-id", default=DEFAULT_OUT_SOURCE_ID)
    out.add_argument("--units", default="microvolts", choices=("volts", "millivolts",
                                                               "microvolts", "nanovolts"),
                     help="unit declared on the republished outlet (never applied)")
    out.add_argument("--types", default="eeg")
    out.add_argument("--chunk", type=int, default=32)
    out.add_argument("--regrid", action="store_true",
                     help="rebuild the timeline on a counted grid instead of "
                          "forwarding the source's stamps")
    out.add_argument("--quiet", action="store_true")

    bench = parser.add_argument_group("bench replay (--mode replay)")
    bench.add_argument("--session", default="datasets/AAD-ANT/session_19-34-06.npz")
    bench.add_argument("--replay-start", type=float, default=0.0)
    bench.add_argument("--replay-seconds", type=float, default=600.0)
    return parser


def main(argv=None) -> int:
    """Resolve, verify and report; exit non-zero rather than hang."""

    args = build_parser().parse_args(argv)
    lines: list[str] = []
    started = time.time()
    if not args.stream_type:
        args.stream_type = None
    try:
        args.electrodes = parse_electrodes(args.electrodes, args.model)
        unit_exponent(args.source_units)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return REFUSAL
    if not math.isfinite(args.sfreq) or args.sfreq <= 0:
        print("error: --sfreq must be finite and positive", file=sys.stderr)
        return REFUSAL

    print(
        f"live_source: mode={args.mode}, {len(args.electrodes)} electrodes by name, "
        f"{args.sfreq:g} Hz, units {args.source_units} "
        f"(exponent {unit_exponent(args.source_units)})",
        flush=True,
    )
    if args.mode == "replay":
        return replay(args, lines)

    stream = None
    try:
        stream, facts, _row = resolve_and_validate(args, lines)
    except RuntimeError as error:
        print(f"\nREFUSED: {error}", file=sys.stderr, flush=True)
        if stream is not None and getattr(stream, "connected", False):
            stream.disconnect()
        return REFUSAL
    except (ValueError, OSError) as error:
        print(f"\nREFUSED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return REFUSAL

    try:
        if args.mode == "bridge":
            return bridge(args, stream, lines)
        print("\nThe amplifier is publishing what this run needs. Nothing was "
              "acquired by this command.\n", flush=True)
        for line in plan_lines(facts, tuple(args.electrodes), args,
                               f"took {time.time() - started:.1f}s"):
            print(line, flush=True)
        return 0
    except (ValueError, OSError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if getattr(stream, "connected", False):
            stream.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
