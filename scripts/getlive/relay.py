"""Republish an LSL outlet with the channel metadata the package requires.

Some control software publishes EEG over LSL without a channel description
that MNE-LSL can read: the inlet then falls back to positional labels
(``0, 1, 2 ...``) and reports ``ch_units`` as *not declared*. The package's
pre-flight check (:func:`nova2026.streaming.preflight.validate_source`) refuses
such a source on purpose, because a stream whose units are unknown cannot be
scaled to uV and a stream whose labels are positional cannot be contracted to
a montage.

This relay is the bridge. It subscribes to the outlet as it is, keeps the
columns that carry electrodes, and republishes them under a new name with
proper labels, types and units. By default nothing is filtered, resampled or
rescaled: the samples and their timestamps are forwarded untouched, so the relay
adds metadata and latency only.

``--regrid`` drops that last guarantee on purpose, for the other half of the
problem: a source whose timestamps cannot be trusted - a recorder that stamps a
whole block once (many samples sharing one timestamp, the shape
:mod:`scripts.getlive.ts_check` diagnoses as chunk-stamped), or one that jitters a
few tenths of a sample below the grid - delivers a timeline ``Repair`` can only
refuse, and no tolerance helps because the unusable steps sit below one sample.

Regridding hands that job to the library's
:class:`nova2026.streaming.TimeBase`: every sample received gets one slot on a
counted grid at the declared rate, the source's stamps are read only to measure how
far they have drifted from that count, and the drift is absorbed by re-locking and
reported. Nothing is invented and nothing is held back, so unlike the
block-spreading algorithm this replaced, the relay adds no latency when it regrids.
The republished ``source_id`` gains a ``+regrid`` suffix, because those timestamps
are the relay's, not the source's.

Run it in its own terminal, leave it running, then point the live test at the
republished outlet:

    python -B -m scripts.getlive.relay --source-name UnicornRecorderRawDataLSLStream \
        --labels Fz,C3,Cz,C4,Pz,PO7,Oz,PO8 --keep 0-7 --regrid

    python -B -m scripts.getlive --stream-name NOVA_Relay --cap declared \
        --sfreq 250 --source-units uV --eog drop --duration 30

``--labels`` is an operator assertion about the montage, exactly like
``--sfreq`` and ``--source-units`` are for the live test: LSL is not telling
us the electrode names, so someone has to, and the run records what was
asserted.
"""

import argparse
import math
import sys
from time import sleep

import numpy as np

from mne_lsl.lsl import StreamInfo, StreamInlet, StreamOutlet, resolve_streams

from nova2026.streaming import GridPolicy, TimeBase

# What the package accepts as a per-channel unit string.
UNIT_CHOICES = ("volts", "millivolts", "microvolts", "nanovolts")

# Appended to the republished source id when --regrid is on, so a consumer (and
# the run's provenance) can tell an invented grid from the source's own stamps.
REGRID_SOURCE_ID_SUFFIX = "+regrid"


def parse_keep(text: str, n_channels: int) -> tuple[int, ...]:
    """Turn ``0-7`` or ``0,1,2`` into column indices, validated against the source."""

    if not text.strip():
        return tuple(range(n_channels))
    kept: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, stop = part.partition("-")
            kept.extend(range(int(start), int(stop) + 1))
        else:
            kept.append(int(part))
    for index in kept:
        if not 0 <= index < n_channels:
            raise ValueError(
                f"--keep names column {index}, but the source publishes "
                f"{n_channels} channels (0..{n_channels - 1})."
            )
    if len(set(kept)) != len(kept):
        raise ValueError("--keep names the same column twice.")
    return tuple(kept)


def build_parser() -> argparse.ArgumentParser:
    """Build the relay's command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("source outlet")
    source.add_argument(
        "--source-name",
        required=True,
        help="LSL name of the outlet to republish, as probe.py prints it",
    )
    source.add_argument(
        "--source-id", help="LSL source id, when two outlets share a name"
    )
    source.add_argument(
        "--resolve-timeout",
        type=float,
        default=10.0,
        help="seconds to keep looking for the source outlet",
    )

    contract = parser.add_argument_group("metadata to publish")
    contract.add_argument(
        "--labels",
        required=True,
        help="comma-separated electrode names for the kept columns, in order",
    )
    contract.add_argument(
        "--keep",
        default="",
        help="source columns to forward, e.g. 0-7 or 0,1,2 (default: all)",
    )
    contract.add_argument(
        "--types",
        default="eeg",
        help="comma-separated channel types, or one type for every channel",
    )
    contract.add_argument(
        "--units",
        default="microvolts",
        choices=UNIT_CHOICES,
        help="unit the source samples are already in; declared, never applied",
    )

    out = parser.add_argument_group("republished outlet")
    out.add_argument("--out-name", default="NOVA_Relay", help="name to publish under")
    out.add_argument("--out-type", default="EEG", help="LSL stream type to publish")
    out.add_argument(
        "--out-source-id",
        default="nova-relay",
        help="LSL source id to publish; keep it stable across restarts",
    )
    out.add_argument(
        "--chunk",
        type=int,
        default=32,
        help="samples pulled and pushed per iteration",
    )
    out.add_argument(
        "--regrid",
        action="store_true",
        help="rebuild the timeline on a counted grid instead of forwarding the "
        "source's stamps: every sample received gets one slot at --regrid-rate, so "
        "a chunk-stamped or jittering source reaches Repair on a grid it accepts. "
        "The library's TimeBase does the work, and nothing is held back",
    )
    out.add_argument(
        "--regrid-jitter",
        dest="regrid",
        action="store_true",
        help="the earlier spelling of --regrid, from when a per-sample jittered "
        "source needed a mode of its own; it selects the same one",
    )
    out.add_argument(
        "--regrid-rate",
        type=float,
        default=None,
        help="samples per second to lay the grid at (default: the rate the source "
        "declares)",
    )
    out.add_argument("--quiet", action="store_true", help="no periodic throughput line")
    return parser


def resolve_source(args: argparse.Namespace):
    """Find exactly one source outlet, or say what is on the network instead."""

    infos = resolve_streams(args.resolve_timeout)
    matched = [info for info in infos if info.name == args.source_name]
    if args.source_id is not None:
        matched = [info for info in matched if info.source_id == args.source_id]
    if not matched:
        available = "\n".join(
            f"    name={info.name!r} type={info.stype!r} "
            f"channels={info.n_channels} sfreq={info.sfreq:g} "
            f"source_id={info.source_id!r}"
            for info in infos
        )
        raise RuntimeError(
            f"No outlet named {args.source_name!r} is publishing.\n"
            f"Available:\n{available or '    (none)'}"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"{len(matched)} outlets are named {args.source_name!r}; "
            "pass --source-id to pick one."
        )
    return matched[0]


def build_outlet_info(args: argparse.Namespace, labels: tuple[str, ...], sfreq: float, dtype) -> StreamInfo:
    """Describe the republished outlet: the metadata the source never declared."""

    types = [name.strip() for name in args.types.split(",") if name.strip()]
    if len(types) == 1:
        types = types * len(labels)
    if len(types) != len(labels):
        raise ValueError(
            f"--types gives {len(types)} entries for {len(labels)} channels; "
            "pass one type, or one per channel."
        )

    # A regridded outlet carries timestamps this relay invented, so say so in
    # the source id: downstream provenance must not have to guess whose clock a
    # window was built on.
    invented = args.regrid
    source_id = (
        args.out_source_id + REGRID_SOURCE_ID_SUFFIX if invented else args.out_source_id
    )
    sinfo = StreamInfo(
        name=args.out_name,
        stype=args.out_type,
        n_channels=len(labels),
        sfreq=sfreq,
        dtype=dtype,
        source_id=source_id,
    )
    sinfo.set_channel_names(list(labels))
    sinfo.set_channel_types(types)
    sinfo.set_channel_units([args.units] * len(labels))
    return sinfo


def main(argv: list[str] | None = None) -> int:
    """Forward one outlet to another, adding the metadata, until interrupted."""

    args = build_parser().parse_args(argv)
    labels = tuple(name.strip() for name in args.labels.split(",") if name.strip())
    if not labels:
        raise SystemExit("--labels must name at least one electrode.")
    if args.chunk < 1:
        raise SystemExit("--chunk must be a positive number of samples.")

    try:
        info = resolve_source(args)
    except RuntimeError as error:
        print(f"error: {error}")
        return 1

    try:
        keep = parse_keep(args.keep, int(info.n_channels))
    except ValueError as error:
        print(f"error: {error}")
        return 1
    if len(keep) != len(labels):
        print(
            f"error: --keep forwards {len(keep)} column(s) but --labels names "
            f"{len(labels)}; they must agree one to one."
        )
        return 1

    print(
        f"source    : name={info.name!r} type={info.stype!r} "
        f"channels={info.n_channels} sfreq={info.sfreq:g} Hz "
        f"source_id={info.source_id!r}"
    )
    print(f"forwarding: columns {list(keep)} as {', '.join(labels)}")

    inlet = StreamInlet(info, max_buffered=10, processing_flags=["clocksync"])
    inlet.open_stream(timeout=10.0)
    # The grid, when one is asked for. It is created inside the try but read in the
    # finally, so it has to exist even if the setup fails before reaching it.
    time_base = None
    try:
        dtype = inlet.dtype
        sinfo = build_outlet_info(args, labels, float(info.sfreq), dtype)
        outlet = StreamOutlet(sinfo, chunk_size=args.chunk)
        print(
            f"publishing: name={args.out_name!r} type={args.out_type!r} "
            f"channels={len(labels)} sfreq={info.sfreq:g} Hz "
            f"units={args.units} source_id={sinfo.source_id!r}"
        )
        if args.regrid:
            rate = float(info.sfreq if args.regrid_rate is None else args.regrid_rate)
            if not math.isfinite(rate) or rate <= 0:
                raise SystemExit("--regrid-rate must be finite and positive.")
            policy = GridPolicy.for_rate(rate)
            time_base = TimeBase(policy)
            print(
                f"regrid    : a counted grid at {rate:g} Hz from the library's time "
                "base; the source's stamps are read only to measure how far they "
                "have drifted from that count"
            )
            print(
                f"            tolerance {policy.tolerance_samples:g} sample(s), "
                f"re-lock past {policy.relock_samples:g} sample(s); every sample "
                "keeps its own slot, so nothing is held back"
            )
        print("Ctrl+C to stop.\n")

        columns = np.asarray(keep, dtype=int)
        forwarded = 0
        since_report = 0
        while True:
            # Every available sample, not --chunk: the source's own pull sizes are
            # what gets forwarded, which keeps the relay's latency to one iteration.
            data, timestamps = inlet.pull_chunk(timeout=1.0)
            if timestamps.size == 0:
                # No samples this turn: yield instead of spinning on the inlet.
                sleep(0.002)
                continue
            data = np.ascontiguousarray(data[:, columns])
            if time_base is not None:
                # Only the timeline changes: the data is forwarded as it arrived,
                # and every sample gets exactly one slot on the counted grid.
                timestamps, _ = time_base.place(np.asarray(timestamps, dtype=float))
            outlet.push_chunk(data, timestamps)
            forwarded += int(timestamps.size)
            since_report += int(timestamps.size)
            if not args.quiet and since_report >= int(info.sfreq) * 5:
                line = f"forwarded {forwarded} samples"
                if time_base is not None:
                    state = time_base.state
                    line += (
                        f" ({len(state.relocks)} re-lock(s), "
                        f"{state.relocked_samples:.1f} sample(s) of drift absorbed, "
                        f"{len(state.large_steps)} suspicious step(s))"
                    )
                print(line)
                since_report = 0
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130
    finally:
        if time_base is not None:
            state = time_base.state
            print(
                f"regrid    : absorbed {state.relocked_samples:.1f} sample(s) of "
                f"drift over {len(state.relocks)} re-lock(s), "
                f"{len(state.large_steps)} suspicious step(s) from the source"
            )
        inlet.close_stream()

    return 0


if __name__ == "__main__":
    sys.exit(main())
