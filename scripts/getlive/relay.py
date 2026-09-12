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
problem: a recorder that stamps a whole block once (many samples sharing one
timestamp, the shape :mod:`scripts.getlive.ts_check` diagnoses as chunk-stamped)
delivers a grid ``Repair`` can only refuse, and no tolerance helps because the
steps inside a block are near zero. Regridding spreads each block over the time
from its own first stamp until the next block's first stamp, so the grid is real
and nothing accumulates. A block that lost data is still spread - keeping its own
stamps would re-inject the near-zero steps - so the loss shows up as one long step
that ``Repair`` stops on, and the relay counts it and says so. The republished
``source_id`` gains a ``+regrid`` suffix, because those timestamps are this
relay's, not the source's.

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

# What the package accepts as a per-channel unit string.
UNIT_CHOICES = ("volts", "millivolts", "microvolts", "nanovolts")

# Two samples whose timestamps differ by less than this many nominal samples were
# stamped together: the block boundary is where the spacing is real. The threshold
# may sit anywhere between the microseconds a chunk-stamped source repeats inside a
# block and the one sample it takes to reach the next one - three orders of
# magnitude apart - so a quarter of a sample is nowhere near either edge.
SAME_STAMP_SAMPLES = 0.25

# A block whose span is more than this multiple of the time its samples can
# account for has lost a whole block: at least half again the nominal block length.
LOST_SPAN_RATIO = 1.5

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
        help="rebuild a per-sample timestamp grid for a chunk-stamped source: each "
        "block is spread over the time from its own first stamp until the next "
        "block's first stamp, so the grid is real and no error accumulates",
    )
    out.add_argument(
        "--regrid-rate",
        type=float,
        default=None,
        help="samples per second to spread each block over (default: the rate the "
        "source declares)",
    )
    out.add_argument(
        "--regrid-lost-ratio",
        type=float,
        default=LOST_SPAN_RATIO,
        help="how much longer than its samples can account for a block's span may "
        "be before the relay reports that the source lost a block (default: 1.5)",
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
    source_id = args.out_source_id + REGRID_SOURCE_ID_SUFFIX if args.regrid else args.out_source_id
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


class Regridder:
    """Rebuild a per-sample timestamp grid for a chunk-stamped source.

    A recorder that stamps one block once leaves ``Repair`` with a grid it can
    only refuse: the steps inside a block are near zero, and a step below one
    sample is fatal at any tolerance. This class replaces those stamps with a real
    grid, and the design has one job - **never accumulate drift**.

    The rule is one sentence: **a block covers the time from its own first stamp
    until the next block's first stamp**. Its samples are spread evenly across
    that span, so

    * the seam between two blocks is exactly one sample - never zero (two samples
      on the same stamp) and never negative (a sample dated before its
      predecessor);
    * nothing is assumed about how many samples a block carries, which on real
      hardware is not a constant;
    * the nominal rate is only ever used to decide whether a span is *plausible*
      (``tolerance``), never to place samples, so a small rate error cannot make
      the grid drift away from the source clock.

    A block whose span is far larger than its sample count implies is a real gap -
    samples the source never sent - and is forwarded with its original stamps, so
    ``Repair`` sees the gap instead of a grid smoothed over it. The source's
    internal spacings are never trusted: on a chunk-stamped source they are
    microseconds by construction.

    One block is always held back, because a block's span is only known once its
    successor arrives; ``flush`` emits the last one at shutdown.

    Args:
        rate: Samples per second the source is expected to run at.
        n_channels: Data columns expected on every call.
        lost_ratio: How much longer than its samples can account for a block's span
            may be before the source is reported as having lost a block. Must be at
            least 1: below that, every ordinary block would look lost.

    Attributes:
        spread_blocks, lost_blocks: Counters for the report.
    """

    def __init__(self, rate: float, n_channels: int, lost_ratio: float) -> None:
        """Validate the settings and start with an empty buffer."""

        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive.")
        if n_channels < 1:
            raise ValueError("n_channels must be positive.")
        if not math.isfinite(lost_ratio) or lost_ratio < 1.0:
            raise ValueError("lost_ratio must be finite and at least 1.")
        self._rate = float(rate)
        self._interval = 1.0 / float(rate)
        self._channels = int(n_channels)
        self._lost_ratio = float(lost_ratio)
        self._data: list[np.ndarray] = []
        self._stamps: list[np.ndarray] = []
        self.spread_blocks = 0
        self.lost_blocks = 0

    @property
    def samples(self) -> int:
        """Samples currently held back."""

        return int(sum(part.size for part in self._stamps))

    def blocks(self) -> int:
        """Blocks spread on the grid, including those that lost data."""

        return self.spread_blocks

    def _buffer(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the buffered samples as two arrays."""

        data = (
            np.concatenate(self._data)
            if self._data
            else np.empty((0, self._channels))
        )
        stamps = np.concatenate(self._stamps) if self._stamps else np.empty(0)
        return data, stamps

    def _groups(self, stamps: np.ndarray) -> np.ndarray:
        """Start index of every block, plus the index one past the last sample.

        A gap between samples ``i`` and ``i + 1`` means sample ``i`` is the last of
        its block, so the next block starts at ``i + 1``.
        """

        boundary = np.diff(stamps) >= SAME_STAMP_SAMPLES * self._interval
        return np.concatenate(([0], np.flatnonzero(boundary) + 1, [stamps.size]))

    def _spread(self, data, stamps, start: float, stop: float):
        """Place one block's samples evenly between two source stamps."""

        count = stamps.size
        if count < 2 or stop <= start:
            return data, stamps
        width = (stop - start) / count
        return data, start + np.arange(count) * width

    def _emit(self, data, stamps, groups: list[tuple[int, int]]):
        """Spread every block over the span its successor's stamp defines."""

        out_data: list[np.ndarray] = []
        out_stamps: list[np.ndarray] = []
        for index, (left, right) in enumerate(groups):
            block_data = data[left:right]
            block_stamps = stamps[left:right]
            if index + 1 < len(groups):
                stop = float(stamps[groups[index + 1][0]])
            else:
                # No successor yet: its own nominal span keeps the tail regular.
                stop = float(block_stamps[0]) + block_stamps.size * self._interval
            expected = block_stamps.size * self._interval
            span = stop - float(block_stamps[0])
            if span > expected * self._lost_ratio:
                # More time than these samples can account for: the source skipped
                # a whole block. The samples are still spread - keeping their own
                # stamps would re-inject the near-zero steps this relay exists to
                # remove - so the loss shows up as one long step and Repair stops
                # with its gap message. Counting it here is what tells an operator
                # the source is dropping data.
                self.lost_blocks += 1
                print(
                    f"regrid    : block of {block_stamps.size} samples covers "
                    f"{span * 1e3:.2f} ms, expected {expected * 1e3:.2f} ms; the "
                    "source appears to have lost a block"
                )
            self.spread_blocks += 1
            block_data, grid = self._spread(
                block_data, block_stamps, float(block_stamps[0]), stop
            )
            out_data.append(block_data)
            out_stamps.append(grid)
        if not out_stamps:
            return np.empty((0, self._channels)), np.empty(0)
        return np.concatenate(out_data), np.concatenate(out_stamps)

    def feed(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add a pulled chunk and return what is ready to be republished.

        Returns:
            ``(data, timestamps)`` with every complete block spread on the grid,
            or empty arrays while only one block is known.
        """

        if data.size == 0:
            return np.empty((0, self._channels)), np.empty(0)
        self._data.append(np.asarray(data))
        self._stamps.append(np.asarray(timestamps, dtype=float))
        data, stamps = self._buffer()
        starts = self._groups(stamps)
        # The last group has no known span yet: it waits for the next block.
        if starts.size < 3:
            return np.empty((0, self._channels)), np.empty(0)
        keep = int(starts[-2])
        groups = [
            (int(starts[index]), int(starts[index + 1]))
            for index in range(starts.size - 2)
        ]
        ready = data[:keep], stamps[:keep]
        self._data = [data[keep:]] if keep < data.shape[0] else []
        self._stamps = [stamps[keep:]] if keep < stamps.size else []
        return self._emit(ready[0], ready[1], groups)

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        """Spread whatever is buffered, for shutdown; the tail is never dropped."""

        if self.samples == 0:
            return np.empty((0, self._channels)), np.empty(0)
        data, stamps = self._buffer()
        self._data, self._stamps = [], []
        starts = self._groups(stamps)
        groups = [
            (int(starts[index]), int(starts[index + 1]))
            for index in range(starts.size - 1)
        ]
        return self._emit(data, stamps, groups)


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
    try:
        dtype = inlet.dtype
        sinfo = build_outlet_info(args, labels, float(info.sfreq), dtype)
        outlet = StreamOutlet(sinfo, chunk_size=args.chunk)
        print(
            f"publishing: name={args.out_name!r} type={args.out_type!r} "
            f"channels={len(labels)} sfreq={info.sfreq:g} Hz "
            f"units={args.units} source_id={sinfo.source_id!r}"
        )
        regridder = None
        if args.regrid:
            rate = float(info.sfreq if args.regrid_rate is None else args.regrid_rate)
            if not math.isfinite(rate) or rate <= 0:
                raise SystemExit("--regrid-rate must be finite and positive.")
            regridder = Regridder(rate, len(labels), args.regrid_lost_ratio)
            print(
                f"regrid    : per-sample grid at {rate:g} Hz; each block is spread "
                "over the time from its own first stamp until the next block's "
                "first stamp"
            )
            print(
                f"            a block covering more than "
                f"{args.regrid_lost_ratio:g}x the time its samples account for is "
                "counted and reported as lost; the grid is invented, so one block "
                "is held back as latency"
            )
        print("Ctrl+C to stop.\n")

        columns = np.asarray(keep, dtype=int)
        forwarded = 0
        since_report = 0
        while True:
            # Every available sample, not --chunk: a block boundary is detected
            # from the source timestamps, and pulling part of a block would look
            # like a boundary that is not there.
            data, timestamps = inlet.pull_chunk(timeout=1.0)
            if timestamps.size == 0:
                # No samples this turn: yield instead of spinning on the inlet.
                sleep(0.002)
                continue
            data = np.ascontiguousarray(data[:, columns])
            if regridder is not None:
                data, timestamps = regridder.feed(data, np.asarray(timestamps, dtype=float))
                if timestamps.size == 0:
                    continue
            outlet.push_chunk(data, timestamps)
            forwarded += int(timestamps.size)
            since_report += int(timestamps.size)
            if not args.quiet and since_report >= int(info.sfreq) * 5:
                line = f"forwarded {forwarded} samples"
                if regridder is not None:
                    line += (
                        f" ({regridder.spread_blocks} block(s) spread, "
                        f"{regridder.lost_blocks} with lost data, "
                        f"{regridder.samples} sample(s) buffered)"
                    )
                print(line)
                since_report = 0
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130
    finally:
        if regridder is not None:
            # Never drop the block that was still incomplete: it is only held
            # back because its successor had not arrived yet.
            tail_data, tail_stamps = regridder.flush()
            if tail_stamps.size:
                try:
                    outlet.push_chunk(tail_data, tail_stamps)
                    print(f"regrid    : flushed {tail_stamps.size} buffered sample(s)")
                except Exception:  # noqa: BLE001 - shutdown must not mask the stop
                    pass
        inlet.close_stream()

    return 0


if __name__ == "__main__":
    sys.exit(main())
