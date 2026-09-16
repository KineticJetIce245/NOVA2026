"""Publish a raw LSL outlet for the getlive diagnostics: no metadata, or jitter.

Two uses, both of them fixtures rather than part of a normal acceptance run:

* default: positional labels and no declared units - the shape ``relay.py``
  exists for (the ``relay_smoke`` flow drives plain source -> relay -> getlive);
* ``--jitter``: a regular grid plus Gaussian jitter, so
  :mod:`scripts.getlive.ts_check` can validate its own probe and show what the
  LSL ``dejitter`` post-processing flag does to the timestamp grid;
* ``--within-chunk-us``: a chunk-stamped source - one stamp per block, with the
  samples inside the block a few microseconds apart - so the probe's chunk
  detector and its verdict can be tested against the shape a live Unicorn
  Recorder produces.

Run it in its own terminal; it publishes until ``--seconds`` elapses:

    python -B -m scripts.getlive.publish_raw --name raw-1 --jitter 0.0003
    python -B -m scripts.getlive.publish_raw --name chunked-1 --chunk 8 --within-chunk-us 12

It declares no channel names, types or units on purpose: the relay is what adds
them, and the package's pre-flight check is what refuses a source without them.

``--stamp-per-chunk`` sends liblsl a scalar time, which its documentation calls
"the acquisition timestamp of the last sample". It does **not** reproduce the
chunk-stamped shape on the measured side: an mne-lsl inlet spreads that value
back over the block and delivers a clean one-sample grid. Use
``--within-chunk-us`` to reproduce the shape instead - it puts distinct
timestamps on the block's samples, all inside one nominal sampling interval,
which is what a recorder that re-stamps a block does.
"""

import argparse
import time

import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock


def main() -> int:
    """Publish the fixture until the requested duration elapses."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="relay-smoke-raw")
    parser.add_argument("--source-id", default="raw-1")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="standard deviation of the per-sample timestamp jitter, in seconds",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="also declare labels and units (default: declare nothing, like a plain recorder)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=16,
        help="samples per push_chunk (a source that stamps per chunk shares one "
        "timestamp across this many samples)",
    )
    parser.add_argument(
        "--stamp-per-chunk",
        action="store_true",
        help="stamp the whole chunk with one timestamp instead of one per sample, "
        "the way a recorder that passes push_chunk a scalar does. NOTE: mne-lsl "
        "then delivers a clean grid anyway; use --within-chunk-us to reproduce a "
        "chunk-stamped source instead",
    )
    parser.add_argument(
        "--within-chunk-us",
        type=float,
        default=0.0,
        help="spread a chunk's timestamps over this many microseconds, ending on "
        "the block's own stamp; 8-12 reproduces the chunk-stamped shape a live "
        "Unicorn Recorder produces (one stamp per ~8 samples)",
    )
    args = parser.parse_args()

    if args.chunk < 1:
        raise SystemExit("--chunk must be a positive number of samples.")

    info = StreamInfo(
        args.name, "EEG", args.channels, args.sfreq, "float32", args.source_id
    )
    if args.metadata:
        info.set_channel_names([f"E{index + 1}" for index in range(args.channels)])
        info.set_channel_types(["eeg"] * args.channels)
        info.set_channel_units(["microvolts"] * args.channels)
    outlet = StreamOutlet(info, chunk_size=args.chunk)

    print(
        f"publishing {args.name!r}: {args.channels} channels, {args.sfreq:g} Hz, "
        f"jitter={args.jitter * 1e3:.2f} ms, metadata={args.metadata}, "
        f"chunk={args.chunk}, stamp_per_chunk={args.stamp_per_chunk}, "
        f"within_chunk={args.within_chunk_us:.1f} us",
        flush=True,
    )

    rng = np.random.default_rng(0)
    chunk = args.chunk
    within = args.within_chunk_us * 1e-6
    # Lead time so an inlet in another process can attach before data flows.
    started = local_clock() + 1.0
    index = 0
    while local_clock() - started < args.seconds:
        if local_clock() < started:
            time.sleep(0.01)
            continue
        grid = started + np.arange(index, index + chunk) / args.sfreq
        stamps = grid + rng.normal(0.0, args.jitter, size=chunk) if args.jitter else grid
        # A scalar timestamp is liblsl's "acquisition timestamp of the last
        # sample", applied to every sample of the chunk; an array stamps each
        # sample individually. --within-chunk-us keeps the block's own stamp on
        # its last sample while the earlier ones sit a few microseconds behind,
        # which is the shape a recorder that re-stamps a block produces: distinct
        # timestamps, all of them inside one nominal sampling interval.
        outlet.push_chunk(
            rng.normal(0, 15.0, (chunk, args.channels)).astype("float32"),
            _shared_block_stamps(stamps, chunk, within, args.stamp_per_chunk),
        )
        index += chunk
        time.sleep(chunk / args.sfreq)

    return 0


def _shared_block_stamps(
    stamps: np.ndarray, chunk: int, within: float, per_chunk: bool
):
    """Return what to hand ``push_chunk`` for one block.

    ``per_chunk`` sends a scalar (liblsl's last-sample timestamp, which this
    mne-lsl path spreads out again). ``within`` sends an array whose samples sit
    a few microseconds apart, ending on the block stamp: the chunk-stamped shape
    with distinct stamps.
    """

    if per_chunk:
        return float(stamps[-1])
    if within <= 0:
        return stamps
    return float(stamps[-1]) - within * np.arange(chunk)[::-1]


if __name__ == "__main__":
    raise SystemExit(main())
