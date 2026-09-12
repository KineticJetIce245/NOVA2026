"""Publish a raw LSL outlet for the getlive diagnostics: no metadata, or jitter.

Two uses, both of them fixtures rather than part of a normal acceptance run:

* default: positional labels and no declared units — the shape ``relay.py``
  exists for (the ``relay_smoke`` flow drives plain source -> relay -> getlive);
* ``--jitter``: a regular grid plus Gaussian jitter, so
  :mod:`scripts.getlive.ts_check` can validate its own probe and show what the
  LSL ``dejitter`` post-processing flag does to the timestamp grid.

Run it in its own terminal; it publishes until ``--seconds`` elapses:

    python -B -m scripts.getlive.publish_raw --name raw-1 --jitter 0.0003

It declares no channel names, types or units on purpose: the relay is what adds
them, and the package's pre-flight check is what refuses a source without them.
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
    args = parser.parse_args()

    info = StreamInfo(
        args.name, "EEG", args.channels, args.sfreq, "float32", args.source_id
    )
    if args.metadata:
        info.set_channel_names([f"E{index + 1}" for index in range(args.channels)])
        info.set_channel_types(["eeg"] * args.channels)
        info.set_channel_units(["microvolts"] * args.channels)
    outlet = StreamOutlet(info, chunk_size=16)

    print(
        f"publishing {args.name!r}: {args.channels} channels, {args.sfreq:g} Hz, "
        f"jitter={args.jitter * 1e3:.2f} ms, metadata={args.metadata}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    chunk = 16
    # Lead time so an inlet in another process can attach before data flows.
    started = local_clock() + 1.0
    index = 0
    while local_clock() - started < args.seconds:
        if local_clock() < started:
            time.sleep(0.01)
            continue
        grid = started + np.arange(index, index + chunk) / args.sfreq
        stamps = grid + rng.normal(0.0, args.jitter, size=chunk) if args.jitter else grid
        outlet.push_chunk(
            rng.normal(0, 15.0, (chunk, args.channels)).astype("float32"), stamps
        )
        index += chunk
        time.sleep(chunk / args.sfreq)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
