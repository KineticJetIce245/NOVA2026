"""Publish a raw LSL outlet for the getlive diagnostics: no metadata, or jitter.

Two uses, both of them fixtures rather than part of a normal acceptance run:

* default: positional labels and no declared units - the shape ``relay.py``
  exists for (the ``relay_smoke`` flow drives plain source -> relay -> getlive);
* ``--jitter``: a regular grid plus Gaussian jitter, so
  :mod:`scripts.getlive.ts_check` can validate its own probe and show what the
  LSL ``dejitter`` post-processing flag does to the timestamp grid.

Run it in its own terminal; it publishes until ``--seconds`` elapses:

    python -B -m scripts.getlive.publish_raw --name raw-1 --jitter 0.0003

It declares no channel names, types or units on purpose: the relay is what adds
them, and the package's pre-flight check is what refuses a source without them.

``--stamp-per-chunk`` reached for the stamping shape a live Unicorn Recorder
produces (many samples per timestamp), and it does hand liblsl a scalar time.
It does **not** reproduce that shape on the measured side: mne-lsl's inlet then
delivers a clean one-sample grid anyway, so a run of
``ts_check --name <this fixture>`` reports ``samples/chunk=1``. Only the real
device produced the chunk-stamped readings, so treat the flag as documentation
of the intent, not as a validated fixture - the measured numbers from the rig
are the evidence to work from.
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
        "the way a recorder that passes push_chunk a scalar does - the shape "
        "ts_check diagnoses as chunk-stamped. NOTE: not reproducible through "
        "mne-lsl on this host (see the module docstring)",
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
        f"chunk={args.chunk}, stamp_per_chunk={args.stamp_per_chunk}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    chunk = args.chunk
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
        # sample individually.
        outlet.push_chunk(
            rng.normal(0, 15.0, (chunk, args.channels)).astype("float32"),
            float(stamps[-1]) if args.stamp_per_chunk else stamps,
        )
        index += chunk
        time.sleep(chunk / args.sfreq)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
