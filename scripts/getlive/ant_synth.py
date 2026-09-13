"""Publish a synthetic ANT-like EEG outlet, so the live route can be rehearsed.

There is no amplifier in this loop. The outlet this publishes has the shape the
real one has: the 24 electrodes of the bring-up rig (``cap.EE511_EEG_CHANNELS``,
deliberately NOT in the chain's order, so that selecting the model's 20 electrodes
**by name** is what gets exercised), 500 Hz, float32, microvolts, 25-sample chunks
stamped per sample. ``scripts/auditory_ui/live.py`` cannot tell it from the
amplifier's own outlet, which is the point: the whole live path -- pre-flight,
contract gate, session, producer, transport, page -- runs for real.

The samples themselves are synthetic and that has a consequence worth stating:
nothing here is correlated with the demo's speech envelopes, so the decoder's
scores are not meaningful and the decisions the page shows are not evidence about
anyone's attention. Use it to prove the plumbing, and
``scripts/getlive/ant_publish.py`` (which replays a recorded ANT session) when you
want decisions that mean something.

    python -B -m scripts.getlive.ant_synth --signal osc --seconds 900
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from scripts.getlive.cap import EE511_EEG_CHANNELS

DEFAULT_STREAM_NAME = "NOVA-ANT-synthetic"
DEFAULT_SOURCE_ID = "ant-synthetic-1"
RATE = 500.0
"""The bring-up rig's measured rate (plan section 3.11), asserted, not probed."""


def levels_block(channels: int) -> np.ndarray:
    """One constant level per column: what a routing test wants to read back."""

    return np.arange(1.0, channels + 1.0)[None, :].astype(np.float32)


def oscillation_block(index: int, channels: int, chunk: int) -> np.ndarray:
    """A synthetic signal: per-column amplitude, 10 Hz + 5 Hz, plus slow drift."""

    t = (np.arange(index * chunk, (index + 1) * chunk) / RATE)[:, None]
    column = np.arange(channels)[None, :]
    amplitude = 4.0 + 2.0 * (column % 7)
    phase = column * 0.7
    signal = amplitude * (
        np.sin(2 * math.pi * 10.0 * t + phase)
        + 0.5 * np.sin(2 * math.pi * 5.0 * t + phase / 2)
    )
    return (signal + 3.0 * np.sin(2 * math.pi * 0.2 * t)).astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=("levels", "osc"), default="osc",
                        help="levels: the flat per-column levels a routing test "
                             "reads back; osc (default): a synthetic oscillation")
    parser.add_argument("--seconds", type=float, default=900.0,
                        help="how long to publish; the demo is meant to be watched")
    parser.add_argument("--chunk", type=int, default=25, help="samples per push_chunk")
    parser.add_argument("--name", default=DEFAULT_STREAM_NAME)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    args = parser.parse_args(argv)

    declared = tuple(reversed(EE511_EEG_CHANNELS))
    info = StreamInfo(args.name, "EEG", len(declared), RATE, "float32", args.source_id)
    info.set_channel_names(list(declared))
    info.set_channel_types(["eeg"] * len(declared))
    info.set_channel_units(["microvolts"] * len(declared))
    outlet = StreamOutlet(info, chunk_size=args.chunk)

    print(f"publishing '{args.name}' id={args.source_id}: {len(declared)} channels, "
          f"{RATE:g} Hz, microvolts, chunk {args.chunk}, signal={args.signal}", flush=True)
    print(f"declared order (the reverse of the chain's, on purpose): {', '.join(declared)}",
          flush=True)
    print("synthetic: uncorrelated with the demo's speech envelopes, so the decisions "
          "this drives are plumbing evidence, not attention evidence.", flush=True)

    # Publish only while something is reading, and re-anchor the clock each time a
    # reader arrives.
    #
    # This is not politeness, it is what makes the source usable. LSL hands a
    # late-joining inlet the outlet's buffered history, so a publisher that pushed
    # while nobody was listening presents the session with blocks whose stamps are
    # seconds old: `Acquire.max_lag` (3 s by default) refuses them, the transport
    # ends with "Source samples are 3.7 seconds old", and the session stops having
    # acquired nothing. Measured, not guessed: that is exactly how the first
    # version of this file failed, with 1525 samples left pending at the end.
    #
    # Idling between readers also means the gap carries no stale stamps: the first
    # sample after a new reader connects is stamped "now", which is what an
    # amplifier that was switched on a moment ago looks like.
    index = 0
    had_consumer = False
    next_push = time.monotonic()
    began = time.time()
    while time.time() - began < args.seconds:
        if not outlet.has_consumers:
            if had_consumer:
                print(f"  {time.time() - began:7.1f}s reader left; pausing rather than "
                      f"queueing stale samples", flush=True)
                had_consumer = False
            time.sleep(0.05)
            next_push = time.monotonic()
            continue
        if not had_consumer:
            print(f"  {time.time() - began:7.1f}s reader connected; publishing", flush=True)
            had_consumer = True
            next_push = time.monotonic()
        if args.signal == "levels":
            block = np.tile(levels_block(len(declared)), (args.chunk, 1))
        else:
            block = oscillation_block(index, len(declared), args.chunk)
        # Stamp the chunk the way liblsl expects: its LAST sample is "now", the
        # earlier ones are as far back as they are old. Stamping ahead of the
        # local clock (the first version added a 50 ms lead) trips the consumer's
        # other guard -- "Source timestamps are ahead of the local LSL clock" --
        # which is a fair reading of a publisher that claims to know the future.
        now = local_clock()
        stamps = now - (args.chunk - 1 - np.arange(args.chunk)) / RATE
        outlet.push_chunk(block, stamps)
        index += 1
        if index % 400 == 0:
            print(f"  {time.time() - began:7.1f}s published ({index * args.chunk} samples)",
                  flush=True)
        # Absolute schedule, not `sleep(chunk/RATE)`: sleeping a fixed amount adds
        # the loop's own overhead every iteration and the stamps then drift behind
        # the wall clock, which is the other way to look stale to the reader.
        next_push += args.chunk / RATE
        delay = next_push - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    print("done publishing", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
