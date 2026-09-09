"""Runnable demo of the full decoupled real-time path.

Run from the repository root:

    .venv/Scripts/python.exe -B -m scripts.streaming_demo --duration 8 --record records

It publishes a synthetic 8-channel recording in volts through PlayerLSL with an
uneven chunk size, then runs the whole chain in one loop:

    Acquire.read()                     (built by StreamSession)
      -> StreamSession.ingest()        channel reorder + raw recording
      -> StreamSession.process()       uV, quality, notch, band, 500->128 Hz
      -> CircularBuffer.push()         2 s windows @ 128 Hz
      -> StreamSession.gate()          warm-up + quality gating
      -> TaskOffloader (or in-loop) analysis

Compare consumer modes with a simulated slow analysis:

    # synchronous analysis stalls acquisition and the lag guard stops the run
    ... --compute 0.8 --workers 0
    # offloaded analysis keeps pulling samples
    ... --compute 0.8 --workers 2

Replace the PlayerLSL/StreamLSL setup with the real amplifier outlet to run
live; nothing else in the loop changes.
"""

from time import monotonic, sleep
from uuid import uuid4

import mne
import numpy as np
from mne_lsl.player import PlayerLSL  # fake EEG source (outlet)
from mne_lsl.stream import StreamLSL  # our reader (inlet)

# Reusable start-up helpers: argument getter + one-shot session assembly.
from nova2026.streaming import TaskOffloader  # runs per-window analysis
from nova2026.streaming.bootstrap import StreamSession, parse_args

# Preprocessing building blocks are DEFINED here, in this script, on purpose.
from nova2026.streaming.preprocess import (
    QualityMonitor,  # observes raw EEG faults (amplitude/saturation/flatline)
    Resampler,  # stateful 500 -> 128 Hz resampling (SoXR)
    SosFilter,  # one stateful causal filter (notch or band-pass)
    design_bandpass,  # designs Butterworth band-pass coefficients
    design_notch,  # designs a notch filter's coefficients
    unit_scaler,  # converts source units (volts) to microvolts
)

# ---------------------------------------------------------------------------
# Demo constants: channel labels and the source's unit exponent.
# ---------------------------------------------------------------------------
# The 8 channel labels the synthetic recording publishes, in column order.
CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")
# The synthetic source publishes samples in volts (10^-0).
SOURCE_UNIT_EXPONENT = 0


def synthetic_recording(seconds: float, sfreq: float) -> mne.io.RawArray:
    """Return a deterministic multi-channel recording in volts.

    Each channel is a clean 10 Hz sine with a slightly different amplitude
    (10..17 uV) so window statistics stay easy to eyeball.
    """

    times = np.arange(round(seconds * sfreq)) / sfreq  # one row per sample
    data = np.vstack(
        [
            (10.0 + index) * 1e-6 * np.sin(2 * np.pi * (10 + index) * times)
            for index in range(len(CHANNELS))
        ]
    )
    info = mne.create_info(list(CHANNELS), sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def main() -> None:
    """Run the synthetic source through the complete processing path."""

    # All common options (geometry, chain, recording, consumer) come from the
    # bootstrap argument getter, so no argparse code lives in this script.
    args = parse_args(description=__doc__)
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative.")

    # ------------------------------------------------------------------
    # 1) Start the synthetic LSL source (PlayerLSL plays in real time).
    # ------------------------------------------------------------------
    name = f"nova-demo-{uuid4().hex[:8]}"  # unique outlet name per run
    player = PlayerLSL(
        synthetic_recording(args.duration + 6.0, args.sfreq),  # extra lead time
        name=name,
        chunk_size=args.chunk_size,  # uneven chunks on purpose
    )
    player.start()

    # ------------------------------------------------------------------
    # 2) Connect our inlet. acquisition_delay=None means MANUAL acquisition:
    #    data is pulled only when we call stream.acquire() (inside Acquire).
    # ------------------------------------------------------------------
    stream = StreamLSL(bufsize=4.0, name=name)
    stream.connect(
        acquisition_delay=None,
        processing_flags=["clocksync"],
        timeout=10,
    )

    # ------------------------------------------------------------------
    # 3a) Define the preprocessing chain EXPLICITLY, right here in the script,
    #     so every stage and its parameters are visible and editable.
    # ------------------------------------------------------------------
    scaler = unit_scaler(SOURCE_UNIT_EXPONENT, desired_exponent=-6)  # V -> uV
    quality = QualityMonitor(n_eeg=len(CHANNELS), sfreq=args.sfreq, warmup_seconds=0.0)
    notch = SosFilter(design_notch(args.notch, args.notch_q, args.sfreq), len(CHANNELS))
    bandpass = SosFilter(
        design_bandpass(args.lpass, args.hpass, args.order, args.sfreq), len(CHANNELS)
    )
    resampler = Resampler(
        args.sfreq, args.out_sfreq, len(CHANNELS), quality=args.resample_quality
    )

    # ------------------------------------------------------------------
    # 3b) StreamSession assembles the REST (contract, recorder, acquire,
    #     buffer) and only executes the chain above in a fixed order.
    # ------------------------------------------------------------------
    session = StreamSession(
        stream,
        args,
        CHANNELS,
        scale=scaler,
        quality=quality,
        filters=(notch, bandpass),
        resample=resampler,
        source_unit_exponent=SOURCE_UNIT_EXPONENT,
    )
    if session.contract.dropped_channels:
        print(f"channel contract: dropping {session.contract.dropped_channels}")

    # ------------------------------------------------------------------
    # 4) Print the configuration so the run is self-explanatory.
    # ------------------------------------------------------------------
    print(
        f"source: {session.n_channels} ch @ {args.sfreq:g} Hz, "
        f"chunk_size={args.chunk_size}\n"
        f"chain : V->uV, notch {args.notch:g} Hz, band {args.lpass:g}-{args.hpass:g} Hz, "
        f"resample {args.sfreq:g}->{session.out_sfreq:g} Hz ({args.resample_quality})\n"
        f"buffer: window={session.window_samples} hop={session.hop_samples} "
        f"capacity={session.capacity_samples} @ {session.out_sfreq:g} Hz\n"
        f"analysis: compute={args.compute:g}s workers={args.workers} queue={args.queue}"
    )
    if session.recorder is not None:
        print(f"record : raw volts -> {session.recorder.path}  (sqlite + fif)")

    # ------------------------------------------------------------------
    # 5) Consumer state: counters and the per-window report function.
    # ------------------------------------------------------------------
    started = monotonic()  # wall clock for the duration loop and prints
    valid_windows = 0  # windows that passed warm-up + quality gating
    rejected_windows = 0  # warm-up or bad-quality windows dropped

    # Display one line per window: peak-to-peak amplitude per channel (uV).
    def report(eeg_window) -> None:
        peak_to_peak_uv = np.ptp(eeg_window.data, axis=0)
        print(
            f"window start={eeg_window.start_sample:6d}  "
            f"at={monotonic() - started:7.3f}s  "
            f"ptp_uV={np.round(peak_to_peak_uv, 1)}"
        )

    # The "analysis" task: optionally sleep to simulate a slow model, then
    # report. This function runs in worker threads when workers > 0.
    def analyze(item) -> None:
        eeg_window = item
        if args.compute > 0:
            sleep(args.compute)
        report(eeg_window)

    # Offloader created once before the loop; workers pick tasks off a queue.
    offloader = (
        TaskOffloader(analyze, workers=args.workers, capacity=args.queue)
        if args.workers
        else None
    )

    # ------------------------------------------------------------------
    # 6) The real-time loop. One iteration = one acquired block; everything
    #    downstream happens in this fixed order every iteration.
    # ------------------------------------------------------------------
    try:
        while monotonic() - started < args.duration:
            # (a) Pull the next raw block from LSL (blocks until ready).
            data, timestamps = session.acquire.read()

            # (b) Contract reorder + save the raw volts before transformation.
            data, timestamps = session.ingest(data, timestamps)

            # (c) Run the preprocessing chain over this block.
            data, timestamps = session.process(data, timestamps)

            # (d) Collect any windows this block completed.
            for window, window_times, start in session.buffer.push(data, timestamps):
                # (e) Package the window with its verdict (warm-up + quality).
                eeg_window = session.wrap(window, window_times, start)
                if not eeg_window.valid:
                    rejected_windows += 1
                    continue
                valid_windows += 1

                # (f) Run the analysis in the loop, or hand it to workers.
                if offloader is None:
                    analyze(eeg_window)
                else:
                    offloader.submit(eeg_window)

    # ------------------------------------------------------------------
    # 7) Shutdown: release everything in the right order.
    # ------------------------------------------------------------------
    except KeyboardInterrupt:
        print("interrupted by user")
    except RuntimeError as error:
        print(f"run stopped: {error}")
    finally:
        session.acquire.close()  # detach callback, drop buffer
        if stream.connected:
            stream.disconnect()  # close the LSL inlet
        player.stop()  # stop the fake source
        if offloader is not None:
            offloader.close(drain=True, timeout=5.0)  # let tasks finish
        session.close(status="completed")  # lock the run, export the FIF
        if session.recorder is not None:
            print(
                f"recorded {session.recorder.samples} raw samples -> "
                f"{session.recorder.path}\n  fif: {session.recorder.fif_path}"
            )

    # ------------------------------------------------------------------
    # 8) Summaries: acquisition stats, resampler output, window counts and
    #    offloader counters.
    # ------------------------------------------------------------------
    print(
        f"\ninput_samples={session.acquire.samples} "
        f"max_lag={session.acquire.max_lag:.3f}s gaps={session.acquire.gaps}"
    )
    if session.resample is not None:
        print(
            f"resampled_output={session.resample.output_samples} "
            f"@ {session.out_sfreq:g} Hz"
        )
    print(
        f"windows: valid={valid_windows} rejected={rejected_windows} "
        f"buffer_windows={session.buffer.windows}"
    )
    if offloader is not None:
        print(
            f"offload: submitted={offloader.submitted} completed={offloader.completed} "
            f"dropped={offloader.dropped} failed={offloader.failed}"
        )


# ---------------------------------------------------------------------------
# Entry point: only runs the demo when executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
