"""Runnable demo of the full decoupled real-time path.

Run from the repository root:

    .venv/Scripts/python.exe -B -m scripts.streaming_demo --duration 8 --record records

It publishes a synthetic 8-channel recording in volts through PlayerLSL with an
uneven chunk size, then runs the whole chain in one loop:

    connect + preflight.prepare(...)  check rate/units/labels/types (B)
    Acquire.read()                     (built by StreamSession)
      -> StreamSession.ingest()        channel reorder + raw recording
      -> STAGES (this script's tuple)  repair, V->uV, quality observer, notch,
      |                                 band-pass, 500 -> 128 Hz
      -> CircularBuffer.push()         2 s windows @ 128 Hz
      -> StreamSession.wrap()          warm-up + judge (quality/repair) gating
      -> TaskOffloader (or in-loop) analysis

Each recorded run keeps a config snapshot and one line per delivered window
(D). Pass --record to keep the raw run. A run stopped by a guard, interrupted,
or hit by a worker failure is closed with status "failed", its error text and a
non-zero exit code, so a truncated session cannot be read as a clean one.

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
from nova2026.streaming import (  # runs per-window analysis
    Recovery,  # A2: reset-and-continue on unrepairable damage
    StreamStats,  # E: per-run counters persisted at close
    TaskOffloader,  # runs per-window analysis on worker threads
    UnrepairableError,  # damage the Repair stage cannot fix
    prepare,  # pre-flight source check -> the run's channel contract (B)
    processing_contract,  # serializable description of this run's chain
    resolve_outlet,  # confirm the outlet is publishing before connecting (B1)
)
from nova2026.streaming.bootstrap import StreamSession, parse_args

# Preprocessing building blocks are DEFINED here, in this script, on purpose.
from nova2026.streaming.preprocess import (
    QualityMonitor,  # observes raw EEG faults (amplitude/saturation/flatline)
    Repair,  # repairs short NaN/Inf runs before the filters see them
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
    # 2a) B1: confirm the outlet is actually publishing before we connect,
    #     so a missing amplifier fails in seconds instead of timing out.
    try:
        resolve_outlet(name=name, timeout=10)
    except RuntimeError as error:
        player.stop()
        raise SystemExit(f"outlet not found: {error}") from None

    stream = StreamLSL(bufsize=4.0, name=name)
    stream.connect(
        acquisition_delay=None,
        processing_flags=["clocksync"],
        timeout=10,
    )

    # 2b) Pre-flight the source BEFORE any sample is read: identity, sampling
    #     rate, numeric dtype, untouched state, labels, types and units (B).
    #     prepare() validates everything AND returns the channel contract the
    #     session will use to reorder every block.
    try:
        contract = prepare(
            stream,
            sfreq=args.sfreq,
            channels=CHANNELS,
            source_unit_exponent=SOURCE_UNIT_EXPONENT,
            n_eeg=len(CHANNELS),
            stream_name=name,
        )
    except RuntimeError as error:
        stream.disconnect()
        player.stop()
        raise SystemExit(f"source rejected: {error}") from None

    # ------------------------------------------------------------------
    # 3a) Define the preprocessing chain EXPLICITLY, right here in the script.
    #     Every stage has the same shape (data, ts) -> (data, ts), and the
    #     order below is exactly the order the loop runs them in. quality is
    #     an OBSERVER, so observe() wraps its feed into that same shape.
    # ------------------------------------------------------------------
    repair = Repair(args.sfreq, source_unit_exponent=SOURCE_UNIT_EXPONENT)
    scaler = unit_scaler(SOURCE_UNIT_EXPONENT, desired_exponent=-6)  # V -> uV
    quality = QualityMonitor(n_eeg=len(CHANNELS), sfreq=args.sfreq, warmup_seconds=0.0)

    def observe(data, timestamps):
        # Watch the raw uV signal here, before any filter can hide faults.
        quality.feed(data, timestamps)
        return data, timestamps

    notch = SosFilter(design_notch(args.notch, args.notch_q, args.sfreq), len(CHANNELS))
    bandpass = SosFilter(
        design_bandpass(args.lpass, args.hpass, args.order, args.sfreq), len(CHANNELS)
    )
    resampler = Resampler(
        args.sfreq, args.out_sfreq, len(CHANNELS), quality=args.resample_quality
    )
    # repair first (source units, before scaling and filters), observe right
    # after scaling so quality sees raw uV, filters, resampling last.
    STAGES = (repair, scaler, observe, notch, bandpass, resampler)

    # Serialized description of exactly this chain: the recorder stores it as
    # provenance so a later replay knows what produced this run (D).
    chain_config = processing_contract(
        eeg_channels=CHANNELS,
        eog_channels=(),
        out_sfreq=resampler.out_sfreq,
        stamp=(
            f"notch{args.notch:g}-q{args.notch_q:g}/"
            f"band{args.lpass:g}-{args.hpass:g}-o{args.order}/"
            f"soxr-{args.resample_quality}"
        ),
    )

    # ------------------------------------------------------------------
    # 3b) StreamSession assembles the REST (contract, recorder, acquire,
    #     buffer). It never runs STAGES: the loop below does. judges are
    #     only asked when wrap() gates a finished window.
    # ------------------------------------------------------------------
    session = StreamSession(
        stream,
        args,
        CHANNELS,
        judges=(quality, repair),
        out_sfreq=resampler.out_sfreq,
        source_unit_exponent=SOURCE_UNIT_EXPONENT,
        contract=contract,  # ChannelContract from prepare() (step 2b)
        recorder_config=chain_config,  # processing_contract snapshot (D1)
        recorder_track_windows=True,
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
        f"resample {args.sfreq:g}->{session.out_sfreq:g} Hz "
        f"(requested {args.resample_quality}, using {resampler.quality}, "
        f"startup delay {resampler.startup_delay_seconds:.2f}s)\n"
        f"buffer: window={session.window_samples} hop={session.hop_samples} "
        f"capacity={session.capacity_samples} @ {session.out_sfreq:g} Hz\n"
        f"analysis: compute={args.compute:g}s workers={args.workers} queue={args.queue}"
    )
    if session.recorder is not None:
        print(f"record : raw volts -> {session.recorder.path}  (sqlite + fif)")

    # ------------------------------------------------------------------
    # 5) Guards, counters and the per-window report function.
    # ------------------------------------------------------------------
    # A2: bounded recovery over every stateful stage of THIS chain. When the
    # Repair stage meets damage it cannot fix, the guard resets all of these
    # and keeps going (up to its limits); segment lets windows say which
    # restart they belong to. Keep persistent_fault_seconds above the window
    # length plus any judge settling: one bad sample invalidates every
    # overlapping window, so a short threshold can turn a single transient
    # into a "persistent" fault and stop the run.
    recovery = Recovery(
        resettable=(repair, quality, notch, bandpass, resampler, session.buffer),
        recorder=session.recorder,
    )
    stats = StreamStats()  # E: counters stored in the run metadata at close

    started = monotonic()  # wall clock for the duration loop and prints

    # Display one line per window: peak-to-peak amplitude per channel (uV).
    def report(eeg_window) -> None:
        peak_to_peak_uv = np.ptp(eeg_window.data, axis=0)
        print(
            f"window seg={eeg_window.segment} start={eeg_window.start_sample:6d}  "
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
    interrupted = False
    failure = None
    try:
        while monotonic() - started < args.duration:
            # (a) Pull the next raw block from LSL (blocks until ready).
            data, timestamps = session.acquire.read()
            stats.blocks += 1
            stats.samples += len(data)

            # (b) Contract reorder + save the raw volts before transformation.
            data, timestamps = session.ingest(data, timestamps)

            # (c) Run THIS script's chain over the block, in our own order.
            #     The Repair stage raises UnrepairableError on damage it
            #     cannot fix; bounded recovery (A2) resets the chain and lets
            #     the run continue, or stops it when the budget is exhausted.
            try:
                for stage in STAGES:
                    data, timestamps = stage(data, timestamps)
            except UnrepairableError as error:
                recovery.handle(error)  # raises when the run must stop
                stats.recoveries = recovery.recoveries
                continue  # this chunk is discarded

            stats.repairs = repair.repaired_samples
            stats.dropped = repair.dropped_rows

            # (d) Collect any windows this block completed.
            for window, window_times, start in session.buffer.push(data, timestamps):
                stats.windows += 1
                # (e) Package the window with its verdict (warm-up + judges).
                eeg_window = session.wrap(window, window_times, start)
                eeg_window.segment = recovery.segment
                recovery.watch(eeg_window)  # persistent-fault guard (A2)
                if session.recorder is not None:
                    # (e2) One provenance line per delivered window (D).
                    session.recorder.log_window(
                        float(window_times[-1]),
                        eeg_window.valid,
                        eeg_window.reasons,
                        eeg_window.segment,
                        eeg_window.artifact_id,
                    )
                if not eeg_window.valid:
                    stats.rejected += 1
                    continue
                stats.valid += 1

                # (f) Run the analysis in the loop, or hand it to workers.
                if offloader is None:
                    analyze(eeg_window)
                else:
                    offloader.submit(eeg_window)

    # ------------------------------------------------------------------
    # 7) Shutdown: release everything in the right order. A run that was
    #    stopped by a guard, interrupted, or hit a worker failure is filed as
    #    "failed" with its own error text: a truncated session must never be
    #    recorded as a clean one.
    # ------------------------------------------------------------------
    except KeyboardInterrupt:
        print("interrupted by user")
        interrupted = True
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised at the end
        print(f"run stopped: {error}")
        failure = error
    finally:
        # Samples still buffered toward the next block never become windows;
        # record them honestly before the acquire handle drops them (D4).
        tail = session.acquire.pending_samples
        session.acquire.close()  # detach callback, drop buffer
        if stream.connected:
            stream.disconnect()  # close the LSL inlet
        player.stop()  # stop the fake source
        if offloader is not None:
            offloader.close(drain=True, timeout=5.0)  # let tasks finish
            stats.offload_dropped = offloader.dropped
            stats.offload_failed = offloader.failed
            if failure is None and not interrupted:
                # Observed on the submitting thread, as TaskOffloader documents.
                try:
                    offloader.raise_error()
                except Exception as error:  # noqa: BLE001 - recorded below
                    failure = error
        if session.recorder is not None and tail:
            session.recorder.mark_not_processed(tail)
        # Persist the transport counters and run statistics (E).
        stats.max_lag = session.acquire.max_lag
        stats.gaps = session.acquire.gaps
        if interrupted:
            recorded = "interrupted by user"
        elif failure is not None:
            recorded = repr(failure)
        else:
            recorded = None
        session.close(
            status="completed" if recorded is None else "failed",
            error=recorded,
            stats=stats.to_dict(),
        )
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
    print(f"resampled_output={resampler.output_samples} @ {session.out_sfreq:g} Hz")
    print(
        f"windows: valid={stats.valid} rejected={stats.rejected} "
        f"buffer_windows={session.buffer.windows}\n"
        f"recovery: events={recovery.recoveries} segment={recovery.segment} "
        f"repairs={repair.repaired_samples} dropped={repair.dropped_rows}"
    )
    if offloader is not None:
        print(
            f"offload: submitted={offloader.submitted} completed={offloader.completed} "
            f"dropped={offloader.dropped} failed={offloader.failed}"
        )

    # ------------------------------------------------------------------
    # 9) A run that stopped early, was interrupted, or lost a worker must not
    #    exit zero: the persisted status and the exit code have to agree.
    # ------------------------------------------------------------------
    if failure is not None:
        raise failure
    if interrupted:
        raise SystemExit(130)


# ---------------------------------------------------------------------------
# Entry point: only runs the demo when executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
