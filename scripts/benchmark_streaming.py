"""Micro-benchmark of the streaming components (no LSL needed).

Times the CPU cost of one 0.1 s block through the real chain and the cost of
packaging one window into an EEGWindow. Run from the repository root:

    .venv/Scripts/python.exe -B -m scripts.benchmark_streaming

Context for the numbers: a 50-sample block at 500 Hz represents 0.1 s of data
(100 000 us of budget); a window arrives every 0.5 s (500 000 us of budget).
"""

import time
from time import perf_counter

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos

from nova2026.streaming import EEGWindow
from nova2026.streaming.circular_buffer import CircularBuffer
from nova2026.streaming.preprocess import (
    QualityMonitor,
    Resampler,
    SosFilter,
    unit_scaler,
)

SFREQ = 500.0
OUT_SFREQ = 128.0
BLOCK = 50  # samples per block (0.1 s)
CHANNELS = 64
WINDOW = 256  # 2 s @ 128 Hz
HOP = 64  # 0.5 s


def best_of(run, repeats: int = 3, steps: int = 3000) -> float:
    """Best average microseconds per call over ``repeats`` runs."""

    results = []
    for _ in range(repeats):
        start = perf_counter()
        run(steps)
        results.append((perf_counter() - start) * 1e6 / steps)
    return min(results)


def main() -> None:
    rng = np.random.default_rng(0)
    volts = rng.standard_normal((BLOCK, CHANNELS)) * 1e-5  # a raw block in volts
    times = np.arange(BLOCK) / SFREQ

    # ------------------------------------------------------------------
    # Rebuild the exact chain a session uses (no LSL, no threads).
    # ------------------------------------------------------------------
    scaler = unit_scaler(0, desired_exponent=-6)
    quality = QualityMonitor(n_eeg=CHANNELS, sfreq=SFREQ, warmup_seconds=0.0)
    sos_notch = tf2sos(*iirnotch(60.0, 30.0, fs=SFREQ))
    sos_band = butter(3, (1.0, 45.0), btype="bandpass", fs=SFREQ, output="sos")

    notch = SosFilter(sos_notch, CHANNELS)
    bandpass = SosFilter(sos_band, CHANNELS)
    resampler = Resampler(SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
    buffer = CircularBuffer(WINDOW, HOP, int(6 * OUT_SFREQ), sfreq=OUT_SFREQ,
                            n_channels=CHANNELS)

    def run_chain(steps: int) -> None:
        # Every step processes a FRESH raw block, as the real loop does; SoXR
        # may emit nothing for many calls, so feeding its output forward would
        # collapse the work to empty arrays.
        for _ in range(steps):
            data = volts
            ts = times
            data, ts = scaler(data, ts)
            quality.feed(data, ts)
            data, ts = notch(data, ts)
            data, ts = bandpass(data, ts)
            data, ts = resampler(data, ts)

    chain_us = best_of(run_chain, repeats=3, steps=4000)

    # ------------------------------------------------------------------
    # Ring buffer: push one output-rate block at a time (writes + copies).
    # ------------------------------------------------------------------
    def run_push(steps: int) -> None:
        data = np.zeros((HOP, CHANNELS))
        ts = np.arange(HOP) / OUT_SFREQ
        for _ in range(steps):
            buffer.push(data, ts)

    buffer_us = best_of(run_push, repeats=3, steps=4000)

    # ------------------------------------------------------------------
    # EEGWindow packaging: the only extra work introduced by the wrapper.
    # ------------------------------------------------------------------
    data = np.ones((WINDOW, CHANNELS))
    eog = np.empty((WINDOW, 0))
    window_times = np.arange(WINDOW) / OUT_SFREQ
    reasons = ()

    def run_wrap(steps: int) -> None:
        for _ in range(steps):
            EEGWindow(
                data=data[:, : CHANNELS],
                eog=eog,
                timestamps=window_times,
                valid=True,
                reasons=reasons,
                start_sample=256,
                channel_names=("F3",) * CHANNELS,
                available_at=1.0,
            )

    wrap_us = best_of(run_wrap, repeats=3, steps=20000)

    # Quality verdict lookup per window (the other half of wrap()).
    def run_reasons(steps: int) -> None:
        for _ in range(steps):
            quality.reasons(1000.0, 1000.0 + WINDOW / OUT_SFREQ)

    reasons_us = best_of(run_reasons, repeats=3, steps=20000)

    # ------------------------------------------------------------------
    # Report against the real-time budgets.
    # ------------------------------------------------------------------
    block_budget = 1e6 * BLOCK / SFREQ  # us of data per block (100 000)
    hop_budget = 1e6 * HOP / OUT_SFREQ  # us between windows (500 000)

    def row(name: str, us: float, budget: float) -> None:
        share = us / budget * 100.0
        print(f"{name:<34s} {us:9.2f} us   {share:6.2f} % of {budget:,.0f} us")

    print("streaming micro-benchmark (best of 3, includes numpy overhead)")
    print("-" * 78)
    row("chain per 0.1 s block (scale+q+filters+soxr)", chain_us, block_budget)
    row("ring buffer push (one 0.5 s hop)", buffer_us, hop_budget)
    row("EEGWindow packaging (one window)", wrap_us, hop_budget)
    row("quality verdict lookup (one window)", reasons_us, hop_budget)

    extra = wrap_us + reasons_us
    print("-" * 78)
    print(
        f"EEGWindow adds {extra:.1f} us per window -> "
        f"{extra / (hop_budget / 1000.0):.3f} % of the 0.5 s window budget"
    )


if __name__ == "__main__":
    main()
