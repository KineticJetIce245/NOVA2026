"""Verify the real-time package on a real EEGLAB recording (offline).

Run from the repository root:

    .venv/Scripts/python.exe -B scripts/verify_realdata.py

It loads a real COG-BCI ``.set`` recording, pushes it block by block through
the exact same preprocessing chain the demo uses (Repair -> uV -> quality ->
notch -> band-pass -> resampler -> ring -> window gate), and checks:

1. The run completes and produces windows (it "runs through").
2. The saved data is correct: the FIF export agrees with the SQLite chunks,
   and replaying the recorded chunks offline reproduces the live windows
   sample-for-sample (save + offline comparison).
3. Edge cases: a short NaN run is repaired identically online and offline; an
   irreparable burst is handled by bounded recovery (run continues, then stops
   once the recovery budget is exhausted).

No LSL or amplifier is involved: the file is the source, which is exactly the
offline-replay mode a recorded run uses later.
"""

import shutil
import sys
from pathlib import Path

import mne
import numpy as np

from nova2026.streaming import Recovery, UnrepairableError
from nova2026.streaming.circular_buffer import CircularBuffer
from nova2026.streaming.preprocess import (
    QualityMonitor,
    Repair,
    Resampler,
    SosFilter,
    design_bandpass,
    design_notch,
    unit_scaler,
)
from nova2026.streaming.recording import (
    RunRecorder,
    RunSpec,
    replay_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 250          # samples per pushed block
WINDOW, HOP = 2.0, 0.5
OUT_SFREQ = 128.0
WARMUP = 2.0

DATASET = ROOT / "datasets" / "COG-BCI" / "sub-01" / "ses-S1" / "eeg" / "RS_Beg_EO.set"


def detect_exponent(raw: mne.io.BaseRaw) -> int:
    """Guess the file's unit exponent from the data's peak magnitude.

    MNE usually hands EEGLAB data back in volts (values ~1e-4). If the values
    are already in microvolts the peak is ~1..1000; if millivolts, ~1e-3..1.
    The chain only needs the exponent consistent with the stored values.
    """

    peak = float(np.max(np.abs(raw.get_data())))
    if peak < 1e-2:
        return 0       # volts
    if peak < 1.0:
        return -3      # millivolts
    return -6          # microvolts


class Runner:
    """Push raw blocks through the demo chain and keep every finished window.

    Both the "live" pass and the "offline replay" pass use this exact class,
    so any difference between them is a real computation drift.
    """

    def __init__(self, channels: int, sfreq: float, exponent: int) -> None:
        self.repair = Repair(sfreq, source_unit_exponent=exponent)
        self.scaler = unit_scaler(exponent, desired_exponent=-6)
        self.quality = QualityMonitor(
            n_eeg=channels, sfreq=sfreq, warmup_seconds=0.0
        )
        self.notch = SosFilter(design_notch(60.0, 30.0, sfreq), channels)
        self.bandpass = SosFilter(design_bandpass(1.0, 45.0, 3, sfreq), channels)
        self.resampler = Resampler(sfreq, OUT_SFREQ, channels, quality="LQ")
        self.buffer = CircularBuffer(
            round(WINDOW * OUT_SFREQ),
            round(HOP * OUT_SFREQ),
            round(6.0 * OUT_SFREQ),
            sfreq=OUT_SFREQ,
            n_channels=channels,
        )
        self.warmup = round(WARMUP * OUT_SFREQ)
        self.records = []  # (window copy, start_sample, valid)

    def feed(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        data, timestamps = self.repair(data, timestamps)
        data, timestamps = self.scaler(data, timestamps)
        self.quality.feed(data, timestamps)
        data, timestamps = self.notch(data, timestamps)
        data, timestamps = self.bandpass(data, timestamps)
        data, timestamps = self.resampler(data, timestamps)
        for window, window_times, start in self.buffer.push(data, timestamps):
            start_t = float(window_times[0])
            end_t = float(window_times[-1])
            reasons = set(self.quality.reasons(start_t, end_t))
            reasons.update(self.repair.reasons(start_t, end_t))
            valid = start >= self.warmup and not reasons
            self.records.append((window.copy(), start, valid))

    def assert_parity(self, other: "Runner", label: str) -> None:
        """Raise AssertionError unless both runners produced identical windows."""

        if len(self.records) != len(other.records):
            raise AssertionError(
                f"{label}: window count differs "
                f"({len(self.records)} vs {len(other.records)})"
            )
        for (window, start, valid), (other_window, other_start, other_valid) in zip(
            self.records, other.records
        ):
            if start != other_start or valid != other_valid:
                raise AssertionError(f"{label}: window start/validity differ")
            if window.shape != other_window.shape or not np.allclose(
                window, other_window, rtol=1e-6, atol=1e-6
            ):
                raise AssertionError(f"{label}: window values differ")


def run_checks(data: np.ndarray, sfreq: float, exponent: int, folder: Path,
               label: str) -> tuple[int, int]:
    """One full pass over ``data``: live record + offline replay parity.

    Returns (windows, valid_windows) from the live pass.
    """

    channels = data.shape[1]
    times = np.arange(len(data)) / sfreq
    recorder = RunRecorder(
        folder, RunSpec("real", "check", f"run_{label}"), tuple(
            f"CH{i:02d}" for i in range(channels)
        ),
        sfreq, unit_exponent=exponent, dtype=np.float64, export_fif=True,
    )
    live = Runner(channels, sfreq, exponent)
    for start in range(0, len(data), BLOCK):
        block = data[start : start + BLOCK]
        recorder.write(block, times[start : start + BLOCK])
        live.feed(block, times[start : start + BLOCK])
    recorder.close(status="completed")

    # Offline replay of the recorded chunks through a fresh pipeline.
    offline = Runner(channels, sfreq, exponent)
    stored = []
    for chunk, chunk_times in replay_chunks(recorder.path):
        stored.append(chunk)
        offline.feed(chunk, chunk_times)
    live.assert_parity(offline, f"{label}: offline parity")

    # FIF round-trip: the exported file equals the SQLite chunks. MNE saves
    # FIF as float32, so compare against the same storage dtype.
    saved = np.concatenate(stored, axis=0)
    fif = mne.io.read_raw_fif(recorder.fif_path, preload=True, verbose=False)
    if not np.allclose(
        fif.get_data().T, saved.astype(np.float32),
        rtol=1e-6, atol=1e-9, equal_nan=True,
    ):
        raise AssertionError(f"{label}: FIF does not match the SQLite chunks")

    return len(live.records), sum(1 for _, _, valid in live.records if valid)


def main() -> int:
    print(f"source file: {DATASET}")
    if not DATASET.is_file():
        print("SKIP: dataset file not found")
        return 0

    raw = mne.io.read_raw_eeglab(DATASET, preload=True, verbose=False)
    raw.crop(tmax=10.0)  # a short clip keeps the check fast
    data = raw.get_data().T
    sfreq = float(raw.info["sfreq"])
    exponent = detect_exponent(raw)
    print(f"file: {data.shape[1]} channels x {data.shape[0]} samples "
          f"@ {sfreq:g} Hz, unit exponent {exponent}")

    folder = ROOT / ".tmp_tests" / "verify_realdata"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # 1+2) Clean real data: runs through, saves, and replays identically.
    windows, valid = run_checks(data, sfreq, exponent, folder, "clean")
    print(f"[1] runs through: {windows} windows, {valid} valid (after warm-up)")
    if windows == 0 or valid == 0:
        raise SystemExit(f"[1] FAIL: expected windows after warm-up, got {valid}")

    # 3a) Edge: a short NaN run is repaired identically online and offline.
    damaged = data.copy()
    row = int(len(data) * 0.4)
    damaged[row : row + 4, 0] = np.nan
    windows, valid = run_checks(damaged, sfreq, exponent, folder, "nan")
    print(f"[2] NaN edge: repaired run replays identically "
          f"({windows} windows, {valid} valid)")

    # 3b) Edge: an irreparable burst triggers bounded recovery, then the
    #     budget limit stops the run.
    channels = data.shape[1]
    times = np.arange(len(data)) / sfreq
    clip = data[: int(len(data) * 0.6)].copy()
    for row in (int(len(clip) * 0.3), int(len(clip) * 0.4)):
        clip[row : row + 60, :] = np.nan  # 60 rows > the 20 ms repair limit

    def feed_with_guard(runner, guard, clip):
        for start in range(0, len(clip), BLOCK):
            block = clip[start : start + BLOCK]
            try:
                runner.feed(block, times[start : start + BLOCK])
            except UnrepairableError as error:
                guard.handle(error)
        return runner, guard

    runner = Runner(channels, sfreq, exponent)
    guard = Recovery((runner.repair,), recorder=None, max_events=3)
    runner, guard = feed_with_guard(runner, guard, clip)
    if guard.recoveries < 1 or guard.segment < 1:
        raise SystemExit("[3] FAIL: bounded recovery did not trigger")
    if not runner.records:
        raise SystemExit("[3] FAIL: no windows after the recovery")
    print(f"[3] recovery edge: {guard.recoveries} recoveries, "
          f"segment {guard.segment}, processing continued")

    runner = Runner(channels, sfreq, exponent)
    guard = Recovery((runner.repair,), recorder=None, max_events=1)
    try:
        feed_with_guard(runner, guard, clip)
    except RuntimeError:
        print("[3] recovery edge: budget of 1 stops the run as expected")
    else:
        raise SystemExit("[3] FAIL: exhausted recovery budget did not stop")

    print("PASS: real-data run-through, save+offline parity and edge cases OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
