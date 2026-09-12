"""Measure the timestamp grid of an LSL outlet against what ``Repair`` requires.

``Repair._check_grid`` accepts a step only when it is within
``tolerance_seconds`` of a whole number of samples, and raises on *any* step
below one sample regardless of the tolerance. So the question "is this source
usable" is answered by two numbers: how far steps stray from 1.0 sample, and how
many steps are compressed to <= 1.0.

Modes:
    --self-test          publish a deliberately jittered synthetic outlet via
                         :mod:`scripts.getlive.publish_raw` and measure it, so
                         the probe itself is validated and the effect of the LSL
                         post-processing flags is visible
    --name <outlet>      measure a live outlet (the real question on the rig)

Every mode measures the same outlet with each post-processing flag set, because
``dejitter`` is the candidate fix and it has to be shown to work before it is
wired into ``outlets.open_inlet``.

    python -B -m scripts.getlive.ts_check --self-test
    python -B -m scripts.getlive.ts_check --name UnicornRecorderRawDataLSLStream --sfreq 250

Read the two percentages, not the raw steps:
    off-grid%  steps ``Repair`` refuses at the package's tolerance
    fatal%     compressed steps that are also off-grid, which NO tolerance can
               accept; only a regularised grid (``dejitter``) can
A non-zero ``fatal%`` means "do not touch ``tolerance_seconds``": loosening the
tolerance cannot rescue a step that is below one sample.
"""

import argparse
import subprocess
import sys
import time

import numpy as np
from mne_lsl.lsl import StreamInlet, resolve_streams

# The package's own default (``Repair.tolerance_seconds``), which is what
# ``live._build_chain`` also passes at any rate below 2 kHz.
PACKAGE_TOLERANCE_SECONDS = 2e-4

FLAG_SETS = (
    ("clocksync", ("clocksync",)),
    ("+dejitter", ("clocksync", "dejitter")),
    ("all", "all"),
)

SELF_TEST_NAME = "ts-check-jittered"
SELF_TEST_SFREQ = 250.0
SELF_TEST_JITTER_SECONDS = 3e-4  # 0.3 ms, above the 0.2 ms leash


def measure(info, flags, sfreq: float, window: float) -> dict:
    """Pull samples through one inlet for a fixed time and describe the steps."""

    inlet = StreamInlet(info, max_buffered=10, processing_flags=flags)
    inlet.open_stream(timeout=10.0)
    try:
        time.sleep(1.0)  # let the clock-sync estimate settle
        collected: list[np.ndarray] = []
        total = 0
        deadline = time.time() + window
        while time.time() < deadline:
            _, stamps = inlet.pull_chunk(timeout=0.5, max_samples=200)
            if stamps.size:
                collected.append(np.asarray(stamps, dtype=float))
                total += int(stamps.size)
    finally:
        inlet.close_stream()

    if not collected:
        return {"flags": flags, "samples": 0}
    stamps = np.concatenate(collected)
    steps = np.diff(stamps) * sfreq
    # The first samples carry the clock-sync settling jump; they are reported
    # separately rather than allowed to dominate the percentiles.
    steps = steps[100:]
    if not steps.size:
        return {"flags": flags, "samples": int(stamps.size)}
    tolerance_steps = PACKAGE_TOLERANCE_SECONDS * sfreq
    error = np.abs(steps - np.round(steps))
    off_grid = error > tolerance_steps
    # A compressed step that is also off-grid cannot be rescued by any
    # tolerance: Repair refuses steps <= 1.0 once the tolerance test fails.
    fatal_compression = off_grid & (steps <= 1.0)
    return {
        "flags": flags,
        "samples": int(stamps.size),
        "min": float(steps.min()),
        "p1": float(np.percentile(steps, 1)),
        "median": float(np.median(steps)),
        "p99": float(np.percentile(steps, 99)),
        "max": float(steps.max()),
        "worst_error": float(error.max()),
        "off_grid": float(off_grid.mean() * 100),
        "fatal_compression": float(fatal_compression.mean() * 100),
        "wild": int((error > 0.5).sum()),
    }


def report(rows: list[dict], sfreq: float) -> None:
    """Print one line per flag set, and the reading of each column."""

    tolerance_steps = PACKAGE_TOLERANCE_SECONDS * sfreq
    print(
        f"\nsfreq {sfreq:g} Hz, tolerance {PACKAGE_TOLERANCE_SECONDS * 1e3:.2f} ms "
        f"= {tolerance_steps:.3f} samples"
    )
    header = (
        f"{'flags':<12}{'n':>6}{'p1':>9}{'med':>9}{'p99':>9}"
        f"{'worst|err|':>11}{'off-grid%':>11}{'fatal%':>8}{'wild':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row.get("samples") or "p1" not in row:
            print(f"{str(row['flags']):<12}  no samples arrived")
            continue
        print(
            f"{str(row['flags']):<12}{row['samples']:>6}{row['p1']:>9.4f}"
            f"{row['median']:>9.4f}{row['p99']:>9.4f}{row['worst_error']:>11.4f}"
            f"{row['off_grid']:>11.2f}{row['fatal_compression']:>8.2f}{row['wild']:>6d}"
        )
    print("\ncolumns are in samples (1.0 = one sample at the nominal rate):")
    print("  off-grid% = steps Repair refuses at the package's tolerance")
    print("  fatal%    = compressed steps that are also off-grid, which NO")
    print("              tolerance can accept; only a regularised grid can")
    print("  wild      = steps more than half a sample out (isolated outliers,")
    print("              e.g. clock-sync corrections)")


def start_self_test() -> subprocess.Popen:
    """Publish the jittered fixture and give it time to register on the network."""

    publisher = subprocess.Popen(
        [
            sys.executable, "-B", "-m", "scripts.getlive.publish_raw",
            "--name", SELF_TEST_NAME,
            "--source-id", "jittered-1",
            "--sfreq", str(SELF_TEST_SFREQ),
            "--channels", "4",
            "--jitter", str(SELF_TEST_JITTER_SECONDS),
            "--seconds", "60",
        ]
    )
    time.sleep(3.0)
    return publisher


def main() -> int:
    """Measure one outlet under each post-processing flag set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="LSL outlet name to measure")
    parser.add_argument("--source-id", help="LSL source id, when names collide")
    parser.add_argument("--sfreq", type=float, default=250.0, help="nominal rate")
    parser.add_argument("--window", type=float, default=4.0, help="seconds measured per flag set")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="measure the jittered synthetic outlet published by publish_raw.py",
    )
    args = parser.parse_args()

    publisher = None
    if args.self_test:
        print(
            f"self-test: publishing {SELF_TEST_SFREQ:g} Hz with "
            f"{SELF_TEST_JITTER_SECONDS * 1e3:.1f} ms of Gaussian jitter"
        )
        publisher = start_self_test()
        name, sfreq = SELF_TEST_NAME, SELF_TEST_SFREQ
    else:
        if not args.name:
            parser.error("--name is required unless --self-test is used")
        name, sfreq = args.name, args.sfreq

    try:
        infos = [
            info
            for info in resolve_streams(5.0)
            if info.name == name
            and (args.source_id is None or info.source_id == args.source_id)
        ]
        if not infos:
            print("No matching outlet found. Is it publishing?")
            return 1
        info = infos[0]
        print(
            f"measuring {info.name!r} source_id={info.source_id!r} "
            f"{info.n_channels} ch, declared {info.sfreq:g} Hz"
        )

        rows = [measure(info, flags, sfreq, args.window) for _, flags in FLAG_SETS]
        for (label, _), row in zip(FLAG_SETS, rows):
            row["flags"] = label
        report(rows, sfreq)
    finally:
        if publisher is not None:
            publisher.terminate()
            try:
                publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - cleanup path
                publisher.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
