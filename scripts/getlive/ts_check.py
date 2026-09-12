"""Measure the timestamp grid of an LSL outlet against what ``Repair`` requires.

``Repair._check_grid`` accepts a step only when it is within
``tolerance_seconds`` of a whole number of samples, and raises on *any* step
below one sample regardless of the tolerance. So two questions decide whether a
source is usable:

1. how many steps are **compressed** below one sample - those are fatal at any
   tolerance, and they are what a chunk-stamped source produces in bulk;
2. how many steps are **off the integer grid** while remaining above one
   sample - those are the only ones a tolerance could rescue.

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

Reading the report:

* ``compressed%`` non-zero and large: the source is **chunk-stamped** - its
  outlet calls ``push_chunk`` with a scalar timestamp, which liblsl documents as
  "the acquisition timestamp of the last sample", so every sample of a chunk
  shares one stamp. The probe also measures the chunk size directly.
* ``zeroish%`` is the fingerprint of that: the share of steps shorter than half
  a sample. A source stamped per sample scores about the share of its jitter
  that lands below half a sample (a fraction of a percent), not tens of percent.
* ``fatal% = compressed% + roundoff%`` is exactly the share of steps ``Repair``
  refuses; a non-zero ``compressed%`` means loosening ``tolerance_seconds``
  cannot help, because the second branch of ``_check_grid`` rejects every
  sub-nominal step once the tolerance test fails.

When every flag set leaves ``fatal%`` above zero the source cannot be repaired
downstream: the fix belongs in the publisher (per-sample timestamps) or in an
adapter that rebuilds the grid from the chunk anchors.
"""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from mne_lsl.lsl import StreamInlet, resolve_streams

# The package's own default (``Repair.tolerance_seconds``), which is what
# ``live._build_chain`` also passes at any rate below 2 kHz.
PACKAGE_TOLERANCE_SECONDS = 2e-4

# A step shorter than this is treated as "the source did not advance its clock":
# the fingerprint of one timestamp shared by a whole chunk.
ZEROISH_STEPS = 0.5

# Above this share of sub-half-sample steps, a source is called chunk-stamped even
# when the chunks are not exactly equal stamps.
CHUNK_ZEROISH_PCT = 10.0

FLAG_SETS = (
    ("clocksync", ("clocksync",)),
    ("+dejitter", ("clocksync", "dejitter")),
    ("all", "all"),
)

SELF_TEST_NAME = "ts-check-jittered"
SELF_TEST_SFREQ = 250.0
SELF_TEST_JITTER_SECONDS = 3e-4  # 0.3 ms, above the 0.2 ms leash

# Number of leading steps dropped before the statistics: they carry the
# clock-sync settling jump and would otherwise dominate the percentiles.
SETTLE_STEPS = 100

# A histogram of every step measured, for the JSON report. Longer tails are
# clipped into the last bucket so a single wild step cannot distort the shape.
HISTOGRAM_MAX_STEPS = 4.0
HISTOGRAM_BUCKETS = 80


def silence_lsl_log() -> bool:
    """Lower liblsl's own verbosity, once the library is already loaded.

    The library's startup banner is printed while the DLL loads, before any
    Python code runs, so it cannot be suppressed from here; this only affects
    messages raised afterwards. Raises nothing: the symbol is looked up
    dynamically and a liblsl build without it keeps its default verbosity.

    Returns:
        Whether the log level was changed.
    """

    from mne_lsl.lsl.load_liblsl import lib

    try:
        lib.lsl_set_log_level.argtypes = [ctypes.c_int]
        lib.lsl_set_log_level.restype = None
        lib.lsl_set_log_level(2)  # 0=verbose 1=debug 2=info/warning 3=error
    except AttributeError:  # pragma: no cover - depends on the liblsl build
        return False
    return True


def pull_stamps(info, flags, window: float) -> np.ndarray:
    """Pull samples through one inlet for a fixed time and return their stamps."""

    inlet = StreamInlet(info, max_buffered=10, processing_flags=flags)
    inlet.open_stream(timeout=10.0)
    try:
        time.sleep(1.0)  # let the clock-sync estimate settle
        collected: list[np.ndarray] = []
        deadline = time.time() + window
        while time.time() < deadline:
            _, stamps = inlet.pull_chunk(timeout=0.5, max_samples=200)
            if stamps.size:
                collected.append(np.asarray(stamps, dtype=float))
    finally:
        inlet.close_stream()

    if not collected:
        return np.empty(0)
    return np.concatenate(collected)


def classify_steps(steps: np.ndarray, tolerance_steps: float) -> dict:
    """Split the steps into the three ways ``Repair`` judges them.

    Args:
        steps: Consecutive timestamp differences, in samples.
        tolerance_steps: ``Repair``'s tolerance expressed in samples.

    Returns:
        Counts and shares: ``compressed`` (below one sample, fatal at any
        tolerance), ``roundoff`` (above one sample but off the integer grid,
        the only repairable class), ``fatal`` (their union) and ``zeroish``
        (shorter than ``ZEROISH_STEPS``, the chunk-stamping fingerprint).
    """

    total = int(steps.size)
    if total == 0:
        return {}
    # A step below 1 - tol fails the tolerance test around 1.0 and then hits the
    # unconditional `steps <= 1.0` branch: no tolerance value can rescue it.
    compressed = steps < 1.0 - tolerance_steps
    # Rounding would snap a near-zero step to 0 and call it "on grid", which is
    # how a chunk-stamped source hides; only k >= 2 counts as a real shortfall.
    nearest = np.where(steps > ZEROISH_STEPS, np.round(steps), np.inf)
    roundoff = (np.abs(steps - nearest) > tolerance_steps) & ~compressed
    zeroish = steps < ZEROISH_STEPS
    fatal = compressed | roundoff

    def share(mask) -> float:
        return float(np.count_nonzero(mask) / total * 100.0)

    return {
        "steps": total,
        "compressed": int(np.count_nonzero(compressed)),
        "compressed_pct": share(compressed),
        "roundoff": int(np.count_nonzero(roundoff)),
        "roundoff_pct": share(roundoff),
        "fatal": int(np.count_nonzero(fatal)),
        "fatal_pct": share(fatal),
        "zeroish": int(np.count_nonzero(zeroish)),
        "zeroish_pct": share(zeroish),
        "negative": int(np.count_nonzero(steps < 0)),
    }


def chunk_structure(stamps: np.ndarray, sfreq: float) -> dict:
    """Measure how many samples share one timestamp, and what rate that implies.

    A source stamped per sample gives ``samples_per_chunk = 1``. A source that
    stamps a whole block once gives the block size, and the effective rate is the
    block size divided by the time between two block anchors - the independent
    check on ``--sfreq``, and the number to use before rebuilding a grid.

    A boundary is a gap of at least ``ZEROISH_STEPS`` samples, the same threshold
    the ``zeroish%`` column uses. It must not be an equality test: a chunk-stamped
    source often shares one timestamp only to within a few microseconds, so equal
    stamps are the special case, not the rule.
    """

    if stamps.size < 2:
        return {}
    gap_seconds = ZEROISH_STEPS / sfreq
    # The first sample of each chunk is where the source advanced its clock; the
    # samples that follow it share that stamp (exactly, or to within a few us).
    boundary = np.concatenate(([True], np.diff(stamps) >= gap_seconds))
    index = np.flatnonzero(boundary)
    if index.size < 2:
        return {"chunks": int(index.size)}
    per_chunk = np.diff(index)
    # Anchor on the first sample of each chunk, never on a per-sample step: on a
    # chunk-stamped source the within-chunk spacings are noise, while the anchor
    # spacing is the sampling interval.
    interval = np.median(np.diff(stamps[index]))
    size = float(np.median(per_chunk))
    effective = None if interval <= 0 else size / interval
    return {
        "chunks": int(index.size),
        "samples_per_chunk_median": size,
        "samples_per_chunk_min": int(per_chunk.min()),
        "samples_per_chunk_max": int(per_chunk.max()),
        "anchor_interval_seconds": float(interval),
        "anchor_rate_hz": None if interval <= 0 else 1.0 / interval,
        "effective_sfreq_hz": effective,
        "sfreq_ratio": None if effective is None else effective / sfreq,
    }


def step_histogram(steps: np.ndarray) -> dict:
    """Bucket the step distribution, clipping the tail, for the JSON report."""

    edges = np.linspace(0.0, HISTOGRAM_MAX_STEPS, HISTOGRAM_BUCKETS + 1)
    counts, _ = np.histogram(np.clip(steps, 0.0, HISTOGRAM_MAX_STEPS), bins=edges)
    return {
        "max_steps": HISTOGRAM_MAX_STEPS,
        "edges": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "below_zero": int(np.count_nonzero(steps < 0)),
    }


def measure(info, flags, sfreq: float, window: float, debug: bool = False) -> dict:
    """Measure one inlet, then describe its steps and its chunk structure."""

    stamps = pull_stamps(info, flags, window)
    row: dict = {"flags": flags, "samples": int(stamps.size)}
    if stamps.size < 2:
        return row
    steps = np.diff(stamps) * sfreq
    if debug:
        print(
            f"\n[debug] {flags}: first 8 stamps {np.round(stamps[:8], 6).tolist()}"
            f"\n[debug] first 16 steps {np.round(steps[:16], 4).tolist()}"
        )
    row["steps"] = int(steps.size)
    row.update(chunk_structure(stamps, sfreq))
    row["histogram"] = step_histogram(steps)

    # The settling head is dropped from the statistics, not from the counts.
    settled = steps[SETTLE_STEPS:] if steps.size > SETTLE_STEPS else steps
    row["dropped_settle_steps"] = int(steps.size - settled.size)
    if settled.size == 0:
        return row

    tolerance_steps = PACKAGE_TOLERANCE_SECONDS * sfreq
    row["tolerance_steps"] = tolerance_steps
    row.update(classify_steps(settled, tolerance_steps))
    row.update(
        {
            "min": float(settled.min()),
            "p1": float(np.percentile(settled, 1)),
            "median": float(np.median(settled)),
            "p99": float(np.percentile(settled, 99)),
            "max": float(settled.max()),
            "worst_error": float(np.abs(settled - np.round(settled)).max()),
            "wild": int(np.count_nonzero(np.abs(settled - np.round(settled)) > 0.5)),
        }
    )
    return row


def report(rows: list[dict], sfreq: float) -> None:
    """Print one line per flag set, and the reading of each column."""

    tolerance_steps = PACKAGE_TOLERANCE_SECONDS * sfreq
    print(
        f"\nsfreq {sfreq:g} Hz, tolerance {PACKAGE_TOLERANCE_SECONDS * 1e3:.2f} ms "
        f"= {tolerance_steps:.3f} samples"
    )
    header = (
        f"{'flags':<12}{'n':>6}{'med':>9}{'p99':>9}{'zeroish%':>10}"
        f"{'compressed%':>13}{'roundoff%':>11}{'fatal%':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row.get("steps") or "fatal_pct" not in row:
            print(f"{str(row['flags']):<12}{row.get('samples', 0):>6}  no steps to judge")
            continue
        print(
            f"{str(row['flags']):<12}{row['samples']:>6}{row['median']:>9.4f}"
            f"{row['p99']:>9.4f}{row['zeroish_pct']:>10.2f}"
            f"{row['compressed_pct']:>13.2f}{row['roundoff_pct']:>11.2f}"
            f"{row['fatal_pct']:>8.2f}"
        )
    print("\ncolumns are in samples (1.0 = one sample at the nominal rate):")
    print("  zeroish%    = steps shorter than half a sample: the source is")
    print("                advancing its clock once per chunk, not per sample")
    print("  compressed% = steps below one sample: Repair refuses them at ANY")
    print("                tolerance, so only a regularised grid can help")
    print("  roundoff%   = steps above one sample but off the integer grid: the")
    print("                only class a wider tolerance could rescue")
    print("  fatal%      = compressed% + roundoff% = what Repair actually rejects")
    _report_chunks(rows, sfreq)


def _report_chunks(rows: list[dict], sfreq: float) -> None:
    """Print the chunk structure measured for each flag set, when there is one."""

    interesting = [
        row for row in rows if row.get("samples_per_chunk_median") is not None
    ]
    if not interesting:
        return
    print("\nchunk structure (one timestamp shared by how many samples):")
    for row in interesting:
        size = row["samples_per_chunk_median"]
        span = (
            f"{row['samples_per_chunk_min']}..{row['samples_per_chunk_max']}"
            if row.get("samples_per_chunk_min") is not None
            else "?"
        )
        line = (
            f"  {str(row['flags']):<12} chunks={row['chunks']:<6} "
            f"samples/chunk={size:g} (min..max {span})"
        )
        if row.get("effective_sfreq_hz"):
            ratio = row["sfreq_ratio"]
            line += (
                f"  anchor rate={row['anchor_rate_hz']:7.2f} Hz"
                f"  effective={row['effective_sfreq_hz']:7.2f} Hz"
                f" ({ratio:+.3f}x of {sfreq:g} Hz)"
            )
        print(line)
    print(
        "  samples/chunk = 1 means the source stamps every sample; a larger\n"
        "  number is the block whose samples share one timestamp (exactly, or to\n"
        "  within a few microseconds). 'effective' is the independent rate estimate:\n"
        "  samples per chunk / anchor interval."
    )


def verdict(rows: list[dict], sfreq: float) -> str:
    """State what the measurements mean, in one short paragraph."""

    judged = [row for row in rows if row.get("fatal_pct") is not None]
    if not judged:
        return "No flag set produced enough samples to judge the grid."
    best = min(judged, key=lambda row: (row["fatal_pct"], row["compressed_pct"]))
    lines = [
        f"best flag set: {best['flags']!r} with fatal% {best['fatal_pct']:.2f} "
        f"(compressed {best['compressed_pct']:.2f}, roundoff "
        f"{best['roundoff_pct']:.2f})"
    ]

    # Stamp granularity is evidence, not inference. A median above one sample per
    # chunk is the block a source stamped once; a median of one with a high
    # zeroish% is the same damage with a few microseconds of spacing inside the
    # block, which is what a recorder that re-stamps a block does.
    sizes = [
        row["samples_per_chunk_median"]
        for row in judged
        if row.get("samples_per_chunk_median")
    ]
    block = max(sizes) if sizes else 0.0
    zeroish = max(row.get("zeroish_pct", 0.0) for row in judged)
    if block > 1.0:
        lines.append(
            f"the source is chunk-stamped: ~{block:g} samples share one timestamp,"
            " so Repair refuses the grid at every flag set and no tolerance helps."
            " Fix it in the publisher (per-sample timestamps, or a chunk size of"
            " 1), or in an adapter that rebuilds the grid from the chunk anchors."
        )
    elif zeroish >= CHUNK_ZEROISH_PCT:
        lines.append(
            f"the source is chunk-stamped too: {zeroish:.2f}% of steps are shorter"
            " than half a sample, but the samples inside a block differ by a few"
            " microseconds, so the exact-equality chunk size stays 1. The grid"
            " still has to be rebuilt from the block anchors."
        )
    elif best["compressed_pct"] > 0.0:
        lines.append(
            f"{best['compressed_pct']:.2f}% of steps stay below one sample without"
            " any chunking; that is the noise floor of a jittered source (samples"
            " stamped per sample), and Repair refuses only those steps: a few per"
            " thousand, not the grid."
        )
    if best["fatal_pct"] > 0.0:
        lines.append(
            "do NOT widen tolerance_seconds: the remaining steps are sub-nominal,"
            " which the second branch of _check_grid rejects unconditionally."
        )
    else:
        lines.append("this flag set leaves the grid `Repair` requires.")
    rates = [
        row["effective_sfreq_hz"]
        for row in judged
        if row.get("effective_sfreq_hz")
    ]
    if rates and (max(rates) - min(rates)) / max(rates) > 0.01:
        lines.append(
            "the flag sets disagree on the effective rate ("
            + ", ".join(f"{rate:.2f} Hz" for rate in rates)
            + "); dejitter fits its own rate, so take the `clocksync` anchor"
            " measurement as the device rate before rebuilding a grid."
        )
    return "\n".join(lines)


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
            "--seconds", "90",
        ]
    )
    time.sleep(3.0)
    return publisher


def build_report(args, rows: list[dict], sfreq: float, name: str, verdict_text: str) -> dict:
    """Assemble the machine-readable report, raw enough to re-analyse offline."""

    return {
        "sfreq": sfreq,
        "window_seconds": args.window,
        "tolerance_seconds": PACKAGE_TOLERANCE_SECONDS,
        "tolerance_steps": PACKAGE_TOLERANCE_SECONDS * sfreq,
        "settle_steps_dropped": SETTLE_STEPS,
        "outlet": name,
        "source_id": args.source_id,
        "self_test": bool(args.self_test),
        "rows": rows,
        "verdict": verdict_text,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def write_report(path: Path, report_dict: dict) -> None:
    """Write the JSON report, creating its parent directory when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")


def main() -> int:
    """Measure one outlet under each post-processing flag set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="LSL outlet name to measure")
    parser.add_argument("--source-id", help="LSL source id, when names collide")
    parser.add_argument("--sfreq", type=float, default=250.0, help="nominal rate")
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="seconds measured per flag set (three flag sets run in sequence)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="measure the jittered synthetic outlet published by publish_raw.py",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the measurements, per flag set, as JSON",
    )
    parser.add_argument(
        "--lsl-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep liblsl's own INFO output (it is silenced when possible)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the first raw stamps and steps of each flag set",
    )
    args = parser.parse_args()
    # liblsl's INFO banner is noise around a three-line report. It is emitted
    # once when the library is loaded, so this only silences messages raised
    # afterwards (including the per-inlet reconnection notices); the startup
    # banner itself cannot be suppressed from Python.
    if not args.lsl_log:
        silence_lsl_log()

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
        print(
            f"estimated {3 * (args.window + 1.0):.0f}s: three flag sets in "
            "sequence, 1s settle each"
        )

        rows = [
            measure(info, flags, sfreq, args.window, debug=args.debug)
            for _, flags in FLAG_SETS
        ]
        for (label, _), row in zip(FLAG_SETS, rows):
            row["flags"] = label
        report(rows, sfreq)
        verdict_text = verdict(rows, sfreq)
        print(f"\nverdict:\n  {verdict_text}")
    finally:
        if publisher is not None:
            publisher.terminate()
            try:
                publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - cleanup path
                publisher.kill()

    if args.json is not None:
        write_report(args.json, build_report(args, rows, sfreq, name, verdict_text))
        print(f"report -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
