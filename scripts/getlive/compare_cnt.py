"""Check recorded LSL runs against the amplifier's own CNT recording.

During bring-up the control software writes its own ``.cnt`` file *while* the
same amplifier publishes LSL, so one signal reaches disk twice: once by the
control software, once by :mod:`scripts.getlive` through the relay. This module
answers the only question that matters about that pair -- **is what we recorded
the same signal the amplifier recorded?** -- by aligning the two and comparing
them sample by sample.

Alignment must not trust the clocks. The run directory name, the LSL timestamps
and the CNT header all describe the same session, but none of them is accurate
to the sample: the LSL grid is regridded in blocks and carries a slow rate
error (:func:`grid_deviation` measures it). The signals themselves are the only
reliable reference, so the offset is found by cross-correlating the **first
difference** of the two signals. Differencing removes the large DC offsets that
dominate raw EEG and would otherwise let a spurious alignment score highly: on
raw data a wrong lag still reaches ``r = 0.9995``, on differences the correct
lag is a lone peak at ``r = 1.0``.

Reading an ANT ``.cnt`` needs the ``antio`` package, which MNE delegates to
(``pip install antio``, or ``uv add antio``). A CNT already decoded to
``.npy`` plus a sidecar ``.json`` can be passed instead with ``--cnt-npy``,
which keeps this script usable where ``antio`` is not installed.

Run from the repository root:

    .venv/bin/python -B -m scripts.getlive.compare_cnt \\
        --cnt ../Lacroix_Flo_2026-09-12_17-13-32.cnt --out records/cnt_check.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

from nova2026.streaming.recording import iter_chunks

# The amplifier quantises to one LSB of 0.0078125 uV (the calibration factor in
# the CNT header), so the two files can differ by up to half an LSB purely from
# rounding the same count differently. The observed disagreement is exactly
# 0.0039 uV on every channel -- independent of that channel's magnitude, and
# far below one LSB -- so one LSB is the threshold for "the same samples".
AMP_LSB_UV = 0.0078125


def load_cnt(path: Path) -> tuple[np.ndarray, float, dict]:
    """Load an ANT ``.cnt`` file as microvolts.

    Args:
        path: Path to a ``.cnt`` file, or to a ``.npy`` holding an already
            decoded ``(n_channels, n_samples)`` array. For a ``.npy``, a
            ``<stem>_meta.json`` sidecar supplies the rate, labels and start
            time; without it only the rate can be reported.

    Returns:
        data: ``(n_channels, n_samples)`` float64 array in microvolts.
        sfreq: Sampling rate in Hz.
        meta: Channel labels, start time and amplifier metadata.

    Raises:
        RuntimeError: When reading a ``.cnt`` and ``antio`` is not installed.
    """

    if path.suffix == ".npy":
        data = np.load(path).astype(np.float64)
        sidecar = path.with_name(f"{path.stem}_meta.json")
        meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        return data, float(meta.get("sfreq", 0.0)), meta

    try:
        import mne
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Reading a .cnt needs mne and antio.") from error

    try:
        raw = mne.io.read_raw_ant(path, preload=True, verbose="ERROR")
    except RuntimeError as error:
        if "antio" in str(error):
            raise RuntimeError(
                "Reading an ANT .cnt needs the 'antio' package: pip install antio "
                "(or pass a decoded array with --cnt-npy)."
            ) from error
        raise

    # MNE scales microvolts to volts; multiply back so both sides are in uV,
    # the unit the recorder stores and the CNT header declares.
    data = raw.get_data() * 1e6
    meta = {"channels": list(raw.ch_names), "meas_date": str(raw.info["meas_date"])}
    return data, float(raw.info["sfreq"]), meta


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one recorded run, through the library's own reader.

    The stored dtype, the chunk order and the SQL all belong to
    :mod:`nova2026.streaming.recording`; this only reshapes them the way the
    comparison wants them. Reimplementing the query here is how the two readers
    would drift apart.

    Args:
        path: Path to ``<run>.sqlite``.

    Returns:
        data: ``(n_channels, n_samples)`` array, oldest sample first, or
            ``(None, None, None)`` for a run that recorded nothing.
        timestamps: First LSL timestamp of every chunk.
        sizes: Sample count of every chunk.
    """

    chunks = list(iter_chunks(path))
    if not chunks:
        return None, None, None
    data = np.concatenate([chunk for chunk, _ in chunks], axis=0).T
    sizes = np.array([chunk.shape[0] for chunk, _ in chunks])
    timestamps = np.array([first for _, first in chunks], dtype=float)
    return data, timestamps, sizes


def find_offset(record: np.ndarray, cnt: np.ndarray) -> tuple[int, float]:
    """Locate a recorded segment inside the CNT by first-difference correlation.

    Args:
        record: ``(n_channels, n_samples)`` recorded block.
        cnt: ``(n_channels, n_samples)`` reference recording.

    Returns:
        offset: Index in ``cnt`` of ``record``'s first sample.
        score: Normalised correlation at that offset; the correct offset
            scores ``1.0`` when the two files hold the same samples.
    """

    # Summing the differences over channels uses every electrode at once and
    # keeps the peak sharp even when one channel happens to be flat.
    reference = np.diff(cnt, axis=1).sum(axis=0)
    signal = np.diff(record, axis=1).sum(axis=0)
    products = correlate(reference, signal, mode="valid", method="fft")
    lags = correlation_lags(len(reference), len(signal), mode="valid")
    best = int(np.argmax(products))
    offset = int(lags[best])
    window = reference[offset : offset + len(signal)]
    denominator = np.sqrt((signal**2).sum() * (window**2).sum())
    return offset, float(products[best] / denominator) if denominator else 0.0


def grid_deviation(timestamps: np.ndarray, sizes: np.ndarray, sfreq: float):
    """Measure how far the LSL chunk stamps stray from a perfect sample grid.

    The recorder stores only each chunk's first timestamp and rebuilds a
    uniform grid from it. When the relay regrids a block-stamped source, the
    rebuilt grid runs slightly ahead of the sample count, which is the drift
    the live reports call "gaps". This quantifies it in samples.

    Returns:
        Peak-to-peak deviation in samples, or ``None`` when a grid cannot be
        rebuilt (empty run or unknown rate).
    """

    if timestamps is None or len(timestamps) < 2 or sfreq <= 0:
        return None
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    deviation = (timestamps - timestamps[0]) * sfreq - starts
    return float(deviation.max() - deviation.min())


def compare(run_dir: Path, cnt: np.ndarray, sfreq: float, names) -> dict:
    """Align and compare one recorded run against the CNT reference."""

    database = run_dir / f"{run_dir.name}.sqlite"
    if not database.is_file():
        return {"run": run_dir.name, "status": "missing-sqlite"}

    record, timestamps, sizes = load_run(database)
    if record is None:
        return {"run": run_dir.name, "status": "no-samples"}

    n_channels = min(record.shape[0], cnt.shape[0])
    offset, score = find_offset(record[:n_channels], cnt[:n_channels])
    length = min(record.shape[1], cnt.shape[1] - offset)

    # A run whose channel order disagrees with the CNT cannot be fixed by a
    # global threshold, so report the worst channel to make that failure legible.
    left = record[:n_channels, :length].astype(np.float64)
    right = cnt[:n_channels, offset : offset + length]
    difference = left - right
    per_channel = np.abs(difference).max(axis=1)
    worst = int(np.argmax(per_channel))
    result = {
        "run": run_dir.name,
        "status": "compared",
        "samples": int(record.shape[1]),
        "compared_samples": int(length),
        "offset": int(offset),
        "offset_score": round(score, 8),
        "rms_uv": float(np.sqrt((difference**2).mean())),
        "max_abs_uv": float(np.abs(difference).max()),
        "outliers": int((np.abs(difference) > AMP_LSB_UV).sum()),
        "worst_channel": (names[worst] if names and worst < len(names) else str(worst)),
        "grid_deviation_samples": grid_deviation(timestamps, sizes, sfreq),
    }
    return result


def print_table(reports) -> None:
    """Print one line per run: where it aligned and how far it differs."""

    columns = (
        f"{'run':24s} {'samples':>8s} {'offset':>8s} "
        f"{'score':>9s} {'rms uV':>10s} {'outliers':>9s}"
    )
    print(columns)
    print("-" * len(columns))
    for report in reports:
        if report["status"] != "compared":
            print(f"{report['run']:24s} {report['status']}")
            continue
        print(
            f"{report['run']:24s} {report['samples']:8d} {report['offset']:8d} "
            f"{report['offset_score']:9.6f} {report['rms_uv']:10.5f} "
            f"{report['outliers']:9d}"
        )


def main(argv=None) -> int:
    """Compare every recorded run under ``--records`` against ``--cnt``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cnt", type=Path, help="ANT .cnt recording.")
    source.add_argument("--cnt-npy", type=Path, help="Decoded .npy + _meta.json.")
    parser.add_argument("--records", type=Path, default=Path("records"))
    parser.add_argument("--out", type=Path, help="Write the report as JSON here.")
    args = parser.parse_args(argv)

    cnt, sfreq, meta = load_cnt(args.cnt or args.cnt_npy)
    names = meta.get("channels") or meta.get("names") or []
    start = meta.get("meas_date")
    start = datetime.fromisoformat(start) if isinstance(start, str) else None

    print(f"reference: {cnt.shape[0]} channels x {cnt.shape[1]} samples @ {sfreq:g} Hz")
    if start is not None:
        print(f"starts    : {start.isoformat()}")

    reports = []
    for database in sorted(args.records.glob("*/*/*/*.sqlite")):
        # exFAT volumes carry an AppleDouble "._name" companion per file; those
        # are not SQLite databases and must not be read as runs.
        if database.name.startswith("._"):
            continue
        report = compare(database.parent, cnt, sfreq, names)
        if report.get("offset") is not None and start is not None:
            first = start + timedelta(seconds=report["offset"] / sfreq)
            report["offset_utc"] = first.isoformat()
        reports.append(report)

    print()
    print_table(reports)

    compared = [r for r in reports if r["status"] == "compared"]
    matched = [r for r in compared if r["outliers"] == 0]
    print()
    print(f"{len(matched)}/{len(compared)} runs match the CNT sample for sample.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=1) + "\n")
        print(f"report written to {args.out}")
    return 0 if compared and len(matched) == len(compared) else 1


if __name__ == "__main__":
    sys.exit(main())
