"""Derived per-trial features for step 5, keyed by the frozen contract.

Running the whole preprocessing chain for 320 trials once per fold is not
affordable, and re-deriving the windows per fold would let two folds disagree
about what a window contains. So the chain runs **once per trial per channel
contract** here, and the result is written to a derived file under
``datasets/auditory_features/<key>/`` (git-ignored; it is a dataset-derived
artifact and never enters git).

One cache file holds exactly what the decoder consumes:

``signal64`` / ``signal20``
    The chain's 64 Hz output in microvolts, as float32, for the 64-channel
    contract and for the 20 shared channels of the live contract. Each comes
    from an :class:`AuditoryProcessor` built for that contract, so the channel
    selection happened by name inside the chain and not by a column slice here.
``times``
    The 64 Hz source-time grid the windows carry.
``envelopes``
    The two reference envelopes aligned on that grid, read from the
    precomputed records under ``datasets/audio`` with
    :func:`~nova2026.auditory.envelopes.load_envelope` (which refuses a missing
    or mismatched record). No envelope is computed at run time.
``labels``
    The per-sample label, present for scoring only. Nothing in this module
    feeds it to the chain, and the decoder never sees it (see
    ``tests/test_aad_decoder.py``).
``valid_<contract>_<tag>``
    The window verdict of the grid whose length is ``<tag>`` seconds and whose
    hop is :data:`STEP_SECONDS`.

The chain runs at :data:`MAX_HISTORY`, and the shorter grids are derived from
the same signal by slicing, because a window is a slice of the processed stream
and its length changes nothing upstream of the buffer. What *does* depend on
the window is the verdict, so the verdicts are captured here, where the judges
still hold their fault history, rather than recomputed later from a signal that
no longer knows which electrode misbehaved.

The cache key is a digest of the frozen contract (``results/kuleuven_audit.md``
section 8): the ``AuditoryConfig``, the channel contracts and their indices,
the chain's stage and filter settings, the window grid and the envelope
generator. Changing any of them moves the key and leaves the old files in
place, so a contract change invalidates the cache instead of silently mixing
incompatible features.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryTrial
from nova2026.auditory.envelopes import GENERATOR_VERSION, load_envelope
from nova2026.auditory.streaming import AuditoryProcessor, stream_config

from .kuleuven_contract import (
    CONVERTED, ENVELOPES, REPO, SHARED_CHANNELS, canonical_story, group_key,
    read_metadata,
)

CACHE_VERSION = "1"
CACHE_ROOT = REPO / "datasets" / "auditory_features"
STAGE = "auditory-current-streaming-v2"
FILTER_ORDER = 3
RESAMPLE_QUALITY = "QQ"
MAX_HISTORY = 60.0
STEP_SECONDS = 1.0
HISTORIES = (5.0, 10.0, 30.0, 60.0)
CHUNK_SECONDS = 1.0
WARMUP_SECONDS = 2.0
CONTRACTS = ("64ch", "20ch")


def history_tag(history):
    """Stable column-name fragment for one window length."""

    return f"{history:g}s"


def sample_rate():
    """The chain's output rate, taken from the code that produces it."""

    return AuditoryConfig().sample_rate


def channel_contracts():
    """The two frozen column sets: the dataset order and the shared subset."""

    names = list(read_metadata()["channel_names"])
    return {"64ch": names, "20ch": [name for name, _ in SHARED_CHANNELS]}


def shared_indices():
    """KU Leuven columns of the 20 shared electrodes, in contract order."""

    return [index for _, index in SHARED_CHANNELS]


def key_material():
    """Everything a change in which must invalidate every cached feature."""

    config = AuditoryConfig()
    return {
        "cache_version": CACHE_VERSION,
        "auditory_config": config.to_dict(),
        "stage": STAGE,
        "filter_order": FILTER_ORDER,
        "resample_quality": RESAMPLE_QUALITY,
        "channel_contracts": channel_contracts(),
        "shared_indices": shared_indices(),
        "histories": list(HISTORIES),
        "step_seconds": STEP_SECONDS,
        "max_history": MAX_HISTORY,
        "chunk_seconds": CHUNK_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "envelope_generator_version": GENERATOR_VERSION,
    }


def cache_key(material=None):
    """The digest naming one cache generation, with the material behind it."""

    material = key_material() if material is None else material
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], material


def cache_directory(key=None):
    """Where one generation of the cache lives."""

    return CACHE_ROOT / (cache_key()[0] if key is None else key)


def cache_path(subject, trial_id, key=None):
    """One trial's cache file inside one generation."""

    return cache_directory(key) / str(subject) / f"{trial_id}.npz"


def window_grid(n_samples, history, step=STEP_SECONDS, rate=None):
    """Sample offsets of the windows a processor at ``history`` would emit.

    The buffer emits a full window as soon as it holds one and then every hop,
    so the grid is the arithmetic sequence that fits whole windows, walked from
    sample zero. Windows the trial is too short to hold are absent, exactly as
    they are from the chain: no partial window is ever emitted.
    """

    rate = sample_rate() if rate is None else rate
    length = round(history * rate)
    hop = round(step * rate)
    if length > n_samples:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_samples - length + 1, hop, dtype=np.int64)


def window_validity(n_samples, history, clean_spans, step=STEP_SECONDS, rate=None):
    """Which grid windows the chain would accept, and why not when it would not.

    A window is rejected while the run is still warming up, and it is rejected
    when a fault the judges saw lies inside it. ``clean_spans`` is the fault
    evidence captured at run time: an empty list means the whole recording was
    fault-free, which lets every window be accepted on the warm-up rule alone.
    A window overlapping a recorded fault span is refused even when the fault
    is narrower than the window, which can only ever drop windows the chain
    would have kept -- never admit one it would have refused.
    """

    rate = sample_rate() if rate is None else rate
    starts = window_grid(n_samples, history, step, rate)
    length = round(history * rate)
    warm_samples = round(WARMUP_SECONDS * rate)
    valid = np.zeros(len(starts), dtype=bool)
    reasons = []
    for position, start in enumerate(starts):
        accept = int(start) >= warm_samples
        overlaps = []
        for left, right, reason in clean_spans:
            if left < start + length and right >= start:
                accept = False
                overlaps.append(reason)
        valid[position] = accept
        reasons.append(tuple(dict.fromkeys(overlaps)))
    return starts, valid, reasons


class Features:
    """One trial's cached features, loaded and self-checked."""

    def __init__(self, path, arrays, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self.signals = {name: arrays[f"signal{name[:2]}"] for name in CONTRACTS}
        self.times = arrays["times"]
        self.envelopes = arrays["envelopes"]
        self.labels = arrays["labels"]
        self.valid = {
            (name, history_tag(history)): arrays[f"valid_{name}_{history_tag(history)}"]
            for name in CONTRACTS
            for history in metadata["histories"]
        }

    @property
    def subject(self):
        return self.metadata["subject"]

    @property
    def trial_id(self):
        return self.metadata["trial_id"]

    @property
    def group_key(self):
        return self.metadata["group_key"]

    @property
    def label(self):
        values = np.unique(self.labels)
        if len(values) != 1 or int(values[0]) not in (0, 1):
            raise ValueError(f"{self.path} does not carry one known label.")
        return int(values[0])

    @property
    def samples(self):
        return len(self.times)

    @property
    def channel_names(self):
        return self.metadata["channel_contracts"]

    @property
    def contract(self):
        return self.metadata["contracts"]

    def signal(self, contract):
        return self.signals[contract]

    def grid(self, contract, history):
        """``(starts, valid, reasons)`` for one contract and window length."""

        tag = history_tag(history)
        starts = window_grid(self.samples, history)
        valid = self.valid[(contract, tag)]
        if len(valid) != len(starts):
            raise ValueError(f"{self.path} holds {len(valid)} verdicts for {len(starts)} windows.")
        return starts, valid, self.metadata["fault_spans"][contract].get(tag, [])

    def windows(self, contract, history, only_valid=True, hop=None):
        """Yield ``(start, stop)`` sample ranges in chain order.

        ``hop`` defaults to the chain's own step, which is what the runtime
        emits. Training and scoring pass ``hop=history`` instead, which is the
        selection ``scripts/auditory/train.py::prepare`` makes: walk the
        chain's grid, keep a window once the previous kept window's history has
        passed, and note that a rejected window does not advance that point.
        The grid is therefore anchored at the first accepted window -- after the
        warm-up -- and not at sample zero. Consecutive kept windows share no
        sample, so one recording cannot be counted many times over.
        """

        hop = STEP_SECONDS if hop is None else float(hop)
        length = round(history * sample_rate())
        valid = self.valid[(contract, history_tag(history))]
        next_start = -np.inf
        for start, keep in zip(window_grid(self.samples, history), valid):
            if only_valid and not keep:
                continue
            if start >= next_start:
                yield int(start), int(start) + length
                next_start = int(start) + round(hop * sample_rate())


def _load_trial(path):
    """Load a converted trial without its (large, unused) candidate audio.

    The chain never reads ``trial.audio``: the reference envelopes come from the
    precomputed records, which is the path the live session takes as well. The
    stub therefore only satisfies the container's shape checks and is never fed
    anywhere.
    """

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        eeg = np.asarray(archive["eeg"], dtype=float)
        timestamps = np.asarray(archive["timestamps"], dtype=float)
        labels = np.asarray(archive["labels"], dtype=int)
    trial = AuditoryTrial(
        eeg,
        timestamps,
        np.zeros((2, 2)),
        float(metadata["audio_rate"]),
        labels,
        metadata["subject"],
        metadata["trial_id"],
        metadata["channel_names"],
        metadata["reference"],
        metadata["upstream_processing"],
        metadata["group"],
    )
    return trial, metadata


def _processors(trial, config):
    """One chain per channel contract, all at the longest window.

    ``check_channels=False`` is a run-policy choice, not a contract change: the
    chain still detects and still reports every electrode fault, it just does
    not reject a window for one. The default policy does reject, and on this
    dataset that is fatal rather than conservative -- KU Leuven carries
    excursions beyond the monitor's 500 uV amplitude limit on single channels,
    and a fault that persists past the recovery window stops the run outright
    (measured: S1/trial_003 raises after 122 s). The decoder is fitted on the
    signal, which no policy touches, and every trial records which policy
    produced its verdicts.
    """

    settings = stream_config(trial, config, MAX_HISTORY, STEP_SECONDS, check_channels=False)
    built = {"64ch": AuditoryProcessor(settings)}
    settings = stream_config(trial, config, MAX_HISTORY, STEP_SECONDS, check_channels=False)
    settings.eeg_channels = tuple(name for name, _ in SHARED_CHANNELS)
    built["20ch"] = AuditoryProcessor(settings, source_channels=trial.channel_names)
    return built


def _run_chains(trial, processors):
    """Feed one trial through every chain and return their windows."""

    rate = trial.sample_rate
    chunk = round(CHUNK_SECONDS * rate)
    windows = {name: [] for name in processors}
    for start in range(0, len(trial.eeg), chunk):
        block = (trial.eeg[start : start + chunk], trial.timestamps[start : start + chunk])
        for name, processor in processors.items():
            windows[name].extend(processor.feed(block))
    return windows


def _rebuild(windows, channels):
    """The full processed stream, reassembled from the windows that carry it.

    The length is read back from the windows instead of predicted from the input
    length: the resampler withholds its last second until the stream ends, so
    the chain covers slightly less than ``len(eeg) / 2`` samples. Predicting a
    longer stream would leave a hole at the end of every trial that no window
    ever fills, which is what this reconstruction refuses to accept.
    """

    if not windows:
        raise RuntimeError("The chain emitted no window at all.")
    samples = int(windows[-1].start_sample) + len(windows[-1].data)
    signal = np.full((samples, channels), np.nan)
    times = np.full(samples, np.nan)
    for window in windows:
        start = int(window.start_sample)
        stop = start + len(window.data)
        signal[start:stop] = window.data
        times[start:stop] = window.timestamps
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(times)):
        raise RuntimeError("The chain did not emit a window over every sample.")
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("The chain's time grid is not increasing.")
    return signal, times


def _fault_spans(windows):
    """Rejected spans of the longest grid, as ``(start, stop, reasons)``."""

    spans = []
    for window in windows:
        if window.reasons:
            spans.append((int(window.start_sample), int(window.start_sample + len(window.data)),
                          list(window.reasons)))
    return spans


def _envelope_grid(trial, times):
    """The two candidate envelopes sampled on the chain's own time grid."""

    candidates = [name for name in str(trial.group).split("|") if name]
    if len(candidates) != 2:
        raise ValueError("A trial must name exactly two candidate stories.")
    columns = []
    provenance = {}
    for candidate in candidates:
        story = canonical_story(candidate)
        record = load_envelope(ENVELOPES / f"{Path(story).stem}.npz", AuditoryConfig())
        values = np.asarray(record.envelope, dtype=float)[:, 0]
        columns.append(np.interp(times, record.timestamps, values))
        provenance[candidate] = {
            "path": str(record.metadata["source_path"]),
            "source_sha256": record.metadata["source_sha256"],
            "envelope_method": record.metadata["envelope_method"],
        }
    return np.column_stack(columns).astype(np.float32), provenance


def build_trial(path, key, material=None):
    """Run both chains over one converted trial and write its feature cache."""

    trial, stored = _load_trial(path)
    config = AuditoryConfig()
    windows = _run_chains(trial, _processors(trial, config))
    arrays = {}
    metadata = {
        "cache_version": CACHE_VERSION,
        "contract_key": key,
        "key_material": key_material() if material is None else material,
        "subject": trial.subject,
        "trial_id": trial.trial_id,
        "group_stored": trial.group,
        "group_key": group_key(str(trial.group).split("|")),
        "candidates": [name for name in str(trial.group).split("|") if name],
        "eeg_samples": int(len(trial.eeg)),
        "eeg_sha256": hashlib.sha256(np.ascontiguousarray(trial.eeg).tobytes()).hexdigest(),
        "sample_rate": float(trial.sample_rate),
        "audio_rate": float(trial.audio_rate),
        "channel_contracts": channel_contracts(),
        "contracts": {},
        "histories": list(HISTORIES),
        "step_seconds": STEP_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "lag_samples": config.lag_samples,
        "quality_policy": "check_channels=False (faults are reported, never rejecting)",
        "fault_spans": {},
        "envelopes": {},
        "windows": {},
    }
    for name, emitted in windows.items():
        signal, times = _rebuild(emitted, len(channel_contracts()[name]))
        samples = len(times)
        arrays[f"signal{name[:2]}"] = signal.astype(np.float32)
        if name == "64ch":
            arrays["times"] = times
            arrays["labels"] = np.asarray(trial.labels, dtype=np.int8)[
                np.searchsorted(trial.timestamps, times, side="right") - 1
            ]
            envelope, provenance = _envelope_grid(trial, times)
            arrays["envelopes"] = envelope
            metadata["envelopes"] = provenance
        spans = _fault_spans(emitted)
        metadata["fault_spans"][name] = {}
        metadata["contracts"][name] = {
            history_tag(history): dict(emitted[0].contract, window_seconds=history,
                                       step_seconds=STEP_SECONDS)
            for history in HISTORIES
        }
        for history in HISTORIES:
            starts, valid, _ = window_validity(samples, history, spans)
            arrays[f"valid_{name}_{history_tag(history)}"] = valid
            metadata["fault_spans"][name][history_tag(history)] = [
                [int(left), int(right), reason] for left, right, reason in spans
            ]
            metadata["windows"][f"{name}_{history_tag(history)}"] = {
                "windows": int(len(starts)), "accepted": int(valid.sum()),
                "warmup_seconds": WARMUP_SECONDS,
            }
    destination = cache_path(trial.subject, Path(path).stem, key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, metadata=json.dumps(metadata), **arrays)
    return metadata


def load(path, key=None, material=None):
    """Read one cache file and refuse one written under a different contract."""

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    expected = cache_key()[0] if key is None else key
    if metadata["contract_key"] != expected:
        raise ValueError(
            f"{path} was built for contract {metadata['contract_key']}, not {expected}."
        )
    if material is not None and metadata["key_material"] != material:
        raise ValueError(f"{path} was built from a different contract material.")
    return Features(path, arrays, metadata)


def build(subjects=None, trials=None, force=False, key=None, material=None, progress=None):
    """Build the cache for every converted trial; report what happened."""

    key, material = cache_key(material) if key is None else (key, material)
    roots = sorted(
        (path for path in CONVERTED.glob("S*") if path.is_dir()),
        key=lambda path: int(path.name[1:]),
    )
    if subjects:
        roots = [path for path in roots if path.name in set(subjects)]
    written = skipped = 0
    records = []
    for root in roots:
        paths = sorted(root.glob("trial_*.npz"))
        if trials:
            paths = paths[: int(trials)]
        for path in paths:
            destination = cache_path(root.name, path.stem, key)
            if destination.is_file() and not force:
                skipped += 1
            else:
                written += 1
                build_trial(path, key, material)
            records.append(str(destination))
            if progress is not None:
                progress(len(records), str(path))
    return {"key": key, "directory": str(cache_directory(key)), "written": written,
            "skipped": skipped, "files": records, "material": material}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", help="subject folders, e.g. S1 S2")
    parser.add_argument("--trials", type=int, help="first N trials per subject")
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    parser.add_argument("--key", action="store_true", help="print the cache key and exit")
    args = parser.parse_args()
    if args.key:
        key, material = cache_key()
        print(json.dumps({"key": key, "directory": str(cache_directory(key)),
                          "material": material}, indent=2))
        return
    summary = build(args.subjects, args.trials, args.force,
                    progress=lambda done, path: print(f"{done}: {path}", flush=True))
    print(f"key {summary['key']} -> {summary['directory']}")
    print(f"written {summary['written']}, skipped {summary['skipped']}")


if __name__ == "__main__":
    main()
