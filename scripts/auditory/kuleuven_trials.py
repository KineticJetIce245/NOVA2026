"""Measure converted KU Leuven trials against the frozen contract.

One function per question the audit asks: what one trial contains
(:func:`audit_trial`), whether it obeys each invariant
(:func:`invariant_results`), what a subject's trials add up to
(:func:`audit_subject`), and how trials distribute over the groups step 5 will
split on (:func:`build_group_map`). Nothing here writes to disk or decides the
verdict; that belongs to the caller.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from nova2026.streaming.preprocess.repair import grid_tolerance_samples

from .kuleuven_contract import (
    EXPECTED_AUDIO_RATE, EXPECTED_CHANNELS, EXPECTED_SAMPLE_RATE, canonical_group,
    canonical_story, group_key, usable_audio_samples,
)

# One rule per line, checked per trial and reported by name. They are the claims
# the conversion made (step 2) plus the step-4 obligations, not a wish list: a
# violation means a decoder trained on that trial would be measuring something
# other than what the contract says it measures.
INVARIANTS = (
    ("lengths_equal", "eeg rows == len(timestamps) == len(labels)"),
    ("channels_named", "64 columns whose stored names match metadata.json"),
    ("sample_rate_nominal", "observed sample rate is 128 Hz within 1e-6"),
    ("timestamps_increase", "every timestamp gap is strictly positive"),
    ("timestamps_on_grid", "timestamp jitter is inside Repair's grid tolerance"),
    ("labels_supported", "every label is unknown (-1), A (0) or B (1)"),
    ("labels_constant_within_trial", "exactly one distinct label value per trial"),
    ("label_names_a_candidate", "the trial's label is 0 or 1, not unknown"),
    ("two_candidates", "the candidate array has exactly two columns"),
    ("audio_rate_nominal", "the candidate rate is 44100 Hz"),
    ("audio_normalized", "every candidate sample lies within [-1, 1]"),
    ("finite", "no non-finite value in eeg, timestamps or candidates"),
    ("dry_names", "both candidates are *_dry.wav files"),
    ("audio_longer_than_eeg", "the candidate array is strictly longer than the EEG"),
    ("truncation_consistent", "the truncation rule reproduces the measured gap"),
)


def natural_subject_key(path):
    """Sort S1..S16 by number; string order would put S10 before S2."""

    suffix = "".join(character for character in Path(path).name if character.isdigit())
    return int(suffix) if suffix else 0


def audit_trial(path, names64):
    """Measure one converted trial and list the invariants it violates."""

    with np.load(path, allow_pickle=False) as archive:
        stored = json.loads(str(archive["metadata"]))
        eeg = np.asarray(archive["eeg"], dtype=float)
        timestamps = np.asarray(archive["timestamps"], dtype=float)
        labels = np.asarray(archive["labels"])
        audio = np.asarray(archive["audio"], dtype=float)

    samples, channels = eeg.shape
    audio_rate = float(stored["audio_rate"])
    gaps = np.diff(timestamps)
    rate = 1.0 / float(np.median(gaps)) if len(gaps) else float("nan")
    duration = float(timestamps[-1] - timestamps[0]) if samples > 1 else 0.0
    audio_seconds = len(audio) / audio_rate
    usable = usable_audio_samples(samples, rate, audio_rate)
    distinct = [int(value) for value in np.unique(labels)]
    candidates = [name for name in str(stored["group"]).split("|") if name]

    record = {
        "file": path.name,
        "subject": stored["subject"],
        "trial_id": stored["trial_id"],
        "index": int(path.stem.split("_")[-1]),
        "eeg_shape": [int(samples), int(channels)],
        "eeg_samples": int(samples),
        "eeg_channels": int(channels),
        "channel_names_match": list(stored["channel_names"]) == names64,
        "duration_seconds": duration,
        "sample_rate": rate,
        "sample_rate_error": abs(rate - EXPECTED_SAMPLE_RATE),
        "grid_deviation_samples": float(np.max(np.abs(gaps - 1.0 / rate)) * rate)
        if len(gaps) else float("inf"),
        "timestamps_increase": bool(len(gaps)) and bool(np.all(gaps > 0)),
        "timestamps_count": int(len(timestamps)),
        "label_count": int(len(labels)),
        "audio_shape": [int(len(audio)), int(audio.shape[1]) if audio.ndim > 1 else 0],
        "audio_rate": audio_rate,
        "audio_seconds": audio_seconds,
        "audio_minus_eeg_seconds": audio_seconds - duration,
        "usable_audio_samples": usable,
        "dropped_audio_seconds": (len(audio) - usable) / audio_rate,
        "labels": distinct,
        "labels_constant": len(distinct) == 1,
        "label_counts": {str(value): int(np.count_nonzero(labels == value))
                         for value in distinct},
        "eeg_abs_max": float(np.max(np.abs(eeg))),
        "audio_abs_max": float(np.max(np.abs(audio))),
        "eeg_finite": bool(np.all(np.isfinite(eeg))),
        "timestamps_finite": bool(np.all(np.isfinite(timestamps))),
        "audio_finite": bool(np.all(np.isfinite(audio))),
        "candidates": candidates,
        "group_stored": str(stored["group"]),
        "group_key": group_key(candidates),
        "canonical_group": canonical_group(stored["group"]),
        "stories": sorted(canonical_story(name) for name in candidates),
    }
    record["violations"] = [rule for rule, holds in
                            invariant_results(record).items() if not holds]
    return record


def invariant_results(record):
    """Whether each named invariant holds for one audited trial."""

    tolerance = grid_tolerance_samples(EXPECTED_SAMPLE_RATE)
    return {
        "lengths_equal": record["eeg_samples"] == record["timestamps_count"]
        == record["label_count"],
        "channels_named": record["eeg_channels"] == EXPECTED_CHANNELS
        and record["channel_names_match"],
        "sample_rate_nominal": record["sample_rate_error"] <= 1e-6,
        "timestamps_increase": record["timestamps_increase"],
        "timestamps_on_grid": record["grid_deviation_samples"] <= tolerance,
        "labels_supported": set(record["labels"]).issubset({-1, 0, 1}),
        "labels_constant_within_trial": record["labels_constant"],
        "label_names_a_candidate": record["labels"] in ([0], [1]),
        "two_candidates": record["audio_shape"][1] == 2,
        "audio_rate_nominal": record["audio_rate"] == EXPECTED_AUDIO_RATE,
        "audio_normalized": record["audio_abs_max"] <= 1.0,
        "finite": record["eeg_finite"] and record["timestamps_finite"]
        and record["audio_finite"],
        "dry_names": bool(record["candidates"])
        and all(name.endswith("_dry.wav") for name in record["candidates"]),
        "audio_longer_than_eeg": record["audio_minus_eeg_seconds"] > 0,
        # The rule in kuleuven_contract.usable_audio_samples against the measured
        # gap: the two may differ only by the sub-sample remainder of the span.
        "truncation_consistent": 0.0 <= record["audio_minus_eeg_seconds"]
        - record["dropped_audio_seconds"] <= 2.0 / record["audio_rate"],
    }


def audit_subject(directory, names64, limit=None):
    """Audit one subject's trials and summarise them."""

    paths = sorted(Path(directory).glob("trial_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    records = [audit_trial(path, names64) for path in paths]
    if not records:
        return {"trials": [], "rollup": None}
    label_counts = Counter()
    for record in records:
        label_counts.update(record["labels"])
    differences = [record["audio_minus_eeg_seconds"] for record in records]
    groups = {}
    for record in records:
        groups.setdefault(record["group_key"], []).append(record["index"])
    return {
        "trials": records,
        "rollup": {
            "trials": len(records),
            "durations_seconds": sorted({round(record["duration_seconds"], 6)
                                         for record in records}),
            "sample_rates": sorted({round(record["sample_rate"], 6)
                                    for record in records}),
            "audio_rates": sorted({round(record["audio_rate"], 6)
                                   for record in records}),
            "label_counts": {str(value): int(label_counts[value])
                             for value in sorted(label_counts)},
            "audio_minus_eeg_seconds": {"min": min(differences), "max": max(differences)},
            "eeg_abs_max": max(record["eeg_abs_max"] for record in records),
            "canonical_groups": len(groups),
            "stored_groups": len({record["group_stored"] for record in records}),
            "trials_per_group": {key: len(value) for key, value in groups.items()},
            "violations": sum(len(record["violations"]) for record in records),
        },
    }


def build_group_map(subjects):
    """How trials distribute over groups, and which stories each group holds."""

    per_subject = {}
    stories = {}
    for subject, entry in subjects.items():
        groups = {}
        stored = set()
        for record in entry["trials"]:
            groups.setdefault(record["group_key"], []).append(record["index"])
            stored.add(record["group_stored"])
            for name in record["candidates"]:
                story = canonical_story(name)
                slot = stories.setdefault(
                    story, {"subjects": [], "trials": 0, "raw_names": []}
                )
                if subject not in slot["subjects"]:
                    slot["subjects"].append(subject)
                slot["trials"] += 1
                if name not in slot["raw_names"]:
                    slot["raw_names"].append(name)
        per_subject[subject] = {
            "groups": groups,
            "distinct_groups": len(groups),
            "distinct_stored_groups": len(stored),
            "trials_per_group": {key: len(value) for key, value in groups.items()},
        }
    for slot in stories.values():
        slot["subject_count"] = len(slot["subjects"])
        slot["raw_names"] = sorted(slot["raw_names"])
    return {
        "rule": "group key = sorted candidate stories with the rep_ prefix folded away",
        "per_subject": per_subject,
        "stories": dict(sorted(stories.items())),
    }
