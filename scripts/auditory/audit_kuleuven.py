"""Audit every converted KU Leuven trial and freeze what step 5 must obey.

    .venv\\Scripts\\python.exe -B -m scripts.auditory.audit_kuleuven

Reads ``datasets/AAD-KULeuven/converted/S*/trial_*.npz`` read-only, prints the
per-subject table, the group map and the frozen contract to stdout, and writes
``results/kuleuven_audit.md`` plus ``results/kuleuven_audit_<stamp>.json``.

The audit exists to falsify things before a decoder is trained on them: the
pre-registered expectations (20 trials per subject, durations inside a stated
range, both label values), the step-4 revisions (labels constant within a trial,
``rep_*`` folded into its base group, the ``source_unit_exponent`` convention),
and every invariant the conversion claimed. A failed check is reported, not
raised -- the report must be written either way, and the exit code carries the
verdict so a scripted caller cannot mistake a violation for success.

A partial run (``--subjects``/``--limit``) is allowed while iterating and marks
itself incomplete in both outputs, so a partial table can never be quoted as a
full audit.
"""

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .kuleuven_contract import (
    CONVERTED, METADATA_PATH, PLAUSIBLE_DURATION_SECONDS, RESULTS,
    channel_subset_report, contract_snapshot, envelope_snapshot, read_metadata,
    rep_identity,
)
from .kuleuven_report import render_markdown
from .kuleuven_trials import INVARIANTS, audit_subject, build_group_map, natural_subject_key

# The pre-registration is quoted in the report: what was expected before the
# audit ran, not what turned out to be convenient afterwards.
PREREGISTERED = (
    ("twenty_trials_per_subject", "the dataset's published 16 x 20 schedule"),
    ("durations_in_plausible_range", "durations inside the stated plausible range"),
    ("both_label_values_per_subject", "each subject attends both candidates"),
    ("labels_constant_within_trial", "one instruction per trial, so one label"),
    ("rep_folds_into_its_base_group", "rep_* and its base share one group"),
    ("unit_exponent_parameterized", "the chain can express uV and V sources"),
)


def collect(subjects):
    """Roll the per-trial records up into the cross-subject measurements."""

    records = [record for entry in subjects.values() for record in entry["trials"]]
    differences = sorted(
        ({"seconds": record["audio_minus_eeg_seconds"],
          "trial": f"{record['subject']}/{record['file']}",
          "duration_seconds": record["duration_seconds"],
          "audio_seconds": record["audio_seconds"]} for record in records),
        key=lambda item: item["seconds"],
    )
    violations = {}
    for rule, _ in INVARIANTS:
        offenders = [f"{record['subject']}/{record['file']}" for record in records
                     if rule in record["violations"]]
        if offenders:
            violations[rule] = offenders
    histogram = Counter(f"{round(record['duration_seconds'], 6):g}" for record in records)
    return {
        "records": records,
        "totals": {
            "subjects": len(subjects),
            "trials": len(records),
            "eeg_samples": sum(record["eeg_samples"] for record in records),
            "audio_samples": sum(record["audio_shape"][0] for record in records),
            "sample_rates": sorted({round(record["sample_rate"], 6)
                                    for record in records}),
            "audio_rates": sorted({round(record["audio_rate"], 6)
                                   for record in records}),
            "bytes_on_disk": sum(record["bytes_on_disk"] for record in records),
            "audio_not_longer_than_eeg": [
                f"{record['subject']}/{record['file']}" for record in records
                if record["audio_minus_eeg_seconds"] <= 0
            ],
        },
        "extremes": {"max": differences[-1], "min": differences[0]},
        "duration_histogram": dict(
            sorted(histogram.items(), key=lambda item: float(item[0]))
        ),
        "nonfinite": {
            key: [f"{record['subject']}/{record['file']}" for record in records
                  if not record[f"{key}_finite"]]
            for key in ("eeg", "timestamps", "audio")
        },
        "violations": violations,
    }


def preregister(subjects, records, unit_measured, unit_detail):
    """The pre-registered expectations plus the three step-4 obligations."""

    low, high = PLAUSIBLE_DURATION_SECONDS
    counts = {subject: entry["rollup"]["trials"] for subject, entry in subjects.items()}
    labels = {subject: set(entry["rollup"]["label_counts"])
              for subject, entry in subjects.items()}
    missing = sorted(subject for subject, values in labels.items()
                     if not {"0", "1"}.issubset(values))
    durations = {round(record["duration_seconds"], 6) for record in records}
    repeats = [record for record in records
               if any(name.startswith("rep_") for name in record["candidates"])]
    not_constant = [f"{record['subject']}/{record['file']}" for record in records
                    if not record["labels_constant"]]
    checks = {
        "twenty_trials_per_subject": bool(counts)
        and all(count == 20 for count in counts.values()),
        "durations_in_plausible_range": bool(durations)
        and all(low <= value <= high for value in durations),
        "both_label_values_per_subject": bool(labels)
        and all({"0", "1"}.issubset(values) for values in labels.values()),
        "labels_constant_within_trial": bool(records) and not not_constant,
        "rep_folds_into_its_base_group": bool(repeats) and all(
            set(record["stories"]) == set(record["canonical_group"].split("|"))
            and set(record["group_key"].split("|")) == set(record["stories"])
            for record in repeats
        ),
        "unit_exponent_parameterized": unit_measured,
    }
    detail = {
        "twenty_trials_per_subject": f"trial counts per subject: {sorted(set(counts.values()))}",
        "durations_in_plausible_range": f"{len(durations)} distinct durations inside "
        f"[{low:g}, {high:g}] s: {sorted(durations)}",
        "both_label_values_per_subject": "subjects missing a label value: "
        f"{missing}",
        "labels_constant_within_trial": f"{len(not_constant)} trial(s) with more than one "
        f"label value {not_constant[:5]}",
        "rep_folds_into_its_base_group": f"{len(repeats)} trial(s) present a rep_ candidate; "
        f"canonical group keys: {sorted({record['group_key'] for record in records})}",
        "unit_exponent_parameterized": unit_detail,
    }
    return checks, detail


def unit_evidence(snapshot):
    """Whether the chain expresses both unit conventions, and how."""

    exponent = snapshot["source_unit_exponent_default"]
    to_uv = snapshot["repair_to_uv_default"]
    ant = snapshot["unit_convention"]["repair_to_uv_at_ant"]
    measured = exponent is not None and to_uv == 1.0 and ant == 1e6
    detail = (f"default exponent {exponent} -> Repair._to_uv {to_uv}; "
              f"exponent 0 -> {ant}" if measured
              else "not measured: the chain could not be built from a converted trial")
    return measured, detail


def console_report(report):
    """The stdout summary: per-subject table, groups, channels, verdict."""

    lines = [f"{'subject':<8} {'trials':>6} {'labels 0/1':>10} {'audio-EEG s':>21} "
             f"{'durations s':>22} groups"]
    for subject, entry in report["subjects"].items():
        rollup = entry["rollup"]
        counts = rollup["label_counts"]
        window = rollup["audio_minus_eeg_seconds"]
        durations = rollup["durations_seconds"]
        span = f"{durations[0]:.3f}..{durations[-1]:.3f} ({len(durations)})"
        lines.append(
            f"{subject:<8} {rollup['trials']:>6} "
            f"{counts.get('0', 0):>4}/{counts.get('1', 0):<5} "
            f"{window['min']:>9.3f} .. {window['max']:<8.3f} "
            f"{span:>22} {rollup['canonical_groups']}"
        )
    totals = report["totals"]
    lines.append(f"total: {totals['trials']} trials, {totals['gib_on_disk']:.2f} GiB, "
                 f"rates {totals['sample_rates']} Hz EEG / "
                 f"{totals['audio_rates']} Hz candidates")
    example = next(iter(report["group_map"]["per_subject"]))
    lines.append(f"\ngroup map (canonical key -> trial indices; {example} shown):")
    for key, indices in report["group_map"]["per_subject"][example]["groups"].items():
        lines.append(f"  {key:<48} {len(indices):>2} trials {indices}")
    groups = report["group_map"]["per_subject"].values()
    lines.append("canonical groups per subject: "
                 f"{sorted({entry['distinct_groups'] for entry in groups})}, "
                 "stored groups per subject: "
                 f"{sorted({entry['distinct_stored_groups'] for entry in groups})}")
    lines.append(f"stories: {sorted(report['group_map']['stories'])}")
    channels = report["channels"]
    lines.append(f"\n20-channel subset indices: {channels['shared_indices']}")
    lines.append(f"names at those indices:    {channels['shared_observed_at_indices']}")
    lines.append("channel subset matches metadata.json order: "
                 f"{'PASS' if channels['shared_matches'] else 'FAIL'}")
    lines.append("live cap channel set equals the 20 shared + 4 live-only: "
                 f"{channels['live_cap_set_matches']}")
    extremes = report["extremes"]
    lines.append(f"\nextremes: audio-EEG {extremes['min']['seconds']:.6f} s on "
                 f"{extremes['min']['trial']} .. {extremes['max']['seconds']:.6f} s on "
                 f"{extremes['max']['trial']}")
    nonfinite = {key: len(value) for key, value in report["nonfinite"].items()}
    violations = {rule: len(trials)
                  for rule, trials in report["invariants"]["violations"].items()}
    lines.append(f"non-finite: {nonfinite}")
    lines.append(f"invariant violations: {violations}")
    lines.append("\npre-registered expectations and step-4 obligations:")
    for name, passed in report["preregistered"].items():
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name} "
                     f"({dict(PREREGISTERED)[name]}): "
                     f"{report['preregistered_detail'][name]}")
    return "\n".join(lines)


def write_report(report, out):
    """Write both products; the JSON is written first so a renderer bug still
    leaves the machine-readable record on disk."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / report["json_name"]
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = out / "kuleuven_audit.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return markdown_path, json_path


def main(argv=None):
    """Audit the converted trials, write both reports, return the verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", help="comma-separated subjects for a partial run")
    parser.add_argument("--limit", type=int, help="trials per subject for a partial run")
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--dry-run", action="store_true", help="write no report files")
    args = parser.parse_args(argv)
    started = time.monotonic()
    names64 = list(read_metadata(METADATA_PATH)["channel_names"])
    wanted = None
    if args.subjects:
        wanted = {name.strip() for name in args.subjects.split(",") if name.strip()}
    directories = [path for path in sorted(CONVERTED.glob("S*"), key=natural_subject_key)
                   if path.is_dir() and (wanted is None or path.name in wanted)]
    if not directories:
        print(f"No converted subjects under {CONVERTED}; run step 2 first.")
        return 2

    subjects, empty = {}, []
    for directory in directories:
        entry = audit_subject(directory, names64, args.limit)
        if entry["rollup"] is None:
            empty.append(directory.name)
            continue
        subjects[directory.name] = entry
        print(f"audited {directory.name}: {entry['rollup']['trials']} trials "
              f"({time.monotonic() - started:.1f}s elapsed)")
    if not subjects:
        print(f"Converted subject directories hold no trials: {empty}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    command = "python -B -m scripts.auditory.audit_kuleuven"
    if args.subjects:
        command += f" --subjects {args.subjects}"
    if args.limit:
        command += f" --limit {args.limit}"
    first_subject = next(iter(subjects))
    example = subjects[first_subject]["trials"][0]
    # Trial size is a disk fact, measured from the files rather than inferred
    # from the arrays that were just read.
    for subject, entry in subjects.items():
        for record in entry["trials"]:
            record["bytes_on_disk"] = (CONVERTED / subject / record["file"]).stat().st_size
    collected = collect(subjects)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step": "final_connection.md step 4 - data audit and contract freeze",
        "command": command,
        "json_name": f"kuleuven_audit_{stamp}.json",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "selection": {"subjects": list(subjects), "limit": args.limit,
                      "complete": args.subjects is None and args.limit is None
                      and not empty,
                      "subjects_without_trials": empty},
        "subjects": subjects,
        "channels": channel_subset_report(names64),
        "contract": contract_snapshot(CONVERTED / first_subject / example["file"]),
        "envelope": envelope_snapshot(),
        "rep_identity": rep_identity(),
        "totals": collected["totals"],
        "extremes": collected["extremes"],
        "duration_histogram": collected["duration_histogram"],
        "nonfinite": collected["nonfinite"],
        "invariants": {"rules": [rule for rule, _ in INVARIANTS],
                       "violations": collected["violations"]},
        "group_map": build_group_map(subjects),
    }
    report["totals"]["gib_on_disk"] = report["totals"]["bytes_on_disk"] / 2 ** 30
    measured, detail = unit_evidence(report["contract"])
    report["preregistered"], report["preregistered_detail"] = preregister(
        subjects, collected["records"], measured, detail
    )

    print("\n" + console_report(report))
    if not args.dry_run:
        markdown_path, json_path = write_report(report, args.out_dir)
        print(f"\nwrote {markdown_path} and {json_path}")
    print(f"done in {time.monotonic() - started:.1f}s")
    failed = (not report["channels"]["shared_matches"] or bool(empty)
              or bool(collected["violations"])
              or not all(report["preregistered"].values()))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
