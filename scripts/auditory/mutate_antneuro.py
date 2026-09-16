"""Show that each step-9.5 test fails under a named mutation, then restore the tree.

Plan section 6.4 requires it: a passing test is not yet evidence that it tests
anything, so every mutation below is applied to the *importer*, the named test is
run, and the test must go red. The mutation is reverted in a ``finally`` block and
the file's bytes are compared before and after, so a run that dies half way cannot
leave a broken tree behind silently.

Every test in ``scripts/auditory/tests/test_antneuro.py`` appears here, and every
mutation is a plausible mistake rather than a syntactic break - two of them
(``report the railed fraction over the wrong axis`` and ``measure the duration one
sample short``) are defects this step actually had.

    .venv\\Scripts\\python.exe -B scripts\\auditory\\mutate_antneuro.py

Exit code 0 only when every mutation was caught by the test it names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = "scripts.auditory.tests.test_antneuro"
MODULE = "scripts/auditory/antneuro.py"

MUTATIONS = (
    {
        "name": "swap two channels in the contract order",
        "find": '    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",',
        "replace": '    "T8", "P7", "Cz", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",',
        "test": "ChannelContractTests.test_the_contract_is_the_twenty_shared_electrodes_in_plan_order",
    },
    {
        "name": "select the contract channels by position instead of by name",
        "find": "    return [names.index(name) for name in expected]",
        "replace": "    return list(range(len(expected)))",
        "test": "ChannelContractTests.test_channels_are_selected_by_name_and_extras_are_dropped_by_name",
    },
    {
        "name": "ignore a contract channel the recording does not have",
        "find": "    if missing:",
        "replace": "    if False:",
        "test": "ChannelContractTests.test_a_missing_contract_channel_is_refused_by_name",
    },
    {
        "name": "accept a channels-by-samples array (MNE's own orientation)",
        "find": "    if data.ndim != 2 or data.shape[1] != len(source.channel_names):",
        "replace": "    if False:",
        "test": "ChannelContractTests.test_a_channel_by_samples_array_is_refused_rather_than_misread",
    },
    {
        "name": "scale volts by 1000 instead of 1e6",
        "find": 'UNIT_SCALES = {"V": 1e6,',
        "replace": 'UNIT_SCALES = {"V": 1e3,',
        "test": "UnitTests.test_volts_become_microvolts_and_the_trial_records_both_units",
    },
    {
        "name": "guess a scale for an unknown unit",
        "find": "    if source_unit not in UNIT_SCALES:",
        "replace": "    if False:",
        "test": "UnitTests.test_an_unknown_unit_is_refused_rather_than_scaled_by_guess",
    },
    {
        "name": "leave non-cue markers out of the table",
        "find": "        rows.append(",
        "replace": '        if disposition == "non-cue":\n            continue\n        rows.append(',
        "test": "MarkerTableTests.test_every_recorded_marker_appears_in_the_table",
    },
    {
        "name": "ignore the operator's exclusion table",
        "find": "        if exclusion is not None:",
        "replace": "        if False:",
        "test": "MarkerTableTests.test_the_operator_exclusions_are_applied_and_reported",
    },
    {
        "name": "treat the exclusion table's entries as optional",
        "find": "    if unmatched:",
        "replace": "    if False:",
        "test": "MarkerTableTests.test_an_exclusion_that_matches_no_marker_is_an_error",
    },
    {
        "name": "report a broken alternation as strict",
        "find": '    return {"ok": not breaks, "breaks": breaks}',
        "replace": '    return {"ok": True, "breaks": breaks}',
        "test": "MarkerTableTests.test_a_broken_alternation_is_recorded_and_not_hidden",
    },
    {
        "name": "exclude the buffer's endpoints from the unknown span",
        "find": "        labels[(times >= onset - buffer_seconds) & (times <= onset + buffer_seconds)] = -1",
        "replace": "        labels[(times > onset - buffer_seconds) & (times < onset + buffer_seconds)] = -1",
        "test": "LabelRuleTests.test_the_buffer_is_symmetric_and_includes_its_endpoints",
    },
    {
        "name": "accept a buffer outside the operator's range",
        "find": "    if not BUFFER_RANGE_SECONDS[0] <= buffer_seconds <= BUFFER_RANGE_SECONDS[1]:",
        "replace": "    if False:",
        "test": "LabelRuleTests.test_a_buffer_outside_the_operators_range_is_refused",
    },
    {
        "name": "label every sample, ignoring the buffers entirely",
        "find": (
            "    for onset in onsets:\n"
            "        labels[(times >= onset - buffer_seconds) & (times <= onset + buffer_seconds)] = -1"
        ),
        "replace": "    for onset in onsets:\n        pass",
        "test": "LabelRuleTests.test_cues_closer_than_twice_the_buffer_leave_no_label_for_the_first",
    },
    {
        "name": "treat the disputed right cue as excluded",
        "find": '                disposition, reason = "ambiguous-cue", ambiguous["reason"]',
        "replace": '                disposition, reason = "excluded", ambiguous["reason"]',
        "test": "LabelRuleTests.test_the_ambiguous_cue_labels_its_own_side_through_a_whole_session",
    },
    {
        "name": "report the conservative buffer as if it were the chosen one",
        "find": "    conservative = labels_from_cues(times, cues, BUFFER_RANGE_SECONDS[1])",
        "replace": "    conservative = labels",
        "test": "LabelRuleTests.test_the_conservative_buffer_is_reported_beside_the_chosen_one",
    },
    {
        "name": "place the audio without the Start anchor",
        "find": "    return start_offset_seconds + (np.asarray(times, dtype=float) - start_seconds)",
        "replace": "    return start_offset_seconds + np.asarray(times, dtype=float)",
        "test": "AudioPlacementTests.test_audio_is_placed_through_the_start_anchor_and_the_session_offset",
    },
    {
        "name": "sample the stimulus from before playback started",
        "find": "    if played_from is not None:\n        inside &= positions >= float(played_from)",
        "replace": "    if False:\n        inside &= positions >= float(played_from)",
        "test": "AudioPlacementTests.test_positions_before_playback_are_zero_and_counted_as_uncovered",
    },
    {
        "name": "record the playable range at 44.1 kHz",
        "find": "                int(round(start_offset * 48000)),",
        "replace": "                int(round(start_offset * 44100)),",
        "test": "AudioPlacementTests.test_the_playable_source_range_is_recorded_for_the_renderer",
    },
    {
        "name": "measure the duration one sample short",
        "find": "    duration = float(eeg.shape[0]) / rate",
        "replace": "    duration = float(times[-1] - times[0])",
        "test": "InterchangeTests.test_the_summary_reports_shape_duration_and_the_label_distribution",
    },
    {
        "name": "lose the trial's group identity",
        "find": '        group=f"antneuro|{session}",',
        "replace": "        group=None,",
        "test": "InterchangeTests.test_the_written_trial_loads_back_with_the_contract_and_its_labels",
    },
    {
        "name": "print only the first few markers in the review document",
        "find": '        for row in summary["markers"]["table"]:',
        "replace": '        for row in summary["markers"]["table"][:5]:',
        "test": "InterchangeTests.test_the_review_document_carries_every_marker_and_the_checks",
    },
    {
        "name": "guess an audio offset for a session the operator never gave one for",
        "find": "    if session not in SESSION_AUDIO_START_SECONDS:",
        "replace": "    if False:",
        "test": "RefusalTests.test_a_session_without_an_operator_offset_is_refused",
    },
    {
        "name": "name any file as a session",
        "find": (
            "        raise ValueError(\n"
            '            f"Cannot name the session from {Path(path).name!r}: expected the recording "\n'
            '            "to end in _HH-MM-SS.cnt."\n'
            "        )"
        ),
        "replace": '        return "19-34-06"',
        "test": "RefusalTests.test_an_unknown_file_name_cannot_name_a_session",
    },
    {
        "name": "use the left envelope for both candidates",
        "find": 'CANDIDATE_ENVELOPES = ("left_mono.npz", "right_mono.npz")',
        "replace": 'CANDIDATE_ENVELOPES = ("left_mono.npz", "left_mono.npz")',
        "test": "RefusalTests.test_a_missing_candidate_envelope_refuses_the_import_by_name",
    },
    {
        "name": "report the railed fraction over the wrong axis",
        "find": "    fractions = (np.abs(eeg) >= RAIL_PEAK_FRACTION * peak).mean(axis=0)",
        "replace": "    fractions = (np.abs(eeg) >= RAIL_PEAK_FRACTION * peak).mean(axis=1)",
        "test": "RailTests.test_a_railed_channel_is_measured_reported_and_not_removed",
    },
)


def digest(path: Path) -> str:
    """SHA256 of one file, so a mutation cannot be left behind unnoticed."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_test(test: str) -> tuple[int, str]:
    """Run one test and return its exit code and a one-line tail of its output."""

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", test],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tail = [
        line
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    return completed.returncode, " | ".join(tail[-3:])[:400]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/antneuro_mutations.json")
    args = parser.parse_args(argv)

    path = REPO / MODULE
    before = digest(path)
    baseline_code, baseline_tail = run_test(SUITE)
    if baseline_code != 0:
        print(f"the unmutated suite is already red; nothing to prove: {baseline_tail}")
        return 1
    print(f"baseline: {SUITE} is green")

    records = []
    for mutation in MUTATIONS:
        original = path.read_bytes()
        # ``newline=""`` on both sides: a text-mode round trip would translate
        # every LF into CRLF, which is a whole-file diff and a false alarm.
        source = path.read_text(encoding="utf-8", newline="")
        if mutation["find"] not in source:
            records.append(
                {
                    "mutation": mutation["name"],
                    "test": mutation["test"],
                    "caught": False,
                    "detail": "the mutation target was not found; the code moved",
                }
            )
            print(f"  [MISSED] {mutation['name']} -> the target moved")
            continue
        path.write_text(
            source.replace(mutation["find"], mutation["replace"], 1),
            encoding="utf-8",
            newline="",
        )
        try:
            code, tail = run_test(mutation["test"])
        finally:
            path.write_text(source, encoding="utf-8", newline="")
            restored = path.read_bytes() == original
        if not restored:
            print(f"FATAL: {MODULE} was not restored")
            return 2
        caught = code != 0
        records.append(
            {
                "mutation": mutation["name"],
                "test": mutation["test"],
                "caught": caught,
                "detail": tail,
            }
        )
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {mutation['name']} -> {mutation['test'].split('.')[-1]}")

    if digest(path) != before:
        print(f"FATAL: {MODULE} differs from its starting state")
        return 2
    report = {
        "kind": "step-9.5 mutation record; every named mutation must be caught",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "suite": SUITE,
        "module": MODULE,
        "baseline_green": baseline_code == 0,
        "mutations": records,
        "caught": sum(1 for record in records if record["caught"]),
        "total": len(records),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mutation record: {out}")
    missed = [record for record in records if not record["caught"]]
    print(f"{report['caught']}/{report['total']} mutations caught")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
