"""Render the committed markdown audit report from the measured record.

One section per question, each returning its own block of markdown, so the
report can be read (and reviewed) out of order and a section can be checked
without re-reading the whole. All numbers are interpolated from the audit's JSON
record: a figure typed into this file by hand would be a figure free to drift
from the measurement.
"""

import textwrap


def _table(headers, rows):
    """A markdown table; every cell is rendered as text."""

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _contract_lines(contract):
    """The frozen chain contract, one field per line."""

    return "\n".join(f"- `{key}`: `{value!r}`" for key, value in sorted(contract.items()))


def _heading(report):
    totals = report["totals"]
    scope = (f"{totals['subjects']} subjects, {totals['trials']} trials, "
             f"{totals['gib_on_disk']:.2f} GiB on disk")
    if not report["selection"]["complete"]:
        scope += " (PARTIAL SELECTION)"
    return textwrap.dedent(f"""\
        # KU Leuven converted-trial audit and frozen feature contract

        Generated: `{report['generated_at']}` by `{report['command']}`.
        Scope: {scope}

        This is step 4 of `final_connection.md`. It measures what the conversion of
        step 2 actually wrote, before step 5 trains anything on it, and it freezes
        the contract step 5 must treat as immutable. Nothing here writes into
        `datasets/`; the machine-readable record of every trial is the
        `results/{report['json_name']}` written in the same run.
        """)


def _verdict(report):
    rows = [(name, "PASS" if passed else "FAIL", report["preregistered_detail"][name])
            for name, passed in report["preregistered"].items()]
    violations = report["invariants"]["violations"]
    detail = "all hold" if not violations else f"{len(violations)} violated"
    body = [_table(("expectation", "result", "detail"), rows), "",
            f"Per-trial invariant checks: **{detail}**."]
    body += [f"- `{rule}`: {len(trials)} trial(s), first: `{trials[0]}`"
             for rule, trials in violations.items()]
    return "## 1. Verdict\n\n" + "\n".join(body)


def _subjects(report):
    rows = []
    for subject, entry in report["subjects"].items():
        rollup = entry["rollup"]
        counts = rollup["label_counts"]
        window = rollup["audio_minus_eeg_seconds"]
        rows.append((
            subject, rollup["trials"], rollup["durations_seconds"],
            rollup["sample_rates"], rollup["audio_rates"],
            f"{counts.get('0', 0)}/{counts.get('1', 0)}",
            f"{window['min']:.3f} to {window['max']:.3f}",
            rollup["canonical_groups"], rollup["stored_groups"],
            f"{rollup['eeg_abs_max']:.1f}",
        ))
    table = _table(("subject", "trials", "durations (s)", "eeg Hz", "audio Hz",
                    "labels 0/1", "audio-EEG (s)", "groups", "stored groups",
                    "peak |EEG|"), rows)
    return textwrap.dedent("""\
        ## 2. Per subject

        """) + table + textwrap.dedent("""

        `labels 0/1` counts trials whose label is candidate A / candidate B;
        `groups` counts the groups after folding `rep_*` into its base, `stored
        groups` counts what the converted `group` field actually carries. `peak
        |EEG|` is the largest absolute sample among the subject's trials, which is
        what identifies the recorded unit: hundreds means microvolts.
        """)


def _label_balance(report):
    """Label share over time: the null a window-level accuracy must beat."""

    records = [record for entry in report["subjects"].values()
               for record in entry["trials"]]
    by_time = {"0": 0.0, "1": 0.0}
    count = {"0": 0, "1": 0}
    for record in records:
        if len(record["labels"]) == 1:
            by_time[str(record["labels"][0])] += record["duration_seconds"]
            count[str(record["labels"][0])] += 1
    total = sum(by_time.values())
    if not total:
        return "- Label balance over time: not measurable (every trial is unknown)."
    return (
        f"- Label balance: {count['0']} trials label A and {count['1']} label B, but "
        f"over time A {by_time['0']:.1f} s and B {by_time['1']:.1f} s. Always answering "
        f"A therefore scores {by_time['0'] / total:.1%} of the known time, so that -- "
        "not 50% -- is the null a window-level accuracy must beat; trial-count balance "
        "and time balance must not be confused."
    )


def _cross_subject(report):
    totals, extremes = report["totals"], report["extremes"]
    histogram = ", ".join(f"`{key}` s -> {value}"
                          for key, value in report["duration_histogram"].items())
    largest, smallest = extremes["max"], extremes["min"]
    nonfinite = report["nonfinite"]
    return "\n".join([
        "## 3. Cross-subject measurements",
        "",
        f"- Distinct sample rates: `{totals['sample_rates']}`; distinct candidate "
        f"rates: `{totals['audio_rates']}`.",
        f"- Duration histogram: {histogram}",
        "- Trials whose candidates are not strictly longer than the EEG: "
        f"{len(totals['audio_not_longer_than_eeg'])}.",
        f"- Largest audio-minus-EEG difference: {largest['seconds']:.6f} s on "
        f"`{largest['trial']}` ({largest['duration_seconds']:.6f} s EEG, "
        f"{largest['audio_seconds']:.6f} s audio).",
        f"- Smallest audio-minus-EEG difference: {smallest['seconds']:.6f} s on "
        f"`{smallest['trial']}` ({smallest['duration_seconds']:.6f} s EEG, "
        f"{smallest['audio_seconds']:.6f} s audio).",
        f"- Non-finite values: EEG {len(nonfinite['eeg'])}, timestamps "
        f"{len(nonfinite['timestamps'])}, candidates {len(nonfinite['audio'])} trial(s).",
        _label_balance(report),
    ])


def _groups(report):
    example = next(iter(report["group_map"]["per_subject"]))
    distribution = report["group_map"]["per_subject"][example]["groups"]
    rows = [(key, len(value), ", ".join(str(index) for index in value))
            for key, value in distribution.items()]
    stories = [(story, entry["subject_count"], entry["trials"])
               for story, entry in report["group_map"]["stories"].items()]
    return "\n".join([
        "## 4. Group map (the key step 5 splits on)",
        "",
        "Merge rule: `rep_partN_trackM_dry.wav` and `partN_trackM_dry.wav` are one",
        "story, so the group key is the sorted pair of candidate stories with the",
        "`rep_` prefix folded away. Both halves of a pair are always presented",
        "together, so a held-out group removes both of that part's stories; the pair",
        "is the finest unit a leakage-free split can hold out.",
        "",
        _table((f"group (canonical, {example} example)", "trials", "trial indices"), rows),
        "",
        "Every subject carries the same canonical groups; the per-subject counts and",
        "trial indices are in the machine-readable report. Stories are shared across",
        "subjects, so a held-out story leaves every subject at once.",
        "",
        _table(("candidate story", "subjects presenting it", "trial appearances"), stories),
        "",
        "**Warning for step 5**: `nova2026.auditory.evaluation.check_split` and",
        "`assert_held_out` compare `trial.group.split(\"|\")` tokens literally. The",
        "converted `group` field still carries the `rep_` prefix, so both guards see",
        "`{part1_track1_dry.wav, part1_track2_dry.wav}` and",
        "`{rep_part1_track1_dry.wav, rep_part1_track2_dry.wav}` as disjoint and will",
        "accept a split that trains and validates on the same 125 s of audio.",
        "Canonicalise every group token with `canonical_story` (or build splits from",
        "`group_key`) before calling either guard.",
    ])


def _channels(report):
    channels = report["channels"]
    rows = list(zip(channels["shared_names"], channels["shared_indices"]))
    verdict = "PASS" if channels["shared_matches"] else "FAIL"
    return "\n".join([
        "## 5. Channel subsets",
        "",
        "### KU Leuven, 64 channels (baseline comparable with the paper)",
        "",
        "Column order is the standard BioSemi64 order recorded in "
        "`datasets/AAD-KULeuven/metadata.json`; `channel_names[47] == \"Cz\"`.",
        "",
        "### Shared subset, 20 channels (live contract)",
        "",
        _table(("channel", "KU Leuven column"), rows),
        "",
        f"- Names read back at those indices: `{channels['shared_observed_at_indices']}`",
        f"- Match against the `metadata.json` order: **{verdict}**",
        f"- Indices unique: {channels['shared_indices_unique']}",
        f"- Live-only electrodes excluded from decoding: `{channels['live_only']}`",
        f"- Live cap order ({channels['live_cap_count']} channels): "
        f"`{channels['live_cap_order']}`",
        "- Live cap channel set equals the 20 shared + 4 live-only, without "
        f"repeats: {channels['live_cap_set_matches']}",
    ])


def _rep_identity(report):
    rows = [(record["base"], record["rep"], f"{record.get('base_seconds', float('nan')):.3f}",
             f"{record.get('rep_seconds', float('nan')):.3f}",
             record.get("compared_samples", 0),
             record.get("max_abs_difference", float("nan")))
            for record in report["rep_identity"]]
    return textwrap.dedent("""\
        ## 6. The `rep_` identity, re-measured

        """) + _table(("base file", "repeated file", "base s", "rep s", "compared",
                       "max abs diff"), rows) + textwrap.dedent("""

        A difference of 0 over the compared prefix is what makes the two files one
        group. Their SHA256 values differ because the files differ in length, which
        is exactly the signal a naive grouping rule mistakes for two stories.
        """)


def _contract(report):
    contract, envelope = report["contract"], report["envelope"]
    if envelope.get("present"):
        envelope_lines = "\n".join([
            f"- Arrays: `{envelope['array_keys']}`; `envelope` "
            f"{envelope['envelope_shape']} {envelope['envelope_dtype']}, `timestamps` "
            f"{envelope['timestamps_dtype']}",
            f"- Metadata keys: `{envelope['metadata_keys']}`",
            f"- Method `{envelope['envelope_method']}`, generator version "
            f"`{envelope['generator_version']}`, {envelope['sample_rate']} Hz, band "
            f"`{envelope['band']}`, source rate {envelope['audio_rate']}",
        ])
    else:
        envelope_lines = f"- Not measured: `{envelope['path']}` is absent."
    if contract["processor_contract"]:
        chain = _contract_lines(contract["processor_contract"]) + "\n" + "\n".join([
            "",
            "`source_unit_exponent` is **not** a contract key: it selects the units of "
            "the endpoint safety check and never rescales a sample, so adding it would "
            "invalidate every trained decoder. The default for this chain is "
            f"`{contract['source_unit_exponent_default']}`, giving `Repair._to_uv = "
            f"{contract['repair_to_uv_default']}`.",
        ])
    else:
        chain = "- Not captured: no converted trial was available to build the chain from."
    units = _table(("field", "value"), list(contract["unit_convention"].items()))
    totals, extremes = report["totals"], report["extremes"]
    return "\n".join([
        "## 7. Frozen feature contract",
        "",
        "### 7.1 Audio-envelope contract",
        "",
        _contract_lines(contract["auditory_config"]),
        f"- `lag_samples`: `{contract['lag_samples']}` (from `lag_seconds`)",
        f"- `MIN_MARGIN`: `{contract['min_margin']}`",
        "",
        "### 7.2 Precomputed envelopes, `datasets/audio/<stem>.npz`",
        "",
        envelope_lines,
        "",
        "### 7.3 Processing chain (compared against trained models)",
        "",
        chain,
        "",
        "### 7.4 Unit convention",
        "",
        "`source_unit_exponent` is the power of ten of the source unit relative to",
        "volts: `0` = volts, `-6` = microvolts. `Repair` multiplies endpoints by",
        "`10 ** (exponent + 6)` for its saturation and amplitude checks and uses the",
        "result for nothing else.",
        "",
        units,
        "",
        "KU Leuven trials are microvolts, so `-6` is the correct exponent for them and",
        "`_to_uv` is `1.0`: the check reads the recorded values unchanged. Feeding",
        "volts under `-6` would read 0.0833 V as 0.0833 uV and pass a saturated signal",
        "through the check, which is why the live path must declare its own exponent.",
        "",
        "### 7.5 Audio truncation rule",
        "",
        "Candidate audio is kept up to and including the last EEG sample:",
        "",
        "```python",
        "usable = floor((len(eeg) - 1) * audio_rate / sample_rate) + 1",
        "audio = audio[:usable]",
        "```",
        "",
        "Equivalently: keep every candidate sample whose time on the trial clock is at",
        "or before `timestamps[-1]`, drop the rest. The candidates are longer than the",
        f"EEG in all {totals['trials']} trials (by {extremes['min']['seconds']:.3f} to",
        f"{extremes['max']['seconds']:.3f} s), so the rule always drops something, and a",
        "window reaching past the last EEG sample would have neither a label nor an",
        "envelope to align against.",
    ])


def _immutable(report):
    return textwrap.dedent("""\
        ## 8. What step 5 must treat as immutable

        1. The `AuditoryConfig` values and the envelope method in 7.1, and the
           envelope record layout in 7.2: a change invalidates every precomputed
           envelope and every trained decoder.
        2. Every field of the processing-chain contract in 7.3, including `stage`,
           `units: "uV"`, the band and the filter order. New run policy belongs in
           `stream_config` arguments and never in the contract dict.
        3. The 64-channel order and the 20 shared indices in section 5, which are the
           only reason a 64-channel and a 20-channel decoder can be compared at all.
        4. The unit convention in 7.4: KU Leuven trials are microvolts and the
           chain's default `source_unit_exponent` is `-6`; the ANT rig's volts are
           `0` and must be declared as such at the call site.
        5. The truncation rule in 7.5.
        6. The group key in section 4: `rep_*` folds into its base before any
           train/validation/test split is drawn.
        """)


def _trials(report):
    rows = []
    for subject, entry in report["subjects"].items():
        for record in entry["trials"]:
            rows.append((
                f"{subject}/{record['file']}",
                f"{record['eeg_shape'][0]}x{record['eeg_shape'][1]}",
                f"{record['duration_seconds']:.6f}",
                f"{record['audio_shape'][0]}x{record['audio_shape'][1]}",
                f"{record['audio_rate']:g}",
                f"{record['audio_minus_eeg_seconds']:.6f}",
                record["labels"],
                record["group_key"],
            ))
    return textwrap.dedent("""\
        ## 9. Per-trial detail

        Columns: trial file, EEG samples x channels, EEG duration in seconds,
        candidate array shape, candidate rate, audio-minus-EEG in seconds, the
        distinct label values inside the trial, and the canonical group.

        """) + _table(("trial", "eeg", "duration (s)", "audio", "audio Hz",
                       "audio-EEG (s)", "labels", "group key"), rows)


def _reproduce(report):
    return textwrap.dedent(f"""\
        ## 10. Reproduce

        ```powershell
        {report['command']}
        ```

        Machine-readable record: `results/{report['json_name']}`.
        """)


def render_markdown(report):
    """Render the committed human-readable audit report."""

    sections = (_heading, _verdict, _subjects, _cross_subject, _groups, _channels,
                _rep_identity, _contract, _immutable, _trials, _reproduce)
    return "\n\n".join(section(report) for section in sections) + "\n"
