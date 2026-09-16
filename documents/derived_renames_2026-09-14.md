# Derived-file renaming, 2026-09-14 (cleanup D-22)

The scratch trees under `tmp/`, `results/` and `records/` had grown one name per
author and per run: `bench_gram2.py` next to `final_report.json` next to
`lsl3_pub.err`. Nothing was deleted - `tmp/` is git-ignored, so a deleted scratch
file is simply gone - but everything was renamed or regrouped so that a file's
**kind** and **subject** can be read off its path alone.

The scheme is `<topic>_<kind>[_run<N>][_<stamp>].<ext>`: subject first, then what
kind of artifact it is, then a repeat index when the same script was run more than
once, then `YYYYMMDD-HHMMSS` for point-in-time artifacts. The same scheme names the
prose in `documents/` and the measurements in `results/`, with a closed vocabulary
of kinds (`audit`, `conversion`, `baseline`, `metrics`, `report`, `sweep`,
`averaging`, `incremental`, `design`, `guide`, `probe`, `smoke`, `bench`, `log`).

## `tmp/` - new layout, by kind

| Directory | Holds |
| --- | --- |
| `tmp/scratch/` | one-off scripts (`.py`, `.ps1`, `.mjs`) left by the subagents |
| `tmp/reports/` | the JSON those scripts produced |
| `tmp/logs/` | captured stdout/stderr, and the KU Leuven conversion logs |
| `tmp/notes/` | commit-message drafts |
| `tmp/antneuro_verify/` | the numbered ANT verification sequence, now `antneuro_verify_<NN>_<subject>.py` |
| `tmp/leftover/` | the empty `fakehome` / `mne_cfg` / `mne_home` / `tmpnjxwu_gz` directories, kept rather than removed |
| `tmp/antneuro_runs/` | **untouched**: the file names (`<run>_raw.fif`, `<run>.json`, `<run>.sqlite`) are the recorder's contract |
| `tmp/antneurodata/` | **untouched**: the user's recording |

## Mapping

Scripts, `tmp/<name>` to `tmp/scratch/<name>`:

| old | new |
| --- | --- |
| `aad_load_smoke.py` | `aad_load_smoke.py` |
| `antneuro_markers.py` | `antneuro_markers.py` |
| `antneuro_probe.py` | `antneuro_probe.py` |
| `antneuro_publisher.py` | `antneuro_publisher.py` |
| `audio_relation.py` | `audio_relation_check.py` |
| `audit_clockjump.py` | `lsl_audit_clockjump.py` |
| `audit_fields.py` | `lsl_audit_fields.py` |
| `audit_lag.py` | `lsl_audit_lag.py` |
| `audit_lost_block.py` | `lsl_audit_lost_block.py` |
| `audit_rate_error.py` | `lsl_audit_rate_error.py` |
| `audit_ts_code.py` | `lsl_audit_ts_code.py` |
| `audit_lsl.ps1` | `lsl_audit_e2e_run1.ps1` |
| `audit_lsl2.ps1` | `lsl_audit_e2e_run2.ps1` |
| `bench_chain.py` | `decoder_bench_chain.py` |
| `bench_gram.py` | `decoder_bench_gram_run1.py` |
| `bench_gram2.py` | `decoder_bench_gram_run2.py` |
| `bench_moments.py` | `decoder_bench_moments.py` |
| `channel_overlap.py` | `cap_channel_overlap.py` |
| `check_models.py` | `decoder_check_models.py` |
| `convert_rest.ps1` | `kuleuven_convert_rest.ps1` |
| `debug_census.py` | `streaming_debug_census.py` |
| `fetch_aad.py` | `aad_fetch.py` |
| `getlive_smoke.py` | `getlive_smoke.py` |
| `getlive_static_check.py` | `getlive_static_check.py` |
| `inspect_aad.py` | `aad_inspect.py` |
| `inspect_converted.py` | `kuleuven_inspect_converted.py` |
| `make_metadata.py` | `kuleuven_make_metadata.py` |
| `make_summary.py` | `kuleuven_make_summary.py` |
| `mr_f01_diag.mjs` | `attune_ui_mr_f01_diag.mjs` |
| `pdf_text_dump.py` | `handout_pdf_text_dump.py` |
| `probe_cache.py` | `decoder_probe_cache_run1.py` |
| `probe_cache2.py` | `decoder_probe_cache_run2.py` |
| `probe_faults.py` | `streaming_probe_faults.py` |
| `probe_kuleuven.py` | `kuleuven_probe_loader_run1.py` |
| `probe_loader.py` | `kuleuven_probe_loader_run2.py` |
| `probe_ridge.py` | `decoder_probe_ridge.py` |
| `publish_raw.py` | `lsl_publish_raw.py` |
| `relay_smoke.py` | `relay_smoke.py` |
| `repro_bad_channels.py` | `getlive_repro_bad_channels.py` |
| `rewrite_report.py` | `decoder_rewrite_report.py` |
| `run_suite.py` | `tests_run_suite.py` |
| `show_results.py` | `decoder_show_results.py` |
| `show_smoke_report.py` | `getlive_show_smoke_report.py` |
| `ts_check.py` | `lsl_ts_check.py` |

JSON evidence, `tmp/<name>` to `tmp/reports/<name>`:

| old | new |
| --- | --- |
| `antneuro_markers.json` | `antneuro_markers.json` |
| `antneuro_report.json` | `getlive_acceptance_antneuro_run1.json` |
| `rep_record.json` | `getlive_acceptance_antneuro_repeat.json` |
| `final_report.json` | `getlive_acceptance_antneuro_final.json` |
| `probe_report.json` | `getlive_probe_outlets.json` |
| `getlive_smoke.json` | `getlive_smoke_acceptance.json` |
| `getlive_smoke_excluded.json` | `getlive_smoke_acceptance_excluded.json` |
| `getlive_smoke_tolerated.json` | `getlive_smoke_acceptance_tolerated.json` |
| `relay_smoke_report.json` | `relay_smoke_acceptance_run1.json` |
| `relay_smoke_auto.json` | `relay_smoke_acceptance_run2.json` |
| `ts_chunked.json` | `ts_check_chunked_run1.json` |
| `ts_chunked2.json` | `ts_check_chunked_run2.json` |
| `ts_raw.json` | `ts_check_raw_run1.json` |
| `ts2_raw.json` | `ts_check_raw_run2.json` |
| `ts_self_new.json` | `ts_check_selfpublished.json` |
| `ts2_relay.json` | `ts_check_relay_run2.json` |

The three `getlive_acceptance_antneuro_*` files were named `antneuro_report.json`,
`rep_record.json` and `final_report.json`; all three carry a getlive acceptance
report whose `source.name` is `eego-test`, so the new names say which device they
describe. `probe_report.json` is the output of `scripts.getlive.probe` (its keys are
`outlets` and `inspected`), hence `getlive_probe_outlets.json`.

Logs, `tmp/<name>` to `tmp/logs/<name>`:

| old | new |
| --- | --- |
| `lsl_pub.err` / `.out` | `lsl_publish_run1.err` / `.out` |
| `lsl2_pub.err` / `.out` | `lsl_publish_run2.err` / `.out` |
| `lsl3_pub.err` / `.out` | `lsl_publish_run3.err` / `.out` |
| `lsl4_pub.err` / `.out` | `lsl_publish_run4.err` / `.out` |
| `lsl_relay.err` / `.out` | `lsl_relay_run1.err` / `.out` |
| `lsl2_relay.err` / `.out` | `lsl_relay_run2.err` / `.out` |
| `relay_smoke.log` | `relay_smoke.log` |
| `ts_raw.log` | `ts_check_raw_run1.log` |
| `ts_relay.log` | `ts_check_relay_run1.log` |
| `convert_logs/S<N>.out` / `.err` | `logs/kuleuven_convert_S<N>.out` / `.err` (N = 2..16) |

Notes, `tmp/<name>` to `tmp/notes/<name>`:

| old | new |
| --- | --- |
| `commit_msg.txt` | `kuleuven_convert_commit_msg.txt` |
| `commitmsg.txt` | `aad_train_commit_msg.txt` |
| `step4_commit_msg.txt` | `kuleuven_audit_commit_msg.txt` |

Verification scripts, `tmp/verify_antneuro/<NN>_<subject>.py` to
`tmp/antneuro_verify/antneuro_verify_<NN>_<subject>.py` (all seven files; the numeric
prefix is kept because the sequence is the point).

Empty directories `tmp/{fakehome,mne_cfg,mne_home,tmpnjxwu_gz}` moved to
`tmp/leftover/empty_<name>`; they hold no files, so nothing could be lost by moving
them, and nothing was removed. `tmp/convert_logs/` and `tmp/verify_antneuro/` were
removed only after their contents had been moved out and they were empty.

## `results/` and `records/`

`records/` holds nothing but `.gitkeep` - there is nothing to rename.

`results/` already conformed to the scheme, with two exceptions, and both were left
alone on purpose:

* `aad_20260913-022712.json` / `.md` carry no kind token (`aad_<stamp>`; the scheme
  wants `aad_metrics_<stamp>`);
* `kuleuven_audit.md` carries no stamp (`kuleuven_audit_<stamp>.md` would pair it
  with the JSON of the same run).

Renaming them is a two-line change in the producers (`scripts/auditory/train_kuleuven.py`
line ~573 and `scripts/auditory/audit_kuleuven.py` line ~220) plus the `.gitignore`
negations - but it also invalidates references this cleanup may not touch:
`src/nova2026/auditory/feature_cache.py` (frozen library), the step-4/step-5
docstrings in `scripts/auditory/`, one comment in `scripts/auditory/tests/`, and the
evidence paths in `final_connection.md` §5. That trade is for the main agent to make,
so the files keep their names and this note records why.

`results/streaming_test_report.md` was untracked and is now committed (see
`.gitignore`): it is the only account of the live-blocking and offline-replay parity
check, and an ignored file is one `git clean -x` away from being lost.

## Not touched

`tmp/antneurodata/**` (the user's recording), `tmp/antneuro_runs/**` (the recorder's
own naming), `results/visual_detect_*.json` and `results/strictTrainEEGNetReg_logrt_*.json`
(already `<topic>_<kind>_<stamp>`), and `dist/` / `.tmp_tests/` (build and test
scratch, ignored, and renaming them would only move the entries their `.gitignore`
lines name).
