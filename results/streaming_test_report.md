# Streaming test report — live blocking check + offline replay parity

Generated: 2026-09-09
Environment: Python 3.13.15, soxr 1.1.0 (SoXR LQ), mne/mne-lsl per `uv.lock`.
Command: `.venv/Scripts/python.exe -B -m unittest discover -s tests/streaming -t .`

## Result

```text
Ran 106 tests in ~27 s
OK
```

| Module | Tests | Scope |
| --- | --- | --- |
| test_acquire | 26 | fixed-size acquisition, reassembly, timeouts, lag guard, uptime independence |
| test_acquisition | 3 | MNE-LSL manual-acquire behaviour baseline |
| test_bootstrap | 9 | arg getter + session assembly + judges gate + stats + contract source check |
| test_channels | 7 | ChannelContract validate/reorder/drop |
| test_circular_buffer | 16 | ring storage + overlapping windows |
| test_e2e | 3 | **live run + record + offline replay parity** |
| test_offload | 11 | worker pool, overflow policies, error transport |
| test_preprocess | 16 | units/filters/quality stages |
| test_recording | 17 | recorder + provenance + tail mark + close safety |
| test_recovery | 11 | Recovery guard + UnrepairableError + StreamStats (A2/E) |
| test_repair | 11 | Repair: NaN/Inf interpolation, gaps, judge interface |
| test_resample | 17 | SoXR resampler + automatic quality selection |
| test_source | 17 | preflight: validate_source + prepare + resolve_outlet (B) |
| test_spatial | 12 | cut_epochs / fit_ssp / SpatialOperator / processing_contract (C) |
| test_window | 5 | EEGWindow value object |
| **total** | **181** | |

## Update — Q1 refactor + A1 repair (`stream`, working tree)

Scope implemented in one pass (user decision: A2 bounded recovery deferred):

1. **Q1: preprocessing fully removed from `StreamSession`.** Deleted
   `process()` and the `scale`/`quality`/`filters`/`resample` constructor
   parameters and defaults. `StreamSession` now only assembles
   (contract/recorder/acquire/buffer) and gates windows. Scripts own an
   explicit ordered `STAGES` tuple of `(data, ts) -> (data, ts)` callables and
   fold it in their own loop; `wrap()` unions verdicts from constructor
   `judges` (objects exposing `reasons(start, end)`), so quality faults and
   repairs both reject windows. Warm-up stays sample-index based.
2. **A1: `Repair` (`preprocess/repair.py`)** — linear, per-channel repair of
   short NaN/Inf runs and small timestamp gaps (default limit 10 samples at
   500 Hz) between finite endpoints; stage + judge in one object. Damage it
   cannot repair (too long, irregular/non-finite timestamps, unsafe endpoints
   when units are known) **raises** — with no A2 the run stops loudly instead
   of poisoning causal filters. Leading damage before the first finite sample
   is dropped. A damaged tail waits for its finite right endpoint across
   blocks. Demo chain is now
   `(repair, scaler, observe(quality), notch, bandpass, resampler)`.

### Issues found and fixed in this milestone

1. **Repair synthesized missing rows at the wrong anchor time.** The grid
   checker updated `_previous_time` before computing gap times, so a
   synthesized row would start at the *current* row's time. Fixed by keeping
   the previous timestamp in a local before assigning.
2. **Trailing-damage test fed 3 data rows with 6 timestamps** →
   `ValueError`. Fixed the test to build `grid(3)` for the second block.

### Regression

- Full suite: 118 tests, all OK (~27 s incl. live PlayerLSL e2e).
- Live demo smoke (`--duration 3 --workers 0`): exit 0, `gaps=0`,
  `max_lag=0.997 s` (connect-time backlog), no Repair misfires on real LSL
  timestamps; 0/3 valid windows is expected at 3 s (entire run is warm-up).

## Update — B + C + D (`stream`, working tree)

Scope: user approved B, C, D after Q1 + A1; A2 stays deferred.

1. **B — `source.validate_source`.** Stateless function run right after
   `connect()`: connected identity, sampling rate, numeric dtype, untouched
   state, unique labels with the required set, channel types (EEG/EOG), and
   per-channel declared units vs the configured exponent; `RuntimeError` on
   any mismatch. Pre-connect outlet resolution (B1) intentionally left out.
   Demo calls it before any sample is read (probe confirmed PlayerLSL
   synthetic outlets declare `'0'` = volts units and `eeg` types).
2. **C — `spatial.py`.** `processing_contract` (JSON-safe run fingerprint),
   `fit_ssp` (EOG-guided SSP from marked epochs; held-out coupling must drop
   by at least half, else raise), `SpatialOperator` (symmetry/idempotence/
   rank checks, content-derived `artifact_id`, no-pickle `.npz` save/load
   that refuses to overwrite, per-window EEG-only application that preserves
   EOG, verdict and timestamps). Calibration-from-recorded-run epoch cutting
   is an integration step needing a real EOG channel.
3. **D — recorder provenance.** `RunRecorder` gains `config=` (stored as
   `config` metadata), `files=` (copied into the run folder + original path
   and SHA-256 stored as `input:<role>`; missing file fails before the run
   folder is created), and `track_windows=`/`log_window()` with an
   `iter_windows` reader. `StreamSession` forwards these via
   `recorder_config` / `recorder_track_windows`.
4. **Demo** validates the source (B) and, when recording, stores a
   `processing_contract` snapshot and logs every window (D). Live smoke with
   `--record`: 3600 raw samples, 10 windows logged (6 valid), config snapshot
   with the exact chain stamp stored — all verified from the SQLite run.

### Issues found and fixed in this milestone

1. **Channel label checks leaked `ValueError` from `ChannelContract`** instead
   of the function's documented `RuntimeError`; replaced with local checks so
   every `validate_source` failure raises the same type.
2. **Fake unit-declaration case was never exercised** (`units=None` fell back
   to defaults in the test fake); an explicit empty list now tests the
   missing-declaration branch.
3. **`processing_contract` stringified `stamp`**, so a NaN stamp silently
   became `"nan"`; the finite check now targets `out_sfreq`, the only numeric
   field.
4. **Demo `--artifact` flag was dropped** after a fence/docstring mismatch: a
   demo without an EOG channel cannot hold a valid EOG-guided operator
   contract, so C ships as library + tests instead of a demo hook.

### Regression

- Full suite: 144 tests, all OK (~27 s incl. live PlayerLSL e2e).
- Live demo smoke (`--duration 6 --record .tmp_tests/demorec --run d1`):
  exit 0, `gaps=0`, 6 valid windows; recorder metadata holds the config
  snapshot and 10 window log lines.

## Update — A2 bounded recovery + E run statistics (`stream`, working tree)

Scope: user picked a standalone `Recovery` guard (option A) over session
integration.

1. **`preprocess/repair.py`** now raises a typed `UnrepairableError(kind,
   gap_seconds=None)` (a `RuntimeError` subclass, so old callers still work)
   instead of bare `RuntimeError`: kinds `nonfinite_timestamp`,
   `irregular_timestamps`, `large_gap` (with seconds), `nonfinite_run`,
   `unsafe_endpoints`.
2. **`recovery.py: Recovery`** — `handle(error)` discards the unrepairable
   chunk, resets every registered stateful component (filters/resampler/ring/
   quality/repair all expose `reset()`), advances `segment` and records the
   decision (+ a `recovery:<kind>` recorder event when recording); raises on
   gaps > `max_gap_seconds` (0.5 s) or more than `max_events` (5) recoveries.
   `watch(window)` stops the run when judge-rejected windows persist longer
   than `persistent_fault_seconds` (5 s); clean windows reset the clock;
   warm-up-only windows (no reasons) are ignored.
3. **`stats.py: StreamStats`** — run counters; `StreamSession.close(stats=)`
   stores them in the recorder metadata (E).
4. **Demo** wires the guard around the chain (`try/except UnrepairableError`),
   stamps `eeg_window.segment`, calls `recovery.watch`, and persists stats.

### Regression

- Full suite: 157 tests, all OK (~27 s incl. live PlayerLSL e2e).
- Recovery unit path proven: bad 15-row NaN chunk raises `UnrepairableError`,
  guard resets the stage, and a later clean chunk processes normally.

## Update — finishing touches + real-recording check (`stream`, working tree)

1. **B1 — `preflight.resolve_outlet`.** Pre-connect identity check that polls
   `resolve_streams` (lazy import; injectable resolver for tests) until an
   outlet matches name/source_id/stype or the timeout expires. Fails fast
   before `connect()`; rates/units still checked by B2 afterwards.
2. **D4 — `recorder.mark_not_processed(rows)`.** Records a buffered tail that
   never became windows as a `not_processed:<rows>` event; the demo calls it
   at shutdown when `Acquire` still holds samples.
3. **C data-input boundary — `spatial.cut_epochs`.** Documented interface for
   ALREADY-PROCESSED data: cuts fixed-length epochs out of continuous
   float `(samples, channels)` EEG/EOG around marked event times; validates
   shapes, ordering, finiteness and that every epoch fits inside the data;
   output feeds `fit_ssp` directly.

### Real-recording check (`scripts/verify_realdata.py`)

Ran against a real COG-BCI `.set` file
(`datasets/COG-BCI/sub-01/ses-S1/eeg/RS_Beg_EO.set`):

| Check | Result |
| --- | --- |
| Runs through (63 ch @ 500 Hz, 10 s clip) | 14 windows, 10 valid after warm-up |
| Save is correct | FIF == SQLite chunks; offline replay windows == live windows (sample-for-sample) |
| Edge: short NaN run (4 rows, 1 channel) | repaired identically online and offline; replay parity holds |
| Edge: 60-row irreparable bursts | 2 recoveries, segment 2, processing continues |
| Edge: recovery budget = 1 | run stops with RuntimeError as expected |

Result: **PASS**.

### Regression

- Full suite: 165 tests, all OK (~27 s incl. live PlayerLSL e2e).

## Update — automatic resampler preset selection (`stream`, working tree)

1. **Budget from the consumer's allowed age.** `Resampler` accepts
   `max_age_seconds` and `reserve_seconds` and derives
   `budget = max_age_seconds - reserve_seconds` (explicit `max_delay_seconds`
   overrides; fallback 2.0 s). A non-positive budget is rejected with an
   explanation, because at that point no preset can help.
2. **Measured, not tabulated, selection.** `quality=None`/`"auto"` measures
   each anti-aliased preset on the installed SoXR build (one zero sample at a
   time until the first output) and keeps the cleanest one inside the budget.
   Measured here at 500 -> 128 Hz: LQ 1.88 s, VHQ 7.12 s, HQ 7.37 s, MQ 7.53 s
   — MQ is slower than HQ/VHQ, so no name-based ordering is safe.
3. **Warnings and control.** `ResamplerQualityWarning` fires when only the
   weakest preset fits, when `QQ` is used, or when nothing meets the budget;
   `strict=True` raises instead. `quality`, `startup_delay_seconds` and
   `max_delay_seconds` expose the decision; `select_quality(...)` exposes it
   without building the stage; `--resample-quality auto` is the CLI switch.
4. **`QQ` stays opt-in** (`allow_qq=True`) for paths whose earlier stages
   already band-limit the signal.

### Regression

- Full suite: 175 tests, all OK (~27 s incl. live PlayerLSL e2e).

## Update — independent audit (same classes as the external review)

Audit scope: documentation-vs-behaviour mismatches, environment/uptime
dependence, and silent misbehaviour — the three classes the external review
found. Everything below was reproduced before being changed.

### Fixed (with reproductions)

| # | Finding | Reproduction | Fix |
| --- | --- | --- | --- |
| 1 | `StreamSession` accepted a `ChannelContract` built for a **different outlet** and silently reordered columns with its permutation (repro: real order F3,F4,C3 + contract C3,F4,F3 → output 3,2,1). | script | session now requires `contract.source_channels == stream.ch_names`; test added |
| 2 | `Recovery.watch` with NaN window timestamps became a permanent silent no-op (`bad_since = nan`, comparisons false) — the persistent-fault guard was disabled for the rest of the run. | script | `watch` now raises when the window has no finite time grid; test added |
| 3 | `test_acquisition` used a **constant outlet name** (collision risk: LSL outlets are network-wide) and an **unbounded** wait loop (hang instead of failure). | code reading | unique `uuid4` name per run + deadline on the wait loop |
| 4 | `RunRecorder.close()` leaked the SQLite handle and masked the original error if FIF export raised. | code reading | export wrapped in `try/finally` so the connection always closes; test added |
| 5 | Stale/incorrect docs: `TaskOffloader.pending` said "queued or being handled"; `Acquire` said it "only drives `stream.acquire()`"; `QualityMonitor` comment still called recovery a "future component". | code reading | wording corrected |

### Recommendations (not changed — need a product call)

- **Load-sensitive timing assertions**: `test_e2e` (`elapsed < duration + 1.5`),
  `test_offload` (`elapsed < 0.1 s`), `test_acquire` (`elapsed < 1.0 s`). These
  can flake on a loaded CI box. Options: loosen, mark as slow, or keep as-is.
- `SpatialOperator.apply_window` cannot check channel order when a window has
  no `channel_names`; it currently proceeds. A strict mode could refuse.
- `processing_contract` does not validate channel-list uniqueness/non-emptiness.
- `StreamSession.close(stats=...)` silently discards the stats when recording
  is off (no recorder to store them in).
- `RunRecorder.write` anchors replay on the first *finite* timestamp, silently
  fabricating times for leading NaN-timestamp rows (unreachable through
  `Repair`, which raises on non-finite timestamps).
- `scripts/verify_realdata.py` detects the unit exponent by magnitude — a
  documented heuristic, not a measurement.
- `preflight.resolve_outlet` can overshoot `timeout` by one resolver call.
- Tests create scratch dirs under `Path.cwd()/.tmp_tests` with no fallback; a
  read-only CWD fails instead of skipping.
- `Resampler` auto-selection with the 2.0 s fallback budget always emits the
  LQ warning at 500→128 Hz (intended alarm, noisy for bare defaults).

### Clean after checking

- No other test compares wall-clock-derived floats exactly; remaining exact
  comparisons use fixed epochs or identical computations.
- Docstrings vs validation/exception behaviour across `Repair`, `Recovery`,
  `preflight`, `spatial`, `recording`, `offload`, `circular_buffer`, `window`:
  no further mismatches found.
- No other unbounded waits or network-visible fixed identifiers in the suite.

### Regression

- Full suite: 181 tests, all OK (~27 s incl. live PlayerLSL e2e).

## E2E scenario results (`test_e2e.py`)

1. **Live run keeps up and records** (`test_live_run_keeps_up_and_records`)
   - MNE `RawArray` played through PlayerLSL (chunk_size 37, 500 Hz, 8 ch).
   - Passed: wall time stayed under `duration + 1.5 s`, acquired samples within
     one block of `duration * sfreq`, `max_lag < 3.0` (lag guard never fired),
     at least one valid window arrived after warm-up, recording closed as
     `completed` and exported `.fif`.
   - Conclusion: **no blocking/stall with an in-loop consumer**.
2. **Recorded run replays identically** (`test_recorded_run_replays_identically`)
   - The SQLite run was read back and the exact raw chunks pushed through a
     brand-new pipeline (same chain code as the live run).
   - Passed: FIF export agrees with the SQLite chunks (round-trip), and every
     offline window matches the live window sample-for-sample
     (`rtol=atol=1e-6`), same starts, same validity.
   - Conclusion: **no computation drift between live and offline**.
3. **Offline pipeline is deterministic** (`test_offline_pipeline_is_deterministic`)
   - Processing the same chunks twice yields bit-identical windows (`atol=0`).

## Performance (micro-benchmark, `scripts/benchmark_streaming.py`)

| Unit | Cost | Share of its real-time budget |
| --- | --- | --- |
| preprocessing chain per 0.1 s block | ~167 µs | 0.17 % of 100 ms |
| ring-buffer push per 0.5 s hop | ~8 µs | ~0 % of 500 ms |
| EEGWindow packaging per window | ~7.4 µs | 0.015 % of 500 ms |
| quality verdict lookup per window | ~0.2 µs | ~0 % |

EEGWindow adds **~7.6 µs/window** — negligible; the consumer never becomes a
bottleneck as long as it is offloaded when slower than ~0.5 s/window.

## Issues found and fixed while writing these tests

1. **Benchmark measured nothing (soxr buffering).**
   SoXR emits 0 samples for many early calls; feeding its output back as the
   next input collapsed every stage to empty arrays (fake 2.7 µs/block).
   Fixed: every timed step processes a fresh raw block. Real cost: ~167 µs.
2. **PlayerLSL drops its first chunk.** Recorded data starts at raw sample 37
   (the chunk size), not sample 0. The old assertion compared the recording
   against `raw[:, :samples]` and failed. Replaced by the meaningful checks:
   FIF ↔ SQLite round-trip, plus live ↔ offline window parity.
3. **`StreamSession.gate()` was renamed to `wrap()` (EEGWindow) but a
   bootstrap test still called `gate`** → AttributeError, which also skipped
   `session.close()`, leaving the SQLite file open so `tearDown` failed with
   `PermissionError [WinError 32]`. Fixed the test to use `wrap`; the lock
   issue disappeared once every path closes the recorder.
4. **`ChainRunner.assert_matches` (plain class) used `unittest` assertions** →
   `AttributeError: no attribute 'assertEqual'`. Rewrote it to raise
   `AssertionError` directly.
5. **Test helper keyword collision** (`eog` as parameter and as an override)
   → `TypeError`. Validation cases now construct `EEGWindow` directly.
6. **Benchmark passed a float capacity** to `CircularBuffer` (int required) —
   fixed with `int(...)`.

## Known limitations (not failures)

- `RunRecorder` keeps raw chunks in per-chunk transactions; it does not yet
  record per-sample timestamps (anchors + sfreq are enough for replay).
- The lag guard measures the age of consumed blocks; the ~1 s `max_lag` seen
  at run start is the connect-time backlog, not a stall.
- Test suite needs `_MNE_FAKE_HOME_DIR` pointing to a writable directory when
  the OS home is read-only for MNE's config file.
