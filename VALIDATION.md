# VALIDATION.md - what is established, what is not, and where to check

This is the closing document of the auditory-attention integration (step 12 of
[`final_connection.md`](final_connection.md)). It is a *reading* of generated evidence, not a
replacement for it: every claim below carries the file, the command or the commit that produced
it, and the scope that measurement actually covers. Where the evidence is narrower than the
sentence around it, the sentence says so in the same line.

Nothing here is a marketing summary. If a number is not in the table, it was not measured.

---

## 1. What is claimed

| # | Claim | Evidence | Scope |
| --- | --- | --- | --- |
| 1 | The pipeline runs end to end on real data, in one command: KU Leuven trial -> streaming chain -> `AttentionSession` -> `AttentionProducer` -> FastAPI transport -> the built `dist/` front end, exit 0 | `results/demo_run_20260914.md`, `results/demo_run_20260914-trial008.json`, commit `3765202` (step 10) | **One** trial, `S1/trial_008` (64 channels, 124.0 s replayed at 1x, 125.4 s wall). No live participant, no amplifier, no browser. |
| 2 | The version-1 transport contract is exercised and re-validated by the client | `src/nova2026/transport/protocol.py`, `apps/attune-ui/src/protocol.js`, `tests/transport` (68 tests, 0 skipped, in the V6 run below), commit `7f2ab43` | Python tests plus loopback runs. The "client" in the demo evidence is Node executing the vendored modules, not a browser. |
| 3 | Markers, cue semantics and the 20-channel contract channel map are exercised on the recorded ANT sessions | `results/antneuro_import_20260913-101225.md` / `.json`, commit `fda17fc` (step 9.5) | Two sessions imported: 155093x20 @500 Hz / 310.19 s and 164317x20 @500 Hz / 328.63 s, 24 cues each, 12 A / 12 B after exclusions. Labels are derived from markers; no decoding was run on this data (see section 3). |
| 4 | Timing is exercised: the browser-reported playback position, the media handshake and the front end's gain gate all run on the real timeline | `results/demo_run_20260914.md` (V5, V3), `results/auditory_media_20260914-trial008.json` (step 9), `results/auditory_media_gate_20260914-trial008.json` | Replay path only. There is no audio device, so block timing and drift in ppm were **not** taken (the record says `available: false`). |
| 5 | The perturbation matrix answers its three questions | `results/perturbation_20260913-062335.md` / `.json`, commit `b10f916` (step 10.5 + V9) | 23 cases = 1 clean + 15 injected faults + 7 adversarial EEG inputs, each in its own process and directory. Crashes 0/23, explicit outcome 23/23, cross-contamination 0/23, exit 0, verdict **PASS WITH FINDINGS** (18 PASS, 5 FINDING). |
| 6 | The offline regression suite and the front end's own suite are green | `results/demo_run_20260914.md` V6: 760 tests / 6 suites / 0 skipped / 489.7 s, all green; front end 53/53 with `ATTUNE_PYTHON` set | Measured on this checkout on 2026-09-14. Suite totals move as steps add tests; the runner fails a suite that reports no tests or all-skips. |

The five FINDINGs are not failures of the invariants; they are missing *reporting* mechanisms.
They are listed in section 4 because they change what a user should expect.

---

## 2. The numbers, each with its scope

### 2.1 Decoder performance (the held-out calibration is the only trustworthy source)

Source: `results/aad_20260913-022712.md` / `.json`, commit `1a6bc3c` (step 5). Model:
Ridge on envelope-lag features, alpha 100, 5 s window, 1 s step, 320 trials, 14 528 windows.

| contract | scheme | accuracy | majority null | balanced accuracy | recall A | recall B |
| --- | --- | --- | --- | --- | --- | --- |
| 64ch | held-out story | 0.6170 | 0.6586 | **0.6195** | 0.6117 | 0.6272 |
| 64ch | leave-one-subject-out | 0.6087 | 0.6586 | **0.6115** | 0.6026 | 0.6204 |
| 20ch | held-out story | 0.5941 | 0.6586 | **0.5974** | 0.5870 | 0.6079 |
| 20ch | leave-one-subject-out | 0.5863 | 0.6586 | **0.5891** | 0.5803 | 0.5980 |

**Raw accuracy is below the majority null in every row.** The null is 0.6586 on the pooled 5 s
window grid, and 0.6609 by labelled time over the dataset (the plan's "66.1 %", §1 fact 13: 48 894 s
of A against 25 087.5 s of B). A classifier that always answered "A" would score higher raw
accuracy than this decoder. **Only the balanced accuracy - chance line 0.50 by construction - says
anything**, and what it says is "real but weak": 0.62 for the 64-channel contract and 0.60 for the
20-channel one on held-out stories. The leave-one-subject-out rows (0.6115 / 0.5891) are the honest
cross-subject statement and are close to the same level; they do **not** support a claim of
cross-subject generalisation beyond "above chance on this corpus".

Scope: one dataset (KU Leuven), the authors' own stimuli, window-level points that are not
independent participants. Per-fold spreads are in the JSON (`fold_balanced_mean` and `folds`).

The channel-count cost is measurable and small: 0.6195 - 0.5974 = **0.022** balanced accuracy for
dropping the 44 electrodes a live cap cannot carry (held-out stories; 0.0224 under LOSO). That is
the number the alignment cost below has to be compared against.

### 2.2 The margin/coverage trade-off, and the operating point actually in use

Source: `results/aad_margin_calibration_20260913-050745.md` / `.json`, commit `bd2da11` (step 9.6),
decisions D-29 and D-31. Method: 4 held-out-story folds of 80 trials each, the model **refitted per
fold**; the deployed 320-trial model is never scored; the fast scorer agrees with `model.score` to
4.6e-16.

The controller refuses to commit while `|corr_A - corr_B| < margin`, and the shipped default
`MIN_MARGIN = 0.5` had never been calibrated against anything. At a 5 s window it covers
**0.0015** of frames - 448 decided frames in the entire held-out corpus - which is why step 8's
first real run had 94 % of its frames say nothing.

| window | margin | coverage | accuracy (decided) | balanced (decided) | per story | first decision | false changes/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 s | **0.05 (in use)** | 0.5242 | 0.6840 | 0.6884 | 0.6680-0.6979 (4/4 above chance) | 10.0 s | 2.427 |
| 60 s | 0.05 (grid best) | 0.3844 | 0.9460 | 0.9483 | 0.9094-0.9723 (4/4) | 64.0 s | 0.109 |
| 5 s | 0.5 (documented default) | 0.0015 | - | - | - | - | - |

**The operating point in use is margin 0.05 with the existing 5 s model** (decision D-29): coverage
0.524, accuracy 0.684, balanced accuracy 0.688 on the decided frames, 4/4 held-out stories above
chance. Its published costs are in the same table and are not small: recall over *all* frames is
0.3477 (A) / 0.3796 (B) - an abstention counts as a miss - the first decision comes after a median
of 10.0 s, and it changes its mind 2.427 times per minute.

The 60 s row is better on every accuracy axis and is **not reachable today**: window length is part
of the decoder contract, and `models/auditory_kuleuven.npz` is a 5 s model. Reaching it requires
training a new model, which no step has done.

`MIN_MARGIN = 0.5` was **not** changed. `CALIBRATED_MARGIN = 0.05` is a separate name that nothing
defaults to; the demo CLI defaults to it (D-31), the evidence CLI defaults to the conservative 0.5,
and both record the effective value in their run records. The margin ladder's lower end (0.05) is
the grid floor: **how far below 0.05 is still worth having was not measured** (D-30).

### 2.3 What a timing offset costs the decoder (step 5.5) - and why it dominates channel count

Source: `results/aad_shift_sweep_20260913-040112.md` / `.json`, commit `a859f40`. Models are fitted
at zero offset and then evaluated at every offset, so this is a deployed model's tolerance, not a
re-training result. The reference envelope was slid over ±300 ms in 12.5 ms steps.

| contract | peak | peak balanced | half-depth width | chance width | cost per 100 ms early | late |
| --- | --- | --- | --- | --- | --- | --- |
| 64ch | -25 ms | 0.6256 | 104 ms | 172 ms | 0.1191 | 0.1496 |
| 20ch | -12.5 ms | 0.5998 | 104 ms | 170 ms | 0.1007 | 0.1175 |

The zero point reproduces step 5 exactly (0.6195 / 0.5974), and the cached pairing was pinned
window-by-window against the raw chain: `max|delta envelope|` 5.3e-10 to 8.6e-10 against a mirror
control of 2.4e-02 to 3.0e-02, `max|delta score|` 3.3e-09 to 8.5e-08, and identical balanced
accuracy at every offset.

**100 ms of residual misalignment costs about 0.10-0.15 balanced accuracy, five to seven times the
entire 64->20 channel cost of 0.022.** By ±250 ms the decoder is back at chance. Consequences that
are written into the design, not just observed:

* the alignment budget is **±100 ms**;
* the front end's `|delta t| <= 0.75 s` clause bounds **report staleness, not alignment** - it is
  about seven times the decoder's whole tolerance, and a run whose residual offset really were
  0.75 s would have chance-level decoding while the UI reported "synchronized"
  (`documents/media_timeline_contract.md` section 2, `documents/auditory_ui_protocol.md` section 7);
* `MIN_MARGIN` is a threshold on the correlation *difference* and this experiment does **not**
  measure it (the margin was calibrated separately, in 2.2).

The first attempt at this sweep (`results/aad_shift_sweep_20260913-033723.md`, commit `6ee0ba5`)
was **retracted**: a sign inversion plus a trim artefact mirrored the curve. The file is kept with
a RETRACTED banner so the wrong direction stays visible; nothing here cites it. The second run
removed both `expectedFailure` markers and turned them into passing tests; five mutations
(sign, rounding, truncation, grid point, inward walk) all go red.

### 2.4 The window-length curve - and the one part of it that is untested

Source: `results/aad_20260913-022712.md`, step 5. Measured on **S1-S4 only** (a documented quarter
of the corpus, so four window lengths stay affordable):

| contract | 5 s | 10 s | 30 s | 60 s |
| --- | --- | --- | --- | --- |
| 64ch balanced | 0.6371 | 0.6979 | 0.7962 | 0.8750 |
| 20ch balanced | 0.6242 | 0.6839 | 0.7744 | 0.8568 |
| windows | 3632 | 1800 | 584 | 288 |

Longer windows are better, monotonically, which is what a weak noise-limited correlation decoder
should do. Two limits belong in the same breath: the 60 s point rests on **288 window-level points
from 4 subjects**, not on 288 independent observations; and **the 30 s and 60 s points were measured
at zero offset only**. Step 5.5 swept 5 s windows; whether a long window is more or less tolerant
of a non-zero offset is **not measured** - it is an untested prediction (plan section 3.17 item 4),
and it is the reason the demo stays on the 5 s model that actually exists.

---

## 3. What is deliberately NOT claimed

Carried over from the plan (section 3.8) and extended with what the later steps added:

1. **No hearing or comprehension benefit.** The system attenuates one of two streams. Whether that
   helps anyone understand anything is not measured, and no listener study was run.
2. **No live-participant result.** All decoding numbers come from KU Leuven, and the demo is a
   replay. Step 11 (eego/LSL hardware, 20-channel contract, loopback measurement) is **TODO**: no
   hardware record exists in this repository.
3. **No cross-subject claim beyond the LOSO rows.** Balanced 0.6115 (64ch) / 0.5891 (20ch) against a
   0.50 chance line, with raw accuracy below the 0.6586 null. That is "above chance on this corpus",
   nothing more.
4. **No latency compensation.** Offsets are *reported*, never corrected. The loopback measurement
   that would justify a correction (`residual_offset_seconds`, tolerance ±30 ms) belongs to step 11
   and has not been taken; the demo record writes `block_timing.available = false` rather than a
   number nobody measured.
5. **No browser-verified UI.** Every frontend result is Node executing the vendored modules
   (`protocol.js`, `state.js`, `decoders.js`, `mediaAudio.js`, `Dashboard.js` via `react-dom/server`).
   No screenshot exists and none is claimed. Layout, interaction, the Web Audio graph and the actual
   gain application in an `AudioContext` are unverified.
6. **No acoustic output measurement.** `TimestampedAudio.diagnostics()` needs a real DAC; on the
   replay path there is none. Block timing, drift in ppm, browser `outputLatency`/`baseLatency` and
   speaker/microphone loopback are all unmeasured.
7. **No claim that the ANT recording's playback offsets are measured.** The two session audio starts
   (0 s and 267.0 s) are **operator testimony** recorded by the importer
   (`start_offset_seconds` in `results/antneuro_import_20260913-101225.json`); within a session the
   `1004/Start` marker is the anchor (`start_anchor_seconds` 5.006 s / 12.838 s). Nothing in this
   repository independently confirms that the stimulus really started at those moments.
8. **The ANT recording is a rehearsal, not the target presentation.** It is *dichotic*: one stream
   per ear, so a listener can use interaural cues. The target form - both streams mixed into both
   ears, the true cocktail-party task - is not implemented: presentation is parameterised
   (`dichotic` wired, `crossmix` implemented and **not** wired), decisions D-20/D-28. Anything
   measured on the ANT data therefore describes the rehearsal, not the goal.
9. **No result on the ANT test set.** V4 of the plan asks for a 20-channel contract result on the
   ANT data. No decoding number for those sessions exists in `results/`; the import produced
   markers, shapes and labels only, and the calibration report states it "says nothing about the
   ANT test set". This is a gap against V4, recorded here rather than papered over.
10. **No amplifier, microphone or headphone calibration** was performed for any number in
    `results/`. `scripts/getlive/` exists so that it can be, with the assertions recorded in the
    run's provenance.
11. **No generalisation from the demo trial's own "accuracy".** The V4 table in
    `results/demo_run_20260914.md` reports window-level accuracy 0.9296 on `S1/trial_008`; that trial
    carries a single label, so its majority null is **1.0000** and the figure beats nothing. The
    balanced accuracy on the same run is 0.4648. The credible numbers remain the calibration's.

---

## 4. Failure modes a user must expect (the V9 findings, D-35)

Recorded as known mechanism gaps: the invariants held in every case, but four gaps are not
*reported* by the system, and one changes what "it started" means. Each comes from
`results/perturbation_20260913-062335.md`.

1. **Duplicates are refused by the chain but not counted at acquisition** (C5). A repeated sample
   block is rejected as `irregular_timestamps`, consumes a recovery restart (the case lost 7 scored
   windows and 4 decisions), and adds no evidence - but `Acquire`'s duplicate counter does not
   exist, so `gaps == 0` and the acquisition layer looks clean. E1 shows the same refusal on the
   adversarial path (`decisions 6 -> 3`).
2. **There is no clock fit anywhere** (C11). With 1.25x drift injected, `|delta t|` reached 2.63 s
   and zero gains were applied - the invariant held - but `sync` stayed
   `status="unobserved", offset_ms=null, drift_warning=null` for the whole run. Drift is visible
   only as a difference between the reported positions; there is no residual, no ppm and no fit to
   quote. This is the same fact as section 2.3: **`0.75 s` bounds staleness, not alignment.**
3. **The transport counts no stalls** (C12). 26 reports were deliberately held back; the stale
   position exceeded the front-end window and 0 stale decisions were applied, but there is no
   dedicated underrun/stall event counter - the count exists only in the hold-back log.
4. **Contract mismatches degrade per window instead of refusing at startup** (C15, E3). Candidate
   lengths of 18 s vs 24 s did **not** fail the start: the uncovered windows were named, no
   confident decision was published while uncovered, and neither candidate was truncated. A sample
   rate of 256 Hz against a declared 128 Hz contract likewise produced `refusal=None` with 6 failed
   windows (`unavailable` 19 / `uncertain` 30) instead of a refusal. **"It started" therefore does
   not mean "the source matches the model."** The checks that *do* refuse at startup are the
   envelope check (C14: HTTP 409 naming the missing file), the channel contract (E4: refused, not
   truncated by column count) and non-monotonic timestamps (E2: refused by both the chain and the
   container).
5. **`signal_quality` was misleading under the relaxed run policy** (plan section 3.17 item 6, fixed
   in step 10 - commit `3765202`). With `check_channels=False` the monitor's `reasons()` is empty by
   construction, so `quality` was constantly 1.0 and a dead electrode reported `artifact=false`. The
   bad-channel census now travels `EEGWindow.bad_channels` -> `AuditoryWindow` -> the frame and the
   summary, and `artifact` is judged from it, **without adding any rejection** (rejecting would
   bring back the mid-run halt the relaxed policy exists to avoid). On the demo trial the census is
   empty (`{}`); the fix is exercised by a synthetic test, not by that trial.

Run policy that must be quoted with any demo number, because it changes verdicts and not signals:
`check_channels=false, max_bad_channels=0, exclude_channels=[], source_unit_exponent=-6,
margin=0.05, warmup=2.0 s, frame=0.25 s` (`results/demo_run_20260914.md`). Under the dataset's
default quality policy the run stops mid-trial - plan §3.17 item 1 records `S1/trial_003` raising
`EEG quality faults persisted beyond the allowed duration (122s)`, measured in step 5 - which is
why the relaxed policy is an explicit, recorded choice rather than a silent default change.

---

## 5. Structural exceptions (files over the 300-line guidance)

`structure-dev` asks for Python modules under 300 lines. The plan registered each exception as it
was granted (D-25, D-27, D-30, D-34 and the per-step notes in section 5). Line counts below were
re-measured on this checkout at HEAD; a few have drifted from the number the plan registered, and
the measured value is the one given here.

| File | Lines | Registered | Why it was left whole |
| --- | --- | --- | --- |
| `scripts/auditory_ui/demorun.py` | 2774 | plan section 5, step 10.5 | The 23 perturbation cases, their fault injectors, their per-case workers and the report writer share one case table; splitting them would separate a case from the criteria that judge it. |
| `scripts/auditory_ui/session.py` | 1075 | D-30 | Step-8 evidence command: starts the app, drives the session, asserts the packet stream. The assertions are the point. |
| `scripts/auditory/margin_calibration.py` | 946 | D-30 | Scorer, ladder, controller replay and the pre-registered selection rule are one argument; the rule must be readable beside the numbers it chose. |
| `scripts/auditory/shift_sweep.py` | 888 | D-25 | Pairing/offset mechanics, curve statistics and the chain-consistency audit are interlocked; the "cache == chain" contract is asserted by tests on all three. |
| `scripts/auditory/antneuro.py` | 864 | plan section 5, step 9.5 | Marker table, exclusion list, anchor arithmetic and label derivation are one audit trail; the marker table is meant to be read as a whole. |
| `src/nova2026/auditory/session.py` | 819 | D-27 | The single timing logic shared by replay, live and transport; the plan's own iron rule 6 is that there is exactly one copy. |
| `scripts/auditory_ui/demo.py` | 904 | plan section 5, step 10 | One command that verifies envelopes, renders media, starts the transport, runs the replay and writes five evidence files; the sequence *is* the deliverable. |
| `scripts/auditory/train_kuleuven.py` | 584 | plan section 5, step 5 | Training, both contracts, both schemes and the window-length curve in one reproducible command. |
| `scripts/auditory/feature_cache.py` | 530 | plan section 5, step 5 | The cache identity (`22e5546fa44abe0f`) is a contract every later step cites; splitting it would scatter that identity. |
| `src/nova2026/transport/media.py` | 458 | D-30 | The timeline rules (freshness, advance allowances, backward tolerance, invalidation) are one state machine. |
| `src/nova2026/transport/server.py` | 374 | step 6 | Snapshot-then-incremental delivery, the 1013 close path and the SPA mount are one request lifecycle. |
| `src/nova2026/auditory/envelopes.py` | 365 | plan section 5, step 3 | "Offline == runtime" is asserted by exact equality; splitting the two entry points across files would split that contract. |
| `src/nova2026/auditory/render.py` | 350 | D-30 | `render_stereo` and its presentation parameter (`dichotic` / `crossmix`) belong together; the mode is the point. |
| `scripts/auditory_ui/media.py` | 341 | D-30 | The browser stand-in used by the demo and the whole perturbation matrix. |

The pattern behind all fourteen: each file is one argument whose parts are asserted as a contract by
tests. The exceptions are recorded rather than hidden, and every one of them was approved by the
plan's decision log before this document existed.

---

## 6. What to run

From the repository root on Windows; `.venv\Scripts\python.exe` is the interpreter (the portable
`.venv/bin/python` form is equivalent). Order matters: envelopes, then a model, then the runs.

```powershell
# 0. once: the offline reference envelopes a session refuses to start without
.venv\Scripts\python.exe -B -m scripts.auditory.envelopes --audio-dir datasets/AAD-KULeuven/stimuli --out datasets/audio
.venv\Scripts\python.exe -B -m scripts.auditory.envelopes --out datasets/audio --verify

# 1. once: the decoder (skipped here if models\auditory_kuleuven.npz already exists)
.venv\Scripts\python.exe -B -m scripts.auditory.train_kuleuven --out results --models models

# 2. the demo (V1-V5): one command, one real trial at 1x, real transport, built front end at /
.venv\Scripts\python.exe -B -m scripts.auditory_ui.demo --browser

# 3. the perturbation matrix (V9) and the unattended demorun: 23 cases, exit 0
.venv\Scripts\python.exe -B -m scripts.auditory_ui.demorun
#    one case only, or a strict policy that turns any FINDING into a failure:
.venv\Scripts\python.exe -B -m scripts.auditory_ui.demorun --only C11 --strict

# 4. the calibration behind the operating point (regenerates results\aad_margin_calibration_*.md)
.venv\Scripts\python.exe -B -m scripts.auditory.margin_calibration --out results --contracts 64ch 20ch --histories 5 10 30 60

# 5. the ANT import (V8): markers, exclusions, 20-channel contract map
.venv\Scripts\python.exe -B -m scripts.auditory.antneuro --data-root tmp/antneurodata --out datasets/AAD-ANT --buffer-seconds 0.5

# 6. the regression suites
.venv\Scripts\python.exe -B scripts/run_tests.py
$env:ATTUNE_PYTHON="$PWD\.venv\Scripts\python.exe"; npm --prefix apps/attune-ui test
```

Notes that change what you see:

* `scripts.auditory_ui.demo --help` and `scripts.auditory_ui.demorun --help` list every knob; the
  demo defaults to the calibrated margin 0.05, while the evidence CLI
  (`scripts.auditory_ui.session`) defaults to the conservative 0.5 - deliberately, and both record
  the effective value (D-31).
* The demo prints a loopback URL that belongs to that run only; the process exits when the replay
  ends unless `--browser` / `--serve-seconds` keep the page up.
* `models/*` and `datasets/*` are git-ignored: a fresh clone has neither data nor weights, so
  commands 1, 2 and 3 need them built first (steps 2-5 of the plan).
* The ANT commands need `tmp/antneurodata/` (the operator's recording) and the `eeg-ant` extra for
  `antio`.

---

## 7. Evidence index

| Step | Commit | Artefact |
| --- | --- | --- |
| 1 baseline | `21a637e` | `results/test_baseline_20260913-005823.txt` (523 tests, 0 skipped) |
| 3 envelopes | `ab63ede` | `datasets/audio/*.npz` (18, `--verify` clean) |
| 4 audit + frozen contract | `48c412d` | `results/kuleuven_audit.md`, `results/kuleuven_audit_20260913-015057.json` |
| 5 decoder | `1a6bc3c` | `results/aad_20260913-022712.md` / `.json`, `models/auditory_kuleuven*.npz` |
| 5.5 offset sweep | `a859f40` (retracted first attempt `6ee0ba5`) | `results/aad_shift_sweep_20260913-040112.md` / `.json` |
| 6 transport | `7f2ab43` | `src/nova2026/transport/`, `tests/transport` |
| 7 front end | `61ca57c` | `apps/attune-ui/`, `apps/attune-ui/PROVENANCE.md` |
| 8 session + producer | `d92a5d6` | `results/auditory_session_trial008.json`, `results/auditory_frontend_trial008.html` |
| 9 media timeline | `9202290` | `results/auditory_media_20260914-trial008.json`, `results/auditory_media_gate_20260914-trial008.json` |
| 9.5 ANT import | `fda17fc` | `results/antneuro_import_20260913-101225.md` / `.json` |
| 9.6 margin calibration | `bd2da11` | `results/aad_margin_calibration_20260913-050745.md` / `.json` |
| 10 demo | `3765202` | `results/demo_run_20260914.md`, `results/demo_run_20260914-trial008.json` |
| 10.5 + V9 demorun | `b10f916` | `results/perturbation_20260913-062335.md` / `.json` |
| 11 live hardware | - | **TODO: no evidence exists** |
| 12 closing docs | this commit | `VALIDATION.md`, `documents/auditory_ui_protocol.md`, `README.md` |

The wire contract between backend and front end is specified in
[`documents/auditory_ui_protocol.md`](documents/auditory_ui_protocol.md); the reasoning behind the
media timeline is in [`documents/media_timeline_contract.md`](documents/media_timeline_contract.md);
the plan, the acceptance criteria V1-V9 and the decision log D-01..D-35 are in
[`final_connection.md`](final_connection.md).
