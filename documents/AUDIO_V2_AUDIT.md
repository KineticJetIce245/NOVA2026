# audio_v2 audit — 12 September 2026

Scope: branch `audio_v2` at `4f4c687` (the commit that merges `stream` into it).
Read-only: no source file was changed by this audit. Every finding below was
re-checked by hand against the cited line after being raised; the claims that did
not survive re-checking are listed under "Rejected after re-checking" so they are
not re-reported in a later pass.

## Resolution status

| # | State | Where |
|---|---|---|
| A1 | reverted at the owner's request — belongs to the `visual_detect` arm, not this branch | `scripts/visual_detect` is byte-identical to HEAD again |
| A2 | fixed | `scripts/streaming_demo.py`; a real lag-guard stop now records `status="failed"` with the error text and exits 1 |
| A3 | fixed + regression | `evaluation._decorrelated_audio`; measured envelope correlation now 0.08 against the 0.5 margin |
| A4 | fixed by wording and record, not by compensation | `timing.py` docstring, `live.py` print, `audio_timing_profile` in `timing.json` |
| A5 | fixed + regression | `assert_held_out` fails closed on a model with no provenance |
| A6 | fixed | `live.py --record/--subject/--session`, real recorded status, recovery and acquisition counters in `timing.json` |
| A7 | fixed + regression | `test_resampler_choice_is_pinned_and_its_precondition_holds`; the doc sentence now describes what the code does |
| A8 | fixed | dead `LiveAuditoryAdapter` removed |
| A9 | fixed + regression | `scripts/auditory/outputs.guard_outputs` and `--force` on `demo`, `evaluate`, `live`, `replay` |
| A10 | fixed + regression | stale evidence is ignored before the invalid branch |
| A11 | fixed + regression | a switch delay needs the new talker held to the end of the label segment |
| A12 | fixed + regression | `TimestampedAudio.align` falls back like its sibling; both reject an empty window |
| A13 | fixed | VALIDATION.md commands use the venv interpreter; the 3.14 claim corrected to 3.13.15 |
| A14 | fixed | offload drop/failure counters in `StreamStats`, written to the run and reported by the demo |
| A15 | fixed | one preallocated float64 array instead of three copies; the sidecar is written before the export error is re-raised |
| A16 | deferred | reverted at the owner's request (it changes shared, tested streaming behaviour); recommendation below |

Remaining recommendation for the `stream` owner: `RunSpec` silently strips
characters out of operator-supplied identities, so `--subject "a b"` is recorded
as `ab` and two distinct names can collapse onto one run directory. The legacy
`scripts/dataproc/streaming/run.py` rejects such names with a regex; matching that
rule in `nova2026/streaming/recording.py` needs one shared test updated
(`tests/streaming/test_recording.py::test_validation`) and is a deliberate
behaviour change, so it is left for a separate change.

## Verification after the fixes

- 181 streaming + 45 auditory + 32 legacy streaming tests pass (258 total), plus
  the 56 `visual_detect` plain-function tests.
- `demo` -> `evaluate` -> `replay` completes. `fault=none` reproduces the
  documented baseline exactly (`known_seconds` 23.99225, wrong suppression 0.0,
  coverage 0.6225447800852358), so the fixes did not disturb the real path.
- `fault=mismatched` now abstains for the whole trial (coverage 0.0, neutral
  23.99225 s) instead of decoding as well as the real audio; per-candidate
  envelope correlation fell from 0.8134 to -0.0839 / 0.0385.
- The output guard refuses a second `demo`/`evaluate` run against the same
  directory (`FileExistsError`) and `--force` overwrites.
- The streaming demo now persists `status="failed"` with
  `error=RuntimeError('Source samples are 3.875 seconds old.')` and exits 1 when
  the lag guard stops it, while a clean run still persists `status="completed"`
  and exits 0. The persisted stats now include `offload_dropped`/`offload_failed`.
- `RunSpec` behaviour is unchanged (A16 deferred).

## How the audit was run

1. Recon of the existing record first, so resolved items were not re-reported:
   `documents/AUDIO_V2_ISSUE_RESOLUTION.md` (ISS-01..26, D1..D11),
   `output/pdf/NOVA2026_Auditory_Independent_Review.txt` (gaps A-D),
   `scripts/auditory/README.md`, `scripts/auditory/VALIDATION.md`.
2. Five parallel read-only reviews: auditory core, auditory runners and CLIs, the
   auditory↔streaming boundary, documentation claim verification, and a
   security/robustness sweep.
3. Independent reproduction by hand: the demo → evaluate → replay chain, all
   unittest suites, and targeted probes (controller dwell, resampler preset
   selection, envelope correlation of the `mismatched` fault).
4. Adversarial re-check of every finding. Three of the five reviews contained
   claims that were wrong on re-reading; they are recorded below rather than
   deleted, because a silent correction is how a wrong finding comes back.

## Environment actually measured

- Python 3.13.15 in `.venv`. There is no 3.14 interpreter on this machine.
- `tests/streaming` 181 OK, `scripts/auditory/tests` 38 OK,
  `scripts/dataproc/streaming/tests` 32 OK (251), plus 56 `visual_detect`
  plain-function tests that `unittest discover` cannot collect and that were run
  with an ad-hoc runner.
- `scripts/dataproc/streaming/tests/player_checks.py` adds 5 tests (33 s) when
  invoked explicitly as its own docstring documents; it is not part of the 251.
- `scripts.auditory.demo` → `evaluate` → `replay` completes. The replay metrics
  reproduce the documented numbers exactly: `known_seconds` 23.99225,
  `wrong_suppression_seconds` 0.0, `coverage` 0.6225447800852358.

## Findings

### A1 — MEDIUM (security) — unrestricted unpickler on an operator-supplied path

`scripts/visual_detect/train_occipital.py:510`

```python
    checkpoint = torch.load(args.dataset, map_location="cpu", weights_only=False)
```

`--dataset` is an operator-supplied path (line 438). The same script loads the
same dataset through the guarded loader at line 553 (`trainer.fetch`), and every
other `visual_detect` module uses `load_chkpt`, which tries `weights_only=True`
first and only falls back with an explicit warning
(`src/nova2026/training/trainer.py:219-228`). Provenance: introduced by
`d364eb9`, not by the `stream` merge. Impact is bounded while the checkpoint is a
local artifact; it becomes arbitrary code execution the first time a checkpoint
arrives from anywhere else.

### A2 — MEDIUM — a stopped live run is persisted as `completed`

`scripts/streaming_demo.py:326-327` and `:343`

```python
    except RuntimeError as error:
        print(f"run stopped: {error}")
...
        session.close(status="completed", stats=stats.to_dict())
```

The caught `RuntimeError`s include the stale-data lag guard, `Recovery.watch` and
the exhausted recovery budget — the failures this loop exists to detect. The
`finally` block then writes `status="completed"` with no error string, and
`RunRecorder.close` persists that status into the run metadata
(`src/nova2026/streaming/recording.py:408`). A `TimeoutError` from
`session.acquire.read()` is an `OSError`, so it takes the same path and only the
traceback distinguishes it. The older runner gets this right
(`scripts/dataproc/streaming/streamer.py:303-308`: `"completed" if error is None
else "failed"`, plus `repr(error)`). Because the demo's own docstring presents
this loop as the template for the real amplifier, the pattern is copyable.

### A3 — MEDIUM — the `mismatched` audio negative control cannot fail on the shipped fixture

`src/nova2026/auditory/evaluation.py:33-34`

```python
    elif kind == "mismatched":
        result.audio = np.roll(result.audio, round(7 * result.audio_rate), axis=0)
```

The fixture's envelopes are 2.7 Hz and 4.1 Hz tones (`scripts/auditory/synthetic.py`),
so a 7 s circular roll is a phase shift, not a decorrelation. Measured with the
real envelope extractor: `corr(original cand0, rolled cand0) = 0.8134`
(theory `cos(2*pi*2.7*7) = 0.809`), against `MIN_MARGIN = 0.5`. The control
therefore never triggers abstention, and `evaluate.py`'s `fault=mismatched` rows
plus the README/VALIDATION claim of fault coverage do not demonstrate audio
specificity on synthetic data. (`zero` and `shuffle` are unaffected and do
abstain.) A roll or shuffle length that is not a near-multiple of the modulation
period would restore the control; on real speech 7 s is fine.

### A4 — MEDIUM/LOW — the measured audio/EEG residual is validated but never applied

`src/nova2026/auditory/timing.py:16-18`, `:34-41`, `:72-78`

```python
    residual = profile["residual_offset_seconds"]
    if not isinstance(residual, (int, float)) or not np.isfinite(residual) or abs(residual) > .030:
        raise ValueError("Measured residual EEG/audio offset exceeds the 30 ms budget.")
...
    def __init__(self, audio, audio_rate, config, *, tolerance=0.030):
```

An exhaustive grep for `residual` over `*.py` finds only the validation, the
equality check against the stored profile and the print in
`scripts/auditory/live.py:36-40`, the persistence in `train.py`, and tests. No
arithmetic consumer exists, and `TimestampedAudio.__init__` accepts no offset
argument, so the timestamp→sample-position mapping carries no correction term.

This may be the intended design — the 30 ms threshold then reads as a tolerance
gate rather than a calibration to subtract — but the run prints the value as
`Verified profile residual`, which invites the opposite reading, and the docs do
not settle it. At 64 Hz the accepted residual is up to ~1.9 samples.

### A5 — MEDIUM/LOW — the held-out guard fails open when a model has no provenance

`src/nova2026/auditory/evaluation.py:40-46` with `decoder.py:37` and `:94`

```python
def assert_held_out(trial, model):
    """Library and command-line evaluation share the same leakage guard."""
    for partition in ("training", "validation"):
        for subject, identity, group in model.training_info.get(partition, []):
```

`fit(examples, training_info=None)` produces `{}` (`decoder.py:94`), which is
saved and loaded verbatim, so the loop body never runs and the guard returns
silently. A decoder fitted programmatically on trial X and then evaluated with
`replay(X, model)` passes the check that the package advertises as shared by
library and CLI, and the resulting metrics would be described as held-out. The
shipped `train()`/CLI path always populates the keys, so this is a library-API
hole, not a live one.

### A6 — LOW/MEDIUM — the live auditory path records nothing, and its failure status is dead code

`scripts/auditory/live.py:54-57` builds its `StreamSession` arguments without a
`record` field and the CLI exposes no `--record`, so
`src/nova2026/streaming/bootstrap/session.py:121-122` finds `record_root is None`
and builds no recorder. Three consequences:

- no raw EEG is kept for the only genuinely live path, so a processing failure
  cannot be re-analysed offline;
- `live.py:141` `session.close(status="failed" if eeg_error else "completed")` is
  a no-op — `session.py:246` acts only `if self.recorder is not None`, so no
  status is ever persisted;
- `Recovery` discards are invisible: `src/nova2026/auditory/streaming.py:55-58`
  constructs `Recovery` without a recorder, `:81-83` turns unrepairable damage
  into `return []`, and `live.py:152` reports only estimates plus
  `provider.diagnostics()`.

### A7 — LOW/MEDIUM — the claimed "strict auto-quality test" does not exist; the production choice is the anti-aliasing-free preset

`documents/AUDIO_V2_ISSUE_RESOLUTION.md:79-81` states that the old
`StreamingResampler(quality="LQ")` test is superseded by "the supported auditory
processor's strict auto-quality test". A grep for `resampl|quality|QQ|strict`
over `scripts/auditory/tests` returns zero matches: no auditory test asserts
anything about resampler quality.

What the production chain actually does
(`src/nova2026/auditory/streaming.py:45-48`):

```python
        self.resampler = Resampler(
            settings.input_sfreq, settings.output_sfreq, n, quality="auto",
            max_age_seconds=3.0, reserve_seconds=1.0, allow_qq=True, strict=True,
        )
```

Measured: with `allow_qq=True` the selection is `'QQ'` at both 128→64 Hz and
500→64 Hz; with `allow_qq=False` it raises, because the fastest clean preset
(LQ) needs 7.672 s against a 2.0 s budget (`3.0 - 1.0`). `strict=True` does not
constrain this — it only raises when the chosen delay exceeds the budget, and
QQ's delay is far inside it. So `allow_qq=True` is not a sloppy default; it is
what makes the chain constructible, and the precondition stated in the streaming
package's own error message ("only if an earlier stage already band-limits the
signal") is satisfied, because the 3rd-order 1-9 Hz band-pass runs before the
resampler (`streaming.py:79-80`). The residual risk is small but real and
unpinned by any test, and `ResamplerQualityWarning` is printed on every run.

### A8 — LOW — dead code: the live adapter is unused, and two aligners coexist

`scripts/auditory/runner.py:96-115` defines `LiveAuditoryAdapter`, which is
referenced nowhere in the tree (only the generated UML and brief under
`output/` mention it). The live path uses `TimestampedAudio`, the replay path
uses `EnvelopeBuffer`: two independent implementations of the same
timestamp→envelope mapping. They need to be kept in sync, and the dead adapter
is the visible sign of the drift.

### A9 — LOW — re-running a CLI silently overwrites the previous run's outputs

`scripts/auditory/live.py:176-179`, `replay.py:141-148`, `evaluate.py:96-98` all
do `mkdir(parents=True, exist_ok=True)` and then write `mixed.wav` /
`timing.json` / `estimates.json` / `metrics.json` in place. Re-running with the
same per-subject `--out` destroys the earlier session with no warning. The rest
of the project deliberately refuses this (`recording.py:155-158` raises on an
existing run database; snapshots are copied with `"xb"`).

### A10 — LOW — an out-of-order invalid estimate clears a fresh decision

`src/nova2026/auditory/controller.py:41-53`: the invalid/non-finite branch runs
before the monotonic guard `if estimate.evidence_end <= self.evidence_end:
return`, so a stale invalid estimate resets `selected` to neutral while
out-of-order *valid* evidence is correctly ignored. Unreachable through the
shipped paths (a single-slot `LatestEstimate` delivers in order), so this is an
API inconsistency that contradicts the D10 wording for the invalid case.

### A11 — LOW — a switch delay is satisfied by a single matching block

`src/nova2026/auditory/evaluation.py:83-89` takes the first block whose choice
matches the new label, with no persistence requirement. A one-block flicker is
therefore reported as a zero-delay successful switch while the surrounding
seconds are counted as wrong suppression. The team's own test brief already
flags "currently first match" as an open question.

### A12 — LOW — `TimestampedAudio.align` rejects `available_at=None`

`src/nova2026/auditory/timing.py:82-84` passes `window.available_at` straight
into `AuditoryWindow`, whose constructor does `float(available_at)`
(`data.py:78`). `EEGWindow.available_at` may be `None`, and the sibling aligner
handles exactly that case (`alignment.py:50-51`), so the same input raises
`TypeError: float() argument must be a string or a real number, not 'NoneType'`.
No shipped caller hits it, because `live.py:128` sets `available_at` first.

### A13 — LOW — documented commands and one documented environment do not hold

- `scripts/auditory/VALIDATION.md:13-17` documents five commands using bare
  `python`. Run verbatim on this machine the first one fails:
  `Ran 15 tests ... FAILED (errors=15)`, exit 1, because bare `python` has no
  `mne`. `scripts/auditory/README.md:218-220` uses `.venv/Scripts/python.exe`
  and works. The two documents disagree.
- `documents/AUDIO_V2_ISSUE_RESOLUTION.md:97` claims "All 251 tests pass on
  Windows/Python 3.14". The count 251 is reproducible; the version is not — the
  venv is 3.13.15 and no 3.14 interpreter exists here.

### A14 — LOW — the demo's offloader can drop windows silently

`scripts/streaming_demo.py:319` submits to `TaskOffloader` with the default
`overflow="drop_oldest"` (`src/nova2026/streaming/offload.py:56`) and never calls
`raise_error()`, although the class docstring says to call it on the submitting
thread (`offload.py:38`). A raising handler is counted in `offloader.failed` and
printed once at the end; it is not part of `StreamStats`, so a `--record` run
persists no record that decisions were dropped or that analysis raised.

### A15 — LOW — the FIF export materialises the whole run in RAM

`src/nova2026/streaming/recording.py:438-443` concatenates every chunk
(`blocks = [data for data, _ in iter_chunks(self.path)]`), then makes a float64
copy for `RawArray`. At 63 channels / 500 Hz / 1 h that is roughly 1.8 GB peak
at the end of an otherwise successful session. No data is lost — chunks are
committed per transaction and the status is written before the export — but the
failure is loud and it skips the JSON sidecar.

### A16 — LOW — operator identities are silently rewritten

`src/nova2026/streaming/recording.py:19,52-55`: `_SAFE_NAME` strips
`. / \ : ? * " < > |` and whitespace, so `--subject "a b"` is recorded as `ab`
with no warning, and two distinct identities can collide onto one directory (the
collision itself then fails loudly). Traversal is impossible and the docstring
admits the behaviour; the issue is only that recorded provenance differs from
what the operator typed without notice.

## Rejected after re-checking (do not re-report)

- **"audio worker errors are silently discarded in `--output wav`"** and the
  related claims about `live.py:144-151`. The reasoning was inverted:
  `live.py:144` raises only when the worker is *still alive*, and line 146
  `if errors: raise errors[0] from eeg_error` runs exactly when it is not — i.e.
  in the normal case. The recorded error is surfaced.
- **"a shutdown-deadline error replaces a real audio error"**. The two are
  mutually exclusive by construction: a worker that raised has already left
  `play()` and is not alive.
- **"`--audio-offset` is not validated for finiteness"**. `runner.py:35` tests
  `not np.isfinite(audio_offset)` as a disjunct, so a non-finite offset always
  raises there; there is no path that skips it.
- **"the margin knob drives two decision rules in one run"**. `replay.py:49`
  does pass the same margin to the controller in every mode, but in
  `hysteresis` the controller's own choice is unused (`selected = held`); only
  the hold rule consumes the margin in that row.
- **"`stream()` has no re-entry guard"**. `stream()` sets `self.initialized =
  False` on entry, so a second call raises `initialize() before stream()`, and
  `initialize()` itself refuses while a worker is alive.
- **"audio errors mask processing errors in the streamer"**. Audio-first with the
  processing error as `__cause__` is the documented rule (D4) and is applied
  consistently in both `live.py` and `streamer.py`.
- **"three consecutive confident windows is not literally enforced"**. Probed
  directly: a sub-margin window resets `pending_count` to 0 unconditionally
  (`controller.py:61-63`), so `strong, weak, strong, weak, strong` never commits
  and three adjacent strong windows do. The claim is accurate as written.
- **"the auditory `strict auto-quality test` exists"** is contradicted, but the
  opposite claim that QQ is an accident is also wrong — see A7.
- **"`player_checks.py` is never run"**. Its own docstring documents the explicit
  invocation, which works: `Ran 5 tests ... OK` in 33 s. It is simply not part of
  the discovered 251.

## Confirmed clean

- **Merge integrity**: no reference anywhere to the moved
  `nova2026.metrics` / `nova2026.trainer`; an AST-level import scan resolves all
  155 `nova2026` imports across 29 modules. `git diff --name-status
  origin/stream HEAD -- src/nova2026/streaming tests/streaming
  scripts/streaming_demo.py` is empty, so the merged streaming package, its tests
  and the demo are byte-identical to the `stream` tip.
- **The four gaps from the earlier independent review are fixed**: channel-width
  broadcast (`decoder.py:26-27` rejects before normalisation), playback stopping
  on EEG failure (`streamer.py:87-102`, `live.py:137-143`), held-out enforcement
  inside the replay library (`replay.py:23`), and the missing final interval in
  duration metrics (`replay.py:94`, `evaluation.py:64`).
- **Controller semantics**: duplicate and equal `evidence_end` cannot re-apply or
  flip a decision; expiry is evaluated from `now` in both `update` and `choice`;
  weak and invalid evidence return to neutral; only the non-selected track is
  ducked. Verified by reading and by probe.
- **Decoder geometry**: design row *t* is exactly `normalised eeg[t..t+lag]`,
  trailing lag rows are excluded in `fit` and `score` alike, and
  `pipeline.predict` reports `evidence_end` at the last scored row. Normalisation
  is frozen at fit and preserved through save/load.
- **Units**: the µV contract holds end to end. `live.py` applies `unit_scaler`
  exactly once, `ingest`/`prepare` apply none, and `Repair(source_unit_exponent=
  -6)` is a no-op — there is no double conversion. PCM scaling in `read_audio` is
  correct for 8/16/24/32-bit and float input.
- **Serialisation**: models and trials load with `np.load(..., allow_pickle=False)`
  and are shape-, finiteness- and contract-checked. `eval`, `exec`, `shell=True`
  and `yaml.load` appear nowhere. A1 is the single exception in the tree.
- **Secrets and machine layout**: no credentials, tokens or absolute user paths
  in any source file; the project root is derived from `.git` discovery.
- **Failure surfacing on the live path** (the class that matters most here):
  `Acquire` raises on disconnect, callback failure, staleness and no-data;
  NaN-timestamp windows stop the run through `Recovery.watch`; `TimestampedAudio`
  marks `audio_unavailable` / `audio_clock_discontinuity`; the audio callback
  turns a device or clock error into `CallbackAbort` plus a re-raised
  `errors[0]`. No bare `except: pass` in product code.

## Limits of this audit

- No hardware was touched: no amplifier, microphone, sound device, loopback
  measurement or human calibration. Every timing and acoustic claim remains
  exactly as unverified as the repository already says it is.
- No real KU Leuven or AASD data was loaded. The Zenodo archive referenced by the
  loader could not be checked, so the open question from the loader review
  stands: if that release contains within-trial attention switches,
  `load_kuleuven`'s single constant label per trial would be wrong after each
  switch.
- The audit is static plus targeted probes. Nothing here is a substitute for the
  missing hardware measurements, and no scientific claim (accuracy, benefit,
  coverage) is re-derived.
- Findings are reported at the severity of their practical consequence on this
  branch, not of their worst-case generality.
