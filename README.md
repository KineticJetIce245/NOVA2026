# NOVA2026

EEG-based **auditory attention** for the NOVA Buildathon 2026. A listener hears two
speech streams; a Ridge decoder on their EEG decides, window by window, which of the
two they are attending to. The decision is published as version-1 packets to the
vendored [attune-ui](https://github.com/yut31/attune-ui) front end, which attenuates
the *un*attended stream - attenuation only, never gain.

This is a **research prototype**, not a product: everything the software does and
does not claim is written down in [`VALIDATION.md`](VALIDATION.md) - what is claimed,
the scope of every number, the failure modes a user must expect, and what was
deliberately not measured - plus the per-step record in §3.8 of
[`final_connection.md`](final_connection.md).

The wire contract between the backend and the vendored front end is
[`documents/auditory_ui_protocol.md`](documents/auditory_ui_protocol.md); the reasoning
behind the media timeline (and why `0.75 s` is a staleness bound, not an alignment
budget) is [`documents/media_timeline_contract.md`](documents/media_timeline_contract.md).

`final_connection.md` is the frozen contract: the locked design decisions (§3), the
acceptance criteria V1-V9 (§4), the step plan and per-step evidence (§5), and the
decision log D-01..D-35 (§8). Any change that deviates from it must change it first -
no subagent may edit it.

## Status

| Step | What | State |
| --- | --- | --- |
| 1-5.5 | baseline, KU Leuven conversion, envelope precompute, data audit + frozen contract, decoder training/evaluation, offset sweep | **DONE** - evidence in `results/` |
| 6-7 | transport layer (`src/nova2026/transport/`), front end (`apps/attune-ui/`) | **DONE** |
| 8-10 | `AttentionSession` + producer, media timeline + gain path, ANT session import, one-command demo | **DONE** - `results/demo_run_20260914.md` |
| 10.5 + V9 | unattended demorun and the 23-case perturbation matrix | **DONE** - 18 PASS, 5 FINDING, exit 0 (`results/perturbation_20260913-062335.md`) |
| 11 | live hardware (eego/LSL, 20-channel contract, loopback measurement) | **TODO** - no hardware record exists; the real-time path is unverified |
| 12 | closing documentation | **DONE** - `VALIDATION.md`, `documents/auditory_ui_protocol.md`, this README |

Regression state at that point: `scripts/run_tests.py` 760 tests in 6 suites, 0 skipped, all
green, and the front end's own 53 tests pass (`results/demo_run_20260914.md` V6).

The per-step deliverables, their evidence paths and the measured numbers live in
[`final_connection.md`](final_connection.md) §5; nothing in this README replaces them,
and [`VALIDATION.md`](VALIDATION.md) is the readable index of what they do and do not
establish.

## Repository layout

```
NOVA2026/
├── apps/attune-ui/       vendored front end (React 19 + Vite); README.md + PROVENANCE.md
├── datasets/             raw datasets (git-ignored, except the KU Leuven metadata.json)
├── documents/            design, audit and handout record (naming scheme below)
├── models/               trained checkpoints (git-ignored)
├── output/               rendered outputs; the committed API-handout bundle is output/pdf/
├── records/              raw run recordings written by the acquisition path (git-ignored)
├── results/              generated measurement reports (only the reviewable ones are committed)
├── scripts/              runnable entry points, one folder per family (below)
├── src/nova2026/         the reusable library
├── tests/                the suites scripts/run_tests.py drives
├── tmp/                  scratch, and the user's ANT recording under tmp/antneurodata/ (git-ignored)
├── final_connection.md   frozen plan, step plan and decision log
├── VALIDATION.md         what is claimed, with scope, and what is deliberately not claimed
├── pyproject.toml        package metadata and dependency groups
├── refs.bib              bibliography (BibTeX)
└── README.md
```

Where to write code: reusable modules go in `src/nova2026/`, runnable scripts go in
`scripts/<family>/`, and anything a live test imports - even an old module - is a
fixture and stays where it is. Raw datasets are never edited or reorganised;
preprocessing belongs in code.

> `datasets/*` and `models/*` are git-ignored (only their `.gitkeep` is tracked), so
> large raw data and weights never get committed. `results/*`, `records/*` and `tmp/*`
> are ignored too, with named exceptions in `.gitignore` for the reports that are
> meant to be reviewable.

## Set up the environment

> **Before running `uv sync`:** if you want the **CUDA build of PyTorch**, uncomment
> the `[tool.uv.sources]` / `[[tool.uv.index]]` `pytorch-cu132` block at the top of
> `pyproject.toml`. Leave it commented for a CPU-only install.

1. **Create the virtual environment** (once):
   ```
   python -m venv .venv
   ```
2. **Install the package** in editable mode - this also installs the runtime
   dependencies (`mne`, `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`,
   `scikit-learn`, `mne-lsl`) declared in `pyproject.toml`:
   ```
   pip install uv
   uv sync
   ```

Requires Python >= 3.12. The integration work is measured on Python 3.13.15 with
numpy 2.5.2 / scipy 1.18.0 / mne 1.12.1 / torch 2.13.0+cu132 / scikit-learn 1.9.0
(see `results/test_baseline_*.txt`).

Two optional groups matter here:

| Group | For | Install |
| --- | --- | --- |
| `transport` | `fastapi` / `uvicorn` / `websockets` / `httpx` - the same-origin API + WebSocket app in `src/nova2026/transport/server.py` | `uv sync --extra transport` |
| `eeg-ant` | `antio`, needed by `mne.io.read_raw_ant` to read the ANT `.cnt` recordings | `uv sync --extra eeg-ant` |

On Windows the interpreter is `.venv\Scripts\python.exe`; the commands below are
written with the portable `.venv/bin/python` form. `sounddevice` is deliberately
**not** a dependency: the browser's Web Audio API is the only playback path.

The front end has its own environment (Node 22.12+, tested with Node 26.7.0):

```
npm --prefix apps/attune-ui ci
```

## Running the scripts

Every family below is a module path run from the repository root, so `-B` keeps the
tree free of `__pycache__`. All of them refuse to overwrite existing outputs unless
`--force` is given.

### 1. Tests

```
.venv/bin/python -B scripts/run_tests.py          # every suite
.venv/bin/python -B scripts/run_tests.py --suite streaming -v
npm --prefix apps/attune-ui test                  # front end (53 tests)
```

| Suite | Covers |
| --- | --- |
| `tests/streaming` | the real-time streaming package and the `getlive` hardware path |
| `tests/tooling` | the test runner itself |
| `tests/transport` | the version-1 packet transport, the WebSocket and the session lifecycle |
| `tests/visual_detect` | post-stimulus visual detection (COG-BCI PVT) |
| `scripts/auditory/tests` | auditory attention |
| `scripts/dataproc/streaming/tests` | the older live implementation, kept as a fixture |

The runner exits non-zero if a suite fails **or reports no tests at all**. That rule
is not decoration: `tests/visual_detect` holds plain `test_*` functions with their own
runners rather than `unittest.TestCase` classes, so `unittest discover` once printed
`Ran 0 tests ... OK` - green, and empty. A suite whose tests all skipped counts as not
green for the same reason.

### 2. KU Leuven conversion

Turns the raw dataset into one `.npz` per trial (EEG + both candidate envelopes +
labels + group), resolving the `hrtf`/`dry` stimulus question and the `rep_*` story
groups. See §1-§3 of `final_connection.md` for the measured facts it encodes.

```
.venv/bin/python -B -m scripts.auditory.convert --kind kuleuven \
    --input datasets/AAD-KULeuven --out datasets/AAD-KULeuven/converted \
    --stimuli datasets/AAD-KULeuven/stimuli \
    --metadata datasets/AAD-KULeuven/metadata.json
```

`--kind aasd` converts an explicitly mapped CNT recording from a manifest instead;
it never guesses triggers or source identities. The converted tree is git-ignored
(15.15 GB for KU Leuven); `metadata.json` is the committed source of the channel
order and reference convention.

### 3. Envelope precompute

A session **refuses to start** without a precomputed envelope for the audio it is
about to play (plan §3.1), so the reference envelopes are built once, offline:

```
.venv/bin/python -B -m scripts.auditory.envelopes --audio-dir datasets/AAD-KULeuven/stimuli \
    --out datasets/audio
.venv/bin/python -B -m scripts.auditory.envelopes --out datasets/audio --verify
```

One `<audio_stem>.npz` per input (`envelope` at 64 Hz, `timestamps`, and a `metadata`
JSON carrying the source path, its SHA256, both rates and the method). `--verify`
re-checks existing envelopes against the audio on disk. Envelopes are derived from a
dataset, so they live under the git-ignored `datasets/audio/`: the recipe is
committed, the megabytes are not.

### 4. Data audit

```
.venv/bin/python -B -m scripts.auditory.audit_kuleuven            # full report
.venv/bin/python -B -m scripts.auditory.audit_kuleuven --subjects S1,S2 --limit 2   # partial
```

Writes `results/kuleuven_audit.md` plus `results/kuleuven_audit_<stamp>.json`, and
carries the verdict in its exit code. The audit's section 4 is the **frozen feature
contract** (what must not change before training), so re-read it before touching the
conversion, the envelopes or the decoder.

### 5. Training and evaluation

```
.venv/bin/python -B -m scripts.auditory.train_kuleuven --out results --models models
```

Trains and scores both channel contracts (64-channel, comparable with the dataset
paper; 20-channel, the subset the live ANT cap can actually feed) under both schemes
(held-out stories, which is leakage-free; and leave-one-subject-out). It refuses to
report a single accuracy: the labels are imbalanced, so every table carries balanced
accuracy and per-class recalls next to the majority-class null.

Smaller loops, useful while iterating:

```
.venv/bin/python -B -m scripts.auditory.demo --out tmp/auditory_demo
.venv/bin/python -B -m scripts.auditory.replay --trial <trial.npz> --model <model.npz> --out <dir>
.venv/bin/python -B -m scripts.auditory.evaluate --trial <trial.npz> --model <model.npz> --out <file.json>
.venv/bin/python -B -m scripts.auditory.live --trial <spec> --model <model.npz> --stream <lsl> --out <dir>
```

`replay` scores a held-out trial on a virtual clock (never on hindsight), `evaluate`
compares controllers and injected fault conditions, `live` runs the same chain
against a real LSL EEG stream (and refuses `--output play` without a measured timing
profile).

### 6. Front end

```
npm --prefix apps/attune-ui test     # node --test, no build needed
npm --prefix apps/attune-ui run build
npm --prefix apps/attune-ui run dev  # Vite dev server, development only
```

The app is a vendored copy of the teammate's client; its source commit and per-file
hashes are in [`apps/attune-ui/PROVENANCE.md`](apps/attune-ui/PROVENANCE.md), and its
own notes - including the `ATTUNE_PYTHON` variable the payload tests need on Windows -
are in [`apps/attune-ui/README.md`](apps/attune-ui/README.md). The supported path is
the **built `dist/`** served by the same origin as the API; the dev server is a
development convenience and is not exposed.

### 7. Transport and the hardware path

`src/nova2026/transport/` is the version-1 packet transport: `protocol` (envelope
validation), `publisher` (bounded ring + latest snapshot), `sessions` (serialized
commands + producer thread), `media` (the browser-reported playback timeline) and
`server` (the same-origin FastAPI app, WebSocket and SPA mount). Import the app with:

```python
from nova2026.transport.server import create_app   # needs the `transport` extra
```

`scripts/auditory_ui/` is the auditory UI path built on top of it: `demo.py` (one command that
verifies the envelopes, renders the stereo asset, starts this app on loopback with the built
`dist/` mounted, drives one real trial at 1x and writes its own evidence), `demorun.py` (the
unattended perturbation matrix), `session.py` (the step-8 evidence command), `media.py` (the
browser stand-in for `POST /api/media/control`) and the Node evidence scripts. Section 8 has the
commands and their order.

`scripts/getlive/` is the EEG-amplifier path: it connects over LSL, runs the same
streaming package the suites cover, and prints an acceptance verdict per window.

```bash
.venv/bin/python -B -m scripts.getlive.probe                       # what is on the network?
.venv/bin/python -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30
.venv/bin/python -B -m scripts.getlive --sfreq 500 --source-units uV --record records/
.venv/bin/python -B -m scripts.getlive.compare_cnt <run-dir> <export>.cnt
.venv/bin/python -B -m scripts.getlive.replay_run records/nova2026/<run>
```

Only two options have no default, because a wrong guess fails *silently*:

| Option | Value | Why it cannot be defaulted |
| --- | --- | --- |
| `--sfreq` | the amplifier's rate, as `probe.py` reports it | the rate is set in the control software and is not reliably advertised over LSL, so the pre-flight check compares the assertion with the outlet and stops the run when they disagree |
| `--source-units` | the outlet's declared `ch_units` (`V`, `mV`, `uV`, `nV`) | LSL carries a unit *label*, not a conversion; a wrong label scales every channel by a power of ten (this amplifier declares `Volt` while sending microvolts) |

`python -m scripts.getlive --help` lists the other 50 options in 11 groups;
[`scripts/getlive/README.md`](scripts/getlive/README.md) explains the failure modes
and the measurement behind each default.

### 8. The auditory UI path: demo, demorun, calibration, ANT import

Order matters: a session refuses to start without the reference envelopes, and the demo and the
perturbation matrix both need a fitted decoder. `models/*` and `datasets/*` are git-ignored, so a
fresh clone builds them first (sections 2-5 above).

```
# the demo (plan step 10, acceptance V1-V5): one command, one real trial at 1x,
# the built front end served from the same origin
.venv/bin/python -B -m scripts.auditory_ui.demo --browser

# the perturbation matrix (V9) and the unattended demorun (step 10.5):
# 1 clean case + 15 injected faults + 7 adversarial EEG inputs, each in its own process
.venv/bin/python -B -m scripts.auditory_ui.demorun
.venv/bin/python -B -m scripts.auditory_ui.demorun --only C11 --strict

# the calibration behind the operating point (margin vs coverage, held-out stories only)
.venv/bin/python -B -m scripts.auditory.margin_calibration --out results \
    --contracts 64ch 20ch --histories 5 10 30 60

# the ANT import (V8): marker table, the two exclusions, the 20-channel contract map
.venv/bin/python -B -m scripts.auditory.antneuro --data-root tmp/antneurodata \
    --out datasets/AAD-ANT --buffer-seconds 0.5
```

Three things a reader should know before quoting anything these produce:

* **The demo's margin is not the shipped default.** `demo.py` defaults to the calibrated
  `0.05` operating point (decision D-29); the evidence CLI `session.py` defaults to the
  conservative `MIN_MARGIN = 0.5`; both write the effective value into their run record.
* **"It started" does not mean the source matches the model.** Missing envelopes refuse the start
  with an HTTP 409 that names the file, and a wrong channel count is refused rather than truncated,
  but a mismatched candidate length or sample rate degrades *per window* instead of refusing at
  startup (perturbation cases C15 and E3, recorded as finding D-35).
* **`sync` says `unobserved` on this path**, because no clock has been fitted anywhere; and the
  front end's `|Δt| ≤ 0.75 s` clause bounds report staleness, not alignment. The alignment budget
  is ±100 ms. See `VALIDATION.md` §2.3/§4 and `documents/auditory_ui_protocol.md` §7.

## Documentation and derived-file naming

The AI-written record is kept in full; only its names were normalised. The scheme is
`<topic>_<kind>[_<stamp>].<ext>` - subject first, then what kind of document it is,
then `YYYYMMDD-HHMMSS` when the file is a point-in-time artifact:

| Where | Kind tokens | Examples |
| --- | --- | --- |
| `documents/` | `design`, `audit`, `guide`, `contract`, `protocol`, plus the retired-module record | `timebase_design.md`, `media_timeline_contract.md`, `auditory_ui_protocol.md`, `audio_v2_audit_2026-09-12.md`, `streaming_guide_from_zero.md`, `riemann_live_pipeline_retired.md` |
| `results/` | `audit`, `conversion`, `baseline`, `metrics`, `report`, `sweep`, `averaging`, `incremental` | `kuleuven_audit_<stamp>.json`, `aad_<stamp>.json`, `test_baseline_<stamp>.txt`, `antneuro_testset_report.md` |
| `tmp/` | never committed; `scratch/`, `reports/`, `logs/`, `notes/` group the kinds | `tmp/scratch/antneuro_probe.py`, `tmp/reports/ts_check_raw_run1.json` |

`documents/` holds the prose (design records, audits of the audio_v2 review, the
from-zero streaming guide, the handout PDFs and their `typst/` sources);
`results/` holds generated measurements and is the home the plan fixes for them
(§3.8); `records/` holds raw run recordings written by the acquisition path. Module
documentation stays **next to the code it documents** - `scripts/auditory/README.md`,
`scripts/getlive/README.md`, `scripts/visual_detect/README.md`,
`scripts/dataproc/streaming/README.md` - because a reader standing in that folder has
to find it there. The two `VALIDATION.md` files that live inside those folders are the
historical audio-v2 records (their streaming sections are explicitly marked
superseded); the root [`VALIDATION.md`](VALIDATION.md) is the one that covers the
integration work. The two places outside `documents/` that name a document,
`tests/streaming/test_timebase.py` and `scripts/getlive/live.py`, both point at
`documents/timebase_design.md`, which is its current name.

## What is deliberately not claimed

- **No hearing or comprehension benefit.** The system attenuates one of two streams;
  whether that helps anyone understand anything is not measured.
- **No live-subject result.** Decoding numbers come from the KU Leuven dataset.
  The user's own ANT recording is a *dichotic* (one stream per ear) rehearsal, not
  the target "both streams in both ears" mixture, and the hardware path has not been
  run on a person.
- **No accuracy headline.** On held-out stories the balanced accuracy is about 0.62
  (64 channels) and 0.60 (20 channels) against a 0.66 majority-class null, so raw
  accuracy alone is *below* the null and is never quoted on its own. Cross-subject
  (leave-one-subject-out) results are near that null; the front end is required to
  show which scheme a number came from.
- **No latency compensation.** Only measured offsets are reported; nothing is
  corrected on the basis of a number that was not measured.
- **No amplifier, microphone or headphone calibration** was performed for the
  numbers in `results/`; `scripts/getlive/` exists so that it can be, with the
  assertions recorded in the run's provenance.
- **No browser has executed the front end.** Every UI result is Node executing the vendored
  modules (`protocol.js`, `state.js`, `decoders.js`, `mediaAudio.js`, `Dashboard.js` under
  `react-dom/server`). No screenshot exists and none is claimed: layout, interaction and the Web
  Audio graph are unverified.
- **No acoustic output measurement.** `TimestampedAudio.diagnostics()` needs a real DAC, so block
  timing and drift in ppm are unavailable on the replay path and the run record says so
  (`available: false`) instead of inventing a figure.
- **No result on the ANT test set.** The operator's recording was imported (markers, labels, the
  20-channel map) but never decoded; acceptance criterion V4's 20-channel ANT number does not
  exist. The ANT recording is also *dichotic* - a rehearsal of the task, not the target
  both-streams-in-both-ears mixture.

The complete list, each item with the evidence that does or does not support it, is
[`VALIDATION.md`](VALIDATION.md) §3; the failure modes a user must expect are in §4.

## References (`refs.bib`)

All citations live in the BibTeX file `refs.bib` at the repo root. Whenever you read
or use a source, add one entry per source and cite it in the docs via its key (e.g.
`@hinss_2022_6874129`). Most sources offer a **Cite / Citation** button with a
`BibTeX` option; paste the entry in and keep the key unique and descriptive.
