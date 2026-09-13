# NOVA2026

Auditory-attention training and replay are documented in
[`scripts/auditory/README.md`](scripts/auditory/README.md).

NOVA Buildathon 2026 project: EEG-based attention-lapse detection using a passive BCI approach.

## Folder structure

```
NOVA2026/
├── datasets/     Raw datasets (git-ignored)
├── documents/    Learning material (PDFs) + typst sources
├── references/   Reference PDFs
├── models/       Saved models (git-ignored)
├── scripts/      Development scripts (scratch / experiments)
├── src/          Python code
├── refs.bib      Bibliography (BibTeX)
└── README.md
```

| Folder | Purpose |
| --- | --- |
| `src/` | **Reusable code goes here.** Organize it into subpackages. |
| `scripts/` | **Development / experimental scripts go here.** Anything still in scratch development (data exploration, prototyping, one-off runs) lives in `scripts/`, organized into subfolders per topic. |
| `datasets/` | **Put raw datasets here.** The folder already exists and is the recommended location. Keep raw data intact (`.set`/`.fdt`, `.cnt`, behavioral logs, channel locations). |
| `models/` | **Saved models / checkpoints go here.** This folder is git-ignored so large weights never get committed. |
| `documents/` | Handouts, challenge briefs, and learning notes (PDF), plus their `typst/` sources. |
| `references/` | Reference papers / device documentation (e.g. `attentivU.pdf`). |
| `refs.bib` | Bibliography — see below. |

> Note: `datasets/*` and `models/*` are git-ignored (only their `.gitkeep` is tracked), so large raw data and weights never get committed.

## Setup the environment

> **Before running `uv sync`:** if you want the **CUDA build of PyTorch**, uncomment the
> `[tool.uv.sources]` / `[[tool.uv.index]]` `pytorch-cu132` block at the top of `pyproject.toml`.
> Leave it commented if you want a CPU-only install.

1. **Create the virtual environment** (only once):
   ```
   python -m venv .venv
   ```
2. **Activate it**:
   - Windows (cmd/PowerShell): `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. **Install the package** in editable mode — this also installs the runtime dependencies (`mne`, `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`, `scikit-learn`, `mne-lsl`) declared in `pyproject.toml`:
   ```
   pip install uv
   uv sync
   ```

Requires Python >= 3.12. The `.venv/` folder is git-ignored, so it never gets committed.

## Tests

Five suites, one command, from the repository root:

```
.venv/bin/python -B scripts/run_tests.py
```

| Suite | Covers |
| --- | --- |
| `tests/streaming` | the real-time streaming package and the `getlive` hardware path |
| `tests/tooling` | the test runner itself |
| `tests/visual_detect` | post-stimulus visual detection (COG-BCI PVT) |
| `scripts/auditory/tests` | auditory attention |
| `scripts/dataproc/streaming/tests` | the legacy live implementation |

`--suite <name>` runs one of them; `-v` streams its output. The runner exits
non-zero if a suite fails **or reports no tests at all**.

That last rule is not decoration. `tests/visual_detect` holds plain `test_*`
functions with their own runners rather than `unittest.TestCase` classes, so
`unittest discover -s tests/visual_detect` prints `Ran 0 tests ... OK` - green,
and empty. It ran that way, unnoticed, until the runner was added, which now
reports its tests like any other suite. A suite whose tests are all skipped
counts as not green for the same reason.

## Running against the real amplifier (`scripts/getlive`)

`scripts/getlive` is the hardware path: it connects to the EEG amplifier over
LSL, runs the same streaming package the test suites cover on the live outlet
for a fixed duration, and prints an acceptance verdict per window. Everything it
needs to know about the amplifier is an argument, because a wrong guess fails
*silently* - the run looks healthy and means nothing.

```bash
# 1. what is on the network, and what does it declare? (metadata only, no samples)
.venv/bin/python -B -m scripts.getlive.probe

# 2. the acceptance run: the two required flags, 30 seconds by default
.venv/bin/python -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30

# 3. the same run, recorded, so it can be replayed offline
.venv/bin/python -B -m scripts.getlive --sfreq 500 --source-units uV --record records/

# 4. check that recording against the amplifier's own .cnt export
.venv/bin/python -B -m scripts.getlive.compare_cnt <run-dir> <export>.cnt
```

On Windows the interpreter is `.venv/Scripts/python.exe`. Steps 1's output is
also where the two required values come from, so it has to run on - or at least
see the multicast of - the machine that publishes the outlet. The full
walk-through, the failure modes and the measurement behind each default live in
[`scripts/getlive/README.md`](scripts/getlive/README.md).

### Required options

Only two, and neither has a default:

| Option | Value | Why it cannot be defaulted |
| --- | --- | --- |
| `--sfreq` | the amplifier's sampling rate in Hz, as `probe.py` reports it | the rate is set in the control software and is not reliably advertised over LSL, so the pre-flight check compares the assertion with what the outlet declares and stops the run when they disagree |
| `--source-units` | the outlet's declared `ch_units` (`V`, `mV`, `uV`, `nV`) | LSL carries a unit *label*, not a conversion; a wrong label scales every channel by a power of ten, which is the difference between an EEG signal and a saturation alarm (this amplifier declares `Volt` while sending microvolts) |

Everything else has a default; `--duration` defaults to 30 s. Run
`python -m scripts.getlive` with neither flag and it names them and points at
the probe.

### All options

52 options in 11 groups. `python -m scripts.getlive --help` prints the same list
with the long help text.

| Group | Option | Default | Meaning |
| --- | --- | --- | --- |
| source and timing | `--sfreq` | **required** | source rate in Hz |
| | `--duration` | `30.0` | seconds to stream |
| | `--out-sfreq` | `128.0` | target rate in Hz after resampling |
| | `--block` | `50` | samples per `Acquire.read()` |
| | `--chunk-size` | `37` | source chunk size (uneven on purpose) |
| windows (defined at the output rate) | `--window` | `2.0` | window length in seconds |
| | `--hop` | `0.5` | window step in seconds |
| | `--capacity` | `6.0` | ring capacity in seconds |
| preprocessing | `--notch` | `60.0` | notch frequency in Hz |
| | `--notch-q` | `30.0` | notch quality |
| | `--lpass` | `1.0` | band-pass low cutoff in Hz |
| | `--hpass` | `45.0` | band-pass high cutoff in Hz |
| | `--order` | `3` | band-pass order |
| | `--warmup` | `2.0` | warm-up seconds before windows are trusted |
| | `--resample-quality` | `LQ` | SoXR preset, or `auto` to measure each preset and pick the cleanest within the delay budget |
| run recording | `--record` | off | recording root, e.g. `records/` |
| | `--subject` | `demo` | subject id stored with the run |
| | `--session` | `synthetic` | session id stored with the run |
| | `--run` | timestamp | run name |
| consumer mode | `--workers` | `0` | `0` keeps analysis in the loop; `N` runs it on N worker threads |
| | `--queue` | `8` | offload queue capacity |
| | `--compute` | `0.0` | simulated analysis seconds per window (a load test) |
| amplifier outlet | `--stream-name` | auto | LSL outlet name, as `probe.py` prints it |
| | `--source-id` | auto | LSL source id, as `probe.py` prints it |
| | `--stream-type` | auto | LSL stream type, exact spelling |
| | `--resolve-timeout` | `10.0` | seconds to keep looking for the outlet |
| | `--connect-timeout` | `10.0` | seconds allowed for the inlet connection |
| | `--expect-channels` | off | with no declared stream type, treat an outlet with at least this many channels as the amplifier |
| channel contract | `--cap` | `auto` | `auto` uses the CA-208 datasheet contract when the declared labels cover it and the declared montage otherwise; `ca-208` demands the datasheet; `declared` always trusts the outlet |
| | `--source-units` | **required** | units the outlet declares |
| | `--channels` | `cap` | `cap`: every EEG electrode of the resolved cap; `model`: only the electrodes the offline classifier uses that this cap also has |
| | `--eog` | `eog` | `eog`: keep the cap's EOG electrode as auxiliary; `drop`: exclude it |
| | `--reference` | profile's | reference already applied by the amplifier, recorded as provenance |
| | `--ground` | profile's | ground of the cap, recorded as provenance |
| | `--upstream` | eego software defaults | upstream processing, recorded as provenance |
| channel quality policy | `--exclude-channels` | none | comma-separated EEG labels known to be dead: still recorded, but never rejecting a window |
| | `--max-bad-channels` | `0` | faulting EEG channels tolerated per fault type and window before quality rejects it |
| | `--no-channel-check` | off | record channel faults but never let them reject a window |
| acceptance | `--min-valid-windows` | `3` | valid windows required for acceptance |
| | `--flat-uv` | `1.0` | mean peak-to-peak below this counts as a flat electrode |
| | `--noisy-uv` | `200.0` | mean peak-to-peak above this counts as a noisy electrode |
| | `--quiet` | off | no per-window lines |
| | `--out` | none | write the acceptance report as JSON |
| | `--role` | `bringup` | run role stored when recording |
| time base | `--timebase` | `grid` | `grid`: place every received sample on a regular grid at `--sfreq` from a counted index, absorbing a drifting or stepping source clock without ever fabricating or dropping a sample; `stamps`: use the source's timestamps unchanged |
| | `--timebase-relock-samples` | `0.5` | grid mode: disagreement tolerated before the grid re-anchors |
| | `--timebase-max-step-samples` | `1.5` | grid mode: a single step at or above this many samples is reported as a suspicious jump |
| | `--timebase-drift-limit` | warn only | grid mode: reject the run when the grid absorbs more than this many samples of drift per 30 s |
| fault tolerance limits | `--repair-amplitude-uv` | `500.0` | largest jump between the two finite samples a repair may bridge, in uV whatever `--source-units` says |
| | `--repair-saturation-uv` | `75000.0` | absolute level that makes a repair unsafe, in uV whatever `--source-units` says; never below `--repair-amplitude-uv` |
| | `--max-recoveries` | `5` | bounded chain restarts allowed before the run stops |
| | `--max-fault-seconds` | `5.0` | longest run of consecutive rejected windows tolerated before the run stops |

The two `--repair-*-uv` limits are operator assertions of a different kind: they
judge the signal in microvolts while `--source-units` decides how the source's
numbers are scaled. A run whose declaration is wrong hits
`--repair-saturation-uv` (a 12.6 mV offset read as volts is 12,600 V) and stops;
raising the rail is then the explicit way to say "I know the declaration is
wrong, keep going". Both are recorded in the run's provenance, so the limits a
run actually used are never a matter of memory.

## Where to write code

- Put reusable Python modules under `src/` (e.g. `src/nova2026/data/`, `src/nova2026/architecture/`).
- Keep it importable: `from nova2026.data.mne_reader import ...`
- Put development / experimental scripts in `scripts/`, organized into subfolders (e.g. `scripts/dataproc/`, `scripts/training/`). Move code here into `src/nova2026/` once it stabilizes.

## Where to put datasets

- Use `datasets/<name>/` — the folder is already set up for this, and it's excluded from git so the large files stay local.
- Don't modify or reorganize the raw files; do preprocessing/cleaning in code instead.
- Save trained models / checkpoints under `models/` (also excluded from git).

## References (`refs.bib`)

All citations live in the BibTeX file `refs.bib` at the repo root. Whenever you read or use a source, add one entry per source and cite it in the docs via its key (e.g. `@hinss_2022_6874129`).

### Getting a BibTeX entry online

Most sources give you a ready-made BibTeX entry via a **Cite / Citation** button, select `BibTex` option. Then just paste the entry into `refs.bib`, keeping a unique, descriptive key.
