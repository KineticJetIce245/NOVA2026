# syncing_to_another_machine.md — what git carries, what it does not, and how to refresh the drive

**中文摘要**：这份文件是出门前读的。`git pull` 只带代码，`models/`、`datasets/`、`tmp/`、
`apps/attune-ui/dist/` 这些被 `.gitignore` 排除的资产它一律不带；`G:\NOVA2026` 上已经放好了
演示需要的那一份（已逐个核对）。本文给出：哪些资产是演示必需、哪些只有训练需要、哪些谁都不需要；
外部盘现状与实测校验结果；**在 Mac 上**重建环境与刷新资产的准确命令；以及会静默出错的几个坑
（包络记录的源路径、`dist/` 是特定提交的构建产物、`.venv`/`node_modules` 不能跨机器、训练特征
缓存不在盘上、`._*` 与 CRLF、Python 版本、shell 语法不通用）。

Companion to `RUN_ON_THIS_MACHINE.md` (on the drive). Where the two disagree, this document is
the newer measurement: it supersedes that file's §1 (Windows-only setup) and §4 (the envelope
recorded path, now fixed and regenerated on the drive).

**Target machine: macOS.** Measured on the source checkout (`C:\Files\git\NOVA2026`, Windows,
2026-09-14) and on the external copy (`G:\NOVA2026`, exFAT, branch `integration` @ `351f76b8`).

---

## 1. What git carries, and what it does not

`git pull` on the Mac brings **code only**: `src/`, `scripts/`, `apps/attune-ui/src/`,
`apps/backend/`, `tests/`, `documents/`, `pyproject.toml`, and the two data files that are
deliberately tracked (`datasets/AAD-KULeuven/metadata.json`, `models/.gitkeep`, `datasets/.gitkeep`,
`datasets/audio/.gitkeep`). Everything below is ignored — measured with
`git check-ignore -v <path>` on this checkout, sizes with `Get-ChildItem -Recurse -File -Force`.

| Ignored path | Ignore rule (measured) | Size here (source) | On `G:` | Needed for the demo | Needed for training |
| --- | --- | --- | --- | --- | --- |
| `models/` | `.gitignore:13 models/*` (`!models/.gitkeep`) | 3 files, < 0.1 MB (2 `.npz` + `.gitkeep`) | 3 files, < 0.1 MB — **hash-identical** | **yes** — both decoders | yes (scoring) |
| `datasets/` | `.gitignore:2 datasets/*` | 3 271 files, 89.9 GB | 2 652 files, 69.1 GB | **yes, but only 4 subtrees ≈ 2.68 GB** | partly |
| ├ `datasets/AAD-KULeuven/stimuli/` | `.gitignore:11 datasets/AAD-KULeuven/*` | 32 files, 725.2 MB | 32 files, 725.2 MB | yes (the 16 `_dry.wav`) | yes |
| ├ `datasets/AAD-KULeuven/converted/` | same | 320 files, 15 515.4 MB (S1–S16) | 40 files, 1 940 MB (**S1, S2 only**) | yes (S1/S2, 40 trials) | no — needs S3–S16 |
| ├ `datasets/AAD-ANT/` | same | 5 files, 15.7 MB | 5 files, 15.7 MB | yes (the user's own session) | no |
| ├ `datasets/audio/` | `.gitignore:22 datasets/audio/*` | 18 files, 1.9 MB | 18 files, 1.9 MB, **regenerated for portability (§4)** | **yes** — a session refuses to start without them | yes |
| └ `datasets/COG-BCI/`, `CAP-POS/`, `NOVA-ORI/` | `.gitignore:2 datasets/*` | 66 999.7 MB (COG-BCI) | 66 294.2 MB (COG-BCI, 2 503 files) | no — unrelated pre-existing data | no |
| `tmp/` | `.gitignore:44 /tmp/` | 171 files, 877.7 MB | 27 files, 340.9 MB | no (only to re-import the ANT raw recording) | no |
| `output/` | **partly ignored**: `.gitignore:39 /output/auditory_demo/`, `:40 /output/auditory_ui/` | 359 files, 256 MB (9 tracked files under `output/pdf/` = 0.1 MB; `output/auditory_ui/` = 255.9 MB ignored renders) | 19 files, 0.1 MB (the 9 tracked + 9 `._*` stubs) | no — rendered per run | no |
| `apps/attune-ui/dist/` | `apps/attune-ui/.gitignore:2 dist/` | 3 files, 0.2 MB | 3 files, 0.2 MB — **present but divergent** (§3) | **yes** — `demo.py` serves it at `/` | no |
| `apps/attune-ui/node_modules/` | `apps/attune-ui/.gitignore:1 node_modules/` | 314 files, 28.2 MB | **absent** | no (only for the interface's own 53 tests) | no |
| `datasets/auditory_features/` | `.gitignore:2 datasets/*` | 320 files, 1 617.4 MB | **absent** | no | **yes** — and it is *derived*, so it must be rebuilt |
| `.venv/` | `.gitignore:33 .venv` | 29 906 files, 3 192.8 MB | **1 file, 1 067 B: an `XSym` stub → `/Users/kji245/venvs/nova2026`** | no — machine-specific | no |

Also ignored and easy to forget: `uv.lock` (`.gitignore:36`) — `uv sync` re-resolves it, which
needs network; `results/*` is ignored except the committed `.md`/`.json` exceptions
(`results/.gitkeep`, `!results/test_baseline_*.txt`, `!results/kuleuven_conversion_*.json`).

**Answer to "what will git not sync that matters":** `models/`, `datasets/audio/`,
`datasets/AAD-KULeuven/stimuli/`, `datasets/AAD-KULeuven/converted/S1+S2`, `datasets/AAD-ANT/`,
and `apps/attune-ui/dist/` — all six are already on the drive. Nothing else is required to run
the demo.

---

## 2. What is already on `G:\NOVA2026`, verified

Method is stated per row because "verified" means different things at different sizes.

| Artifact | How it was verified | Result |
| --- | --- | --- |
| `models/auditory_kuleuven.npz` | SHA256 both sides | identical — `86AE5989DCB4…` |
| `models/auditory_kuleuven_live20.npz` | SHA256 both sides | identical — `93A1C8F1B695…` |
| 16 `stimuli/*_dry.wav` + `left_mono.wav` + `right_mono.wav` | the recorded `source_sha256` inside each regenerated envelope equals the old one — i.e. the drive's WAVs hash to what the source-machine envelopes were built from | identical |
| 16 `stimuli/*_hrtf.wav` | file count + total bytes (725.2 MB both sides) + name sets | identical names/sizes; not individually hashed (unused by the demo) |
| `datasets/audio/*.npz` (18) | regenerated on the drive from the drive's own audio; old→new arrays compared elementwise | `max|Δenvelope| = 0`, `max|Δtimestamps| = 0`, `source_sha256` unchanged; only `source_path` and `generated_at` differ (§4) |
| `datasets/AAD-KULeuven/converted/S1,S2` (40 files, 1 940 MB) | count + bytes + name sets vs source | present, same names; **deliberately partial** (S3–S16 absent, ~13 GB) |
| `datasets/AAD-ANT/` (5 files, 15.7 MB) | count + bytes + name sets | present, same names |
| `tmp/antneurodata/` (27 files, 340.9 MB) | count + bytes vs source (13 files, 872 MB) | partial by design: the raw `.cnt`/`.evt` and the 3 actually-played WAVs; 6 WAVs are duplicates/unplayed and stay behind |
| `apps/attune-ui/dist/` (3 files) | asset file names both sides | **STALE/DIVERGENT — see §3** |
| `output/pdf/` (9 tracked files) | present on the drive | carried by git anyway |
| `datasets/COG-BCI` | file count + bytes, before and after this work | **2 503 files / 69 514 534 399 bytes — unchanged** |

**Missing from the drive and known to matter:**
`datasets/auditory_features/` (training), `apps/attune-ui/node_modules/` (the interface's own
tests), a usable `.venv/`, and the 14 other KU Leuven subjects. **Missing and harmless:**
`output/auditory_ui/` (255.9 MB of rendered demo audio, regenerated per run).

---

## 3. `dist/` is a build artifact of a specific commit — measured divergence

The built interface on the drive and the one in this checkout are **different builds**:

```text
source (C:): index.html | assets/index-CBYAnOBp.js | assets/index-PFFYOI6A.css
drive  (G:): index.html | assets/index-Bl4SuvjD.js | assets/index-BQ2Co7ya.css
```

Content-hashed asset names differ, so the JS/CSS the drive serves is **not** the source you get
from `git pull`. Symptom if you ignore it: the served interface silently disagrees with the
`apps/attune-ui/src/**` you are reading, and no test fails. `apps/attune-ui/src/**` is under active
edit in the source checkout right now, so treat any `dist/` older than your checkout as wrong.

Fix on the Mac, after pulling (this is also the only way to be sure):

```sh
npm --prefix apps/attune-ui ci
npm --prefix apps/attune-ui run build
```

Copying the source machine's `dist/` over is a valid shortcut **only** if you know the source tree
was not modified after its last build — it currently is, so rebuild.

---

## 4. The envelope `source_path` (fixed and regenerated on the drive)

Every `.npz` under `datasets/audio/` records the audio file it came from, and a session re-hashes
that file to catch a stimulus that changed under a stale envelope. The old envelopes recorded
**absolute** paths on the generating machine:

```text
before  C:\Files\git\NOVA2026\datasets\AAD-KULeuven\stimuli\part1_track1_dry.wav
after   datasets/AAD-KULeuven/stimuli/part1_track1_dry.wav
```

The old retry (`Path(recorded)` then `envelope_path.parents[1] / recorded`) is a **no-op for an
absolute path**, so on any other machine nothing resolved, `source_sha256` was never compared, and
the demo started with the stale-envelope guard silently off — while
`… -m scripts.auditory.envelopes --out datasets/audio --verify` failed loudly, because that entry
point passes the recorded location explicitly. Same tree, opposite verdicts.

All 18 envelopes on the drive are now recorded **relative to the repository root, POSIX
spellings**, and verified there:

```text
18/18 envelopes loaded with the SHA256 comparison armed; no resolved path is on C:
$ python -B -m scripts.auditory.envelopes --out datasets/audio --verify   # exit 0, 18 × ok
$ cd / && python -B -m scripts.auditory.envelopes --out /Volumes/<drive>/NOVA2026/datasets/audio --verify   # exit 0
```

Rules that follow from this:

* **Never overwrite `G:\NOVA2026\datasets\audio\` from the Windows checkout.** The source
  checkout's own envelopes still record `C:\…`, which is exactly the shape that goes inert when
  copied. If you do overwrite them, regenerate at the destination.
* Regenerate on the machine that will read them (`--out datasets/audio --force`, §6). The
  regenerate is content-preserving: `max|Δenvelope| = 0` and `max|Δtimestamps| = 0` measured over
  all 18 files, with the band (`[1.0, 9.0]`), rate (64 Hz) and generator version unchanged.
* A record that resolves nowhere is a *failure* under `--verify` and an *unchecked* pass at session
  start. The verification code now also re-roots an **absolute** record at the repository-relative
  tail of it, and normalises `\`-separated relative records to `/`, so both spellings survive the
  Windows → macOS move. A Windows-absolute record read on macOS cannot be repaired by any fallback
  (POSIX `pathlib` sees `C:\a\b` as one file name); the drive no longer contains one.

---

## 5. Traps, each with its symptom

1. **Envelope recorded paths** — symptom: the demo starts and looks fine on the Mac, but the
   stimulus hash is never checked; `--verify` fails with `Source audio is missing: C:\…`. Cause and
   fix in §4.
2. **`dist/` is a build artifact of a specific commit** — symptom: the served interface disagrees
   with `apps/attune-ui/src/**` and nothing fails. Fix: `npm --prefix apps/attune-ui ci && npm
   --prefix apps/attune-ui run build` after every pull that touches `src/**`. Measured divergence
   in §3.
3. **`node_modules/` is machine-specific** — symptom: `npm test`/`npm run build` fail with missing
   native binaries if you copy the tree between machines. The drive has none; `npm ci` on the Mac.
4. **`.venv/` cannot be carried, and the one on the drive is a foreign stub** — measured: the
   entry is a single **1 067-byte `XSym` file** whose target is `/Users/kji245/venvs/nova2026`, a
   *different* checkout's environment written by whoever prepared the drive. On Windows it is not
   even a directory; on the Mac it is a dangling symlink that `python -m venv .venv` will refuse or
   write through. Symptom: `No module named fastapi`, or `venv` errors on an existing path. Fix:
   `rm -f .venv` before creating the environment, and never "repair" it. On macOS the interpreter
   lives at **`.venv/bin/python`**, not `.venv\Scripts\python.exe`.
5. **`datasets/auditory_features/` is absent, so training cannot run there** — 320 files,
   1 617.4 MB of derived per-trial features (`scripts/auditory/feature_cache.py`,
   `CACHE_ROOT = REPO/"datasets"/"auditory_features"`), read by training and the shift sweep. It is
   *derived*, so it can be rebuilt, but rebuilding needs the converted trials for S3–S16 (280 files,
   ~13 GB) which are also absent. The demo never reads it. Symptom:
   `FileNotFoundError`/empty feature set from `train_kuleuven`, not a demo failure.
6. **Line endings** — symptom on macOS: `bad interpreter: /bin/sh^M`. Measured: the repository has
   **no `.gitattributes`**, and its only `.sh` file in the tracked tree
   (`scripts/getlive/live_demo.sh`) currently has **0 CRLF lines**, so nothing is broken today. If
   you add a `.sh`, or edit one on Windows with `core.autocrlf=true`, make it LF; the durable,
   narrow fix is a repo-root `.gitattributes` containing exactly
   `*.sh text eol=lf`. *(Deliberately not added here: it is outside the files this change was
   allowed to touch.)*
7. **`._*` AppleDouble sidecars and `.DS_Store`** — measured: **833 `._*` files already on the
   drive**, 0 `.DS_Store`. macOS on exFAT writes one beside every file it touches. Symptom: any
   glob, count or copy silently includes them — `*_dry.wav` counts, `du` totals, `for f in *`
   loops that then treat `._part1_track1_dry.wav` as audio. Always exclude `._*` and `.DS_Store`
   (see the commands in §6) and count data with `find … ! -name '._*' | wc -l`.
8. **Python version** — `pyproject.toml` says `requires-python = ">=3.12"`; the macOS system
   `python3` is usually older and will fail on the type syntax in this tree. Get 3.12+ first:
   `brew install python@3.12`, or the python.org installer, or `uv python install 3.12`, then build
   the venv with that interpreter explicitly.
9. **Shell syntax is not portable** — `$env:ATTUNE_PYTHON='…'` is PowerShell only;
   `ATTUNE_PYTHON=… npm …` is sh/zsh only; `.\` separators do not exist in sh; `python3` is not
   `.venv/bin/python`. Use the form for the shell you are actually in (§6 shows both).
10. **Local modifications on the drive block `git pull`** — the drive currently has three modified
    tracked files (the §4 fix) and one untracked file (`RUN_ON_THIS_MACHINE.md`). Symptom:
    `error: Your local changes to the following files would be overwritten by merge`. Fix in §6
    step 1; those three files are byte-identical to what this change commits (SHA256 prefixes
    `980B23ABCB1E`, `167ABC081DAB`, `35506B9993AC`), so discarding the local copies loses nothing.
11. **Never run `git clean -fdx` (or `robocopy /MIR`) on the drive root** — symptom: it deletes
    every ignored asset, including 66 GB of unrelated `datasets/COG-BCI` data and the demo's
    models/stimuli/envelopes. `/MIR` is acceptable only for `apps\attune-ui\dist` (§6).

---

## 6. The exact commands

### 6.1 First setup on the Mac (sh/zsh)

```sh
cd /Volumes/<drive>/NOVA2026

# 1. code: drop the drive's local copies of the portability fix, then pull
git status                       # expect: 3 modified files + RUN_ON_THIS_MACHINE.md untracked
git checkout -- src/nova2026/auditory/envelopes.py \
                scripts/auditory/envelopes.py \
                scripts/auditory/tests/test_envelopes.py
git pull

# 2. a Python >= 3.12, then the environment (the shipped .venv is a foreign stub, not a venv)
brew install python@3.12         # or: uv python install 3.12
rm -f .venv
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip uv
.venv/bin/uv sync --extra transport --extra eeg-ant
#   CUDA: only on a CUDA machine -- uncomment the pytorch-cu132 block in pyproject.toml first.
#   On a Mac leave it commented (CPU/MPS wheels).

# 3. the interface: node_modules is not carried, and dist/ must match the pulled source
npm --prefix apps/attune-ui ci
npm --prefix apps/attune-ui run build

# 4. check the assets you travelled with (all three must pass)
.venv/bin/python -B -m scripts.auditory.envelopes --out datasets/audio --verify
find datasets/COG-BCI -type f ! -name '._*' | wc -l      # 2503
.venv/bin/python -B -m scripts.auditory_ui.demorun        # 23-scenario fault matrix, headless

# 5. run the demo (defaults: trial_008, models/auditory_kuleuven.npz, margin 0.05)
.venv/bin/python -B -m scripts.auditory_ui.demo --browser
```

The interface's own tests need the interpreter named explicitly — and the assignment syntax is
shell-specific:

```sh
# macOS / sh / zsh
ATTUNE_PYTHON="$PWD/.venv/bin/python" npm --prefix apps/attune-ui test
```

```powershell
# Windows / PowerShell
$env:ATTUNE_PYTHON = 'G:\NOVA2026\.venv\Scripts\python.exe'
npm --prefix apps/attune-ui test
```

### 6.2 Re-regenerating the envelopes (either machine, run from the repository root)

```sh
.venv/bin/python -B -m scripts.auditory.envelopes \
    --audio-dir datasets/AAD-KULeuven/stimuli --pattern "*_dry.wav" --out datasets/audio --force
.venv/bin/python -B -m scripts.auditory.envelopes \
    --audio-dir tmp/antneurodata/audio_files/experiment --pattern "*_mono.wav" \
    --out datasets/audio --force
.venv/bin/python -B -m scripts.auditory.envelopes --out datasets/audio --verify
```

### 6.3 Refreshing the drive from this Windows checkout (run before travelling)

Re-runnable as-is; `/XO` copies only newer files, `/XF` skips macOS sidecars. It is **not** `/MIR`
except for `dist`, so nothing else on the drive is deleted.

```powershell
$s = 'C:\Files\git\NOVA2026'; $d = 'G:\NOVA2026'
robocopy "$s\models"                                   "$d\models"                                   /E /XO /XF ._* .DS_Store
robocopy "$s\datasets\AAD-KULeuven\stimuli"            "$d\datasets\AAD-KULeuven\stimuli"            /E /XO /XF ._* .DS_Store
robocopy "$s\datasets\AAD-KULeuven\converted\S1"       "$d\datasets\AAD-KULeuven\converted\S1"       /E /XO /XF ._* .DS_Store
robocopy "$s\datasets\AAD-KULeuven\converted\S2"       "$d\datasets\AAD-KULeuven\converted\S2"       /E /XO /XF ._* .DS_Store
robocopy "$s\datasets\AAD-KULeuven\metadata.json"      "$d\datasets\AAD-KULeuven"                    /XO
robocopy "$s\datasets\AAD-ANT"                         "$d\datasets\AAD-ANT"                         /E /XO /XF ._* .DS_Store
robocopy "$s\tmp\antneurodata"                         "$d\tmp\antneurodata"                         /E /XO /XF ._* .DS_Store
# dist/ is a pure build artifact, so purging stale hashed assets here is correct and safe:
robocopy "$s\apps\attune-ui\dist"                      "$d\apps\attune-ui\dist"                      /MIR /XF ._* .DS_Store
# datasets\audio is deliberately NOT copied: the drive's envelopes are the portable ones (§4).
```

Sanity check afterwards (must print the same three numbers as before you started):

```powershell
(Get-ChildItem 'G:\NOVA2026\datasets\COG-BCI' -Recurse -File).Count        # 2503
(Get-ChildItem 'G:\NOVA2026\datasets\audio' -Recurse -File -Filter *.npz).Count   # 18
Get-PSDrive G | Select-Object Free
```

macOS equivalent, if the drive is reachable from the Mac and you are pulling data **from** it
(`--exclude` keeps the sidecars out):

```sh
rsync -a --exclude '._*' --exclude '.DS_Store' \
      "/Volumes/<drive>/NOVA2026/models/" "$PWD/models/"
```

---

## 7. What the other machine still cannot do

* **Anything that needs the amplifier.** No ANT eego cap or amplifier has ever been connected to
  this project; every "live" path is a replay of a recording. The Mac cannot verify acquisition,
  impedance, triggering or the ANT SDK path, and re-importing raw `.cnt` needs `antio`
  (`uv sync --extra eeg-ant`) plus the raw files, of which only one session's subset travels.
* **A browser-based verification.** All interface evidence in `results/` was produced headlessly
  under Node. The layout, the controls, real Web Audio playback and the sound card's actual latency
  have never been seen in a browser — opening the page on the Mac is the first real browser test,
  and it *creates* evidence rather than reproducing it. Nothing on the drive can substitute for it.
* **The audio loopback measurement.** The audio-to-EEG offset needs a physical loopback (played
  WAV out, recorded back in) on real hardware; there is no software-only substitute, no recorded
  artifact for it, and the drive carries no loopback capture.
* **Training and evaluation.** `datasets/auditory_features/` is absent and S3–S16 (280 trials,
  ~13 GB) are absent, so the training/evaluation numbers in `results/` cannot be reproduced there —
  the demo can be run, the study cannot.
* **The interface's own test suite** until `npm ci` has run on the Mac and `ATTUNE_PYTHON` points
  at a working interpreter (§6.1).
* **Anything that depends on the source checkout's uncommitted state.** `git pull` carries
  committed code only; work in flight in this checkout (modified `apps/attune-ui/src/**`,
  `src/nova2026/auditory/session.py`, `scripts/auditory_ui/**`, …) reaches the Mac only after it is
  committed and pushed.
