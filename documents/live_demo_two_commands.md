# The live demo, in two commands

Amplifier publishing on the network → command 1 → command 2 → a browser page
showing the live decision. This document is the whole procedure, including the
parts that are not ready yet and the failure modes that will actually happen.

Written 2026-09-14 on branch `integration`. Everything below marked **measured**
was measured in this repository; everything marked **not yet measured** has not
been, and the command output says so too.

> **Two things must be settled before any number from this demo means
> anything.** Neither is a step you can skip.
>
> 1. **The audio-to-EEG loopback offset.** It has never been measured on this
>    rig. 100 ms of misalignment costs **0.119–0.150 balanced accuracy**
>    (`results/aad_shift_sweep_*`) — five to seven times the 0.022 that
>    replacing 44 electrodes with the 20 the headset has costs. Until it is
>    measured, nobody can say whether the system is aligned. Command 1 below is
>    the calibration, and command 2 prints a warning and records
>    `audio_alignment.measured: false` until a profile exists.
> 2. **The 20-channel model.** `models/auditory_kuleuven_live20.npz` is the only
>    contract this headset can carry (20 electrodes, 128 Hz input, 64 Hz
>    output). It was fitted on KU Leuven data — **Cz derivation** — while this
>    rig records **CPz at 500 Hz**. That mismatch is not repaired anywhere in
>    this path, and it caps how much any decision can be trusted. Command 2
>    records it under `trust.reference_mismatch` and prints it.

---

## 0. Prerequisites

| Need | Detail |
| --- | --- |
| Amplifier | eego connected, cap mounted, control software open and **recording**, and **Application options → Network Operation → Enable LSL EEG streaming ticked**. Without that tick nothing is published and both commands refuse. |
| Network | This host on the **same network** as the amplifier PC. LSL is multicast: a firewall that blocks multicast blocks LSL. `./live_demo.sh check` lists what is visible before anything else is tried. |
| Rate and gain | Set in the control software; **not** advertised over LSL. They are operator assertions (`--sfreq 500 --source-units uV`) and the pre-flight checks them against what the outlet declares. |
| Audio | A working output device and **headphones**. Candidate A plays in the left ear, candidate B in the right; the unattended ear is attenuated. |
| Browser | Any current browser on this machine. The interface is served from the same origin as the API, so no extra dev server is needed. |
| Python | **3.12 or newer**, with the package installed (`pip install -e .`). Recreate `.venv` on the machine you run from — it cannot be copied between Windows and macOS. |

---

## 1. The two commands

Both launchers resolve their own directory, so they work from anywhere. The
manual command under each is exactly equivalent.

### Command 1 — is the amplifier publishing what this run needs?

```powershell
# Windows
scripts\getlive\live_demo.ps1 source
```
```sh
# macOS / Linux
scripts/getlive/live_demo.sh source
```
```sh
# the manual equivalent, either platform (interpreter path differs)
.venv/Scripts/python.exe -B -m scripts.getlive.live_source      # Windows
.venv/bin/python        -B -m scripts.getlive.live_source       # macOS / Linux
```

**What it prints, in order:** the electrode list it will demand (read from the
model, not typed in), the rate and unit it will assert, then the resolved
outlet's own declaration — name, type, channel count, rate, units, and every
channel label — then which columns carry the 20 electrodes, then
`pre-flight: accepted 20 electrodes by name`, then a **counter-check** that
proves acceptance is a check rather than a formality (the same stream with the
wrong unit declaration is refused), and finally the exact command 2 to run.

**Exit 0** means: the amplifier is publishing the 20 electrodes this headset's
model needs, at the rate and in the units asserted. Nothing was acquired.

```sh
# When the outlet declares positional labels, no units, or a non-EEG lead:
scripts/getlive/live_demo.sh source --mode bridge      # republishes as NOVA_Live
# Bench rehearsal with no amplifier at all (publishes a RECORDING):
scripts/getlive/live_demo.sh source --mode replay
```

`--mode bridge` republishes exactly the 20 electrodes, in model order, typed
`eeg`, with the unit declared, under `NOVA_Live` — use it when the package's
pre-flight refuses the amplifier's own declaration. It runs until Ctrl+C.

### Command 2 — the session in front of the browser

```powershell
# Windows
scripts\getlive\live_demo.ps1 run --sfreq 500 --source-units uV --open-browser
```
```sh
# macOS / Linux
scripts/getlive/live_demo.sh run --sfreq 500 --source-units uV --open-browser
```
```sh
# the manual equivalent
.venv/Scripts/python.exe -B -m scripts.auditory_ui.live --sfreq 500 --source-units uV   # Windows
.venv/bin/python        -B -m scripts.auditory_ui.live --sfreq 500 --source-units uV    # macOS/Linux
```

It resolves the amplifier, connects, runs the package's pre-flight on the 20
electrodes, verifies the two reference envelopes, renders the two candidates to
a stereo file, **checks the model contract against the chain contract before a
port is opened**, serves the built interface from the same origin, prints the
URL, and starts the session when the page connects.

**What the operator should see**

1. The terminal prints the outlet it resolved, the policy in force, the
   envelope check, the rendered media, the gate-key line, and then
   `OPEN THIS URL: http://127.0.0.1:<port>/`.
2. The page at that URL shows the attention card. **Press play in the page.**
   The two candidates start, one per ear.
3. Within about 10 seconds the first decisions appear; when the score gap
   exceeds `--margin` (default 0.05, the calibrated value) the unattended ear is
   attenuated and the card names the attended candidate. When the gap is small
   the card says it is not sure and the volumes stay equal — that is the
   designed behaviour, not a failure (section 3.3: at margin 0.05 it commits on
   52 % of frames; at the old 0.5 it committed on 0.15 %).
4. `Ctrl+C` in the terminal ends the run and writes
   `results/live_demo_<stamp>.json` plus a `.md` beside it. The record holds
   counters, the policy, the declared contract and decision words — **no EEG
   sample, no packet payload, no participant data.**

**Exit 2** from either command means "refused", and the text says which wall was
hit. That is the only exit code the launchers translate into a hint.

### One command, if you do not want to remember the flags

```sh
sh scripts/run_antneuro_demo.sh --sfreq 500 --source-units uV
```

It runs the pre-flight and then the session, with the EEG traces switched on. The
wrapper is transparent: it consumes `--sfreq`, `--source-units`,
`--eeg-display-channel` and `--no-preflight`, and passes everything else to command
2 unchanged. It refuses before starting anything if the audio pair is missing,
because the candidate WAVs live under `tmp/`, which git ignores and which therefore
does not travel with the repository.

`--eeg-display-channel Cz` is what makes the page draw the two EEG traces. Without
it nothing publishes an `eeg_display` packet and the panel reads "Awaiting EEG
display data" for the whole session -- honestly, because nothing was published. The
same flag was added to `scripts/auditory_ui/serve.sh` for the replay route, which
has the same kind of entry point:

```sh
sh scripts/run_kul_demo.sh        # 60 s of a recorded KU Leuven trial, traces on
```

---

## 2. The calibration step, for both platforms

This is deliverable #1 of the two things above: the audio-to-EEG delay. The
full operator procedure is built into the command:

```sh
.venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --procedure   # Windows
.venv/bin/python        -B -m scripts.getlive.calibrate_loopback --procedure    # macOS/Linux
```

Short form:

```sh
# 1. build the click track (no hardware needed)
.venv/bin/python -B -m scripts.getlive.calibrate_loopback --make-clicks
# 2. connect the loopback: headphone output -> a spare electrode input
# 3. measure
.venv/bin/python -B -m scripts.getlive.calibrate_loopback --live --method cable
# 4. hand the profile to command 2
.venv/bin/python -B -m scripts.auditory_ui.live --calibration results/loopback_calibration.json
```

**What it measures:** `offset = (the instant the click appears on the chain's
own time axis) − (the instant playback was requested)`. That includes the sound
card's output buffer, the cable, the amplifier, and LSL delivery — every term
between "playback was requested" and "the sample carries the click". The two
clocks involved (the LSL stamp clock and this process's monotonic clock) are
mapped by one paired reading at the instant playback is requested, so nothing
assumes they are the same clock. The profile records the offset, the per-click
spread, how many clicks were matched and missed, and a **status**: a spread
wider than ±30 ms is reported as "measured, but the spread exceeds the
tolerance", and no clicks at all is "not measurable" rather than a number.

`--method acoustic` (headphones on a real head) works but measures a **larger,
different** quantity — it adds the acoustic path and the auditory evoked
response. The profile says so in its own text: it is an upper bound, not the
value to compensate with.

`--require-calibration` turns a missing profile into a refusal instead of a
warning. Use it for any run whose numbers will be quoted.

---

## 3. Running this from a Mac

The two launchers are equivalent; the differences below are facts, not
preferences.

| Fact | Consequence |
| --- | --- |
| **A `.venv` cannot be carried between platforms.** The interpreter and every compiled wheel are platform-specific. | `.venv/` must be **rebuilt on the Mac**, and so must `apps/attune-ui/node_modules/` (npm installs are platform-specific too). A Windows `.venv` on the external drive will not run. |
| `scripts/auditory_ui/serve.ps1` hardcodes `.venv\Scripts\python.exe`. | That path exists only on Windows. Both launchers here **detect** the interpreter: `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on macOS/Linux, then `python3`/`python` on PATH, and refuse with the creation command if none exists. |
| macOS ships an older system Python; `pyproject.toml` requires 3.12+. | Install one first: `brew install python@3.12`, or python.org, or `uv python install 3.12`. Both launchers check the version and say which one they found. |
| Shell syntax does not mix. | `$env:X='…'` is PowerShell only; `X=… cmd` and `export` are `sh`/`zsh` only; `.\` separators do not work in `sh`, and `PYTHONPATH` uses `;` on Windows and `:` on macOS/Linux. Each launcher uses its own shell's forms. |
| **macOS on exFAT writes `._*` AppleDouble sidecars** beside every file it touches, plus `.DS_Store`. | Anything that globs or counts files must ignore `._*` and `.DS_Store`. Neither launcher globs, so neither can mistake one for an input — but a census of the drive will, and the drive already has some. |
| This checkout has `core.autocrlf=true` and **no `.gitattributes`**. | The committed blob for `live_demo.sh` is LF (verified: `git cat-file blob HEAD:scripts/getlive/live_demo.sh`), and the file on disk is LF, so the drive copy is safe. A *Windows* re-checkout would turn it into CRLF, and a CRLF shebang fails on macOS with `bad interpreter: …^M`. If the launcher is ever re-checked-out on Windows before going to the Mac, run `dos2unix scripts/getlive/live_demo.sh` (or add `*.sh text eol=lf` to `.gitattributes`, which is a repository-wide file and a decision for the main agent). |

**Safe to carry across, verified by inspection:** `models/*.npz` and
`datasets/audio/*.npz` hold only NumPy arrays and JSON strings, native byte
order, **no pickle**, so they read identically on macOS; `apps/attune-ui/dist/`
is static and platform-independent. The source tree itself is plain text.

Build on the Mac, once:

```sh
cd /Volumes/<drive>/NOVA2026
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
npm --prefix apps/attune-ui ci          # only if the built dist/ is not carried over
chmod +x scripts/getlive/live_demo.sh   # an exFAT volume does not keep the mode
```

Then:

```sh
scripts/getlive/live_demo.sh source
scripts/getlive/live_demo.sh run --sfreq 500 --source-units uV --open-browser
```

**Not executed here.** This machine is Windows: `bash -n` is unavailable, so
`live_demo.sh` was reviewed line by line and **not run**. It is `sh`-syntax
only (no bashisms), uses `CDPATH= cd …` for self-location, quotes every path,
and is `set -eu`. The first Mac run should start with `./live_demo.sh` with no
arguments, which only prints usage.

---

## 4. Known failure modes, and what to do

| Symptom | Cause | Action |
| --- | --- | --- |
| `REFUSED: No LSL outlet appeared within 10s` (exit 2) | LSL streaming not enabled, amplifier not connected, wrong network, multicast blocked, or a firewall | Tick **Enable LSL EEG streaming**, confirm the network, run `live_demo.ps1 check` / `live_demo.sh check` |
| `2 outlets could be the amplifier` | Two publishers with the same identity (a stale helper, a second control-software instance) | Stop the other publisher; LSL cannot tell them apart. Or pin `--stream-name` / `--source-id` |
| `The source sampling rate (…) does not match` | Amplifier rate differs from `--sfreq` | Set the rate in the control software, or pass the probed value |
| `Source voltage units for … do not match` | Outlet declares e.g. `V` while `--source-units uV` was passed | Pass the units the probe printed. **This is the single most damaging mismatch**: it scales every value by 10⁶ |
| `Source channel types must identify EEG and EOG correctly` | The outlet declares an EOG lead as `eeg`, or a status channel as `eog` | `--mode bridge` on command 1, which republishes only the 20 electrodes |
| `The outlet does not publish N of the 20 electrodes` | Wrong cap, wrong montage, or the cap is not on the head | Run `probe`; check the montage against plan section 3.11 |
| `REFUSED: the live chain contract disagrees with the model on input_sfreq` (exit 2) | `--pre-resample off` (or an explicit rate) delivers 500 Hz to a chain that must declare 128 Hz | Leave `--pre-resample auto`; it reads the rate off the model's own contract |
| `REFUSED: … on units` | `--source-units V` on a source carrying microvolts, or the reverse | Match the unit the outlet declares |
| Windows rejected `flatline`, run stops after ~15 s | A dead electrode (a dry cap always has some) and `--strict-policy` | Drop `--strict-policy` (the default records faults without rejecting), or name the electrode with `--exclude-channels` |
| Windows rejected `amplitude`, no decision ever commits | Drift beyond the 500 µV default — normal for this hardware (section 5.6) | Declare `--amplitude-limit-uv`, and read the record: with a limit high enough to admit the recording, the check can no longer separate drift from a real spike. Saturation and flatline still judge the window |
| `Input samples are 3 s old` / high lag | Source publishes in large bursts, or the host is busy | Reduce the amplifier's LSL chunk size; close other load |
| `Cannot repair source damage (irregular_timestamps)` repeatedly | The source's timestamps jitter below one sample (`Repair` refuses that at any tolerance) | Command 2 defaults to `--timebase grid`; keep it |
| `REFUSED: no Python interpreter found` | No `.venv` on this machine (or it was built for the other platform) | Build one — the refusal prints the exact command for the platform you are on |
| `REFUSED: … is Python 3.x; this repository requires 3.12` | macOS system Python | `brew install python@3.12` (or python.org / `uv`), recreate `.venv` |
| The page loads but nothing ever happens | The session was not started, or play was never pressed | Press play in the page. The gain gate refuses to act on a playback position the browser did not report (decision D-02) |
| Decisions are all "uncertain" | The margin is too high, or the EEG is not reaching the decoder | Check the packet counters in the run record; the default margin 0.05 commits on ~52 % of frames on our data |

---

## 5. What is verified, and what only the operator can confirm

| Claim | Status |
| --- | --- |
| The contract gate refuses a source whose processing keys disagree, naming the keys | **verified by test** (`tests/streaming/test_live_demo.py`, mutation-verified) |
| The 20 electrodes are selected **by name**, in model order, from a montage published in a different order, over real LSL | **verified by test**, against a real outlet in-process |
| The quality policy, every flag in it, and where each value came from reach the run record | **verified by test** |
| The loopback arithmetic recovers a known injected offset, reports the sign convention, and says "not measurable" instead of inventing a number | **verified by test** on synthetic clicks |
| Command 1 refuses rather than hangs when no outlet is publishing | **verified by test** |
| The amplifier publishes LSL, with which name, labels, order, units and rate | **only the operator can confirm.** No eego hardware was present; run command 1 first, and `probe` prints the declaration |
| The audio-to-EEG loopback offset | **only the operator can confirm**, with the calibration above |
| That the decisions are right | **unproven on this rig.** The model carries a CPz-for-Cz mismatch and the accuracy numbers from KU Leuven are not an expectation for it |
| That a listener hears better | **not claimed anywhere.** The measurements are about a decision, not about hearing |

---

## 6. Where the rest is written down

* `scripts/getlive/README.md` — the acquisition path: outlet resolution, the
  250/500 Hz handling, `TimeBase`/`grid`, the quality flags, the failure table.
* `scripts/getlive/live_source.py` — command 1, in full.
* `scripts/auditory_ui/live.py` — command 2, in full, including what it refuses
  and why.
* `scripts/getlive/calibrate_loopback.py` — the calibration, in full.
* `documents/where_the_demo_stands.md` — what the demo can and cannot do today.
* `final_connection.md` §3.11, §3.17, §4 — the rig's measured properties, the
  constraints inherited from them, and the acceptance criteria (V4).
