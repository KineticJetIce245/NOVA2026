# getlive: does the streaming package work with the real cap?

`nova2026.streaming` was validated against PlayerLSL and recorded data only.
This folder is the hardware counterpart. It answers two questions on the rig:

1. Is the amplifier publishing LSL, and what does the outlet declare?
2. Does the package's preprocessing chain run on that outlet for a whole
   session, and does every electrode of the cap deliver signal?

It is **not tied to one cap**. The channel contract is resolved at run time:
when the outlet publishes exactly the electrodes of a datasheet profile (the
ANT Neuro **waveguard original CA-208** today, 64 channels on an **EEgo EE-22x**
amplifier) that profile is used and its order, reference and ground are checked;
otherwise the contract is built from the montage the outlet declares, at any
electrode count, and the run says so. Known-dead electrodes, which a dry cap
always has, can be excluded from the fault verdict while still being recorded.

## Rig prerequisites

1. Amplifier connected over USB, cap mounted, control software open and
   recording.
2. **Application options -> Network Operation -> tick `Enable LSL EEG
   streaming`.** Without it nothing is published on the network and every
   script here fails at resolution. (Tick `Enable Network Events` only if you
   also need LSL *markers*; this folder does not.)
3. Sampling rate and gain are set in the control software and are **not**
   advertised over LSL. Note them; the rate goes into `--sfreq`.
4. Montage: whichever cap is on the head, with its reference and ground.
5. LSL is multicast: the host must be on the same network as the amplifier
   PC, and a firewall that blocks multicast blocks LSL. Run the probe first.

## Step 1: probe the outlet

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive.probe
.venv/Scripts/python.exe -B -m scripts.getlive.probe --json records/probe.json
.venv/Scripts/python.exe -B -m scripts.getlive.probe --all --cap declared
```

It lists every LSL outlet, then connects to the EEG-like one and prints the
identity, the declared channel labels, types and units, **the cap profile the
live run would use**, and how the declared montage compares with it. Connecting
is **metadata-only**: no sample is acquired, no thread is started and nothing is
recorded. `--no-connect` stays at the resolve level; `--name` / `--source-id` /
`--stream-type` inspect one outlet explicitly; `--all` inspects every outlet.

Read the declared `sfreq` and `ch_units` from here: they are the values Step 2
must be told.

## Step 2: the live acceptance test

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30
```

`--sfreq` and `--source-units` have no defaults on purpose: they are operator
assertions about the amplifier, and the package's pre-flight check compares
them against what the outlet declares. A wrong setting must fail the run, not
pass against a guess. `Ctrl+C` is a normal way to end the run: the report is
still printed and written.

```powershell
# CA-208, all 64 electrodes, 30 s
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30

# Another cap: trust what the outlet declares
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 1000 --source-units V `
    --cap declared --duration 30

# Dry cap with two known-dead electrodes
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --exclude-channels Fp1,F7 --duration 60

# Record the raw run for offline replay
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --record records --subject S01 --session dry --out records/acceptance.json

# A source whose grid is sound: keep its timestamps untouched
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --timebase stamps --duration 30
```

| Option | Meaning |
| --- | --- |
| `--sfreq`, `--source-units` | Required. Declared rate (Hz) and unit (`V`, `mV`, `uV`, `nV`). |
| `--duration` | Seconds to stream; default 30. |
| `--cap auto\|ca-208\|declared` | `auto` (default): the CA-208 contract when the declared labels cover it, else the declared montage. `ca-208`: demand the datasheet. `declared`: always trust the outlet. |
| `--channels cap\|model` | `cap` (default): every EEG electrode of the resolved cap. `model`: the electrodes the offline classifier uses that this cap also has. |
| `--eog eog\|drop` | `eog` (default): keep the cap's EOG electrode as auxiliary. `drop`: exclude it. |
| `--exclude-channels` | Comma-separated EEG labels known to be dead. They are still recorded; they never reject a window. |
| `--max-bad-channels N` | Faulting EEG channels tolerated per fault type and window before quality rejects it. Must be below the EEG electrode count. |
| `--no-channel-check` | Record channel faults but never let them reject a window. |
| `--expect-channels N` | With no declared stream type, treat an outlet with at least this many channels as the amplifier. |
| `--stream-name`, `--source-id`, `--stream-type` | Pin the outlet when several publish. |
| `--notch` | 60 Hz default; use `--notch 50` on 50 Hz mains. |
| `--resample-quality` | `LQ` default (lowest latency); `auto` measures the presets. |
| `--timebase grid\|stamps` | `grid` (default): place every received sample on a regular grid at `--sfreq` from a counted index. `stamps`: keep the outlet's own timestamps. See [The timeline](#the-timeline---timebase). |
| `--timebase-relock-samples` | `grid` only: disagreement tolerated before the grid re-anchors; default 0.5 samples. |
| `--timebase-max-step-samples` | `grid` only: a single timestamp step this large is reported as suspicious; default 1.5. |
| `--timebase-drift-limit` | `grid` only: reject the run when the grid absorbs more than this many samples of drift per 30 s of stream. Unset by default, so drift is warned about and never rejects. |
| `--reference`, `--ground` | Provenance strings; default to what the resolved profile asserts. |
| `--record` with `--subject/--session/--run` | Write the raw run (SQLite + FIF) under that root. |
| `--out` | Write the acceptance report as JSON. |
| `--quiet` | No per-window lines. |

### The timeline: `--timebase`

The outlet's own timestamps are not always usable for placement. On the EE-511 rig
0.48% of the steps were shorter than half a sample, which `Repair` refuses at any
tolerance, so the run died after about two seconds with `Too many data faults` -
every time, on data whose samples were provably intact.

`--timebase grid` stops trusting the stamps for placement and counts instead:
sample *n* is placed at `anchor + n / --sfreq`, and the stamps are read for one
thing only - to measure how far the anchors have drifted from that count. That
measurement is reported, and past `--timebase-relock-samples` the grid re-anchors
by slewing the correction over enough samples that no single step leaves the
consumer's tolerance.

What it will not do is fabricate or drop a sample. The grid has exactly one slot
per sample received, which is the point: the code before it read a stamp step as
lost samples, inserted slots for them, and `Repair` then repaired rows that never
existed.

Use it whenever the source's stamps cannot be trusted:

```powershell
# the rig, and any source whose ts_check fatal% is not 0
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --cap declared --eog drop --timebase grid --duration 30

# a chunk-stamped recorder lands on a clean grid the same way
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 250 --source-units uV `
    --timebase grid --cap declared --eog drop --duration 30
```

Read the result in three places: the `timebase` line of the verdict, and
`timebase_relocked_samples`, `timebase_anchor_rate`, `timebase_relocks` and
`timebase_large_steps` in the JSON report and in the recording's `meta`.
`timebase_relocked_samples` is how much drift the grid absorbed - the number the
old reports called "gaps"; `timebase_anchor_rate` is the source clock's own rate,
so its distance from `--sfreq` is that clock's ppm error.

Two things to know before reading those numbers:

* a **chunk-stamped** source reports one suspicious step per block. That is its
  stamping shape, not damage: fed a fixture with 8 samples per block, the grid came
  out at exactly one sample per step and `Repair` accepted it.
By default the drift is only a warning, because whether it makes a run unusable
depends on what the windows are for. `--timebase-drift-limit N` makes it a verdict
when the analysis is time-locked: N counts samples of drift per 30 s of stream, so
the limit does not depend on how long the run happened to be. The 2-9 samples per
30 s measured on the rig would fail a limit of 1.

* `--timebase` defaults to `grid`, and that move was made on evidence rather than
  on an amplifier run: the rig stopped being available, and the only amplifier this
  project measured fails without the grid (the 0.48% above), while grid mode is a
  near-no-op on a clean source and also regularises a chunk-stamped one. The
  failure and the fix are reproduced against synthetic timelines in
  `documents/streaming_explained.md` §11.15. `--timebase stamps` is the escape
  hatch for a source whose grid is sound and whose timestamps you want untouched.

### Which cap contract is used

| `--cap` | Behaviour when the outlet declares ... |
| --- | --- |
| `auto` (default) | every electrode of the CA-208 datasheet: use it, and check order, reference and ground. Anything else: build the contract from the declared labels and types, and say so. |
| `ca-208` | anything short of the full CA-208 contract: fail with the missing electrodes named. |
| `declared` | always build the contract from the declared labels and types. |

A **declared** contract classifies each channel as EEG or auxiliary:

1. a name with an auxiliary prefix (`EOG`, `EKG`, `ECG`, `EMG`, `RESP`, `GSR`,
   `Temp`, `Status`, `TRIG`, `Marker`, `Misc`, ...) is auxiliary, even when the
   outlet declares it as EEG — control software routinely labels the EOG drop
   lead as EEG;
2. a channel declared `eog` is the EOG auxiliary;
3. everything else declared `eeg`, or with no usable type, is EEG;
4. any other declared type (ECG, EMG, stim, misc) is auxiliary but not EOG, so
   it is reported and dropped rather than contracted.

`--channels model` intersects the cap with the offline classifier's electrode
list, so on a cap other than CA-208 it usually covers only part of it; the
`model_channels` check reports exactly how much and names what is missing.

### What runs

Same chain as `scripts/streaming_demo.py`, in this order:

```
Repair (source units, short NaN/gap runs; excluded columns are held, never fatal)
  -> source units to uV
  -> QualityMonitor (observes raw uV: amplitude, saturation, flatline; names the channels)
  -> 60 Hz notch
  -> 1-45 Hz band-pass (3rd order)
  -> SoXR resample to 128 Hz
  -> 2 s windows every 0.5 s, 2 s warm-up, bounded recovery on damage
```

128 Hz / 2 s is the geometry the offline model uses
(`src/nova2026/config.py`); the run reports whether it matches.

### Bad channels on a dry cap

A dry cap has electrodes that are dead for the whole session. The package's
quality monitor detects them per electrode (`flatline` for a dead contact,
`saturation` for a railed one, `amplitude` for a pop or a drift), and by default
a single one of them rejects every window it touches and the run stops after
`persistent_fault_seconds`. That is the wrong default for a dry cap, so this
script offers three ways out:

* `--exclude-channels Fp1,F7` — the operator's assertion "this electrode is
  dead". Still detected and still printed, never counted, never fatal. The same
  list reaches the repair stage, so a column that never reports a value is held
  at its last finite level instead of exhausting the recovery budget.
* `--max-bad-channels N` — tolerate up to N faulting electrodes **per fault
  type**; the N+1-th still rejects the window. Keeps the "electrode fell off
  mid-run" guard alive.
* `--no-channel-check` — record and never reject. The blunt version.

Every faulting electrode is recorded either way: the per-window line carries
`bad=...`, and the report carries `bad_channel_windows` (label -> window count)
plus `held_rows`. That census is what decides the next session's exclusion list.

### The verdict

Three statuses; only `fail` blocks acceptance. Exit codes: `0` accepted,
`1` rejected or failed, `130` stopped with Ctrl+C.

| Check | `fail` when |
| --- | --- |
| `connection` | the run raised |
| `source_rate` | declared rate differs from `--sfreq` |
| `channel_contract` | the outlet publishes fewer channels than the contract needs |
| `data_flow` | under half of `--duration` x `--sfreq` samples arrived |
| `input_lag` | a consumed block was 3 s old (the package's own guard) |
| `windows` | no valid window was produced |
| `resampler` | startup delay above 3 s |
| `electrodes` | **every** electrode is flat |

Everything else that is worth knowing only warns: dropped extra channels,
timestamp gaps, a slow-but-working source, rejected windows (with the reason
histogram), chain restarts, repaired rows, the flattest and noisiest
electrodes, a geometry that differs from the offline model, the channel-quality
policy in force (`channel_scope`), the faulting electrodes that were seen
(`bad_channels`), and a contract that misses electrodes the offline model needs
(`model_channels`). A dead electrode is a gel/contact problem, not a streaming
problem, so it warns - and the report names the electrodes.

The per-electrode summary uses two prototype thresholds (`--flat-uv` 1 uV,
`--noisy-uv` 200 uV mean peak-to-peak). They are heuristics for a first look,
not amplifier specifications. `--out` writes the full per-channel table.

## Step 3: check the recording against the amplifier's own file

While the control software streams over LSL it also writes its own `.cnt`. That
gives a second, independent copy of the same session, and comparing the two is
the only way to know the relay path changed nothing - it would otherwise be
possible to drop, reorder or filter samples and still produce a run that "looks
fine" on its own.

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive.compare_cnt `
    --cnt ..\Lacroix_Flo_2026-09-12_17-13-32.cnt --out records/cnt_check.json
```

The offset is found by cross-correlating the **first difference** of the two
signals, never the clocks: the run directory name, the LSL stamps and the `.cnt`
header all describe the session but none is exact to the sample. Differencing
also matters for correctness of the verdict - on raw EEG a *wrong* alignment
still reaches `r = 0.9995`, because the amplifier's large DC offsets dominate;
on differences the correct lag is a lone peak at `r = 1.0`.

Reading a `.cnt` needs the `antio` package (`pip install antio`), which MNE
delegates to. Where it cannot be installed, decode the file elsewhere and pass
`--cnt-npy <array.npy>` with a `<stem>_meta.json` sidecar.

Result on the 2026-09-12 EE-511 bring-up session (24 channels, 500 Hz, cap on a
real head), `../Lacroix_Flo_2026-09-12_17-13-32.cnt` as the reference:

| | |
| --- | --- |
| Runs compared | 6 of 7 (the first 30 s run recorded no samples and failed on lag) |
| Sample pairs compared | 88 500, over 2.6 min of signal |
| Alignment | every run at `r = 1.000000`, found independently of any clock |
| Worst disagreement | 0.0039 uV = **half the amplifier's 0.0078125 uV LSB** |
| Samples differing by more than one LSB | **0** |

So the LSL/relay path delivers the amplifier's samples unaltered: same values,
same channel order, no dropped or interpolated samples. That also cross-checks
the run tree's own metadata - each run's first sample lands within 2 s of the
timestamp in its directory name.

One caveat belongs to the *timestamps*, not the data: the chunk stamps rebuild a
grid that runs up to 9 samples (18 ms) ahead of the sample count over a 31 s run.
That is what the reports call "gaps", and `grid_deviation_samples` in the JSON
report carries the number. It does not affect sample values, but it is the error
a time-based downstream analysis inherits.

## Replaying a recorded run without the cap

`records/` outlives the amplifier. `replay_run.py` feeds a recorded run back
through **the live script's own chain and its own acceptance rules** -
`live._prepare`, `live._loop`, `live._finish`, with acquisition swapped for the
recorded blocks - so "would the current code have handled that session?" is a
question you can answer offline:

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive.replay_run records/nova2026/bringup30/run-20260912-172216
.venv/Scripts/python.exe -B -m scripts.getlive.replay_run <run-dir> --timebase stamps
.venv/Scripts/python.exe -B -m scripts.getlive.replay_run <run-dir> --out replay.json --record records/replay/
```

It prints the header, per-electrode table and acceptance verdict a live run
prints, and it says what it replayed. Measured on the six recordings that hold
samples - a seventh run directory is an early bring-up whose chunk table is empty
- each one replayed in both modes, with what the live run of the day produced:

| Recording | live, 2026-09-12 | replay `grid` | replay `stamps` |
| --- | --- | --- | --- |
| `bringup/171856` | 23/38 valid, 15 `interpolated` | **37/41, 0 repairs, USABLE** | 24/41, 6 repairs |
| `bringup30/171935` | 40/58 valid, 14 `interpolated` | **54/58, 0 repairs, USABLE** | 40/58, 5 repairs |
| `bringup30/172216` | **failed**: persistent quality faults, 28/57 | **54/58, 0 repairs, USABLE** | **reproduces the stop**, 0/11 valid |
| `bringup30/172323` | 17/43 valid, 2 recoveries | **50/54, 0 repairs, USABLE** | 30/54, 6 repairs |
| `offload1/172814` | 39/58 valid, 17 `interpolated` | **54/58, 0 repairs, USABLE** | 39/58, 9 repairs |
| `offload8/172727` | 47/58 valid, 7 `interpolated` | **54/58, 0 repairs, USABLE** | 48/58, 2 repairs |

The `stamps` column is the fidelity check that matters: replaying reproduces the
window counts of the day (40 -> 40, 39 -> 39, 47 -> 48) and, on the run that
died, the same `EEG quality faults persisted beyond the allowed duration` stop.
In `grid` every recording reaches the end with repairs at zero and only the four
warm-up windows rejected - which is the claim the live grid path never got to
make, because the rig was gone by then.

What a replay cannot do, and what the tool says so about:

* **No transport.** `input_lag` and `timing_gaps` report `not measured` instead of
  scoring the zeros a replay would otherwise invent; the timeline is scored by the
  `timebase` rule, which a replay does measure.
* **No outlet.** A recording holds the columns the run *contracted*, so the
  amplifier's original montage is not re-checked.
* **No inside-block stamps.** A recording carries the recorder's rebuilt grid, so
  `--timebase stamps` re-tests the steps at the block seams, not the source's
  per-sample stamps. The six recordings above show that is where the rig's steps
  landed.

Options default to what the recording itself used - contract, exclusions, channel
check, window, output rate, limits - so a replay reproduces the session;
`--timebase` excepted, which defaults to `grid` (the live script's current
default) and prints the mode the recording was made with.

## When something fails

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No LSL outlet appeared` | LSL streaming not enabled, amplifier not connected, wrong network/firewall | Tick `Enable LSL EEG streaming`, re-run the probe |
| `2 outlets could be the amplifier` | two publishers with the same identity (a stale helper process, or a second instance of the control software) | stop the other publisher; LSL cannot tell them apart |
| `Source sampling rate (...) does not match` | amplifier rate differs from `--sfreq` | set the rate in the software, or pass the probed value |
| `units ... do not match` | outlet declares e.g. `-6` while `--source-units V` was passed | pass the units the probe printed |
| `Source channel types must identify EEG and EOG correctly` | the outlet declares the EOG electrode as `eeg` (or a status channel as `eog`) | `--eog drop` to test the EEG electrodes, or `--channels model` |
| `Source is missing required channels` | the montage is neither the CA-208 contract nor the declared profile | compare the probe labels with `cap.py`, or run `--cap declared` |
| `cap profile rejected: ... does not publish the CA-208 contract` | `--cap ca-208` was demanded on another cap | `--cap auto` or `--cap declared` |
| `--exclude-channels names channel(s) that are not EEG electrodes` | typo, or the label only exists on another cap | the error lists the contract's EEG electrodes |
| many windows rejected `amplitude`/`saturation` | wrong gain or wrong `--source-units`, or a saturated electrode | check gain/units; look at the named electrodes |
| windows rejected `flatline`, run stops after ~15 s | a dead electrode (a dry cap always has some) | `--exclude-channels` the one the report names, or `--max-bad-channels` |
| `Input samples are 3 s old` / high `input_lag` | source publishes in large bursts, or the host is busy | reduce the amplifier's LSL chunk size, close other load |
| `no valid window` with `data_flow` ok | warm-up plus resampler delay exceeded the run | stream longer than 10 s |
| flat electrodes | gel/contact, or an electrode not connected | fix contact, re-run; impedance lives in the control software, not in the LSL stream |
| `Cannot repair source damage (irregular_timestamps)`, repeatedly | the source's timestamps jitter by more than `Repair`'s tolerance | `--timebase grid`; measure with `ts_check` (below) first if you want the numbers |
| `Too many data faults` after a second or two | the same, and the recovery budget ran out before a single window | `--timebase grid`; the EE-511 rig does this on every run without it |
| the same, and `ts_check` reports a high `compressed%` | the source is chunk-stamped: a whole block shares one timestamp | `--timebase grid` lands it on a clean grid; the relay's `--regrid` (below) is the older route |
| `Cannot repair source damage (unsafe_endpoints)` | the endpoints are read far above the rails, which normally means the unit declaration is wrong (`--source-units V` on a source sending microvolts scales every value by 1e6) | fix `--source-units`; if the declaration cannot be fixed, raise `--repair-saturation-uv` past the scaled level |

Four limits are operator-settable rather than fixed in the library, and the run
records the values it used in its provenance:

| Flag | Default | What it bounds |
| --- | --- | --- |
| `--repair-amplitude-uv` | 500 | jump between the two finite endpoints a repair may bridge, in uV whatever `--source-units` says |
| `--repair-saturation-uv` | 75 000 | absolute endpoint level above which a repair is unsafe, in uV whatever `--source-units` says |
| `--max-recoveries` | 5 | chain restarts before the run stops |
| `--max-fault-seconds` | 5.0 | consecutive judge-rejected windows before the run stops; keep it above the window length plus the judges' settling, because one transient invalidates every window that overlaps it |

## Repairing a chunk-stamped source: `--timebase grid`

A recorder that stamps a whole block once hands out a grid `Repair` can only
refuse, and measuring it does not make it usable. Two ways to fix it, and the
first is the one to reach for:

```powershell
# the live script grids by default, so this is the whole job
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 250 --source-units uV `
    --cap declared --eog drop --duration 30
```

If the source also declares no usable channel metadata, put the relay in front of
it - the relay is what adds labels, types and units - and let either the relay or
the live script rebuild the timeline. Both use the same code, the library's
`TimeBase`, so they agree by construction:

```powershell
# terminal 1: republish the metadata the source never declared, on a counted grid
.venv/Scripts/python.exe -B -m scripts.getlive.relay --source-name UnicornRecorderRawDataLSLStream `
    --labels Fz,C3,Cz,C4,Pz,PO7,Oz,PO8 --keep 0-7 --regrid

# terminal 2: measure it
.venv/Scripts/python.exe -B -m scripts.getlive.ts_check --name NOVA_Relay --sfreq 250
```

`--regrid` puts the timeline on a counted grid (`--regrid-jitter` is the earlier
spelling and selects the same mode). What it does, in one sentence: **every sample
received gets one slot at the declared rate**, so the source's stamps cannot place
anything - they are read only to measure how far they have drifted from that count.
Three consequences matter on real hardware:

* no step is ever zero or backwards, whatever the source's stamps do;
* nothing is assumed about how many samples a block carries, which is not a
  constant on real hardware;
* nothing is held back, so unlike the block-spreading algorithm this replaced,
  regridding costs no latency. The price is that a hole closes up: the grid gives
  one slot per sample received, so a sample the source never sent cannot appear as
  a step in it. It is reported as a suspicious step and counted as absorbed drift
  instead, and the run's own report carries both.

Measured on the fixture, chunk-stamped on purpose with 8 samples per block, through
the real transport:

| outlet | `zeroish%` | `fatal%` (clocksync) | `samples/chunk` | median step |
| --- | --- | --- | --- | --- |
| raw fixture | 88.64 | 88.64 | 8 | 0.0030 |
| through `--regrid` | 0.68 | 0.68 | 1 | 1.0000 |

The step distribution through the relay has a median *and* a p99 of exactly
1.0000 samples: the grid is regular. The 0.68% that `ts_check` still counts as
fatal is what the receiving inlet's own clock sync adds to a grid that arrives
regular - the same order as the 0.5-1.1% the `+dejitter` column shows on the raw
fixture, and not a block structure that survived.

## Diagnosing a bad timestamp grid

`Repair._check_grid` accepts a step only when it is within
`tolerance_seconds` of a whole number of samples, and raises on **any** step
below one sample regardless of the tolerance. Two classes of damage follow:

* a step **compressed** below one sample is fatal at **any** tolerance: the
  tolerance test around 1.0 fails first, and the next branch rejects every
  sub-nominal step unconditionally;
* a step **off the integer grid** while still above one sample is the only
  class a wider tolerance could rescue.

So the first move is to measure, not to widen the tolerance.

```powershell
# validate the probe itself against a jittered synthetic outlet
.venv/Scripts/python.exe -B -m scripts.getlive.ts_check --self-test

# the real question: the outlet on the rig
.venv/Scripts/python.exe -B -m scripts.getlive.ts_check --name UnicornRecorderRawDataLSLStream --sfreq 250
# and the relay's second hop, if one is in the path
.venv/Scripts/python.exe -B -m scripts.getlive.ts_check --name NOVA_Relay --sfreq 250
# keep the raw distribution of every step for offline comparison
.venv/Scripts/python.exe -B -m scripts.getlive.ts_check --name <outlet> --sfreq 250 --json records/ts.json
```

Each run measures the same outlet under three LSL post-processing flag sets
(`clocksync`, `+dejitter`, `all`) so the candidate fix is shown to work before
it is wired in. Three flag sets run in sequence, 1 s of clock-sync settling each,
so allow about `3 x (--window + 1)` seconds.

| Reading | Meaning |
| --- | --- |
| `zeroish%` high | the source is **chunk-stamped**: many samples share one timestamp. The `samples/chunk` line gives the block size directly |
| `compressed%` > 0 | sub-nominal steps: no `tolerance_seconds` value helps, in any flag set |
| `roundoff%` > 0 with `compressed%` at 0 | off-grid but above one sample: a wider tolerance could absorb this class |
| `fatal%` = 0 | this flag set leaves the grid `Repair` requires |
| all flag sets dirty | the damage is not fixable downstream; look at the publisher, not the package |

### A chunk-stamped source

liblsl documents `StreamOutlet.push_chunk(data, timestamp=float)` as *"the
acquisition timestamp of the last sample"*: one float stamps **every** sample of
that block with the same time. Recorders that re-stamp a block themselves produce
the same shape with distinct stamps a few microseconds apart, so the steps inside
a block are ~0 while the steps between blocks are the block size. `median` then
sits near 0, the mean at 1.0, and `zeroish%` is roughly `1 - 1/chunk_size`. Such
a source is faithfully delivered but its grid is unusable: `Repair` raises on the
first intra-block step. The three flag sets behave as follows.

| Flag set | Effect on a chunk-stamped source |
| --- | --- |
| `clocksync` | near-zero steps stay, so almost every step is fatal |
| `clocksync+dejitter` | the grid is rebuilt and the median lands on 1.0, but `monotize` is missing, so clock re-corrections leave **negative** steps behind |
| `all` (`clocksync+dejitter+monotize`) | no negative steps; the residual includes a few steps below one sample that dejitter could not place |

`dejitter` is therefore necessary and **not sufficient**: an adapter that rebuilds
the grid from the chunk anchors, or a publisher that stamps per sample
(`push_chunk(data, t0 + arange(n) / sfreq)`), is what actually removes the
sub-nominal steps. A publisher with a chunk size of 1 makes the scalar form
equivalent to per-sample stamping.

The `samples/chunk` and `effective` rate lines are the check on `--sfreq`, and the
two numbers to use before rebuilding a grid. A source stamped per sample reports
1 sample/chunk. On a chunk-stamped source, `samples/chunk` is the block size and
`effective` is the block size divided by the anchor interval - the device rate
measured without dejitter's own rate fit, so the `clocksync` row is the honest
one and the flag sets may disagree with each other by design.

Reproduce the shape locally with the fixture - one stamp per block, the samples
inside it a few microseconds apart:

```powershell
python -B -m scripts.getlive.publish_raw --name chunked-1 --chunk 8 --within-chunk-us 12 --seconds 90
python -B -m scripts.getlive.ts_check --name chunked-1 --sfreq 250 --window 5
```

That reports `zeroish% ~88`, `compressed% ~88` on `clocksync`, `samples/chunk=8`
and `effective` back at 250 Hz, which is what the detector has to say on a source
whose blocks are stamped once. `--stamp-per-chunk` (a scalar time) does **not**
reproduce it: an mne-lsl inlet spreads that value back over the block and delivers
a clean one-sample grid.

A source that publishes a regular grid with Gaussian jitter is what
`publish_raw.py --jitter` simulates. Two things it does **not** cover, and both
matter before making `dejitter` the default: a source that genuinely **drops**
samples (smoothing must not hide that from `Repair`'s gap logic), and the
**step-shaped** clock error of a cross-host clock sync, which differs from
per-sample noise.

## What is still unverified

* The **outlet name, type and channel order are unconfirmed offline**: no eego
  software was available while writing this. The probe exists precisely because
  of that; run it before trusting any default here.
* Only the **CA-208 datasheet profile** is curated in `cap.py`. Any other cap
  runs through the declared profile, where electrode order and completeness
  cannot be checked against a second source - the probe output is the only
  evidence, and the report says which profile was used.
* **Gain and upstream processing** are not advertised over LSL. The software's
  filters/gain settings must be recorded by the operator; the run stores the
  `--reference` / `--ground` / `--upstream` strings as provenance, and they are
  assertions, not measurements.
* The **reference** and **ground** come from the cap profile; the package does
  not re-reference, so whatever the amplifier applied is what the windows
  contain.
* LSL adds network latency and jitter. The run reports both; it does not
  compensate them.
* **No impedance/contact measurement**: that is the control software's job.
* Acceptance thresholds are prototype values. They say "this run behaved", not
  "this amplifier is calibrated".
* Excluding a bad channel from the verdict does **not** remove it from
  `window.data`: the decoder still receives that column, because removing it
  would break the model's channel contract.

## Files

| File | Responsibility |
| --- | --- |
| `cap.py` | cap profiles (CA-208 datasheet, declared), profile selection, channel contracts |
| `outlets.py` | resolve, filter, select, connect and inspect LSL outlets |
| `checks.py` | pure acceptance rules, including the channel-quality policy |
| `electrodes.py` | per-electrode peak-to-peak statistics |
| `probe.py` | the metadata probe CLI |
| `live.py` | the live acceptance run and its channel-quality policy |
| `report.py` | console and JSON rendering |
| `relay.py` | republish an outlet that declares no usable channel metadata |
| `ts_check.py` | measure a source's timestamp grid against `Repair`'s rules |
| `compare_cnt.py` | align recorded runs against the control software's own `.cnt` and diff them sample by sample |
| `replay_run.py` | feed a recorded run back through the live chain and score it, with no amplifier |
| `publish_raw.py` | fixture publisher (metadata-less, optionally jittered) |
| `__main__.py` | `python -m scripts.getlive` |

Checks (hardware-free, they cover contracts, selection and scoring):

```powershell
.venv/Scripts/python.exe -B -m unittest tests.streaming.test_getlive -v
```

## References

* `datasets/UDO-SM-0215rev09 CA-208 Datasheet 2020-12-14.pdf`, pages 1-3:
  cap description, layout and pinning (CPz reference, AFz ground, EOG lead).
* `datasets/UDO-SM-0120_ENrev11 eego amplifier EE-22x User Manual
  2025-07-01_02.pdf`, page 6: up to 64 referential channels, sampling rate and
  gain set in the control software.
* [ANT Neuro: Third-Party Integrations with eego Systems](http://academy.ant-neuro.com/blog/application-notes-2/third-party-integrations-with-eegotm-systems-86):
  `Enable LSL EEG streaming`, LSL best practices.
* [MNE-LSL StreamLSL API](https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.stream.StreamLSL.html)
