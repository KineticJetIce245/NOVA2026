# NOVA2026 Real-Time EEG Streaming — Explained From Zero

This guide assumes **no background knowledge** about EEG, LSL, threading or
this codebase. Read it top to bottom; every term is explained the first time it
appears. After the "big picture" section you can jump to either the demo tour
(Section 4) or the module-by-module guide (Section 5).

> Short version: this code reads voltages from an EEG device **while it is
> recording**, chops the continuous signal into fixed-length "windows", cleans
> each window with filters, and hands windows to whoever wants to analyse them
> (today: printing statistics; tomorrow: a trained model).

---

## 1. What are we trying to do?

An EEG amplifier measures the tiny electrical voltages on a person's scalp,
many times per second. We want to **use those voltages live**, not only after
the recording ends:

1. Receive data as it arrives, over the network (the protocol is called LSL).
2. Turn the endless stream of numbers into small fixed-length pieces that
   algorithms can work with ("windows").
3. Clean the signal (remove mains hum at 60 Hz, keep only the interesting EEG
   frequency band, slow it down from 500 samples/second to 128 samples/second).
4. Make sure the data is usable: reject windows that contain artefacts
   (a spike, a dead channel, the very first seconds before the filters have
   "warmed up").
5. Hand each good window to a consumer. Today the consumer prints statistics
   so you can watch the pipeline work. Later the consumer will be a trained
   model that classifies the window (e.g. "concentrating" vs "resting").

A long-term design rule for this code: **every piece is small, does one thing,
and can be tested or replaced on its own.** The script that runs the demo is
just the glue that assembles the pieces in the order we want.

---

## 2. Background concepts (no prior knowledge needed)

- **Channel / electrode.** A sensor at one position on the scalp. Each channel
  has a name, e.g. `F3`, `Cz`, `O1`.
- **Sample.** One measurement: the voltage of *every* channel at one instant.
- **Sampling rate (`sfreq`).** How many samples arrive per second, in Hz.
  500 Hz = 500 samples per second.
- **Units.** The amplifier delivers volts (tiny numbers like `0.000012`).
  EEG is normally discussed in microvolts (µV): `1e-6` of a volt, so multiply
  volts by `1 000 000`. `10^-0` = volts, `10^-6` = µV.
- **Data layout.** One data block is a 2D array `(samples, channels)`.
  Column `j` belongs to channel number `j`.
- **Chunk.** Whatever piece of data happens to arrive at once. The sender
  decides the size (we deliberately use uneven sizes in the demo to prove the
  code does not care).
- **LSL (Lab Streaming Layer).** A standard for streaming time-series data
  between programs/network. An *outlet* sends, an *inlet* receives.
- **Manual acquisition.** We open the inlet with `acquisition_delay=None`,
  which means "do not pull data automatically; pull it only when I ask you to".
  This keeps control in our code and makes threading simple.
- **Window.** A fixed-length slice of the signal, e.g. 2 seconds at 128 Hz =
  256 samples per channel. Windows can **overlap**: with a "hop" of 0.5 s, a
  new window starts every 0.5 s but each window still contains 2 s of data.
- **Real-time loop.** A `while` loop that, roughly every 0.1 s: gets a block,
  processes it, and (sometimes) produces a finished window.
- **Stateful.** An object that remembers something between calls (a filter
  remembers its previous outputs). Stateless = no memory between calls.
- **Thread.** An independent line of execution inside one program. Our
  acquisition loop runs on the main thread; we let slow "analysis" work run on
  worker threads so it never blocks data collection.
- **Copy vs view.** If a component keeps data in a ring buffer that is later
  overwritten, handing out a *view* would let the next write silently change
  already-delivered data. We always hand out *copies*.

---

## 3. The big picture

ASCII pipeline (time flows top to bottom for one run):

```text
   EEG amplifier / PlayerLSL (fake source for the demo)
        |   publishes chunks of (samples, channels) in VOLTS
        v
   StreamLSL inlet (connected with manual acquisition)
        |
        v
   Acquire.read()                 --> one fixed-size block (e.g. 50 samples)
        |
        v
   StreamSession.ingest()         --> channel reorder (ChannelContract)
        |                            + save raw volts (RunRecorder, optional)
        v
   STAGES (defined in the script) --> one ordered tuple, one line per stage:
        |                            repair short NaN/Inf runs (Repair)
        |                            scaler (V->uV)
        |                            quality observer (wrapped as a stage)
        |                            notch 60 Hz, band-pass 1-45 Hz (filters)
        |                            SoXR resampler 500 -> 128 Hz
        v
   CircularBuffer.push()          --> 0..N finished windows
        |
        v
   StreamSession.wrap()           --> EEGWindow with its verdict
        |                            (data/eog split, warm-up + judge reasons)
        v
   Consumer: print, model, ...    --> runs in the loop, or on TaskOffloader
        |                            worker threads
        v
   RunRecorder.close()            --> lock the run, export .fif (optional)
```

The runtime ordering of one run:

```text
parse args  ->  start fake source  ->  connect inlet
           ->  define the STAGES tuple + judges (in the script!)
           ->  StreamSession(...)     [builds contract/recorder/acquire/buffer]
           ->  while running:  read -> ingest -> run STAGES -> push -> wrap -> consume
           ->  finally:        close acquire, disconnect, close recorder
```

Two rules that keep the pipeline correct:

- **The script owns preprocessing order**: `STAGES` is an ordered tuple the
  script defines, and the session never runs or reorders it. Convention:
  repair first (source units, before filters), quality observes the
  raw-but-scaled signal before filters, then filters, resampling last.
- **Copy out of ring buffers**: never let a later write corrupt an earlier
  window that is already on its way to the consumer.

---

## 4. Tour of `scripts/streaming_demo.py`, step by step

Run it from the repository root:

```powershell
.venv/Scripts/python.exe -B -m scripts.streaming_demo --duration 8 --record records
```

The file has three layers: a docstring (the help text), helper functions, and
`main()`. `main()` is numbered 1-8 in comments; we follow those numbers.

### The docstring

Everything between the first `"""` and its closing `"""` is the **module
docstring**. Python stores it in `__doc__`, and argparse prints it when you run
`--help`. So this text explains the demo *and* is the help screen — one source
of truth. It shows the exact command, the pipeline, and how to compare
"analysis in the loop" (`--workers 0`) with "offloaded to worker threads"
(`--workers 2`).

### Imports

- `from time import monotonic, sleep` — `monotonic()` is a clock that only
  moves forward (for timing the loop and lag); `sleep()` simulates a slow
  model.
- `from uuid import uuid4` — makes a unique name for the fake stream so two
  runs never collide.
- `import mne, numpy as np` — MNE is the Python EEG toolbox (used to build the
  fake recording and read recordings back); NumPy is the array library every
  block of data is made of.
- `PlayerLSL` — the *outlet* that "plays" our fake recording in real time.
  `StreamLSL` — the *inlet* that receives it.
- `TaskOffloader` — runs per-window work on worker threads (see
  `offload.py`).
- `StreamSession, parse_args` — the bootstrap helpers (see `bootstrap/`).
- `QualityMonitor, Repair, Resampler, SosFilter, design_bandpass,
  design_notch, unit_scaler` — the preprocessing parts, imported here on
  purpose: the demo wants you to *see* the chain being defined.

### Constants

- `CHANNELS` — the 8 channel names our fake device publishes, in column order.
- `SOURCE_UNIT_EXPONENT = 0` — the source sends volts (`10^0`), not µV.

### `synthetic_recording(seconds, sfreq)`

Builds the fake data. One clean 10 Hz sine wave per channel; each channel has a
slightly different amplitude (`(10 + index)` µV, i.e. 10..17 µV), so the
statistics printed later are easy to eyeball. Returns an `mne.io.RawArray`
holding volts. In real life you would delete this function and point
`PlayerLSL` (or the inlet) at the real amplifier.

### `main()` — step 1, start the fake source

- `args = parse_args(description=__doc__)` — the bootstrap argument getter
  defines all command-line options (rates, window sizes, filter settings,
  recording identity, consumer mode) with defaults; `description=__doc__`
  makes the module docstring the `--help` text. Nothing in this script
  declares argparse options.
- A unique `name`, then `PlayerLSL(...)` is created and `start()`ed. It
  "plays" `duration + 6` seconds of data (extra lead time for filter warm-up)
  in chunks of `--chunk-size` (37 by default — deliberately uneven).
- `--workers < 0` is rejected early.

### Step 2, connect the inlet

```python
stream = StreamLSL(bufsize=4.0, name=name)
stream.connect(acquisition_delay=None, processing_flags=["clocksync"], timeout=10)
```

`acquisition_delay=None` = manual acquisition (we pull when we want).
`clocksync` = synchronize the device clock with ours. `bufsize=4.0` = the
inlet keeps up to 4 seconds of buffered data.

### Step 3a, define the preprocessing chain explicitly

This is the heart of "the script owns the pipeline":

```python
repair    = Repair(args.sfreq, source_unit_exponent=SOURCE_UNIT_EXPONENT)
scaler    = unit_scaler(SOURCE_UNIT_EXPONENT, desired_exponent=-6)  # V -> uV
quality   = QualityMonitor(n_eeg=len(CHANNELS), sfreq=args.sfreq, ...)

def observe(data, timestamps):        # quality is an observer, not a stage:
    quality.feed(data, timestamps)    # wrap its feed into the stage shape
    return data, timestamps

notch     = SosFilter(design_notch(args.notch, args.notch_q, args.sfreq), n)
bandpass  = SosFilter(design_bandpass(args.lpass, args.hpass, args.order, ...), n)
resampler = Resampler(args.sfreq, args.out_sfreq, n, quality=args.resample_quality)
STAGES    = (repair, scaler, observe, notch, bandpass, resampler)   # the chain
```

Every stage, its parameters, and its order are visible and editable here.
Every stage has the same shape `stage(data, timestamps) -> (data,
timestamps)`; the loop just folds `STAGES` in order. `design_notch`/
`design_bandpass` compute filter coefficients; `SosFilter` applies one filter
continuously (it remembers its state between blocks).

### Step 3b, assemble the rest via `StreamSession`

```python
session = StreamSession(stream, args, CHANNELS,
                        judges=(quality, repair),
                        out_sfreq=resampler.out_sfreq, ...)
```

`StreamSession` (see `bootstrap/session.py`) builds the boring run-once
pieces: the channel contract, the optional recorder, the `Acquire` object and
the `CircularBuffer`. It does **not** define or run the preprocessing — the
script folds `STAGES` itself. The `judges` are the objects `wrap()` asks for a
verdict (quality faults and repaired samples both reject windows). If the
source sent channels we do not want, the contract prints them here as
"dropping ...".

### Step 4, print the configuration

One block of `print()` that shows what will happen: source rate, chain
settings, resampler target, window/hop/capacity in *samples at the output
rate*, consumer mode, and the recording destination. Printing the plan makes
every run self-explanatory.

### Step 5, consumer state and helpers

- `started = monotonic()` — the wall clock used to stop after `--duration`.
- `valid_windows` / `rejected_windows` — counters for the final summary.
- `report(eeg_window)` — prints one line per window: the peak-to-peak
  amplitude (max - min, in µV) of the window's EEG data. This is our
  stand-in for "a model".
- `analyze(item)` — receives an `EEGWindow`, optionally `sleep()`s to pretend
  analysis is slow, then reports.
- `offloader = TaskOffloader(analyze, workers=..., capacity=...)` — created
  *once*, only when `--workers > 0`. It is a bounded queue plus worker
  threads; workers pick up `EEGWindow` tasks and run `analyze`.

### Step 6, the real-time loop

```text
while monotonic() - started < args.duration:      # until time is up
    (a) data, ts = session.acquire.read()         # pull one fixed block
    (b) data, ts = session.ingest(data, ts)       # reorder + record raw
    (c) for stage in STAGES: data, ts = stage(data, ts)   # the script's chain
    (d) for window, w_times, start in session.buffer.push(data, ts):
        (e) eeg_window = session.wrap(window, w_times, start)  # -> EEGWindow
            if not eeg_window.valid: count & skip  # warm-up or judge reasons
        (f) analyze(eeg_window)                    # or offloader.submit(...)
```

Every iteration is one acquired block. `push()` returns the list of windows
*completed by this block* — often empty, sometimes several. `wrap()` packages
each window into an `EEGWindow` that carries its own verdict (`valid` +
`reasons`, see `window.py`): warm-up windows and windows flagged by a
registered judge (a quality fault, or a repaired sample) come out
`valid=False` and are skipped.

### Step 7, shutdown

The `finally:` block always runs, whether the loop ended normally, the user
pressed Ctrl+C (`KeyboardInterrupt`), or the lag guard raised `RuntimeError`.
Order matters:

1. `session.acquire.close()` — detach the callback and drop leftover samples;
2. `stream.disconnect()` — close the LSL inlet;
3. `player.stop()` — stop the fake source;
4. `offloader.close(drain=True)` — let queued analysis finish;
5. `session.close(status="completed")` — locks the run, exports the `.fif`.

### Step 8, summaries

Prints acquisition statistics (`input_samples`, `max_lag`, `gaps`), how many
samples the resampler emitted, how many windows were valid/rejected, and the
offloader counters. `max_lag` is the oldest age of any consumed block — if it
ever exceeds 3 s the lag guard stops the run instead of silently analysing
stale data.

### Entry point

```python
if __name__ == "__main__":
    main()
```

Runs `main()` only when the file is executed directly (`python -m
scripts.streaming_demo`), not when it is imported.

---

## 5. Module-by-module guide to `src/nova2026/streaming`

Layout:

```text
src/nova2026/streaming/
  __init__.py                # public exports (one import line)
  acquire.py                 # Acquire          - pull fixed-size blocks
  circular_buffer.py         # CircularBuffer   - ring storage -> windows
  offload.py                 # TaskOffloader    - run analysis on workers
  preflight.py               # ChannelContract + validate_source/prepare (B)
  recording.py               # RunSpec/RunRecorder + read-back helpers
  recovery.py                # Recovery         - bounded recovery (A2)
  spatial.py                 # SpatialOperator, fit_ssp, processing_contract (C)
  stats.py                   # StreamStats      - run counters (E)
  window.py                  # EEGWindow        - window + verdict for consumers
  preprocess/
    __init__.py              # re-exports the preprocessing pieces
    units.py                 # unit_scaler      (stateless)
    filters.py               # design_* + SosFilter (stateful)
    quality.py               # QualityMonitor   (stateful observer)
    repair.py                # Repair           (stateful, stage + judge)
    resample.py              # Resampler        (stateful, soxr)
  bootstrap/
    __init__.py              # re-exports
    args.py                  # make_parser / parse_args (argument getter)
    session.py               # StreamSession    (run-once assembly)
```

Companion scripts and tests live next to the package: `scripts/streaming_demo.py`
(the runnable walkthrough), `scripts/benchmark_streaming.py` (per-stage
micro-benchmark), `scripts/verify_realdata.py` (real-recording check: run
through, save + offline parity, edge cases), and `tests/streaming/test_*.py`
(unit + end-to-end tests, 181 in total).

Rule of thumb used everywhere: **stateless -> function, stateful -> class**.
Classes hold their own state, validate arguments in the constructor, expose a
call/method per data block, and offer `reset()` for a new segment.

---

### 5.1 `__init__.py` — the public door

Re-exports the most-used names so scripts can write one import:

```python
from nova2026.streaming import Acquire, ChannelContract, CircularBuffer,
    TaskOffloader, Resampler, RunRecorder, RunSpec
```

Why: keep one stable public surface; anything else can still be imported from
its own module (`from nova2026.streaming.preprocess import SosFilter`).

---

### 5.2 `acquire.py` — `Acquire`

**Problem.** LSL delivers chunks of any size, whenever they arrive. Analysis
is much easier if we always get *exactly N samples per call*.

**Why a class.** It must remember partially received samples between calls
(state: leftover data + timestamps), remember the newest sample time to detect
gaps, and remember the first error raised while receiving.

**Public surface.**

| Member | Purpose |
| --- | --- |
| `Acquire(stream, block_samples, *, sfreq, poll_interval, no_data_timeout, max_lag_seconds, max_future_seconds, stop_event)` | Build one. `stream` must be connected with `acquisition_delay=None` (manual). |
| `read(timeout=None) -> (data, timestamps)` | Return exactly `block_samples` rows and their timestamps, oldest first. Blocks internally until ready. |
| `stop()` | Request shutdown from another thread. |
| `close()` | Detach the callback, drop leftover samples. Does NOT disconnect the stream. |
| `stopped`, `pending_samples` | Properties for the caller. |
| `blocks`, `samples`, `gaps`, `max_gap`, `max_lag` | Public counters/stats. |

**How it works inside.** MNE-LSL calls a callback each time we call
`stream.acquire()`; because acquisition is manual, the callback runs on the
same thread that calls `read()`, so **no lock is needed**. `_on_chunk`
validates shapes and copies the chunk (MNE reuses its buffer), `_fill` loops
pulling until at least one full block is buffered, `_take_block` slices one
block off the front and keeps the rest, `_check_age` rejects blocks that are
too old (stale data) or from the future.

**Expected usage** — usually not directly; `StreamSession` builds it:

```python
acquire = Acquire(stream, block_samples=50, sfreq=500.0)
data, timestamps = acquire.read()          # (50, n_ch)
```

---

### 5.3 `preflight.py` — the source contract and its validation (B)

**Problem (channels).** The device publishes channels in *its* order and set;
our pipeline/recording expects channels in *our* order (EEG first, aux like
EOG after). If we silently paired columns with the wrong names, every number
downstream would be attached to the wrong electrode — a quiet, hard-to-find
bug.

**Why a class.** One place to keep the validated mapping (indices,
selected/dropped names) built once and reused every block.

**Public surface.** `ChannelContract(source_channels, expected_channels)`;
attributes `source_channels`, `expected_channels`, `selected_channels`,
`dropped_channels`; method `reorder(data) -> data`.

**Behaviour.** Constructor checks for duplicate labels and for *missing*
required channels (raising `ValueError` with the missing names — fail at
startup, not mid-recording), and computes the column permutation. `reorder`
takes the requested columns in canonical order; when the source already
matches exactly it returns the array untouched (zero copy on the hot path).

**Problem (validation).** The pipeline trusts whatever outlet it connects to;
connecting to the wrong stream, or to the right stream with a different rate,
units or channel types, quietly records plausible but wrong data. Validation
must run right after `connect()` and before the first `read()`.

**Public surface (validation).** `validate_source(stream, *, sfreq, channels,
source_unit_exponent, n_eeg=None, stream_name=None, source_id=None,
stream_type=None)` — a stateless function that raises `RuntimeError` on any
mismatch: connected identity, sampling rate, numeric dtype, untouched state
(no filters/callbacks/unread samples), duplicate/missing labels (reusing the
constructor's rules), channel types (EEG vs EOG), and each channel's declared
voltage units against the configured exponent. `prepare(...)` is the same
check but *returns* the `ChannelContract` for the run — the one-call entry
point.

**Expected usage** (at run start; `StreamSession` accepts the pre-built
contract so nothing is checked twice). The contract must come from *this*
outlet: the session checks that its `source_channels` equal the connected
stream's labels and rejects a contract built for another source, because such
a contract would silently reorder columns by name:

```python
from nova2026.streaming import prepare
contract = prepare(stream, sfreq=args.sfreq, channels=EXPECTED,
                   source_unit_exponent=0, n_eeg=len(EXPECTED))
session = StreamSession(stream, args, EXPECTED, ..., contract=contract)
data = contract.reorder(data)     # every block (inside session.ingest)
```

---

### 5.4 `circular_buffer.py` — `CircularBuffer`

**Problem.** Turn a continuous stream of blocks into fixed-length windows,
possibly overlapping, without losing or duplicating samples, while bounding
memory.

**Why a class.** Ring buffers are stateful by definition (write pointer,
logical row count, next window boundary, timestamp anchor).

**Public surface.** Constructor
`CircularBuffer(window_samples, hop_samples, capacity_samples, sfreq,
n_channels, dtype)`; `push(data, timestamps) -> list[(window, window_times,
start_sample)]`; `reset()`; attributes `total_written`, `windows`.

**How it works.** A fixed array ("ring") plus a *logical* counter
(`total_written`, which never wraps). On each push it writes up to the next
window boundary, and the moment a boundary is crossed it **copies** the window
out (later ring writes can never corrupt it), then advances the boundary by
`hop`. Windows that straddle the end of the ring are reassembled from two
pieces. Timestamps are a uniform grid at `sfreq` anchored to the first finite
timestamp pushed. Pure storage only: no units, filters or validity decisions.

**Expected usage** (built by `StreamSession`; `session.buffer`):

```python
windows = buffer.push(data, timestamps)
for window, window_times, start in windows:
    ...  # e.g. eeg_window = session.wrap(window, window_times, start)
```

---

### 5.5 `offload.py` — `TaskOffloader`

**Problem.** If the consumer (e.g. a model) is slow, and it runs in the same
loop as acquisition, data collection stalls: we measured the lag guard firing
because analysis could not keep up. Solution: hand analysis to a pool of
worker threads through a bounded queue; the loop submits and immediately
continues.

**Why a class.** A pool has lifecycle (start workers, stop them), a queue with
an overflow policy, and shared counters.

**Public surface.**

| Member | Purpose |
| --- | --- |
| `TaskOffloader(handler, *, workers=1, capacity=8, overflow="drop_oldest", on_result=None, on_error=None)` | handler runs on a worker thread for each submitted item. |
| `submit(item) -> bool` | Enqueue and return immediately; False when dropped/closed. |
| `raise_error()` | Re-raise the first handler failure on the caller's thread. |
| `close(*, drain=False, timeout=5.0)` | Stop accepting, (optionally) wait for the backlog, join workers. |
| `pending`, `submitted`, `completed`, `failed`, `dropped` | State/counters. |

**Overflow policy** matters when producers are faster than consumers:
`"drop_oldest"` keeps the freshest items (right for real-time), `"drop_newest"`
keeps history, `"raise"` propagates `queue.Full`.

**Notes.** Items must be self-contained — this is exactly why
`CircularBuffer` hands out copies. Handlers must be thread-safe. If the
average computation time exceeds the arrival rate, no architecture can keep
up; the only question is what you drop.

**Expected usage:**

```python
offloader = TaskOffloader(analyze_window, workers=2, capacity=8)
offloader.submit(eeg_window)            # non-blocking; EEGWindow is a copy
...
offloader.raise_error()                 # surface worker failures
offloader.close(drain=True, timeout=5.0)
```

---

### 5.6 `window.py` — `EEGWindow`

**Problem.** A window leaves the ring buffer as a raw tuple
`(data, window_times, start_sample)`. Whoever consumes it (a model, a
recorder) would have to re-derive "may I use this window and why?". Bundle
the data with that verdict once, at the buffer boundary.

**Why a class.** It is a value object with fixed semantics: validated shapes,
EEG/EOG split, and a verdict (`valid`/`reasons`) that travels with the data.

**Public surface.**

| Member | Purpose |
| --- | --- |
| `EEGWindow(data, eog, timestamps, valid, reasons, start_sample, segment=0, artifact_id=None, channel_names=(), contract=None, available_at=None)` | One window. Arrays are **copied** on construction, so the object is safe to hand to worker threads. |
| `len(window)` | Number of samples. |
| `.data`, `.eog` | µV samples: EEG columns and auxiliary columns separately. |
| `.valid`, `.reasons` | Verdict: valid windows have empty reasons. |
| `.timestamps`, `.start_sample`, `.segment`, `.channel_names`, `.contract`, `.available_at` | Provenance for models and recording. |

**Expected usage** — never constructed by hand in a script;
`StreamSession.wrap()` builds it:

```python
eeg_window = session.wrap(window, window_times, start)   # gates + splits
if eeg_window.valid:
    model(eeg_window.data)
else:
    log(eeg_window.reasons)
```

---

### 5.7 `recording.py` — `RunSpec`, `RunRecorder` and read-back helpers

**Problem.** Keep the raw data (before any filtering) so that a run can be
re-analysed later, survives a crash, and carries enough context (who, when,
status, task events) to be meaningful.

**Why classes + module functions.** `RunSpec` is a tiny identity value;
`RunRecorder` manages one SQLite database and the FIF export; reading is done
by stateless module functions.

**Public surface.**

| Name | Purpose |
| --- | --- |
| `RunSpec(subject, session, run, role="run")` | Identity; unsafe characters stripped so a name can never escape the folder tree. |
| `RunRecorder(root, spec, channels, sfreq, *, ch_types, unit_exponent, dtype, export_fif, config=None, files=None, track_windows=False)` | Creates `root/subject/session/run/run.sqlite`; refuses to overwrite an existing run. `config` stores a serializable run description; `files` copies + SHA-256-hashes provenance inputs; `track_windows` enables the window log. |
| `recorder.write(data, timestamps)` | Commit one chunk (data BLOB + row count + first timestamp) in its own transaction — crash-safe. |
| `recorder.mark(label, timestamp=None)` | Thread-safe task event (e.g. `"blink"`), for the controller thread. |
| `recorder.log_window(ts, valid, reasons, segment, artifact_id)` | One line per delivered window, when `track_windows=True`. |
| `recorder.close(status="completed", error=None, stats=None)` | Lock the run; if `export_fif`, read the DB back and write `<run>_raw.fif` next to it; write a JSON sidecar. |
| `recorder.mark_not_processed(rows, timestamp=None)` | Record a buffered tail that never became windows (`not_processed:<rows>` event) when the loop stops. |
| `read_metadata(path)` | All metadata as a dict (includes `config` and `input:<role>` entries when provided). |
| `iter_chunks(path)` | Yield `(data, first_timestamp)` for every chunk in order. |
| `iter_events(path)` | Event list `(timestamp, label)`. |
| `iter_windows(path)` | Window log as `(ts, valid, reasons, segment, artifact_id)`; empty for runs without tracking. |
| `chunk_timestamps(first, n, sfreq)` | Rebuild a uniform time grid from a chunk's anchor. |
| `replay_chunks(path)` | Yield `(data, timestamps)` — feed a recorded run through the same chain later (offline evaluation). |

**Design note.** We store *per-chunk* first timestamps, not per-sample
timestamps: the whole pipeline works on anchor + uniform grids anyway, so
per-sample storage would just bloat the file.

**Expected usage** (recording is optional; `StreamSession` wires it when
`--record` is given):

```python
rec = RunRecorder(Path("records"), RunSpec("s1", "a", "trial01", "trial"),
                  CHANNELS, sfreq=500.0, unit_exponent=0)
rec.write(data, timestamps)      # every block
rec.mark("blink", t)             # from the task thread
rec.close(status="completed")
```

---

### 5.8 `preprocess/` — the preprocessing package

Shared contract for transform stages:
`stage(data, timestamps) -> (data, timestamps)`, called once per block,
state kept across blocks. `QualityMonitor` is an *observer*, not a transform.

#### `preprocess/units.py` — `unit_scaler`

**Why.** The chain works in µV; the source sends volts (or another exponent).
Scaling is stateless, so it is a *function factory* that returns a stage
closure. Validates exponents (0/-3/-6/-9) and returns
`(data * 10^(source-desired), timestamps)`.

#### `preprocess/filters.py` — `design_bandpass`, `design_notch`, `SosFilter`

**Why.** Filtering needs two different jobs, kept separate: *design* the
coefficients once (`butter`/`iirnotch` from SciPy, in numerically stable
second-order-section form) and *apply* them continuously.

- `design_bandpass(low, high, order, sfreq)` -> SOS coefficients.
- `design_notch(frequency, quality, sfreq)` -> SOS coefficients.
- `SosFilter(sos, n_channels)` is stateful: it initialises the filter state
  (`zi`) on the first block so there is no start-up transient, calls
  `sosfilt` per block while carrying `zi`, and offers `reset()`.

**Expected usage:**

```python
notch = SosFilter(design_notch(60.0, 30.0, 500.0), n)
data, ts = notch(data, ts)     # every block
```

#### `preprocess/quality.py` — `QualityMonitor`

**Why.** Detect bad raw data (large excursion, saturation, flatline) *before*
filtering, so filters cannot hide a stuck electrode. Reports faults by
*timestamp interval*, so results do not depend on how the stream was chunked.

**Public surface.** `QualityMonitor(n_eeg, sfreq, ...limits...)`;
`feed(data_uv, timestamps)` (observer, per block); `reasons(start, end) ->
tuple[str, ...]` (ask about a finished window); `reset()`.

**Expected usage:** scripts wrap `feed` in a stage closure and register the
monitor as a judge on `StreamSession`; `wrap` then asks `reasons(window_start,
window_end)` for every finished window. Feed every block after scaling and
before filters.

#### `preprocess/repair.py` — `Repair`

**Why.** A stream occasionally delivers a few broken samples (an isolated
`NaN`/`Inf`, or a short missing run). One broken sample fed into a causal
filter corrupts its state for the whole settling time afterwards, so damage
must be fixed *before* any filter. Repair bridges short runs (default up to
20 ms) linearly between the two finite samples on either side, per channel,
never rewriting healthy values.

**Public surface.** `Repair(sfreq, *, max_seconds=0.02, ...)` is both a
transform stage — `(data, timestamps) -> (data, timestamps)` — and a judge
with `reasons(start, end)` (returns `("interpolated",)` when a window touches
a repaired span, so those windows are rejected); `reset()`.

**Behaviour.** Missing rows are detected from timestamp gaps and repaired the
same way. A damaged tail without a finite right endpoint is held back and
returned with the next block. Damage that cannot be repaired — a run longer
than `max_seconds`, irregular or non-finite timestamps, or extreme endpoints
when units are known — **raises**, stopping the run loudly rather than feeding
bad values into filters. Damage before the first finite sample is dropped (the
run simply starts later). Damage it cannot repair raises
`UnrepairableError`; Section 9 A2's `Recovery` guard turns that into a bounded
restart or, beyond its limits, a stop.

#### `preprocess/resample.py` — `Resampler`

> **New to resampling? Read this box first.**
>
> **What a sample rate is.** 500 Hz means "500 measurements every second".
> Picture them as dots on a line, 2 mm apart. We want dots 7.8 mm apart
> (128 Hz).
>
> **Why we cannot simply throw dots away.** If a 200 Hz tone is hiding in the
> signal and we just keep every fourth dot, that tone comes back as a ghost at
> about 72 Hz — a frequency that was never there, and once it is mixed in it
> cannot be separated again. The ghost appears because the signal was not
> "smoothed" before being thinned out.
>
> **What SoXR does instead.** It first builds a smoothing filter that removes
> anything too fast for the new dot spacing, and then reads the new dots off
> the smoothed curve rather than off the raw dots. That is all a resampler is:
> "smooth, then re-space".
>
> **Why that costs time.** To draw the smoothed curve it needs to see a run of
> neighbouring dots, so it must collect a batch of input before it can hand
> out the first output. The batch size is counted in **samples, not seconds**:
> the standard setting (`LQ`) wants about 950 of them, which is about 1.9 s at
> 500 Hz but about 7.4 s at 128 Hz. Halve the input rate and you double the
> delay in seconds.
>
> **The shortcut and its price.** `QQ` skips the smoothing almost entirely, so
> its delay is tiny — but the ghost tones come back. It is only safe if
> something earlier in the chain has already removed the high frequencies.
> Here the anti-ghost margin is thin, so this wrapper refuses `QQ` outright
> (the paragraph on latency below explains why).
>
> **One-line takeaway.** SoXR trades "clean signal" against "delay", and its
> delay is a number of samples — so the same setting means very different
> latencies at different sample rates.

**Why.** Change the sample rate (500 -> 128 Hz) *continuously* while
streaming. SoXR (`soxr`) provides a stateful `ResampleStream`; this class
wraps it in the chain contract and regenerates an output-rate timestamp grid.

**Why a class.** Resampling is stateful (internal history, output index,
anchor) and its output row count per call varies (0 on early calls while SoXR
buffers).

**Dependency note.** `soxr` is imported lazily, only when a `Resampler` is
constructed; the rest of the package works without it (the demo requires it).

**Latency, and why `QQ` is not accepted.** SoXR's delay is an internal *sample
count*, not a duration, and it is deliberately not compensated (timestamps stay
anchored to the input grid). LQ buffers roughly 950 input samples: about 1.9 s
at 500 -> 128 Hz, but about 7.4 s at 128 -> 64 Hz, because halving the input
rate doubles the delay in seconds. So keep the input rate high, or band-limit
early and resample straight to the rate the consumer wants. SoXR also offers a
`QQ` preset, which this wrapper excludes by default: `QQ` performs essentially
no anti-aliasing, and this path's third-order 1-45 Hz band-pass leaves the
output Nyquist (64 Hz) too close to the stopband edge, so energy would fold
back in. Any future "upgrade" to `QQ` on this path would need a steeper
anti-alias filter first, not just a changed word.

**Choosing the preset automatically (the time budget).** Which preset is right
depends on how late a sample may be when the consumer uses it, so the class
derives a delay budget instead of guessing one:

```text
budget = max_delay_seconds                        # explicit override
budget = max_age_seconds - reserve_seconds        # from the consumer's expiry
budget = 2.0 s                                     # fallback when neither is set
```

With `quality=None` (or `"auto"`) the class does not read a table: it
**measures** each anti-aliased preset on the installed SoXR build (feeding one
zero sample at a time until the first output appears) and picks the cleanest
one whose measured delay fits the budget. Measuring matters because the delay
is a sample count that varies by build — on the build here, for example, MQ is
slowest (7.55 s at 500 -> 128 Hz) and VHQ is fastest of the strong presets
(7.17 s), which no name-based ordering would predict. It raises
`ResamplerQualityWarning` when the pick is the weakest preset (LQ), when only
`QQ` would fit (and `allow_qq=True` let it through), or when nothing meets the
budget at all; `strict=True` turns those warnings into errors, and
`resampler.quality` / `resampler.startup_delay_seconds` report what was chosen
and what it costs.

**What "signal error" would mean here (not auto-tuned yet).** The time budget
above is about *when* data arrives. The other kind of error is about *how
faithful* the conversion is: `QQ` barely filters, so energy that is too fast for
the new sample spacing folds back and appears as a frequency that was never
there (in the review's measurement, a tone that should vanish came through at
about 97 %); `LQ` removes it but slightly changes amplitudes in the passband
(about a 9 % error on the tested tone). Those two numbers, not the preset
names, are what "signal error" means. Measuring them per build would mean
feeding test tones and reading the leakage, which is a heavier probe; it is
deliberately left out for now, and the safe rule of thumb stands: keep a proper
anti-aliasing preset (never `QQ`) unless an earlier stage already band-limits
the signal hard.

**Public surface.**
`Resampler(in_sfreq, out_sfreq, n_channels, quality=None, *,
max_age_seconds=None, reserve_seconds=0.0, max_delay_seconds=None,
allow_qq=False, strict=False)`; `quality=None`/`"auto"` selects automatically,
an explicit name (`"LQ"`, `"MQ"`, `"HQ"`, `"VHQ"`) keeps full control, and
`"QQ"` needs `allow_qq=True`. Also: `(data, timestamps) -> (out,
out_timestamps)`; `reset()`; `output_samples`; public `in_sfreq`, `out_sfreq`,
`quality`, `startup_delay_seconds`, `max_delay_seconds`; the helper
`select_quality(...)` exposes the choice without constructing a stage. The CLI
counterpart is `--resample-quality auto`.

---

### 5.9 `bootstrap/` — start-up helpers so scripts stay small

#### `bootstrap/args.py` — the argument getter

**Why.** Every live script needs the same options (rates, windows, filters,
recording identity, consumer mode). Instead of rewriting argparse in every
script, define them once here.

- `make_parser(description="", extra=None)` -> `ArgumentParser` with all
  common options, grouped: source/timing, windows (at the output rate),
  preprocessing, run recording (`--record/--subject/--session/--run`),
  consumer (`--workers/--queue/--compute`).
- `parse_args(description="", extra=None, argv=None)` -> parsed namespace.
  `extra` lets a script register its own flags: `parse_args(extra=lambda p:
  p.add_argument("--model"))`.

**Expected usage:** `args = parse_args("my script")`, then use
`args.sfreq`, `args.window`, `args.run`, ... everywhere.

#### `bootstrap/session.py` — `StreamSession`

**Why.** The "run-once" assembly (channel contract, optional recorder,
Acquire, CircularBuffer) is identical across scripts. `StreamSession` does it
once and exposes loop helpers, while the *preprocessing stages stay defined in
the script* and are passed in.

Constructor:

```python
StreamSession(stream, args, channels, *,
              judges=(), out_sfreq=None,
              source_unit_exponent=0, role="run", ch_types=None, n_eeg=None)
```

Attributes: `contract`, `recorder`, `acquire`, `buffer`, `judges`,
`eeg_count`, `n_channels`, `out_sfreq`, `window_samples`, `hop_samples`,
`capacity_samples`, `warmup_samples`.

Methods:

| Method | Purpose |
| --- | --- |
| `ingest(data, ts)` | Contract reorder + write raw block to the recorder (if any). |
| `wrap(window, window_times, start) -> EEGWindow` | Package one window with its verdict (warm-up + every judge's reasons) and split EEG/EOG. |
| `close(status, error)` | Finalize the recorder (lock run + export FIF). |

What it does NOT do: connect the stream, start threads, process data, run a
chain, consume windows, or define the preprocessing. There is no `process()`
method and no preprocessing defaults — the script owns `STAGES` entirely, and
the session only registers `judges` (objects exposing
`reasons(start, end) -> tuple[str, ...]`, e.g. `QualityMonitor` and `Repair`)
whose verdicts `wrap()` unions onto each window. Warm-up stays session-owned:
it is measured in samples written, not in judge time.

**Expected usage** — see the demo; the minimum script is:

```python
session = StreamSession(stream, args, CHANNELS,
                        judges=(quality, repair),
                        out_sfreq=resampler.out_sfreq)
while running:
    data, ts = session.acquire.read()
    data, ts = session.ingest(data, ts)
    for stage in STAGES:              # the script's own ordered tuple
        data, ts = stage(data, ts)
    for w, w_ts, start in session.buffer.push(data, ts):
        eeg_window = session.wrap(w, w_ts, start)   # verdict included
        if eeg_window.valid:
            consume(eeg_window)
session.close()
```

---

### 5.10 `spatial.py` — eye-artefact projection (C)

**Why.** Blinks and eye movements share a spatial direction across frontal EEG
channels. A fixed projector `P = I - U @ U.T` learned from marked EOG
calibration removes that direction from the EEG columns without touching EOG,
timing or validity. Pure NumPy; no LSL, threads or files except the operator
`.npz`.

**Public surface.**

| Name | Purpose |
| --- | --- |
| `processing_contract(eeg_channels, eog_channels, out_sfreq, units="uV", stamp="")` | JSON-safe description of the chain an operator is calibrated/applied under; exact equality decides compatibility. |
| `cut_epochs(eeg, eog, timestamps, event_times, seconds=1.0)` | **The data-input boundary for calibration.** Cut 1 s epochs out of ALREADY-PROCESSED continuous EEG/EOG (float `(samples, channels)` arrays plus their timestamps) around marked event times; rejects epochs that would fall outside the data. |
| `fit_ssp(eeg, eog, contract, n_components=1, ...)` | Offline fit from marked epochs (events × samples × channels, in uV): mean-subtract, fit on the first events, hold the last third out, fail unless held-out EEG–EOG coupling drops by at least half. |
| `SpatialOperator(matrix, contract, report)` | Validates symmetry/idempotence/rank, derives `artifact_id`, applies per window, saves/loads `.npz` without pickle (never overwrites). |
| `operator.apply_window(window)` | Project the window's EEG columns once (`artifact_id` prevents a second correction), preserve EOG/verdict/timestamps. |

**Where it plugs in:** after `wrap()` and before the consumer — `P` is linear
and time-invariant, so per-window projection equals projecting the continuous
signal first. The script loads an operator and checks `operator.validate(
processing_contract(...))` once before the loop, then calls `apply_window` on
every finished window. Cutting calibration epochs out of a recorded run (raw
chunks -> the same chain -> one-second windows around `blink` events) is an
integration step that uses the recorder's events and window log (Section 9, D).

**The data-input boundary (what "already-processed data" means here).**
`cut_epochs` is where already-processed data enters calibration, and its
docstring spells out exactly what it expects so external producers can feed it
without re-running this package's chain: two float arrays shaped
`(samples, channels)` (EEG columns in the contract's EEG order, EOG columns in
its EOG order, same units the contract declares — microvolts for our chain),
one strictly-increasing timestamp per row on the same clock as the event
times, and epoch windows that fit entirely inside the data. Its output
`(eeg_epochs, eog_epochs)` goes straight into `fit_ssp`.

---

### 5.11 `recovery.py` — bounded recovery (A2)

**Why.** `Repair` fixes what it can and raises `UnrepairableError` for what it
cannot. Killing the whole run on the first glitch is too brittle for a live
stream, so `Recovery` gives the run a bounded number of clean restarts and
stops loudly only when the budget is gone or a single fault is too large. It
is deliberately NOT part of `StreamSession`: the script registers the stateful
components it owns.

**Public surface.** `Recovery(resettable, *, max_events=5, max_gap_seconds=0.5,
persistent_fault_seconds=5.0, recorder=None)`; `handle(UnrepairableError)`
(discard the chunk, reset every registered component, advance `segment`);
`watch(EEGWindow)` (stop after judge-rejected windows last longer than
`persistent_fault_seconds`; a window without finite timestamps raises instead
of silently disabling the watch); `reset()`; attributes `segment`,
`recoveries`, `events`.

**Choosing `persistent_fault_seconds`.** A single bad sample invalidates every
window that overlaps it, so the bad stretch a judge can report is roughly
`window length + settling` (QualityMonitor appends `warmup_seconds`, Repair
appends `settle_seconds`). Set this threshold comfortably above that — about
twice the window length is the safe rule — otherwise one transient spike looks
like a persistent fault and stops the run. The demo's 2 s window, zero quality
settling and 5 s threshold sit on the safe side of that rule; a 5 s analysis
window with the default 2 s quality settling would not.

**Expected usage** (in the script's loop):

```python
try:
    for stage in STAGES:
        data, ts = stage(data, ts)
except UnrepairableError as error:
    recovery.handle(error)          # resets or raises when the run must stop
    continue
# per window: eeg_window.segment = recovery.segment; recovery.watch(eeg_window)
```

### 5.12 `stats.py` — run counters (E)

**Why.** A run's terminal output vanishes; the structured summary should live
with the recording. `StreamStats` counts what happened and
`StreamSession.close(stats=...)` stores it in the recorder metadata.

**Public surface.** Counters `blocks`, `samples`, `windows`, `valid`,
`rejected`, `recoveries`, `repairs`, `dropped`, `gaps`, `max_lag`; `to_dict()`.

**Expected usage.** Increment at each loop stage, then
`session.close(status="completed", stats=stats.to_dict())`.

---

## 6. Data-flow recap and the ordering rules

Who calls what, per run and per block:

```text
RUN level:
  parse_args()                      # bootstrap/args.py
  PlayerLSL(...).start()            # demo only
  StreamLSL.connect(...)            # manual acquisition
  build STAGES tuple + judges       # in the script
  session = StreamSession(...)      # builds the rest
  loop { session.acquire.read() ... }       # the script's while
  session.acquire.close(); stream.disconnect(); session.close()

BLOCK level (inside the loop):
  read()      -> (block, ts)                     # exactly --block rows
  ingest()    -> reorder; recorder.write(raw)    # raw volts are kept
  STAGES      -> for stage in STAGES: data, ts = stage(data, ts)
  buffer.push() -> list of windows completed by this block
  per window: wrap(w, w_ts, start) -> EEGWindow; consume or reject by .valid
```

Invariants to respect when writing a new script:

1. The script owns `STAGES`: every stage is `(data, ts) -> (data, ts)`, and
   the loop folds them in the tuple's order. Convention: repair first, then
   scale -> quality observation -> filters -> resample (filters must not run
   before quality sees the raw signal).
2. `ingest` before the stages (record the untouched volts).
3. Register every verdict provider (`QualityMonitor`, `Repair`) as a
   `judge` so `wrap()` can gate windows on it.
4. Never mutate a window returned by `buffer.push` and expect neighbours to be
   unaffected — each window is already a private copy.
5. Shut down in order: close the acquire handle, disconnect the inlet, stop
   the source, close the offloader, close the recorder.
6. Create run-once objects (offloader, recorder, session) once, outside the
   loop; inside the loop only call their methods.

---

## 7. Where `EEGWindow` lives (implemented)

`CircularBuffer.push()` still returns plain tuples
`(window, window_times, start_sample)` — the buffer stays pure storage.
Between the buffer and the consumer, `StreamSession.wrap()` packages each
tuple into an `EEGWindow` (defined in `src/nova2026/streaming/window.py`, at
the top level of the package: it is about the consumer boundary, not
preprocessing). The class bundles the data with its verdict:

- µV EEG columns (`data`) and auxiliary columns (`eog`) split apart;
- `valid` plus rejection `reasons` (warm-up, quality faults);
- provenance: `start_sample`, `segment`, `channel_names`, `timestamps`,
  optional `contract` and `available_at`.

Arrays are copied on construction, so an `EEGWindow` can be handed straight to
`TaskOffloader` workers without ever being corrupted by later ring writes.
Consumers only need to check `window.valid` once.

```python
eeg_window = session.wrap(window, window_times, start)
if eeg_window.valid:
    model(eeg_window.data)
else:
    reject(eeg_window.reasons)
```

---

## 8. Running, testing, and exploring

```powershell
# From the repository root, with the virtualenv active:
python -B -m scripts.streaming_demo --duration 8 --record records

# Compare consumer modes:
python -B -m scripts.streaming_demo --duration 8 --compute 0.8 --workers 0
python -B -m scripts.streaming_demo --duration 8 --compute 0.8 --workers 2

# Full test suite for the package (181 tests):
python -B -m unittest discover -s tests/streaming -t .

# Real-recording check (needs the COG-BCI dataset in datasets/):
python -B scripts/verify_realdata.py

# Micro-benchmark (no LSL needed):
python -B -m scripts.benchmark_streaming
```

Tests live in `tests/streaming/` (one file per module, plus end-to-end tests
in `test_e2e.py` that stream a real PlayerLSL outlet, record the run, and
replay it offline to confirm identical windows). They run fast and need no
LSL except those PlayerLSL end-to-end tests. If MNE complains about its config
directory, point it somewhere writable first:
`$env:_MNE_FAKE_HOME_DIR = "path/to/a/writable/.mne"`.

`scripts/verify_realdata.py` is the same idea on a *real file*: it loads a
COG-BCI `.set` recording from `datasets/`, pushes about ten seconds through
the exact demo chain (Repair -> uV -> quality -> notch -> band-pass ->
resampler -> windows), records it with `RunRecorder`, replays the saved chunks
offline and checks three things: the run produces windows after warm-up; the
save is correct (the exported FIF equals the SQLite chunks, and the offline
windows match the "live" windows sample-for-sample); and edge cases behave —
a short NaN run is repaired identically online and offline, while an
irreparable burst triggers bounded recovery (the run continues, then stops
once the recovery budget is exhausted). Executed result: PASS on
`sub-01/ses-S1/RS_Beg_EO.set` (63 channels @ 500 Hz, 10 s).

---

## 9. Robustness: what can go wrong, and the guardrails (A–E)

> **Status: implemented.** A1 (`Repair`), A2 (`Recovery`), B
> (`preflight.resolve_outlet` before connecting, then `prepare` /
> `validate_source`), C (`spatial.SpatialOperator` / `fit_ssp` /
> `processing_contract`, with `cut_epochs` as the already-processed-data
> input boundary), D (recorder `config` snapshot, provenance file hashes,
> per-window log, `not_processed` tail mark) and E (`StreamStats`, persisted
> at close) are implemented and covered by tests. A real-recording check
> (`scripts/verify_realdata.py`, a real COG-BCI `.set` file) confirms the
> path runs through, saves correctly (FIF == SQLite) and replays identically
> offline. The demo wires A1, A2, B, D and E; C's end-to-end operator fit on
> a recording with marked blinks still awaits such a recording to exercise.

A live stream is fragile in ways an offline file is not. The sender can stall,
skip samples, emit `NaN`, send the wrong metadata, or degrade slowly over time.
One corrupted sample can poison a stateful filter for seconds. The wrong outlet
can quietly record garbage with plausible channel names. A slow consumer can
fall further and further behind. The Section 3 pipeline assumes a well-behaved
source; the guardrails below close the failure classes one by one:

| Failure class | Example | Guardrail |
| --- | --- | --- |
| Broken values, short | isolated `NaN`/`Inf`, a few ms of missing samples | **A1** — repair short spans by interpolation (**implemented**) |
| Broken values / timing, severe | a whole damaged chunk, gaps > 0.5 s, overlapping or irregular timestamps | **A2** — bounded recovery: discard and restart, or fail (**implemented**) |
| Slow decay | minutes of near-flat or saturated signal | **A2** — persistent-fault watch on windows (**implemented**) |
| Wrong source | wrong sampling rate, units, channel labels/types | **B** — validate the source before the first sample (**implemented**) |
| Eye artefacts | blinks/saccades leaking into frontal EEG | **C** — calibrated spatial projection (EOG-guided SSP) (**implemented**) |
| No provenance | cannot replay a run, cannot tell which operator/files were used | **D** — richer run metadata and snapshots (**implemented**) |
| No audit trail | a run stops and leaves no structured summary | **E** — run statistics persisted at close (**implemented**) |

The letters A–E match the design discussion. Each guardrail below follows the
same format: *the problem*, *where it plugs into the Section 3 pipeline*, *why
that spot*, and *what it needs from the other components*.

Where the guardrails sit, on the Section 3 diagram:

```text
   outlet -> StreamLSL
              |   [B1] resolve_outlet() first   (implemented)
              |   [B2] validate_source() after   (implemented)
              v
   read() -> ingest()                 # reorder + record RAW volts
              |                       #   (raw damage is kept on purpose)
              v
   [A1] Repair stage (implemented)    # short NaN/Inf spans, source units,
              |                       #   first stage of the script's STAGES
   [A2] Recovery guard (implemented) -> catches UnrepairableError from A1:
              |                         reset the chain + segment++, or stop
              v
   STAGES (the script's tuple)       # scale -> quality -> filters -> resample
              v
   push() -> wrap() -> EEGWindow     # [A2 watch] persistent faults (impl.)
              |                      # [D3] window log (implemented)
              v
   [C2] operator.apply_window()      # implemented; the script wires it between
              |                      #   wrap() and the consumer
              v
   consumer
```

---

### A. Surviving a damaged signal: repair (A1) and bounded recovery (A2) — implemented

**The problem.** EEG amplifiers occasionally produce a few broken samples:
an isolated `NaN` or `Inf`, or a short run of missing samples. Two things make
this dangerous even though the damage is tiny:

1. A stateful causal filter keeps history. Feed it a `NaN` once and its output
   is corrupted for the filter's settling time afterwards — many times longer
   than the damage itself.
2. A quality monitor that sees the raw `NaN` records a fault, even though the
   damage was fixable.

**A1 — short-span repair (implemented as `Repair`).** When a broken run has a
finite sample on *both* sides and lasts at most `max_seconds` (default 20 ms),
fill the gap by linear interpolation between those two endpoints. Everything
longer than that is not A1's business.

- **Where:** the first stage of the script's `STAGES` tuple — after
  `ingest()`, before the unit scaler, before quality and filters. Repair
  happens in source units on the original signal; the recorder keeps the
  *unrepaired* raw volts so offline replay can repair identically and the raw
  damage stays visible for diagnosis.
- **Why there:** scaling cannot fix values and filters must never see a `NaN`;
  A1 is the earliest point where the data is in one canonical shape.
- **How it behaves (v1):** a row is damaged when any of its values is
  non-finite; missing rows are detected from timestamp gaps, synthesised and
  repaired the same way. Repair is linear, per channel, and never rewrites
  healthy values. Finished repairs are remembered as time intervals; `Repair`
  is registered as a judge, so any window overlapping a repaired span comes
  out `valid=False` with reason `"interpolated"`. A damaged tail without a
  finite right endpoint is held back and returned with the next block, so the
  chain only ever sees settled rows.
- **What it cannot repair:** damage longer than `max_seconds`, irregular or
  non-finite timestamps, and — when source units are known — unsafe endpoints
  (saturated or an extreme jump) raise `UnrepairableError` with a
  machine-readable kind and, for gaps, the gap size. Damage before the first
  finite sample is dropped instead (the run simply starts later). The bounded
  recovery guard (A2) catches the error and decides the run's fate.
- **Needs:** its own stateful tail and a `reset()` — both implemented and
  called by A2 on every restart.

**A2 — bounded recovery (implemented as `Recovery` in `recovery.py`).** For
damage that cannot be repaired — a chunk that is still non-finite, timestamps
that are irregular or overlap, a gap between chunks — the run has two options:
quietly keep going on garbage, or stop. Both are wrong in general, so the rule
is *bounded recovery*: a limited number of times, the run may throw the bad
chunk away and start a fresh processing **segment**, and beyond that it stops
loudly. `Repair` raises `UnrepairableError` in exactly these situations; the
script's loop catches it and hands it to the guard.

- **Where:** two spots. `Recovery.handle(error)` is called when `Repair`
  raises (the chunk is discarded and the chain restarts); `watch(window)` runs
  right after `wrap()`, because only finished windows reveal "the signal has
  been bad for five seconds straight".
- **How a recovery works:** `handle` increments the recovery count, then
  resets *every* stateful stage the script registered — filters, resampler,
  ring buffer, quality history and `Repair`'s tail — and advances the segment.
  Warm-up repeats, so windows from the restart are rejected until warm-up
  passes again. The demo stamps `eeg_window.segment = recovery.segment`, so no
  window ever spans two segments; recorded runs also get a
  `"recovery:<kind>"` event per restart.
- **When it fails instead:** more than `max_events` recoveries (default 5), a
  single gap longer than `max_gap_seconds` (0.5 s), or judge-rejected windows
  persisting longer than `persistent_fault_seconds` (5 s, `watch`). These stop
  the run with an error — a live system must never silently ship garbage.
- **The architectural cost:** the stateful stages live in the script, so the
  script tells the guard which components are resettable. The **reset
  protocol** is one rule — every registered component implements `reset()` —
  and `Recovery` stays completely independent of `StreamSession`.

---

### B. Validate the source before the first sample (implemented)

**The problem.** The pipeline trusts whatever outlet it connects to. Connect to
the wrong stream, or to the right stream with a different sampling rate or
different declared units, and the run records plausible-looking but wrong data
— the worst kind of corruption, because nothing complains later.

**Where:** in two phases, both before any sample is read or recorded. Both are
implemented in `nova2026.streaming.preflight`.

- **B1, before connecting (implemented as `resolve_outlet`).** Resolve the
  available outlets and require one whose `name` / `source_id` / `stype`
  match the configuration, polling up to a timeout (a freshly started source
  can take a moment to appear). This fails fast — "the amplifier is not
  publishing" — instead of timing out while connected to an unrelated stream.
  Only identity is visible without connecting; rates and units still need B2.
- **B2, right after connecting (implemented):** check, from the connected
  inlet's metadata: the sampling rate matches the configured `sfreq`; every
  required channel declares voltage units consistent with the configured
  source exponent (V / mV / µV / nV); labels are unique and contain the
  required set; the channel types identify EEG and EOG correctly; samples are
  numeric; and no filters, callbacks or unread samples exist before setup.
  The reference and upstream-processing descriptions cannot be verified
  automatically — they are carried by the run's config snapshot (D1) as
  operator assertions for later audits.

**Why both phases:** the pre-connect check answers "is this the outlet we
mean?", the post-connect check answers "does this outlet deliver what we
declared?" — the second needs metadata that only exists after connecting, so it
cannot be moved earlier.

---

### C. Eye-artefact removal by calibrated projection (implemented as `spatial.py`)

> **Still too technical? Read this first.** Imagine you are taking photos
> through a window and the glass has a smudge. Every photo has the same
> smudge in the same place. You could try to clean each photo by hand — or
> you could first learn exactly where the smudge is (one calibration photo),
> and then subtract it from every photo automatically. Blinks and eye
> movements are the "smudge" of EEG: when the eye moves, all the frontal
> electrodes see one shared wobble. The EOG channel is our "reference photo":
> it sits next to the eye and directly measures that wobble. Guardrail C
> learns the shared direction of the wobble once, then removes it from the
> EEG columns of every window. That is all it does.

**The problem.** When somebody blinks or moves their eyes, the electrical
signal travels a little way across the scalp, so *several* EEG channels see it
at the same time — like the smudge appearing in the same spot on many photos.
A classifier trained on clean windows will happily learn "the smudge = a real
brain event", because it never saw the smudge removed. We want to erase the
eye wobble from the EEG *before* the classifier ever looks at a window.

**One mental model for "direction".** At any instant, one sample of N EEG
channels is just a list of N numbers. You can think of that list as a point
in an N-dimensional space. When the eye moves, the points all slide together
along one particular direction in that space (say, "front channels up,
back channels down" is one such direction). If we know that direction — call
it `U` — we can build a simple linear filter `P = I - U @ U.T` that says:
"take any signal, erase the part that points along `U`, keep everything else".
Applying `P` is called *projecting*. This is a well-known technique called
signal-space projection (SSP); it is NOT ICA (blind source separation) and NOT
sample interpolation — we are not guessing missing values, we are removing a
learned direction.

**Why we need a calibration run at all.** We do not know `U` in advance — it
depends on the person's head, electrode positions and the amplifier. So we
learn it from a short, boring recording where the person deliberately blinks
and moves their eyes on command, while a special EOG channel measures the eye
movements. That recording is only used once, offline, to produce a small file
called the *operator* (`.npz`). Every later run can use the same operator.

**What a calibration session looks like (for the person running it).**

1. Record a run with role `artifact_calibration` (raw EEG + a real EOG
   channel + the recovery/quality guardrails active). Use the recorder's
   `mark("blink")` / `mark("eyes_left")` / ... to stamp each event at the
   moment you see it.
2. Collect at least six events, spaced about a second apart (more is better),
   with a few seconds of clean recording before the first and after the last.
3. The recorder keeps the raw data, the event times, and the window log (D3)
   — that is everything calibration needs later.

**C1 — calibration, step by step (offline, one time).**

| Step | What happens | Which function |
| --- | --- | --- |
| 1 | Replay the recorded run through the **exact same chain** the live runs use, so the data matches what a real window would look like (same repair, scaling, filters, resampling). | `replay_chunks` + your chain |
| 2 | Cut a 1-second piece (an *epoch*) around every `blink`/`eyes_*` event. This is where ALREADY-PROCESSED data enters: the input boundary. | `cut_epochs(...)` |
| 3 | Fit `U` using only the *first* events; keep the *last third* untouched as a test set. Never test on data you fitted on. | `fit_ssp(eeg, eog, contract)` |
| 4 | Check the result honestly: measure how much EEG still "moves together with" EOG in the held-out events. It must drop by at least half — otherwise the calibration is useless and the function refuses to build an operator. | inside `fit_ssp` |
| 5 | Save the operator (matrix + contract + a report with the numbers above). The file refuses to overwrite an existing operator. | `operator.save(path)` |

**What exactly `cut_epochs` expects (so you never have to guess).** It wants
data that is already processed, and its docstring spells the rules out. In
plain words:

- `eeg` and `eog` are two 2-D grids of numbers: rows = samples, columns =
  channels. EEG columns must be in the contract's EEG order; EOG columns in
  the contract's EOG order; values in the contract's units (microvolts for
  this chain). A tiny worked example: 10 blinks, 1 s each at 128 Hz, 8 EEG
  channels → after cutting you get an `(10, 128, 8)` array.
- `timestamps`: one time per row (same clock the events use), strictly
  increasing.
- `event_times`: the marked event centres, sorted.
- `seconds`: how long each epoch is. If any epoch would stick out of the data,
  `cut_epochs` refuses — a silently shortened calibration would be worse than
  none.
- Output is `(eeg_epochs, eog_epochs)`, shaped `(events, samples, channels)`,
  ready for `fit_ssp`.

**C2 — application (online, every window).** Once an operator exists, a run
may choose to use it. In the script, before the loop:

```python
operator = SpatialOperator.load("eye_operator.npz")
operator.validate(processing_contract(...))   # refuse if the run differs
```

Then, right after `wrap()` and before the consumer, every window passes
through `operator.apply_window(eeg_window)`. `P` is *linear and
time-invariant*, which is a fancy way of saying "it does not matter whether
we correct each 2-second window or the whole continuous signal first — the
result is identical". That is why we can correct window-by-window and keep
the ring buffer completely untouched. Four safety rules are built in:

1. **Check the table first**: the operator remembers the channel order, rates,
   units and a `stamp` of the chain it was calibrated under; if the current
   run's contract differs, it refuses instead of applying a wrong correction.
2. **Only the EEG columns change**; EOG is left alone (it is measurement, not
   noise here).
3. **Stamp once**: `apply_window` writes an `artifact_id` on the window, so a
   window can never be corrected twice.
4. **Never rescue a bad window**: if the window was already rejected by the
   quality/recovery guards, correction keeps it rejected — a projector can
   never turn garbage into a valid window.

**Honest warnings.** (1) If some *real* brain activity happens to point in
the same direction as the eye wobble, it is removed too — that is physics, not
a bug, so inspect the calibration report before trusting an operator on human
data. (2) After projection the data has one fewer "direction" of freedom
(mathematically: its rank drops by one); a future classifier working on
covariances must account for that. (3) Nothing ever auto-selects the newest
operator file; the choice is explicit in the script and auditable (D2 stores a
copy + hash of the exact file). (4) If no calibration recording exists yet,
this whole guardrail is simply skipped — the pipeline runs exactly as before,
only without eye-artefact removal — and can be added later the moment such a
recording (EOG channel + marked events) is available. The mechanics above are
implemented and unit-tested; only that real recording is missing for a
full end-to-end demonstration.

---

### D. Run provenance: what happened, exactly (implemented)

**The problem.** A recorded run is only science if someone can later answer:
what settings produced it, which operator/baseline/model files went in, which
windows were actually delivered, and did it finish. The recorder now stores
these.

- **D1 — configuration snapshot (implemented).** The recorder accepts a
  `config` dict (`RunRecorder(..., config=...)`; `StreamSession` forwards
  `recorder_config=`) and stores it under the `config` metadata key — the
  script describes its own chain (rates, notch/band-pass settings, order,
  resampling quality, window geometry, units) via
  `spatial.processing_contract`. Without it, offline replay cannot know what
  processing a run went through.
- **D2 — selected-input snapshots (implemented).** The recorder accepts
  `files={"artifact": path, ...}`; each file is copied into the run folder and
  its original path and SHA-256 hash are stored as `input:<role>` metadata. A
  missing file fails the run before the run folder is created. Nothing
  auto-selects the newest file — the choice is explicit and auditable.
- **D3 — window event log (implemented).** With `track_windows=True`, one line
  per delivered window (timestamp, `valid`, reasons, segment, `artifact_id`)
  is written by `recorder.log_window(...)` and read back with
  `iter_windows(path)`. This is what lets an offline replay reproduce exactly
  which windows were delivered, and what calibration (C1) aligns its epochs
  against. The demo logs every window after `wrap()`.
- **D4 — honest endings (implemented).** The run is locked as `completed` or
  `failed` with the error text at `close()`, and a crash leaves a clearly
  `recording`-status database that cannot be mistaken for a finished one.
  When the loop stops with a tail of samples still buffered, the demo records
  them with `recorder.mark_not_processed(rows)` (an event labelled
  `not_processed:<rows>`) instead of silently dropping them.

**Where:** all of D lives in the recorder; the config snapshot and provenance
files are passed at construction (through `StreamSession` when a script uses
it), and the per-window log is written right after `wrap()` in the loop.

---

### E. Run statistics (implemented as `stats.py`)

**The problem.** A run can stop for any of the reasons above, or finish
normally, and the terminal output is gone. The next question — "how many
windows were delivered, how many faults were recovered, how bad was the
transport?" — needs a structured answer attached to the run itself.

**What it does.** `StreamStats` is a small counter object (`blocks`, `samples`,
`windows`, `valid`, `rejected`, `recoveries`, `repairs`, `dropped`, `gaps`,
`max_lag`). The script increments it at each stage of the loop, then hands
`stats.to_dict()` to `StreamSession.close(status=..., stats=...)`, which stores
it in the recorder metadata (`stats` key) — see D1's metadata home.

---

### What this section deliberately does not promise

Reference application, ICA-based removal, per-subject adaptive calibration, and
validation against real amplifier hardware are all out of scope for A–D. The
guardrails protect against transport and recording failures; they do not turn
synthetic checks into proof that neural activity survives a projector.

---

## 10. Glossary (quick lookup)

| Term | Meaning |
| --- | --- |
| channel | one electrode position; column of the data |
| sfreq / Hz | samples per second |
| sample | one measurement of every channel at one instant |
| µV | microvolt, 1e-6 V; what EEG is measured in |
| chunk | whatever piece of data arrives at once |
| block | the fixed size `Acquire.read()` returns |
| window | a fixed-length slice (e.g. 2 s) the consumer analyses |
| hop | time between consecutive window starts |
| warm-up | initial seconds before filters/resampler output can be trusted |
| manual acquisition | `acquisition_delay=None`; we pull data explicitly |
| stateful | remembers something between calls |
| ring buffer | fixed array whose write pointer wraps around |
| drop-oldest / drop-newest | overflow policies of the offloader queue |
| FIF | MNE's native EEG file format (`.fif`) |
| epoch | a fixed-length slice of data cut around an event (calibration) |
| projector / SSP | a matrix that removes a learned spatial direction (`P = I - U Uᵀ`) |
| recovery / segment | bounded restart after unrepairable damage; each restart is a new segment |
| not_processed | recorder event marking samples buffered when the run stopped |
| pre-flight | checks done before the first sample: outlet resolution + metadata validation |
