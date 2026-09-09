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
   StreamSession.process()        --> scaler (V->uV)
        |                            quality observer
        |                            notch 60 Hz, band-pass 1-45 Hz (filters)
        |                            SoXR resampler 500 -> 128 Hz
        v
   CircularBuffer.push()          --> 0..N finished windows
        |
        v
   StreamSession.gate()           --> drop warm-up windows and windows with
        |                            quality faults
        v
   Consumer: print, model, ...    --> runs in the loop, or on TaskOffloader
        |                            worker threads
        v
   RunRecorder.close()            --> lock the run, export .fif (optional)
```

The runtime ordering of one run:

```text
parse args  ->  start fake source  ->  connect inlet
           ->  define preprocessing chain (in the script!)
           ->  StreamSession(...)     [builds contract/recorder/acquire/buffer]
           ->  while running:  read -> ingest -> process -> push -> gate -> consume
           ->  finally:        close acquire, disconnect, close recorder
```

Two rules that keep the pipeline correct:

- **Preprocessing order is fixed**: units first, quality observes the *raw
  cleaned-but-unfiltered* signal, then filters, then resampling last.
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

### The docstring (lines 1-26)

Everything between the first `"""` and its closing `"""` is the **module
docstring**. Python stores it in `__doc__`, and argparse prints it when you run
`--help`. So this text explains the demo *and* is the help screen — one source
of truth. It shows the exact command, the pipeline, and how to compare
"analysis in the loop" (`--workers 0`) with "offloaded to worker threads"
(`--workers 2`).

### Imports (lines 28-48)

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
- `QualityMonitor, Resampler, SosFilter, design_bandpass, design_notch,
  unit_scaler` — the preprocessing parts, imported here on purpose: the demo
  wants you to *see* the chain being defined.

### Constants (lines 53-56)

- `CHANNELS` — the 8 channel names our fake device publishes, in column order.
- `SOURCE_UNIT_EXPONENT = 0` — the source sends volts (`10^0`), not µV.

### `synthetic_recording(seconds, sfreq)` (lines 59-74)

Builds the fake data. One clean 10 Hz sine wave per channel; each channel has a
slightly different amplitude (`(10 + index)` µV, i.e. 10..17 µV), so the
statistics printed later are easy to eyeball. Returns an `mne.io.RawArray`
holding volts. In real life you would delete this function and point
`PlayerLSL` (or the inlet) at the real amplifier.

### `main()` — step 1, start the fake source (lines 82-95)

- `args = parse_args(description=__doc__)` — the bootstrap argument getter
  defines all command-line options (rates, window sizes, filter settings,
  recording identity, consumer mode) with defaults; `description=__doc__`
  makes the module docstring the `--help` text. Nothing in this script
  declares argparse options.
- A unique `name`, then `PlayerLSL(...)` is created and `start()`ed. It
  "plays" `duration + 6` seconds of data (extra lead time for filter warm-up)
  in chunks of `--chunk-size` (37 by default — deliberately uneven).
- `--workers < 0` is rejected early.

### Step 2, connect the inlet (lines 101-106)

```python
stream = StreamLSL(bufsize=4.0, name=name)
stream.connect(acquisition_delay=None, processing_flags=["clocksync"], timeout=10)
```

`acquisition_delay=None` = manual acquisition (we pull when we want).
`clocksync` = synchronize the device clock with ours. `bufsize=4.0` = the
inlet keeps up to 4 seconds of buffered data.

### Step 3a, define the preprocessing chain explicitly (lines 112-120)

This is the heart of "the script owns the pipeline":

```python
scaler    = unit_scaler(SOURCE_UNIT_EXPONENT, desired_exponent=-6)  # V -> uV
quality   = QualityMonitor(n_eeg=len(CHANNELS), sfreq=args.sfreq, ...)
notch     = SosFilter(design_notch(args.notch, args.notch_q, args.sfreq), n)
bandpass  = SosFilter(design_bandpass(args.lpass, args.hpass, args.order, ...), n)
resampler = Resampler(args.sfreq, args.out_sfreq, n, quality=args.resample_quality)
```

Every stage, its parameters, and its order are visible and editable here.
`design_notch`/`design_bandpass` compute filter coefficients; `SosFilter`
applies one filter continuously (it remembers its state between blocks).

### Step 3b, assemble the rest via `StreamSession` (lines 126-137)

```python
session = StreamSession(stream, args, CHANNELS,
                        scale=scaler, quality=quality,
                        filters=(notch, bandpass), resample=resampler, ...)
```

`StreamSession` (see `bootstrap/session.py`) builds the boring run-once
pieces: the channel contract, the optional recorder, the `Acquire` object and
the `CircularBuffer`. It does **not** define the preprocessing; it only
guarantees the order in which it will run them. If the source sent channels we
do not want, the contract prints them here as "dropping ...".

### Step 4, print the configuration (lines 142-152)

One block of `print()` that shows what will happen: source rate, chain
settings, resampler target, window/hop/capacity in *samples at the output
rate*, consumer mode, and the recording destination. Printing the plan makes
every run self-explanatory.

### Step 5, consumer state and helpers (lines 157-182)

- `started = monotonic()` — the wall clock used to stop after `--duration`.
- `valid_windows` / `rejected_windows` — counters for the final summary.
- `report(window, start)` — prints one line per window: the peak-to-peak
  amplitude (max - min, in µV) of each channel. This is our stand-in for
  "a model".
- `analyze(item)` — unpacks a submitted `(window, start)`, optionally
  `sleep()`s to pretend analysis is slow, then reports.
- `offloader = TaskOffloader(analyze, workers=..., capacity=...)` — created
  *once*, only when `--workers > 0`. It is a bounded queue plus worker
  threads; workers pick up `(window, start)` tasks and run `analyze`.

### Step 6, the real-time loop (lines 188-212)

```text
while monotonic() - started < args.duration:      # until time is up
    (a) data, ts = session.acquire.read()         # pull one fixed block
    (b) data, ts = session.ingest(data, ts)       # reorder + record raw
    (c) data, ts = session.process(data, ts)      # uV/quality/filters/resample
    (d) for window, w_times, start in session.buffer.push(data, ts):
        (e) accepted, reasons = session.gate(start, w_times)
            if not accepted: count & skip          # warm-up or bad quality
        (f) analyze(window, start)                # or offloader.submit(...)
```

Every iteration is one acquired block. `push()` returns the list of windows
*completed by this block* — often empty, sometimes several. The gate decides
which windows deserve the consumer: windows whose `start` is still inside the
warm-up period, or that overlap a recorded quality fault, are rejected and
counted.

### Step 7, shutdown (lines 217-233)

The `finally:` block always runs, whether the loop ended normally, the user
pressed Ctrl+C (`KeyboardInterrupt`), or the lag guard raised `RuntimeError`.
Order matters:

1. `session.acquire.close()` — detach the callback and drop leftover samples;
2. `stream.disconnect()` — close the LSL inlet;
3. `player.stop()` — stop the fake source;
4. `offloader.close(drain=True)` — let queued analysis finish;
5. `session.close(status="completed")` — locks the run, exports the `.fif`.

### Step 8, summaries (lines 239-256)

Prints acquisition statistics (`input_samples`, `max_lag`, `gaps`), how many
samples the resampler emitted, how many windows were valid/rejected, and the
offloader counters. `max_lag` is the oldest age of any consumed block — if it
ever exceeds 3 s the lag guard stops the run instead of silently analysing
stale data.

### Entry point (lines 262-263)

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
  channels.py                # ChannelContract  - check/reorder channels
  circular_buffer.py         # CircularBuffer   - ring storage -> windows
  offload.py                 # TaskOffloader    - run analysis on workers
  recording.py               # RunSpec/RunRecorder + read-back helpers
  preprocess/
    __init__.py              # re-exports the preprocessing pieces
    units.py                 # unit_scaler      (stateless)
    filters.py               # design_* + SosFilter (stateful)
    quality.py               # QualityMonitor   (stateful observer)
    resample.py              # Resampler        (stateful, soxr)
  bootstrap/
    __init__.py              # re-exports
    args.py                  # make_parser / parse_args (argument getter)
    session.py               # StreamSession    (run-once assembly)
```

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

### 5.3 `channels.py` — `ChannelContract`

**Problem.** The device publishes channels in *its* order and set; our
pipeline/recording expects channels in *our* order (EEG first, aux like EOG
after). If we silently paired columns with the wrong names, every number
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

**Expected usage** (usually inside `StreamSession`):

```python
contract = ChannelContract(stream.ch_names, EXPECTED)
data = contract.reorder(data)     # every block
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
    ...  # e.g. session.gate(start, window_times)
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
offloader.submit((window, start))       # non-blocking
...
offloader.raise_error()                 # surface worker failures
offloader.close(drain=True, timeout=5.0)
```

---

### 5.6 `recording.py` — `RunSpec`, `RunRecorder` and read-back helpers

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
| `RunRecorder(root, spec, channels, sfreq, *, ch_types, unit_exponent, dtype, export_fif)` | Creates `root/subject/session/run/run.sqlite`; refuses to overwrite an existing run. |
| `recorder.write(data, timestamps)` | Commit one chunk (data BLOB + row count + first timestamp) in its own transaction — crash-safe. |
| `recorder.mark(label, timestamp=None)` | Thread-safe task event (e.g. `"blink"`), for the controller thread. |
| `recorder.close(status="completed", error=None, stats=None)` | Lock the run; if `export_fif`, read the DB back and write `<run>_raw.fif` next to it; write a JSON sidecar. |
| `read_metadata(path)` | All metadata as a dict. |
| `iter_chunks(path)` | Yield `(data, first_timestamp)` for every chunk in order. |
| `iter_events(path)` | Event list `(timestamp, label)`. |
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

### 5.7 `preprocess/` — the preprocessing package

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

**Expected usage** (wired by `StreamSession` as `session.quality`; the gate
uses `reasons`): feed every block after scaling and before filters; when a
window is finished, ask `reasons(window_start_time, window_end_time)`.

#### `preprocess/resample.py` — `Resampler`

**Why.** Change the sample rate (500 -> 128 Hz) *continuously* while
streaming. SoXR (`soxr`) provides a stateful `ResampleStream`; this class
wraps it in the chain contract and regenerates an output-rate timestamp grid.

**Why a class.** Resampling is stateful (internal history, output index,
anchor) and its output row count per call varies (0 on early calls while SoXR
buffers).

**Dependency note.** `soxr` is imported lazily, only when a `Resampler` is
constructed; the rest of the package works without it (the demo requires it).

**Public surface.** `Resampler(in_sfreq, out_sfreq, n_channels, quality="LQ")`;
`(data, timestamps) -> (out, out_timestamps)`; `reset()`; `output_samples`;
public `in_sfreq`, `out_sfreq`.

---

### 5.8 `bootstrap/` — start-up helpers so scripts stay small

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
              scale=None, quality=None, filters=(), resample=None,
              source_unit_exponent=0, role="run", ch_types=None)
```

Attributes: `contract`, `recorder`, `acquire`, `buffer`, `scale`, `quality`,
`filters`, `resample`, `n_channels`, `out_sfreq`, `window_samples`,
`hop_samples`, `capacity_samples`, `warmup_samples`.

Methods:

| Method | Purpose |
| --- | --- |
| `ingest(data, ts)` | Contract reorder + write raw block to the recorder (if any). |
| `process(data, ts)` | Run the chain in the fixed order: scale -> quality.feed -> each filter -> resample. |
| `gate(start_sample, window_times) -> (accepted, reasons)` | Warm-up + quality gating for one finished window. |
| `close(status, error)` | Finalize the recorder (lock run + export FIF). |

What it does NOT do: connect the stream, start threads, consume windows, or
define the preprocessing. Sensible defaults exist (`scale`/`quality` fall
back to a unit scaler and a monitor over all channels), but the intended style
is the explicit demo.

**Expected usage** — see the demo; the minimum script is:

```python
session = StreamSession(stream, args, CHANNELS,
                        scale=scaler, quality=quality,
                        filters=(notch, bandpass), resample=resampler)
while running:
    data, ts = session.acquire.read()
    data, ts = session.ingest(data, ts)
    data, ts = session.process(data, ts)
    for w, w_ts, start in session.buffer.push(data, ts):
        if session.gate(start, w_ts)[0]:
            consume(w, start)
session.close()
```

---

## 6. Data-flow recap and the ordering rules

Who calls what, per run and per block:

```text
RUN level:
  parse_args()                      # bootstrap/args.py
  PlayerLSL(...).start()            # demo only
  StreamLSL.connect(...)            # manual acquisition
  define scaler/quality/filters/resampler   # in the script
  session = StreamSession(...)      # builds the rest
  loop { session.acquire.read() ... }       # the script's while
  session.acquire.close(); stream.disconnect(); session.close()

BLOCK level (inside the loop):
  read()      -> (block, ts)                     # exactly --block rows
  ingest()    -> reorder; recorder.write(raw)    # raw volts are kept
  process()   -> uV; quality.feed; filters; resample
  buffer.push() -> list of windows completed by this block
  per window: gate(start, w_ts) -> consume or reject
```

Invariants to respect when writing a new script:

1. Preprocessing order is scale -> quality observation -> filters -> resample
   (filters must not run before quality sees the raw signal).
2. `ingest` before `process` (record the untouched volts).
3. Never mutate a window returned by `buffer.push` and expect neighbours to be
   unaffected — each window is already a private copy.
4. Shut down in order: close the acquire handle, disconnect the inlet, stop
   the source, close the offloader, close the recorder.
5. Create run-once objects (offloader, recorder, session) once, outside the
   loop; inside the loop only call their methods.

---

## 7. Where will `EEGWindow` live (next step)?

Today `CircularBuffer.push()` returns plain tuples
`(window, window_times, start_sample)` and the script performs gating by hand
with `session.gate`. The old dataproc code packaged one window together with
its verdict: data, µV EEG vs EOG, timestamps, `valid`, rejection `reasons`,
`start_sample`, `segment`, `channel_names`, `contract`, ...

When we add that layer it will be a small value class, naturally placed at
`src/nova2026/streaming/window.py` (top level of the package — it is about the
consumer boundary, not preprocessing), produced between the buffer and the
consumer (by `StreamSession` or a tiny assembler). `session.gate` would then
*create* an `EEGWindow` instead of returning `(accepted, reasons)`, so the
consumer receives one object that already knows whether it may be used and
why.

---

## 8. Running, testing, and exploring

```powershell
# From the repository root, with the virtualenv active:
python -B -m scripts.streaming_demo --duration 8 --record records

# Compare consumer modes:
python -B -m scripts.streaming_demo --duration 8 --compute 0.8 --workers 0
python -B -m scripts.streaming_demo --duration 8 --compute 0.8 --workers 2

# Full test suite for the package:
python -B -m unittest discover -s tests/streaming -t .
```

Tests live in `tests/streaming/` (one file per module). They run fast and need
no LSL except a few PlayerLSL end-to-end tests. If MNE complains about its
config directory, point it somewhere writable first:
`$env:_MNE_FAKE_HOME_DIR = "path/to/a/writable/.mne"`.

---

## 9. Glossary (quick lookup)

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
