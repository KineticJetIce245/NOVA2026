# The time base: one owner for the LSL timeline

**Status: proposal, for review. Nothing in this document is implemented.**
Scope: Phase 1 of the streaming refactor — the time base only. The channel
policy, the recording reader, the offload wiring and the legacy
`scripts/dataproc/streaming` tree are deliberately out of scope.

Related: `documents/streaming_explained.md` §11 (the bring-up post-mortem),
`scripts/getlive/compare_cnt.py` (the ground-truth tool),
`tests/streaming/test_rig_bringup.py` (the frozen regression tests).

## 1. The symptom

On the 2026-09-12 bring-up the pipeline reported, on every run, "gaps": samples
the source never delivered. The reports looked like this:

| run | samples | gaps | repairs | `interpolated` windows |
| --- | --- | --- | --- | --- |
| `bringup/run-171856` | 11 200 | 4 | 6 | 15 of 39 |
| `bringup30/run-171935` | 15 600 | 5 | 5 | 14 of 58 |
| `offload8/run-172727` | 15 550 | 2 | 2 | 7 of 58 |
| `offload1/run-172814` | 15 600 | 6 | 9 | 17 of 58 |

Those gaps were not free: `Repair` synthesises a NaN row for every sample the
grid implies and the source did not send, interpolates it, and flags every window
that overlaps the repair as `interpolated` - 14 of 58 windows in the 30 s run.
(The `unsafe_endpoints` faults seen that day were a separate symptom: they came
from the wrong unit assertion, not from the electrode offset. Corrected in
`streaming_explained.md` 11.8.)

## 2. Two explanations were tried and both are wrong

**"Network packet loss."** LSL runs over UDP multicast, so ~0.14% loss was
plausible. It is disproved: the control software writes its own `.cnt` file
while the same amplifier publishes LSL, and a sample-by-sample comparison of
the two (`scripts/getlive/compare_cnt.py`, 6 runs, 88 500 sample pairs, 2.6 min
of signal) finds **every run aligned at r = 1.000000**, the worst disagreement
at **half the amplifier's LSB** (0.0039 uV of 0.0078125 uV), and **zero samples
differing by more than one LSB**. A single integer offset aligns the entire run,
so nothing was dropped, duplicated or interpolated anywhere in it.

**"The amplifier's rate is wrong."** Dividing the total drift by the run length
gives an "effective rate" of 499.71-499.84 Hz (-322 to -578 ppm). That number is
an artefact of the division; see §3.

## 3. What the measurements actually say

The recorder stores only each chunk's first timestamp and rebuilds a uniform
grid from it (`src/nova2026/streaming/recording.py:7`,
`chunk_timestamps()`). Comparing those anchors against the sample count
(`grid_deviation`, and reproduced independently for this document):

| run | residual, last chunk | ≥ 0.5-sample steps | ≥ 1-sample steps | largest step |
| --- | --- | --- | --- | --- |
| `run-171856` | +5.99 | 5 | 1 | **+2.00** |
| `run-171935` | +5.01 | 4 | 1 | **+2.00** |
| `run-172216` | +6.97 | 7 | 1 | **+1.00** |
| `run-172323` | +5.97 | 6 | 1 | **+1.00** |
| `run-172814` | +9.00 | 8 | 2 | **+2.00** |
| `run-172727` | +2.00 | 2 | 1 | **+1.01** |

The residual is a **staircase, not a ramp**. Its increments are ~0 between
steps, and the steps are **whole samples**. That settles the rate question: over
30 s the nominal 500 Hz grid matches the anchors to better than about 100 ppm,
so **the rate is fine and does not need calibrating**.

It does *not*, on its own, settle where the steps come from, and the shape
cannot: a clock correction and a genuine loss of k samples both appear as a
permanent +k step in the residual. The integrality of the steps is not evidence
either, because the relay's own arithmetic is integral — it inserts
`rint(step) - 1` slots per chunk (`GridBuilder.feed`), so the staircase's shape
is what that formula produces whatever the source did.

What can be said with the numbers in hand: the relay's own log reported "45
samples missing over 115 s", i.e. ~12 phantom slots per 30 s, the same order as
the 5-9 samples of drift measured here. **The drift is therefore at least
largely the relay's own inserted slots, not the source's behaviour**, and every
one of those slots is a sample that never existed.

## 4. The mechanism as implemented today

```
two machines' clocks: LSL clocksync makes an occasional whole-sample correction
        |
        v   (the sample stream is intact - the CNT comparison proves it)
the source's timestamps step by +1 or +2 samples
        |
        v
relay/GridBuilder treats the step as "the source lost samples" and advances its
grid index by rint(step) - 1                     <-- pure invention, no data
        |
        v
the recorder stores the relay's synthetic stamps as chunk anchors
        |
        v
replay rebuilds the grid at the declared rate, freezing the staircase in place
        |
        v
Repair sees a 2-sample step, synthesises the missing row, repairs it, and flags
the window "interpolated"
```

The pipeline is fabricating damage in response to a timekeeping artefact, and
then repairing the damage it fabricated.

## 5. Design

### 5.1 Principles

1. **Never fabricate a sample, never discard one.** The grid has exactly one
   slot per sample actually received. A timeline anomaly may change where the
   next sample lands; it may not create or destroy rows.
2. **Time is a measured quantity with an error bar, not a fact.** The
   disagreement between the anchors and the sample count is *reported* per run;
   it is not silently absorbed and not "repaired".
3. **A re-lock is an event, not an accident.** When the disagreement grows past
   a bound, the grid re-anchors — smoothly, and on the record.
4. **One rule, one owner.** The tolerance that decides "on the grid" lives in
   one object and is handed to `Repair`; it is not re-derived in four places.

### 5.2 The one owner: `src/nova2026/streaming/timebase.py`

```python
@dataclass(frozen=True)
class GridPolicy:
    """How a timestamps-damaged source becomes a usable grid."""

    nominal_sfreq: float
    tolerance_samples: float = 0.1        # jitter absorbed; Repair's rule, once
    relock_samples: float = 0.5           # disagreement tolerated before re-lock
    relock_slew_samples: int = 50         # samples a re-lock is spread over
    max_step_samples: float = 1.5         # larger = report as a suspicious jump


@dataclass(frozen=True)
class TimeBaseEvent:
    kind: str                # "relock" | "large_step"
    index: int               # sample index where it happened
    size_samples: float


class TimeBase:
    """Place every received sample on a regular grid at the nominal rate."""

    def __init__(self, policy: GridPolicy, *, anchor: float | None = None) -> None: ...

    def place(
        self, stamps: np.ndarray, n_samples: int
    ) -> tuple[np.ndarray, TimeBaseState]: ...


@dataclass(frozen=True)
class TimeBaseState:
    """What the run has to report about its own timeline."""

    residual_samples: float          # anchor-vs-count disagreement, now
    residual_peak_samples: float     # worst seen this run
    anchor_rate: float               # rate implied by the anchors
    relocks: tuple[TimeBaseEvent, ...]
    large_steps: tuple[TimeBaseEvent, ...]
```

Behaviour:

* **Placement.** Sample `n` lands at `anchor + n / nominal_sfreq`, counted from
  the samples actually received. This is the `GridBuilder` idea, minus the gap
  insertion: the index never advances without a sample behind it.
* **Residual.** After each chunk, compare the chunk's anchor against the grid
  position of its first sample. The difference is the residual, in samples.
  Report it; do not act on it until it crosses `relock_samples`.
* **Re-lock.** When `|residual| > relock_samples`, shift the anchor by the
  residual and spread the correction over enough samples that **no single step
  leaves the tolerance**: `slew = max(relock_slew_samples, ceil(|residual| /
  (tolerance / 2)))`. The floor alone is not enough, and treating it as the whole
  rule was a bug waiting to happen — a 9-sample correction spread over 50 samples
  moves each sample by 0.18, while the consumer's tolerance is 0.1, so the
  re-lock would break the very rule this design exists to satisfy. The event
  records both the residual and the slew actually used.
  No sample is dropped, duplicated or re-dated by more than a fraction of a
  sample; the discontinuity is auditable instead of hidden.
* **Large steps.** A single anchor step beyond `max_step_samples` is recorded as
  a `large_step` event and left to the scoring rules; it is **not** treated as
  loss and **not** repaired.
* **Never a loss verdict.** The time base does not decide that samples were
  lost. If a source really loses samples, that is a property of the source, and
  the honest instruments for it are the run's residual/step report and the
  external CNT comparison — not a counter that cannot distinguish the two cases.

### 5.3 Who stops doing what

| Component | Today | After Phase 1 |
| --- | --- | --- |
| `relay.py` `GridBuilder` | inserts `rint(step)-1` slots per chunk | thin shell over `TimeBase`, or deleted for this path (Phase 2). Its `lost_samples` counter stops claiming loss |
| `recording.py` | stores anchors only | stores anchors plus the `TimeBaseState` in `meta` (rate, peak residual, relocks); a reader can then rebuild the grid the run actually used |
| `Repair` | re-derives `min(2e-4, 0.4/sfreq)` (`live.py:406`) | receives `GridPolicy.tolerance_samples`; the four copies (`live.py`, `relay.py`, `ts_check.py`, `repair.py` default) collapse to one |
| `checks.py` | reports `gaps` | scores `residual_peak_samples` and `relock`/`large_step` counts |

### 5.4 What the operator sees

* `grid_deviation_samples` and the step count move from an incidental field to a
  first-class part of the report and of the verdict.
* `interpolated` windows caused by phantoms disappear; `interpolated` keeps its
  meaning for *real* damage (NaN/Inf runs the source actually sent).
* `unsafe_endpoints` stops firing on re-locks, because no row is synthesised.
* A flag on `scripts/getlive`, `--timebase {grid,stamps}`. It shipped defaulting
  to `stamps` so that the first step changed nothing, and now defaults to `grid` -
  see the status in §10 for why it moved without an amplifier run. `stamps` stays
  as the pass-through for a source whose grid is sound, and
  `--timebase-drift-limit N` turns the drift warning into a verdict (N samples per
  30 s of stream).

## 6. What does not change

* The model contract: windows are still 256 samples at 128 Hz; the nominal rate
  is still what the amplifier declares.
* The recording schema (Phase 1 adds keys to `meta`, no new tables, no format
  break).
* `Repair`'s behaviour on data damage; only where its tolerance comes from
  changes.
* The relay's metadata role (filling in labels/types/units a source never
  declared) — that is a separate, still-needed job.

## 7. Tests

New, in `tests/streaming/`:

1. **The measured staircase.** A fixture built from §3's numbers: ~0 slope with
   a handful of +1/+2 steps. Assertions: exactly one grid slot per received
   sample; the grid is strictly increasing; no sample moves by more than
   `residual / slew`; every step beyond `max_step_samples` appears as an event;
   the peak residual is reported.
2. **A real gap stays visible.** A source that genuinely skips samples must not
   be silently closed: the anomaly must appear in the reported residual/events
   even though no row is fabricated. (This is the test that stops Principle 1
   from becoming "hide everything".)
3. **Re-lock never reorders.** Property test over random staircase patterns:
   the grid stays monotonic, one slot per sample, slewed by at most the bound.
4. **Cross-checks against today's failures.** Every assertion in
   `tests/streaming/test_rig_bringup.py` keeps passing; in particular the relay
   must still regularise the pure jitter fixture.

End-to-end, on the rig: the CNT gate in §8.

## 8. Acceptance gates

| Gate | Today | Target |
| --- | --- | --- |
| `timebase_relocked_samples`, 30 s run | 2 - 9 samples of drift, reported as "gaps" and repaired | absorbed, **not repaired**, and reported as this number |
| `timebase_anchor_rate` | not measured | reported in ppm against the nominal rate |
| `interpolated` windows | 14 - 17 | warm-up only |
| `unsafe_endpoints` fatal faults | present only under the wrong unit assertion, never once the units were right | **none** |
| CNT alignment | r = 1.000000, 0 samples > 1 LSB | unchanged |
| CNT sample count vs recorded count | not asserted | **equal** for the same interval |

The last row is new and is the strongest statement available: it proves no
sample was fabricated or dropped, which the value comparison alone cannot.

`residual_peak_samples` is deliberately **not** a gate. The policy bounds it by
construction (past `relock_samples` the grid corrects itself), so it can never
show how far the source's clock and the grid disagreed over a session. The two
numbers that can are `relocked_samples` - how much drift was absorbed - and
`anchor_rate`, the source clock's own ppm error.

Whether that drift should **reject** a run is the operator's call, and it is now
one: the drift is a warning by default, and `--timebase-drift-limit N` (samples of
drift per 30 s of stream, so the limit travels between run lengths) turns it into a
FAIL. Only the operator knows whether the windows feed a time-locked analysis.

## 9. The experiment this design still needs

§3 cannot distinguish "the source's own stamps step" from "the relay invented
the steps", and the design differs:

* if the source itself steps → `TimeBase` must classify steps (this design);
* if only the relay steps → the relay is the whole cause and Phase 2 (delete it)
  is the fix, with `TimeBase` a safety net.

**Experiment: record 10 s bypassing the relay and measure the same residual.**
Cheap (10 s of cap time), decisive, and it should be run before the code is
written.

**Update: the rig stopped being available, so this experiment cannot be run.** The
contrast it asks for was reproduced against the repository's own fixture publisher
instead - 24 channels, 500 Hz, 0.3 ms of per-sample jitter: `stamps` died in 300
samples with six recoveries, `grid` ran 11 850 samples with none
(`streaming_explained.md` §11.15). That settles what the grid does with a stepping
timeline. It cannot settle whether the amplifier's own steps survive a relay,
because the relay is no longer in the path.

## 10. Migration and rollback

* One commit per step: (a) `timebase.py` + tests, no wiring; (b) wire it into
  `scripts/getlive` behind `--timebase`; (c) move `Repair`/relay onto the shared
  `GridPolicy`; (d) make `grid` the default once the rig passes §8.
* Each step leaves the rig able to run: `(a)` changes nothing at all, `(b)`
  keeps `stamps` as the default, `(c)` is behaviour-preserving because the
  constant is the same one.
* Rollback is a flag flip up to `(d)`, then a revert of one commit.

**Status.** Steps (a), (b) and (c) are implemented. `timebase.py` and its tests
land with no wiring; `scripts/getlive` gains `--timebase {stamps,grid}` (default
`stamps`, so a run that does not ask for the grid is unchanged),
`--timebase-relock-samples` and `--timebase-max-step-samples`; the policy is
recorded in the run's provenance, the time base's verdict in the run counters
(`timebase_relocked_samples`, `timebase_anchor_rate`, `timebase_relocks`,
`timebase_large_steps`), which flow into both the JSON report and the recording's
`meta`. The four copies of the tolerance constant are down to one: `Repair` owns
`grid_tolerance_seconds()` and the time base, the relay and `ts_check` ask it.
The acceptance rules score the timeline as a WARN, `--timebase-drift-limit` turns
that into a FAIL when the operator asks for one, and `main()` is back under the
50-line method limit. Step (d) - flipping the default to `grid` - **is done**, made
after the rig stopped being available and on the evidence in §9's update rather
than on an amplifier run: the only amplifier this project measured fails without
it, grid mode is a near-no-op on a clean source, and it also regularises a
chunk-stamped one. `--timebase stamps` remains as the escape hatch. The experiment
in §9 cannot be run and the relay's grid modes are now the redundant third
implementation of the same idea.

Two details settled during implementation and reflected above: the re-lock slew
is derived rather than floored (§5.2), and the residual peak is not a gate (§8).
A third is placement: the time base runs **before** the recorder sees a block,
because `StreamSession.ingest` writes the raw block on its way past. If it ran
inside `run.stages` instead, the recording would carry the source's untrusted
stamps while the run report claimed a grid.

## 11. Non-goals

* The legacy `scripts/dataproc/streaming` tree (3 731 lines, external dependents
  are riemann's five files plus one import in `tests/streaming/test_getlive.py`).
  Assessed and deferred; it should be frozen with a deprecation note, not
  refactored here.
* The channel/dry-cap policy split across `Repair` and `QualityMonitor`.
* A recording reader in `src/`.
* The exFAT working volume (AppleDouble files, no hard links).

## 12. Open questions for review

1. ~~Run the §9 experiment first?~~ **Moot:** the rig is no longer available, and
   the relay has left the path anyway. The contrast was reproduced against the
   fixture publisher instead (§9's update).
2. ~~Should exceeding the residual bound fail a run, or warn?~~ **Answered, by
   making it the operator's choice.** The drift warns by default and
   `--timebase-drift-limit N` (samples per 30 s of stream) makes it fail, because
   only the operator knows whether the analysis is time-locked. For scale: the
   rig's measured 2-9 samples per 30 s would fail a limit of 1.
3. **`relock_samples = 0.5`** (half a sample, matching `Repair`'s fatal limit) is
   a proposal, not a measurement, and it is exposed as
   `--timebase-relock-samples` so the rig can settle it. `relock_slew_samples`
   is no longer in question: it is a floor, and the length is derived from the
   correction (§5.2).
4. **Does the relay survive Phase 2** as a metadata-only tool, or does the
   library grow a metadata-filling source wrapper and the script disappear?
