# Live streaming and artifact calibration

The live implementation keeps the two-thread runner and adds recording,
bounded recovery, and a reusable fixed spatial correction. This round changes only
this streaming folder. The existing offline preprocessing and model code remain
separate.

On `riemann`, bounded interpolation and the existing-architecture inference pipeline
are now available. See `../riemann/README.md` for the exact hookup and trainer
interfaces. Model training and model evaluation remain deferred.

## The earlier five implemented points

1. **Simple components:** ordinary constructors, focused files, project-style
   docstrings, clearer spacing, and Pyright/Ruff checks.
2. **Explicit runs and recording:** subject/session/run identity, four run roles,
   original sample timestamps, task markers, configuration, and selected input
   snapshots with hashes.
3. **Bounded recovery:** discard damaged chunks, restart processing and warm-up,
   and record the decision. Persistent faults still stop the run.
4. **Artifact calibration:** marked EOG events produce a saved EEG spatial
   projector, held-out diagnostics, and a before/after inspection plot.
5. **Consistent application and replay:** the same processor/operator serves
   baseline, labeled training, and trial runs. Disk reconstruction and real
   PlayerLSL checks verify the end-to-end path.

## Run it now

Use the existing project environment from the repository root. No amplifier or
extra configuration is needed for the synthetic PlayerLSL check:

```powershell
.venv/Scripts/python.exe -B -m scripts.dataproc.streaming.replay --duration 12 --units uV
```

This creates a local outlet, connects the real runner, checks received signals,
and shuts down. To use hardware, the sender must already be publishing LSL with
confirmed identifiers, channel labels/types, units, sampling rate, reference,
and upstream processing. Those active amplifier settings are still unknown.

The application entry point stays small:

```python
from scripts.dataproc.streaming.streamer import Streamer

# config is a StreamConfig with the confirmed sender settings.
# consume_window accepts an EEGWindow, with data shaped samples by EEG channels.
streamer = Streamer(config, pipeline=consume_window)
streamer.initialize()
streamer.stream()
```

`stream()` blocks on the calling thread. Use `duration=30`, Ctrl+C, or
`streamer.stop()` from a controller/callback to stop. Call `initialize()` immediately
before each new run. Use `close()` to release a prepared source without streaming.
Pipeline and observer callbacks run synchronously and must return promptly.

For an existing generic `Pipeline`, use `pipeline=my_pipeline.rundown` if its tubes
accept `EEGWindow`. An offline pipeline expecting MNE Raw needs an explicit adapter;
existing trained models have not been validated for this live preprocessing.

## Component boundaries

```text
Acquisition worker: LSL acquire -> copy/reorder -> bounded AcquisitionQueue
Calling thread:    queue.get -> raw recorder -> RunProcessor.feed
                                              -> CircularBuffer.feed/spit
                                              -> optional SpatialOperator
                                              -> valid window -> pipeline
                                              -> every window -> observer
```

The runner creates one acquisition worker. The caller is the second processing
thread; LSL/PlayerLSL have their own internal transport workers. Processing,
recording, and calibration components do not create threads.

| File | Responsibility |
| --- | --- |
| `config.py`, `run.py` | Processing contract and explicit run identity/selections |
| `streams.py`, `channel_selection_contract.py` | Validate the source and map inlet channels to canonical order |
| `acquisition_queue.py` | Copy/reorder, bound the queue, transport original worker errors |
| `voltage_converter.py`, `quality.py` | Convert source units to uV; check raw EEG quality |
| `filters.py`, `streaming_filter.py` | Design coefficients; separately maintain causal filter state |
| `streaming_resampler.py`, `circular_buffer.py` | Resample continuously; retain samples and form windows |
| `buffer_factory.py`, `processor.py` | Assemble processing; recover and apply the optional operator |
| `window.py`, `stats.py` | Explicit window/result containers |
| `recording.py`, `run_reader.py` | Write raw runs; read them without mutation |
| `artifact_calibration.py`, `spatial_operator.py` | Fit artifact directions; save/load/apply the fixed operator |
| `streamer.py`, `recorded_replay.py` | Live orchestration; reconstruction and calibration entry points |

A future thread architecture can call `RunProcessor.feed((data, timestamps))`
directly. Its input is canonical source-unit data; its output is a list of
`EEGWindow` objects. It owns no LSL inlet, file, or thread. The simpler buffer API
is still `circular_buffer.feed(item)` followed by `circular_buffer.spit()`.
Direct buffer use is strict: bounded recovery belongs to `RunProcessor`.

## Record a run

```python
from pathlib import Path
from scripts.dataproc.streaming.run import RunSpec
from scripts.dataproc.streaming.streamer import Streamer

run = RunSpec("subject01", "session01", "baseline01", "baseline")
streamer = Streamer(config, pipeline=consume_window, run=run, record_root=Path("runs"))
streamer.initialize()
streamer.stream(duration=30)
```

This creates `runs/subject01/session01/baseline01/run.sqlite`. Existing run
folders are never overwritten. Supported roles are `artifact_calibration`,
`baseline`, `labeled_training`, and `trial`.

The database commits original chunk boundaries and canonical, selected EEG/EOG
samples **before** scaling, filtering, resampling, or correction. NaN values are
retained for diagnosis. Extra unselected inlet channels are not recorded. Metadata
retains the inlet order, canonical order, source identity, complete configuration,
run status, error, and final statistics.

`RunSpec` optionally takes `artifact_path`, `baseline_path`, and `model_path`.
Choose them explicitly; nothing selects the newest file automatically. The recorder
copies the exact selected files and stores their original paths and SHA-256 hashes.
For a recorded baseline, select its `run.sqlite` file. Baseline/model selection
records provenance; loading a classifier and baseline normalization remain the
responsibility of the downstream pipeline. Only `artifact_path` activates a
processing operation here.

Call `streamer.mark("stimulus")` from the task/controller while recording. An
optional timestamp must use the same local LSL clock (`mne_lsl.lsl.local_clock`)
and represent the original event time. The bounded marker queue is drained on
the processing thread. This API does not automatically consume a separate LSL
marker outlet or PlayerLSL annotations.

Normal stop/duration produces `completed`; an exception or Ctrl+C produces `failed`.
An abrupt process loss can leave `recording` status. Committed chunks remain
inspectable with `RunReader`; `replay_run` requires a completed run. Queued tail
chunks are saved as `not_processed`. Window events preserve the exact delivery
cutoff, including a stop from within a callback. Incomplete windows and the pending
resampler tail are not flushed into inference.

## Calibration and application

Record a run with role `artifact_calibration`, actual EOG channels, and no existing
artifact operator. After initial warm-up, collect at least six separated marked
blinks/eye movements, preferably more. Use `blink`, `eyes_left`, `eyes_right`,
`eyes_up`, or `eyes_down` markers at the observed event center. Space centers at
least one second apart; two or more seconds is easier to review. Allow several
seconds of clean recording before the first event and after the last one.

With the default 500 -> 128 Hz processing, allow roughly six seconds before the
first event to cover startup/resampler buffering. The extractor requires each
one-second epoch to fit entirely in a valid window. Missing, overlapping, warm-up,
and rejected events produce an error rather than a silently reduced event set.

```powershell
.venv/Scripts/python.exe -B -m scripts.dataproc.streaming.recorded_replay runs/subject01/session01/eyes01 --calibrate runs/subject01_eye_operator.npz
```

The output parent directory must exist. The command creates an `.npz` operator,
`.json` report, and `.png` inspection plot without overwriting existing files.

Calibration uses the first events for fitting and the last third (at least two)
for held-out evaluation. EEG/EOG cross-covariance identifies a small spatial
subspace; the fixed matrix is `P = I - U @ U.T`. There must be measurable EOG in
both sets and held-out EEG/EOG coupling must decrease by at least half. With the
single CA-208 EOG channel, this implementation supports one removed direction.
It is an EOG-guided SSP implementation, not ICA or interpolation.

Review the held-out traces, removed spatial weights, and retained energy before
selecting an operator for a human run. Projection also removes neural activity
that shares its spatial direction. Synthetic success cannot establish neural
preservation or generalize to every eye movement, muscle, or motion artifact.

Select the same reviewed operator for subsequent roles:

```python
run = RunSpec(
    "subject01",
    "session01",
    "trial01",
    "trial",
    artifact_path=Path("runs/subject01_eye_operator.npz"),
)
streamer = Streamer(config, pipeline=consume_window, run=run, record_root=Path("runs"))
streamer.initialize()
streamer.stream()
```

Use the identical selection for baseline and labeled-training runs. The operator
checks EEG/EOG order, input/output rate, reference, upstream processing, and filter/
resampler settings. It applies after causal preprocessing, affects EEG only,
and preserves timestamps and pre-correction validity. `artifact_id` prevents a
window from being corrected twice. Projection reduces rank; future covariance
classifiers must regularize accordingly. No training/classification code is changed.

Reconstruct the delivered windows from a completed run:

```python
from scripts.dataproc.streaming.recorded_replay import replay_run

for window in replay_run(Path("runs/subject01/session01/trial01")):
    if window.valid:
        consume_window(window)
```

Reconstruction uses original chunks, recovery decisions recomputed by the same
processor, and the run's hash-verified operator snapshot. `apply_saved_operator=False`
returns uncorrected processing; `operator_path=...` explicitly chooses a compatible
replacement. Recordings themselves are never edited. Keep the same dependency
versions when exact numerical reproducibility matters.

## Timing, units, and recovery

Defaults: causal 60 Hz notch, third-order 1-45 Hz band-pass, SoXR LQ to 128 Hz,
2-second windows, and 0.5-second steps. `EEGWindow.data` is samples by EEG channels
in uV; `eog` is separate (zero columns when absent). It also carries timestamps,
validity/reasons, segment number, per-segment start sample, and artifact ID.

LSL performs clock synchronization without gap-hiding dejittering. Output times
follow `first_input_time + sample_index / output_sfreq` within each segment.
Causal phase delay is not compensated. SoXR buffers across calls; its processing
delay is not added to source timestamps. LQ first output was measured near 1.88 s
at 500 -> 128 Hz; HQ near 7.4 s, exceeding the default lag guard.

Short missing sample runs and isolated NaN/Inf values can now be linearly repaired
with finite endpoints, up to 20 ms by default. Repairs are recorded and marked on
windows. A window is rejected when its repair count exceeds the configured limit
(default 5% of nominal source rows). Unrepairable chunks use bounded recovery.
Each recovery resets filters, resampling, ring storage, quality history, and warm-up.
No window spans an unrepaired segment boundary. Defaults allow
five recovery events, at most a 0.5-second source gap, and five seconds of persistent
bad-window quality before failing. These are configurable prototype limits.

The first two output seconds of each segment are warm-up; all overlapping windows
are rejected. Raw EEG checks flag excursions above 500 uV relative to the initial
level, saturation, and 0.5 seconds of near-flat signal. Bad intervals include a
warm-up-length settling period. EOG does not drive EEG amplitude/flatline checks.
Correction never turns an already-rejected window into a valid one.

Channel/shape errors, large gaps, too many recoveries, persistent quality faults,
queue overflow, stale/future samples, stale windows, no data, and callback failures
stop the run and release its resources. Reconnection is explicit. `max_queue_lag`
reports the greatest observed age of a consumed source chunk against local LSL time.

The cap mapping uses 57 EEG labels shared by CA-208 and COG-BCI, in cap order,
with EOG separate. CA-208 specifies CPz reference and AFz ground. The actual sender
must still be checked; reference/upstream descriptions are operator assertions.
Metadata types/units are validated without rewriting them. Source exponents are
0 (V), -3 (mV), -6 (uV), and -9 (nV).

The 75000 uV saturation default reflects half the manual's lowest 150 mV
peak-to-peak range, not a confirmed gain setting. Confirm hardware gain, labels,
units, rate, reference, timing tolerance, and quality thresholds before hardware
validation. COG-BCI replay retains its native, unconfirmed reference/processing.

## Checks

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s scripts/dataproc/streaming/tests -v
.venv/Scripts/python.exe -B -m unittest scripts.dataproc.streaming.tests.player_checks -v
.venv/Scripts/python.exe -B -m scripts.dataproc.streaming.replay --duration 30 --recording datasets/COG-BCI/sub-01/ses-S3/eeg/RS_Beg_EO.set
```

The deterministic tests exercise recovery, quality, state, contracts, recording,
calibration, and replay. PlayerLSL checks additionally exercise real transport,
shutdown, operator application, a nonfinite source sample, and exact reconstruction.
See `VALIDATION.md` for executed results. The environment includes NumPy, SciPy,
MNE, MNE-LSL, SoXR, and Matplotlib; SQLite is in the Python standard library.

## References

- User-provided CA-208 datasheet, pages 2-3: channel mapping and reference/ground.
- User-provided eego EE-22x manual, pages 6 and 26: acquisition and gain ranges.
- [MNE signal-space projection](https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html)
- [MNE-LSL StreamLSL API](https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.stream.StreamLSL.html)
- [MNE-LSL PlayerLSL API](https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.player.PlayerLSL.html)
- [Python-SoXR streaming API](https://python-soxr.readthedocs.io/en/stable/soxr.html)


## Pipeline results and interpolation compatibility

`Streamer(..., on_result=callback)` forwards the pipeline's return value unchanged.
For `Pipeline.rundown`, the callback receives `(result, final_data)`.
`pipeline_on_invalid=True` is intended for pipelines such as RiemannPipeline that
explicitly produce rejected results without calling a classifier. The default
remains valid-window-only for existing consumers.

Old recordings missing interpolation settings use rejection-only replay.
Operators calibrated without repair require `interpolation_max_seconds=0` or a
new calibration under the desired contract. `window.channel_names` and
`window.contract` make channel and preprocessing compatibility explicit.
