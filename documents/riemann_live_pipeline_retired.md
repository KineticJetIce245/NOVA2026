# Live Riemann pipeline

This uses the existing `nova2026.data.pipeline.Pipeline` without changing `src`.
It prepares checklist steps 8-9 for an external trainer. No model was trained,
no held-out accuracy was measured, and no four-model comparison was run.

## Data and channel order

Set `StreamConfig(expected_channels=(...))` to choose EEG channels in order.
This aliases `eeg_channels`; EOG remains in `eog_channels`. The existing
`ChannelSelectionContract` selects and reorders inlet columns. The pipeline does
not reorder again. Each `EEGWindow` carries channel names and a processing contract.

The live input is `(samples, EEG channels)` in uV. `WindowAdapter` makes a
`(windows, samples, channels)` batch. `CovarianceEstimator` returns
`(windows, channels, channels)`. `TangentSpace` returns
`(windows, channels * (channels + 1) // 2)`. The classifier produces one probability
per saved class. No EEG channel count is hardcoded in these components.

For another model, `WindowAdapter.channels_first(data)` returns
`(windows, channels, samples)`. With `convolution_axis=True`, it returns
`(windows, 1, channels, samples)`. This changes axes, not channel order or units.
An external model may still need its own explicit normalization or dimensions.

## Components your trainer can use

The following is an example for your trainer, not an operation run in this task:

```python
from sklearn.linear_model import LogisticRegression
from scripts.dataproc.riemann.adapter import WindowAdapter
from scripts.dataproc.riemann.covariance import CovarianceEstimator
from scripts.dataproc.riemann.tangent import TangentSpace
from scripts.dataproc.riemann.model import RiemannModel

# config, X_train, y_train and artifact_id belong to your training preparation.
# X_train contains valid live-preprocessed EEG: windows x samples x channels.
adapter = WindowAdapter(
    config.expected_channels, config.sample_count(config.window_seconds)
)
covariance = CovarianceEstimator(ridge=1e-6)
C_train = covariance.transform(adapter.batch(X_train))
tangent = TangentSpace().fit(C_train)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(tangent.transform(C_train), y_train)

model = RiemannModel(
    config,
    tangent,
    classifier,
    label_definition="0 = normal response; 1 = lapse (define the threshold)",
    artifact_id=artifact_id,
    ridge=1e-6,
    training_info={"split": "describe held-out subjects/sessions"},
)
model.save(model_path)
```

Generate windows using `RunProcessor` or `replay_run`, not zero-phase offline
filtering. Exclude rejected windows. Keep overlapping windows and related trials
within the same subject/session split. Fit the tangent reference and classifier
on training data only. Validation/test covariances use `tangent.transform()` only.
Choose the RT-derived class labels and causal prediction time before training.
This classifier predicts classes/probabilities, not reaction time in milliseconds.

The covariance uses per-window Ledoit-Wolf shrinkage plus a relative positive
ridge, including after rank-reducing spatial projection. TangentSpace explicitly
uses a log-Euclidean training reference and a whitened matrix-log tangent map.
It does not estimate an iterative affine-invariant mean. Both fitting and
inference use exactly the same feature definition.

RiemannModel saves only numeric arrays and JSON in an NPZ file. It saves class
order, weights, intercept, reference, ridge, labels, provenance and preprocessing
settings; load() does not fit or load executable pickle objects. The supplied
serializer supports direct sklearn LogisticRegression on tangent features.
Extra scalers or other classifier types need their own explicit pipeline stage
and persistence support. Treat fitted components as fixed after bundling.

## Connect the actual pipeline

```python
from pathlib import Path
from scripts.dataproc.riemann.model import RiemannModel
from scripts.dataproc.riemann.pipeline import RiemannPipeline
from scripts.dataproc.streaming.streamer import Streamer
from scripts.dataproc.streaming.run import RunSpec

model_path = Path("models/your_fitted_riemann.npz")
model = RiemannModel.load(model_path)
pipeline = RiemannPipeline(model)

# Use confirmed source identifiers; keep training processing settings unchanged.
config = model.config.updated(stream_name="CONFIRMED_OUTLET", source_id=None)
run = RunSpec(
    "subject01",
    "session01",
    "trial01",
    "trial",
    model_path=model_path,
    artifact_path=artifact_path,
)


def receive(result):
    prediction, window = result
    print(prediction.to_dict())
    # Optionally persist the result on the same processing/recording thread.
    if streamer.recorder is not None:
        streamer.recorder.event(
            prediction.emitted_at, "prediction", prediction.to_dict()
        )


streamer = Streamer(
    config,
    pipeline=pipeline.rundown,
    on_result=receive,
    pipeline_on_invalid=True,
    run=run,
    record_root=Path("runs"),
)
streamer.initialize()
streamer.stream()
```

`artifact_path` must name the operator required by `model.artifact_id`, or be None
when the model was trained without correction. Do not substitute another operator.
The selected model file should be the same file passed to load(). The run recorder
snapshots selected files. Inference checks the actual window contract and operator
ID; a matching array size alone does not establish compatibility.

`rundown(window)` returns `(Prediction, original_window)`. The four tubes are
input validation, covariance, tangent mapping and prediction. Inherited
`feed(window)` / `step()` also work. Invalid windows pass through these tubes
without computing features or calling the classifier; their probabilities are
None. `pipeline_on_invalid=True` is an explicit opt-in for this behavior.
Other pipelines remain valid-window-only by default.

`on_result` receives the exact pipeline return value; it is not silently unpacked.
Callbacks execute on the processing thread, must return promptly and can forward
results through a bounded queue. Pipeline errors stop the run and are propagated.

## Timing, repair and replay

The live default now linearly repairs at most 20 ms of missing sample rows or
NaN/Inf values, using finite endpoints. It preserves finite values in partly bad
rows. It never estimates a missing timestamp, extrapolates a run boundary, or
repairs saturation/flatlines. Unrepairable data goes to the existing recovery
policy. No valid output crosses an unrepaired discontinuity.

`interpolation_max_seconds=0` disables repair. The maximum permitted setting is
0.1 seconds. A pending tail is bounded by source-sample count; if the source
stops, the runner's no-data timeout applies. Shutdown discards pending repair
state while retaining the original recorded samples. Repair spans touching a
window are counted conservatively; more than `interpolation_max_fraction`
(default 5%) of a window's nominal input rows rejects it. The `interpolated` flag
also covers the configured settling interval after a repair; it is not an exact
infinite IIR impulse-response mask.

`Prediction` includes source start/end, conservative source-batch availability (including interpolation endpoints),
actual emission time from the local LSL clock, segment, validity, reasons,
repair information, model ID, class order and probabilities. Batch availability includes the latest source sample consumed for resampling;
large offline chunks can make it conservative. Actual emission additionally
includes queue/processing delay. Offline emission
time is when replay ran, not the historical delivery time.

Old recordings without interpolation settings replay with repair disabled. Old
spatial operators are not silently relabeled as compatible with the new default:
disable interpolation for their original contract, or create a new calibration
with the desired settings. Do not mix processing contracts when retraining.

`replay_run(directory)` reconstructs delivered windows; feed those to the same
pipeline and retain rejection flags. A trial labeled `completed` can still have
rejected windows and an unprocessed tail. Model evaluation and PlayerLSL
prediction-comparison runs are deliberately deferred.

## Software checks performed

`python -B -m scripts.dataproc.riemann.checks` checks interpolation, explicit
layouts, positive-definite covariance, fixed-reference matrix operations and
Pipeline tube flow with a stub classifier. It does not train/evaluate a model.
Static checks target streaming and riemann only. Existing PlayerLSL recovery
fixtures now inject a burst larger than the interpolation limit; they were not
rerun as model evaluations in this task.
