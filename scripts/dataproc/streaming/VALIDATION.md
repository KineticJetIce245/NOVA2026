# Earlier validation: rt_artifact

These measurements precede the riemann/interpolation changes. Current software-only
checks and deferred model evaluation are described in `../riemann/README.md`.

# Live streaming and artifact validation

Executed locally on 7 September 2026 on `rt_artifact`, using the existing project
environment. This round changes only `scripts/dataproc/streaming`. The pre-existing
`pyproject.toml` change and deletion of the old acquisition-queue file were left
as found. No commit or push was made.

Environment: Windows, Python 3.14.3, NumPy 2.5.2, SciPy 1.18.0, MNE 1.12.1,
MNE-LSL 1.14.0, SoXR 1.1.0, Matplotlib, and Python's SQLite library.

## Automated checks

- **32 deterministic tests passed.** Existing channel/unit, timing, resampling,
  quality, queue, and lifecycle checks remain covered. New checks exercise raw
  sample/marker round trips including NaNs; selected-file snapshots and tamper
  detection; missing inputs and run collisions; failed-run metadata; recording
  finalization after pipeline failure; recovery and fresh warm-up; long gaps,
  repeated faults and persistent saturation; marked calibration; operator
  compatibility; double-application rejection; and identical reconstruction for
  baseline, labeled training, and trial roles.
- **5 real PlayerLSL tests passed.** Source interruption, wrong rate, wrong units,
  repeated initialization with callback stop, and recorded spatial correction with
  an actual NaN delivered through LSL. The last test also checks a marker, recovery,
  fresh warm-up, raw fault retention, exact delivery cutoff, and disk reconstruction.
- **Pyright:** zero errors and warnings using the project's Python environment and
  the user's installed Neovim/Mason Pyright.
- **Ruff:** lint and formatting checks pass using the user's installed Ruff.
- All production classes and methods have docstrings. Containers use ordinary
  constructors. Streaming Python files use LF endings.

Commands for the tests and replay are in `README.md`. Static checks target only
this streaming folder. The interruption test intentionally produces a liblsl
transmission/reconnection message before the runner times out and closes.

## Measured PlayerLSL results

| Check | EEG / EOG | Input samples | Output samples | Delivered windows | Valid / rejected | Recoveries | Maximum window age |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12-second synthetic replay, reversed cap order, source uV | 57 / 1 | 6590 | 1526 | 20 | 16 / 4 | 0 | 1.799 s |
| Recorded trial with fitted operator and a NaN, stopped by callback | 2 / 1 | 7140 | 1744 | 19 | 11 / 8 | 1 | 1.797 s |

The clean replay matched independently rechunked offline processing with maximum
EEG difference **0.0 uV**. Window counts, timestamps, units/order, validity gating,
and clean shutdown all passed. Maximum source-chunk age was 0.055 s; maximum
queue size was two chunks.

The recorded trial discarded one damaged chunk, reset state, and delivered valid
windows both before and after recovery. The eight rejected windows cover initial
and post-recovery warm-up. Its maximum source-chunk age was 0.189 s and queue size
was ten chunks. Replaying the SQLite run reproduced every delivered EEG/EOG array,
timestamp, segment, rejection reason, and operator identifier exactly, including
stopping partway through a processing batch. The NaN remained in the raw recording.

Counts include samples buffered during LSL setup and vary with scheduling. Output
counts include generated samples even when a callback stops delivery early. The
pending resampler tail is not flushed. Every real check left no acquisition worker
alive and closed its inlet.

## Synthetic artifact diagnostics

A 32-second known signal contains two EEG channels with neural direction `[1, -1]`,
a shared ocular direction `[1, 1]`, and one measured synthetic EOG channel. It is
passed through the same causal preprocessing and recorded with ten blink markers.
Seven events fit the operator; three are held out. Each epoch has 128 samples.

- One removed spatial direction, approximately `[0.6953, 0.7188]`.
- Held-out EEG/EOG coupling ratio after/before: **0.00001945**.
- Held-out total EEG energy retained: **28.60%**. This signal is deliberately
  artifact-heavy; retained energy is not a measure of neural preservation.
- An independent known-direction check bounds neural-direction distortion below
  2% and residual unit ocular-direction amplitude below 0.02.
- Saved operator/report/plot generation succeeds. The held-out plot was visually
  inspected: correction exposes the known oscillation while preserving separate
  EOG diagnostics and labeled spatial weights.

This validates the implementation on a known synthetic mixture. It does not
validate an operator for a person. Projection also removes neural activity sharing
an artifact direction, and the single EOG channel limits the fitted subspace to
one direction in this implementation.

## Earlier prototype checks

Before this round, the `rt_prototype` implementation passed 60 seconds of synthetic
PlayerLSL replay and 30 seconds of the existing COG-BCI
`datasets/COG-BCI/sub-01/ses-S3/eeg/RS_Beg_EO.set` recording. Those earlier results
were 116 windows (112 valid) and 58 windows (39 valid), respectively, with exact
reconstruction and clean shutdown. They are historical results, not new hardware
or human-artifact validation on this branch.

The existing deterministic signal checks still exercise SoXR passband and alias
rejection. Earlier measurements found roughly 1.88 seconds to the first LQ output
batch at 500 -> 128 Hz, versus 7.4 seconds for HQ. This explains the LQ default and
why recovery tests must allow enough clean data to refill the resampler and window.

## Remaining external verification

No amplifier was connected and no human EOG calibration was available. Actual
outlet identifiers, channel metadata, units, reference, rate, upstream processing,
and selected gain still need confirmation. Thresholds, timing tolerance, warm-up,
and the learned operator need review on the intended hardware and participant.

Baseline/model files are explicitly snapshotted for provenance; their downstream
normalization/classification is not implemented here. Existing offline models
were not modified or declared compatible. The operator must be used consistently
when collecting baseline/training/trial data, and rank reduction must be handled
by future covariance classifiers. No automatic source reconnection is added.
