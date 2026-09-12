# Post-stimulus visual detection (COG-BCI PVT)

A standalone task: predict, from the occipital/posterior EEG **after** stimulus
onset, whether the subject registered the stimulus. It shares nothing with the
pre-stimulus attention pipeline in `scripts/dataproc/` -- different question,
different preprocessing, its own pipeline class, no shared code.

```
raw PVT .set (500 Hz, 62 EEG ch)
        |
        |  build_dataset.py    <- 4-20 Hz, resample, epoch at stimulus onset
        v
datasets/COG-BCI/outputs/PVT_VIS_<channels>_m50_500ms_128.0Hz_VisualPipeline.pt
        |
        +--> decode.py          algorithmic arm: waveform-keeping decoders
        +--> train_occipital.py EEGNet arm, for comparison
        v
results/visual_detect_*.json
```

| File | What it is |
| --- | --- |
| `pipeline.py` | The preprocessing pipeline. Own class, own defaults. |
| `device.py` | Channel sets (8 / 17 / 62), output paths, checkpoint tag. |
| `build_dataset.py` | Reads the PVT recordings, pairs stimulus with response, cuts epochs. |
| `decode.py` | The algorithm arm: decoders that keep the time axis. |
| `block_decode.py` | Does averaging K consecutive trials rescue the decoder? |
| `metric_report.py` | Why the ranking is AUC: what accuracy would hide at other thresholds. |
| `incremental.py` | Does the EEG beat reaction time? (Spoiler: it does not.) |
| `models.py` | `EEGNetWindow`, the network arm, for comparison. |
| `train_occipital.py` | Training and evaluation for the network arm. |

## Run it

```powershell
$env:_MNE_FAKE_HOME_DIR = "$PWD\.tmp_mne"   # only if ~/.mne is not writable
.venv\Scripts\python.exe -m scripts.visual_detect.build_dataset --channel-set occipital
.venv\Scripts\python.exe -m scripts.visual_detect.build_dataset --channel-set posterior
.venv\Scripts\python.exe -m scripts.visual_detect.build_dataset --channel-set all
.venv\Scripts\python.exe -m scripts.visual_detect.decode
.venv\Scripts\python.exe -m scripts.visual_detect.block_decode
.venv\Scripts\python.exe -m scripts.visual_detect.incremental
.venv\Scripts\python.exe tests\visual_detect\test_decode.py
.venv\Scripts\python.exe tests\visual_detect\test_block_decode.py
.venv\Scripts\python.exe tests\visual_detect\test_visual_detect.py
```

## The pipeline

Order: `notch -> band-pass -> resample -> band-pass -> [center/scale/clip]`.

| Setting | Default | Why |
| --- | --- | --- |
| Pass-band | 4-20 Hz | Carries the 100-150 ms evoked positivity (+3.7 uV) and attenuates the pre-motor negativity (-5 uV in this band against -13 uV in 1-40 Hz). |
| Per-channel `center/scale/clip` | **off** (`--normalize` to enable) | It divides by 8 uV, shrinking that same positivity to ~0.5 uV. |
| Sample rate | 128 Hz (`--sample-rate 500` to skip) | 26 samples over 100-300 ms, 57 over 50-500 ms. |
| Stored epoch | -50 to +500 ms | The classification window is chosen at training time, so changing it costs no rebuild. |
| Units | microvolts, float32 | The raw int16 encoding is the precision limit of this copy. |

Channel sets: `occipital` (8), `posterior` (17, the 8 plus the surrounding
parietal-occipital ring), `all` (62, as a control).

Trial pairing: each stimulus (code 13) takes the earliest response (code 14)
after it. A premature press (code 12) makes its own trial **and** the one before
it ambiguous, so both RTs are dropped -- 56 trials in the whole dataset.

## What the data actually supports

Grand average over 1340 pooled trials (15 sessions, occipital, 128 Hz):

| Window | Amplitude | t over trials | Fast − slow |
| --- | --- | --- | --- |
| 50-100 ms | -0.2 uV | -0.7 | -1.1 uV |
| **100-150 ms** | **+3.7 uV** | **+13.6** | +1.8 uV |
| 150-200 ms | +1.1 uV | +4.1 | +0.8 uV |
| 200-250 ms | -4.1 uV | -14.1 | **-6.1 uV** |
| 250-300 ms | -0.4 uV | -1.2 | +2.9 uV |

1. A stimulus-locked visual response exists and peaks at 100-150 ms, and it is
   essentially RT-invariant -- which is what a sensory component looks like.
2. The late part of the 100-300 ms window is **not** visual: the 200-250 ms
   negativity is more than three times larger for fast than for slow trials, so
   it tracks response preparation.
3. Single-trial SNR is roughly 3.7 / 2.8 ~ 1.3. Per-trial accuracy is therefore
   inherently limited; read every number against the `window_mean` control
   below, which can only win through slow level differences.

## Protocol

`decode.py --split cross_session` (default) trains on two of a subject's three
sessions and tests on the third, in both directions. Trials of one recording
share an electrode mount and a slow drift, so splitting a single session at
random lets the model score on structure that cannot transfer; that is what
`--split same_session` measures, and the gap between the two is a result in
itself.

`train_occipital.py` additionally offers `--protocol loso` (leave one subject
out) and `--protocol within` (per subject, session-grouped).

## What has been measured

Mean over 25 subjects of the per-subject AUC. `window_mean` is the control that
sees only the window's average level, not its shape.

| Decoder | 8 ch, 100-300 | 17 ch, 100-300 | 62 ch, 100-300 | 62 ch, 50-500 |
| --- | --- | --- | --- | --- |
| `riemann` (covariance + tangent space) | 0.516 | 0.522 | **0.566** | **0.575** |
| `peak_to_peak` | 0.511 | 0.515 | 0.537 | 0.550 |
| `xdawnd` (xDAWN + template) | 0.512 | 0.513 | 0.513 | 0.516 |
| `template` (matched filter) | 0.510 | 0.510 | 0.510 | 0.511 |
| `linear` (flattened + logistic) | 0.509 | 0.512 | 0.509 | 0.509 |
| `window_mean` (control) | 0.497 | 0.498 | 0.495 | 0.494 |

Same-session split, the optimistic control (8 and 62 channels, 100-300 ms):

| Decoder | 8 ch | 62 ch |
| --- | --- | --- |
| `riemann` | 0.547 | **0.658** (25/25 subjects above 0.5) |
| `linear` | 0.523 | 0.522 |
| `template` | 0.497 | 0.498 |
| `window_mean` (control) | ~0.49 | ~0.49 |

What this says:

* **More electrodes help, and the occipital eight are not enough.** Going
  8 -> 17 -> 62 channels lifts the best decoder from 0.516 to 0.575, and lifts
  the same-session control from 0.547 to 0.658. The response is spatially wider
  than the visual cortex patch, or the extra electrodes simply give the
  covariance estimate more to work with.
* **Spatial covariance beats temporal template.** The one decoder that reliably
  works is the log-Euclidean tangent-space projection of the per-trial
  covariance. Matched filtering with the grand-average waveform -- the textbook
  VEP detector -- does nothing above chance here, which is what an SNR of ~1.3
  predicts.
* **A single session is not enough training data for a transferable decoder.**
  The same-session control scores ~0.1 AUC higher than the cross-session
  protocol, so a large part of any within-session result is session-specific.
  Any number above ~0.6 obtained from a random split inside one session should
  be treated as an upper bound, not as evidence of a usable single-trial
  detector.
* **The `window_mean` control stays under 0.50 everywhere**, including on 62
  channels. Slow level differences between the stimulus and silence epochs are
  therefore not what any of the above scores are reading.
* Window width makes little difference (0.566 -> 0.575 on 62 channels), so
  widening 100-300 to 50-500 is not where the gain is; the extra samples mostly
  add noise.

Honest summary: a small but reproducible single-trial effect exists in the
spatial covariance of the response (mean subject AUC 0.575, 23/25 subjects above
chance, and a same-session ceiling of 0.658). It is far from a usable detector.
The occipital-only hypothesis is not supported by these numbers.

The network arm (`EEGNetWindow` in `models.py`) sits at chance on this task
(0.507-0.514 AUC) under every geometry tried so far, including `--pool2 0`. It
remains as the comparison arm, not as the recommended path.

## Does averaging trials rescue it?

`block_decode.py` asks how the score grows when K consecutive trials are
combined. Two ways to combine, and they behave in opposite directions:

* `--mode score` averages the **decoder scores** of K consecutive same-class
  trials. The per-trial decoder is fitted once on all trials, exactly as in a
  real-time system. The decision window grows to K stimuli (~6 s each, so K=12
  is about a minute).
* `--mode block` averages the **EEG** of K consecutive trials and then fits a
  decoder on the averaged epochs. A block is never a mix of classes or sessions,
  and never spans a pause longer than `--max-gap-s`.

Mean subject AUC, cross-session, 100-300 ms, `riemann`:

| K | 8 ch, score | 62 ch, score | 8 ch, block | 62 ch, block |
| --- | --- | --- | --- | --- |
| 1 | 0.516 | 0.566 | 0.516 | 0.566 |
| 2 | 0.519 | 0.581 | 0.511 | 0.558 |
| 3 | 0.524 | 0.592 | 0.520 | 0.564 |
| 5 | 0.524 | 0.600 | 0.527 | 0.544 |
| 8 | 0.530 | **0.615** | 0.526 | 0.529 |
| 12 | 0.536 | **0.640** | 0.446 | 0.441 |

What this says:

* **Averaging decoder scores works, and keeps working.** On the full 62-channel
  cap it climbs monotonically from 0.566 at one trial to 0.640 at twelve. That
  is the useful result: the per-trial decoder does not have to be good, it has
  to be slightly better than chance, and averaging turns a small edge into a
  usable one.
* **Averaging the EEG does not.** Block mode is flat or worse, and at K=12 it
  collapses to 0.44 -- below chance. The reason is sample size, not signal:
  K=12 turns 13500 training trials into 701, and a covariance-based decoder
  needs many trials to estimate a covariance. Score mode is the right way to
  spend the averaging.
* **The decision window is the price.** K=12 means roughly a minute of stimuli
  per decision (median inter-stimulus interval is ~6 s), so this buys a
  vigilance-level readout, not a per-trial detector. K=5 (~30 s) already gives
  0.600 on 62 channels if a shorter window matters.
* **Electrode count still dominates.** At K=12, 8 channels give 0.536 and 62
  channels give 0.640. The occipital-only restriction costs about 0.1 AUC even
  after averaging.

Practical read: **62 electrodes, `riemann`, score-averaged over 5-12 consecutive
trials, mean subject AUC 0.60-0.64.** That is a usable direction for a
vigilance monitor and still not a single-trial "did he see it" detector.

## Why the scores are AUC, not accuracy

`metric_report.py` scores one decoder's predictions several ways to show what
accuracy would hide. Every number below comes from the *same* predictions; only
the decision threshold changes.

`vr` (balanced, 50/50):

| threshold | accuracy | balanced acc | macro F1 | recall of rare class | predicted-positive rate |
| --- | --- | --- | --- | --- | --- |
| -5.97 | 0.5204 | 0.5204 | 0.4291 | 0.120 | 0.900 |
| -1.33 | 0.5481 | 0.5481 | 0.5403 | 0.418 | 0.630 |
| 0.00 | 0.5456 | 0.5456 | 0.5452 | 0.519 | 0.526 |
| +6.84 | 0.5230 | 0.5230 | 0.4321 | 0.923 | 0.100 |

`rt` (imbalanced, 96.9 % fast). AUC here is 0.7160:

| threshold | accuracy | balanced acc | macro F1 | recall of rare class |
| --- | --- | --- | --- | --- |
| 0.00 | **0.9680** | 0.5108 | 0.5135 | 0.023 |
| +5.01 | 0.8892 | 0.6184 | 0.5488 | 0.329 |
| +6.78 | 0.7185 | **0.6785** | 0.4781 | 0.636 |
| +7.81 | 0.5154 | 0.6268 | 0.3789 | 0.746 |
| +9.69 | **0.1307** | 0.5457 | 0.1266 | 0.988 |

Three things this pins down:

1. **Accuracy spans 0.13 to 0.97 on identical predictions.** Only the threshold
   moved. Reporting one accuracy number without the threshold is reporting an
   arbitrary choice as if it were a property of the model.
2. **On `rt`, the best-looking accuracy is the useless one.** At threshold 0 the
   model scores 0.9680 -- barely below the 0.9687 of a model that always answers
   "fast" -- and it finds 2.3 % of the slow trials. The row that actually works
   (0.6785 balanced accuracy, 64 % of slow trials found) has an accuracy of
   0.7185, which looks far worse.
3. **AUC and average precision do not move with the threshold.** That is why the
   sweeps are ranked by AUC. On `rt` the average precision is 0.101 against a
   0.031 random baseline, a x3.2 lift, which is the honest way to state the same
   fact the 0.716 AUC states.

The `window_mean` control from the decoder sweep is under 0.50 AUC everywhere,
so the ranking ability is not coming from slow level drift.

## Does the EEG add anything beyond reaction time?

`incremental.py` asks the question that decides whether this pipeline is worth
running at all: if a slow reaction time already tells you the subject was
lapsing, what does the EEG add? Three predictors, same held-out session, target
= the **next** block's mean RT (so nothing predicts itself):

| K | `rt_only` rho | `eeg_only` rho | `eeg_plus_rt` rho | change from adding EEG |
| --- | --- | --- | --- | --- |
| 3 | **+0.453** | +0.418 | +0.441 | **-0.012** |
| 5 | **+0.508** | +0.428 | +0.473 | **-0.035** |
| 8 | **+0.515** | +0.481 | +0.497 | **-0.018** |

**Reaction time alone beats every combination that includes the EEG.** Adding
EEG to the behavioural baseline makes the prediction slightly *worse* at every
block size, so on this data the EEG is not just redundant -- it is noise on top
of a better signal.

Two honest qualifications before anyone writes the approach off:

* The comparison is small: 598-1765 block transitions from 25 subjects, and the
  EEG-only correlations (0.42-0.48) are not trivial. The EEG does carry
  information about the upcoming block; reaction time just carries more, and the
  EEG's share is not additive.
* The target is reaction time, which is the strongest vigilance measure there
  is. A different target -- a subjective sleepiness score, or a lapse defined by
  behaviour in a *different* task -- could come out differently.

What would have to be true for the EEG to earn its place: it must help where
reaction time cannot, meaning **before** behaviour degrades, or **without** a
button press at all (a subject who is not responding, a covert-attention task, a
locked-in patient). None of that is demonstrated here, and the runs above are
evidence against the easy version of the claim.

## Useful knobs

```powershell
# decoders, channel sets and windows are all command-line choices
-m scripts.visual_detect.decode --decoders riemann peak_to_peak --channel-sets all
-m scripts.visual_detect.decode --windows 50-500 100-300 100-200
-m scripts.visual_detect.decode --split same_session      # optimistic control
-m scripts.visual_detect.decode --base -200 0             # longer baseline

# averaging: how far does K take it?
-m scripts.visual_detect.block_decode --block-sizes 1 2 3 5 8 12 --mode both
-m scripts.visual_detect.block_decode --mode score --decoders riemann --channel-set all
-m scripts.visual_detect.block_decode --max-gap-s 30      # tolerate longer pauses

# rebuild only if you change the pre-stimulus context or the pipeline settings
-m scripts.visual_detect.build_dataset --channel-set posterior --band 1 40
```
