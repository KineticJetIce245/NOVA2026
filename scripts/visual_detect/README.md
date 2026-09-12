# Post-stimulus visual detection (COG-BCI PVT)

Predict, from the occipital EEG **100-300 ms after stimulus onset**, whether the
subject registered the stimulus. This is the mirror image of
`scripts/dataproc/cogbci_pvt.py`, which predicts a lapse from the 2 s *before*
the stimulus.

```
raw PVT .set (500 Hz, 62 EEG ch)
        |
        |  build_dataset.py    <- 4-20 Hz, resample, epoch at stimulus onset
        v
datasets/COG-BCI/outputs/PVT_VIS_occipital_m50_500ms_128.0Hz_VisualPipeline.pt
        |
        |  train_occipital.py  <- slice 100-300 ms, baseline, EEGNet, LOSO
        v
results/visual_detect_<scheme>_<tag>_<timestamp>.json
```

## Run it

```powershell
$env:_MNE_FAKE_HOME_DIR = "$PWD\.tmp_mne"   # only needed if ~/.mne is not writable
.venv\Scripts\python.exe -m scripts.visual_detect.build_dataset
.venv\Scripts\python.exe -m scripts.visual_detect.train_occipital
.venv\Scripts\python.exe tests\visual_detect\test_visual_detect.py
```

## The two label schemes

Both live in the same checkpoint, so switching schemes costs no rebuild.

`vr` (default) — *was it registered at all?*
: 6750 epochs aligned to real stimuli (label 1) against 6750 epochs drawn from
  the same session's event-free silence (label 0), with a 700 ms guard around
  every stimulus, response and premature press. Balanced by construction and
  needs no behavioural threshold. **Caveat:** the null epochs are drawn
  uniformly in time while the stimulus is a brief flash, so part of the
  separability is stimulus-locking (onset/offset transient, evoked response)
  rather than a cognitive decision. It answers "is there a stimulus here", not
  "did the subject see it".

`rt` — *was it registered in time?*
: 6694 stimulus epochs labelled by RT `< 500 ms` (the PVT-lapse convention used
  elsewhere in this repo) against slower trials. Only 174 trials are slow, so
  the training script re-weights the loss (`--class-weight`) and undersamples
  negatives (`--neg-cap 20`) and reports macro F1 and mean per-subject AUC
  against a permutation null. This is the closer proxy for "seen", and a much
  harder task.

Continuous log-RT is stored as `labels["rt_ms"]` / `labels["rt_log_z"]` for a
threshold-free analysis.

## Geometry decisions

| Choice | Value | Why |
| --- | --- | --- |
| Channels | `O1 Oz O2 PO7 PO3 POz PO4 PO8` | Visual evoked response sites, caudal enough to be clear of the motor plan for the button press. `--channel-set posterior` adds the parietal-occipital ring as a sensitivity check. |
| Pass-band | 4-20 Hz | Carries the 100-150 ms positivity (+3.7 uV) and attenuates the pre-motor negativity that 1-40 Hz lets through (-5 uV vs -13 uV in 200-250 ms). Below the 60 Hz line frequency, so no notch is needed. |
| Amplitude scaling | none (`--no-normalize`) | AttentivU's divide-by-8-uV per-channel `centre/scale/clip` shrinks the evoked response to ~0.5 uV. Per-trial scaling is applied at training time instead. |
| Sample rate | 128 Hz (`--sample-rate 500` to disable) | Matches the rest of the repo; 26 samples over 100-300 ms. |
| Stored epoch | -50 to +500 ms | Keeps the analysis window and its baseline available at training time, so the window can be changed without rebuilding. |
| Trial pairing | stimulus (13) takes the earliest response (14) after it | A premature press (12) makes its own trial **and** the trial before it ambiguous, so both RTs are dropped (56 trials in the whole dataset). |
| `vr` label | stimulus-present vs. silence | The 56 ambiguous trials keep their `vr` label, because "a stimulus appeared" is true for them; they are excluded only from the `rt` scheme. |

The montage in `chanlocs/get_chanlocs.txt` is not in the 10-20 frame (Cz sits
at `(-92, +7, -5)` mm), so channels are selected by name only and nothing here
depends on electrode coordinates.

## What the data actually supports

Measured on 1340 pooled PVT trials (15 sessions, occipital, 128 Hz):

| Window | Grand-average amplitude | t over trials | Fast − slow |
| --- | --- | --- | --- |
| 50-100 ms | -0.2 uV | -0.7 | -1.1 uV |
| **100-150 ms** | **+3.7 uV** | **+13.6** | +1.8 uV |
| 150-200 ms | +1.1 uV | +4.1 | +0.8 uV |
| 200-250 ms | -4.1 uV | -14.1 | **-6.1 uV** |
| 250-300 ms | -0.4 uV | -1.2 | +2.9 uV |

Three conclusions drove the defaults:

1. **A stimulus-locked visual response exists** and peaks at 100-150 ms
   (+3.7 uV, t=+13.6). It is essentially RT-invariant (+1.8 uV fast−slow), which
   is what a sensory component should look like.
2. **The late part of the 100-300 ms window is contaminated by a pre-motor
   potential.** The 200-250 ms negativity is more than three times larger for
   fast than for slow trials, so it tracks response preparation rather than
   seeing. Keep it in the window (that is the literature convention) but read
   late-window results with that in mind.
3. **Two processing choices were destroying the signal.** AttentivU's
   per-channel `centre/scale/clip` divides by 8 uV and shrank the +3.7 uV
   response to about 0.5 uV, and the 1-40 Hz band lets the pre-motor negativity
   through (it reaches -13 uV there against -5 uV in 4-20 Hz). Hence
   `--band 4 20` and `--no-normalize` as the defaults, and a training-time
   scaling that divides by the window SD **without** centring, because the
   response lives in the window mean.

Single-trial SNR is roughly 3.7/2.8 ~ 1.3, so per-trial accuracy is inherently
limited and the numbers to beat are the ones in the summary JSON, not 0.5.

## Protocols

`--protocol loso` (default) leaves one subject out; `--protocol within` trains on
one subject and holds out one of its sessions, which is the realistic
passive-BCI calibration setting. Every epoch is centred and scaled with
statistics from the training trials only — without that, LOSO sits at chance
because the ~3 uV response is drowned by between-session offsets in microvolts.

```powershell
-m scripts.visual_detect.train_occipital --protocol loso    # 25 folds
-m scripts.visual_detect.train_occipital --protocol within  # 25 x 3 session folds
```

Reported per run: pooled AUC, mean per-subject AUC (also the best-epoch metric),
macro F1, accuracy against the majority-class rate, and a label-permutation
null.

## What has been observed so far

Run on the local COG-BCI copy (25 subjects x 3 sessions, 6750 stimuli):

| Reading | Value |
| --- | --- |
| Grand-average 100-150 ms evoked positivity, occipital, 4-20 Hz | +3.7 uV (t = +13.6 over 1340 trials) |
| Single-trial SD in the same window | ~2.8 uV per channel |
| Linear probe, stimulus vs. silence, **within** one session | 0.88-0.91 AUC |
| Same probe **across** subjects (grouped-by-subject CV) | ~0.51 AUC |
| EEGNet LOSO, `vr` scheme, 30 epochs | 0.514 AUC |
| EEGNet `--protocol within`, 30 epochs | 0.507 AUC |
| EEGNet `rt` scheme, 100-300 ms, LOSO | 0.714 pooled AUC / 0.680 mean subject AUC, but only 174 slow trials in the whole dataset, so several folds have fewer than 10 positives |

Read that as: **the evoked response is real and single-trial decodable by a
linear model, but the current EEGNet configuration does not reach it.** The
likely reason is geometry, not signal: `pool1=2, pool2=4` reduces the 26-sample
window to 3 time steps, so the shape that a linear probe reads directly is
averaged away. Worth trying before concluding anything — `--pool2 0` (keep the
whole time axis), a wider window (`--win-lo 50 --win-hi 500`), and a spatial
filter (xDAWN) in front of the network, which is the standard VEP-detection
front end.

Treat the `rt` numbers with care until the slow class is bigger; and note that
both EEGNet runs above landed at chance, so the honest description of the
current state is "pipeline verified, model not yet tuned to the task".

## Model

`EEGNetWindow` (in `models.py`) is EEGNet with the temporal kernel and pooling
made explicit, because the shared `nova2026.architecture.cnn.EEGNet` hard-codes
a 500 ms kernel and a 32x temporal reduction, which collapses a 26-sample
window to zero width. Defaults: 125 ms kernel, pool `(1, 2)` then `(1, 4)`,
giving 48 features. It should move into `src/nova2026/architecture/` once the
geometry settles.

## Protocol

Leave-one-subject-out over the 25 subjects, 20 % of the remaining subjects held
out for best-epoch selection, EEGNet max-norm projection after every step.
Reported: pooled AUC, mean per-subject AUC, macro F1, accuracy against the
majority-class rate, and a label-permutation null with the prediction rate held
fixed.

**Expect the `rt` task to be hard.** In a single session the fast/slow difference
in the 100-300 ms occipital mean is on the order of 1 uV against a single-trial
SD of 2-3 uV, so a mean per-subject AUC of ~0.55-0.65 would already be a real
effect; the `vr` task should be well above 0.7. Read every number against the
permutation null rather than against 0.5 alone.

## Useful knobs

```powershell
# a different analysis window inside the same stored epoch (no rebuild)
-m scripts.visual_detect.train_occipital --win-lo 50 --win-hi 250

# the wider posterior montage (needs its own build)
-m scripts.visual_detect.build_dataset --channel-set posterior
-m scripts.visual_detect.train_occipital --dataset <that file>

# keep the best-epoch weights of every LOSO fold for later inspection
-m scripts.visual_detect.train_occipital --save-models

# both label schemes at once
-m scripts.visual_detect.train_occipital --scheme vr
-m scripts.visual_detect.train_occipital --scheme rt --neg-cap 20
```
