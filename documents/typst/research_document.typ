#set page(paper: "us-letter")
#set par(justify: true)
#set document(
  title: [Research Document on NOVA Buildathon 2026],
)

#let cover(
  doc,
) = {
  place(
    horizon + center,
    {
      title()
    },
  )

  doc
}

// Cover page
#show: cover
#pagebreak()

// Menu
#outline()
#pagebreak()

= Attention Lapse Detection
== COG-BCI Dataset
The *COG-BCI database* #cite(<hinss_2022_6874129>) is a multi-session, multi-task EEG dataset designed for research on *passive brain-computer interfaces* (pBCI). Each subject performed the same battery of cognitive tasks across three sessions while *63 EEG channels* were recorded at *500 Hz* with a *common-average reference*. For every condition, the dataset also provides trial-level *behavioral logs*, which makes it possible to align neural activity with overt performance.

=== Dataset Structure
The local copy follows a *BIDS-like layout*, organized per subject (`sub-XX`) and session (`ses-S1`--`ses-S3`):
- `eeg/` -- raw EEGLAB files (`.set` header + `.fdt` binary data) for each condition;
- `behavioral/` -- trial-level behavioral logs (`.mat`) mirroring the same conditions;
- `chanlocs/` -- channel names and 3D coordinates.
Each session contains the same task battery: the N-back task (0/1/2-back), the Flanker task, the Multi-Attribute Task Battery (MATB), the Psychomotor Vigilance Task (PVT), and resting-state blocks (eyes open / eyes closed) at the start and end of the session.

=== Structure of Data on Psychomotor Vigilance Task
The *Psychomotor Vigilance Task* (PVT) is a classic *sustained-attention* paradigm. Participants watch a screen and press a button as soon as a stimulus appears. Stimuli are presented at *pseudo-random inter-stimulus intervals* of 2--10 s, so the task cannot be solved by rhythm or anticipation; performance therefore tracks the current level of vigilance.\
The primary outcome is the *reaction time* (RT) on each trial. Slow responses and *lapses* are the hallmark of reduced vigilance, which is observed under sleepiness, fatigue, and *mind wandering*. When the mind wanders, attention is diverted away from the stimulus, so the participant reacts late or not at all -- precisely the lapse signature that the PVT measures. PVT performance is therefore a standard *behavioral proxy* for mind-wandering episodes, and, combined with the simultaneously recorded EEG, it allows searching for the neural correlates of attentional drift.\
The PVT EEG is stored in *EEGLAB format*: a `.set` header file together with a `.fdt` binary file that holds the raw samples. Inspecting the MATLAB struct of `PVT.set` gives the following picture:

```matlab
EEG =
  filename: 'PVT.set'   datfile: 'PVT.fdt'
  nbchan:  63           trials: 1
  pnts:    306060       srate:  500
  xmin:    0            xmax:   612.118
  data:    [63×306060 single]
  ref:     'common'
  event:   [272×1 struct]
```
The various fields have the following meaning:
- `nbchan: 63`, `srate: 500` -- the recording contains *63 EEG channels* sampled at *500 Hz*.
- `trials: 1`, `pnts: 306060`, `xmax: 612.118` -- a *single continuous recording* of 306060 sample points, i.e. roughly *612 seconds (\~10.2 minutes)*, that is *not* segmented into epochs.
- `data: [63×306060 single]` -- the raw amplitude values (in microvolts, single-precision floats) laid out as *channels × time points*.
- `ref: 'common'` -- the signal has already been *average-referenced*.
- `chanlocs: [1×63 struct]` -- the channel locations, matching `chanlocs/get_chanlocs.txt`.
- `event: [272×1 struct]` -- event markers for the 90 PVT trials (stimulus onsets and responses) together with session start/end markers; these events make it possible to *segment the continuous signal into trials*.
- `datfile: 'PVT.fdt'` -- the companion binary file containing the actual samples.

=== Corrupted Behavioral Log
The behavioral log `PVT.mat` describes the same 90 trials at the performance level. In theory, taken together, `PVT.set` / `PVT.fdt` provide the continuous neural signal while `PVT.mat` provides the per-trial behavioral ground truth.
However, an inspection on the `PVT.mat` files using *MD5* shows that the files seem to be *identical across sessions and subjects*, which is unexpected.\
A further inspection of the EEG files shows that `sub-17` and `sub-27`, `sub-25` and `sub-28` have identical EEG recordings for the PVT. Those four subjects are removed from the analysis, leaving 25 subjects with valid data.\
Since we are training for a model that detects lapses from EEG, we need to have the reaction times for each trial in order to label the trials as *lapse* or *non-lapse*. Hence, we need to *reconstruct the behavioral log* from the EEG event markers.\
We use the `mne` package to read the EEG files as `raw` and extract the event markers via `raw.annotations`. The event labeled as `13` indicates the *stimulus onset* and the event labeled as `14` indicates the *response*. The difference between the two timestamps gives the *reaction time* for each trial.\

#pagebreak()
== Training EEGNet Using the COG-BCI Dataset
=== Data Labeling
In order to obtain a clean training set, each PVT trial must be labeled and its EEG segment extracted before being fed to the model.\
We use the `mne` package to read the raw EEG recording as a `Raw` object and extract the event markers through `raw.annotations`. As described above, the annotation labeled `13` marks the *stimulus onset* and the annotation labeled `14` marks the *response*; the difference between the two timestamps gives the *reaction time* of the trial.\
A trial is labeled as a *lapse* when the participant fails to respond in time. A global threshold such as the classic 500 ms does not generalize here: an analysis of the data shows that for some participants all reaction times are below 500 ms. We therefore use a *per-participant, data-driven threshold*: for each participant, the trials whose reaction times fall in the *longest 10%* are labeled as lapses, and all the others as non-lapses.\
Since we want the model to learn the neural pattern preceding an attention lapse, the EEG of interest is the one *right before the stimulus*. For every trial we extract the *2000 ms of signal ending 100 ms before the stimulus onset*; the 100 ms gap ensures the segment does not include the stimulus-evoked response itself.\
Because the extracted segment sits immediately before the stimulus, it must not be contaminated by the *remnant of the previous response or error trial*. A trial is therefore kept only if the previous response/error happened at least *2000 ms + 200 ms* before the current stimulus onset: 2000 ms so that the whole segment is clear of the previous event, plus a *200 ms buffer* so that any residual activity from the last response/error has time to disappear. Trials that do not satisfy this spacing requirement are discarded.

#figure(
  image("images/data_labeling_spacing.svg", width: 90%),
  caption: [
    Spacing criterion for keeping a trial: the previous response/error must occur
    at least *2000 ms + 200 ms* before the current stimulus onset, so that the
    extracted 2000 ms EEG segment is not contaminated by residual activity.
  ],
) <fig:trial-spacing>

=== Model Input
Before being fed to the model, the extracted segments are prepared as input. Depending on the session, the recordings contain *63 or 64 channels*, one of which is an *ECG channel*; we therefore remove that channel, which reduces the data to *62 channels* (63 to 62, 64 to 63). To keep a single consistent input shape across all subjects and sessions, the network is fed exactly these *62 channels*.\
The raw signal, originally recorded at *500 Hz*, is *downsampled to 128 Hz*. For every trial the model receives *2 s of signal* (the 2000 ms segment ending 100 ms before the stimulus onset), which corresponds to *256 time samples* at 128 Hz. The final input to the network is therefore a *62-channel × 256-sample* window.

=== Training EEGNet
The *EEGNet* architecture #cite(<Lawhern_2018>) is a compact convolutional neural network designed for EEG-based brain-computer interfaces. It learns temporal features with regular convolutions and spatial features with *depthwise and separable convolutions*, which makes it well suited to small datasets with limited training data. We *replicated the model in PyTorch*.\
To obtain an unbiased estimate of cross-subject generalization, we use *leave-one-subject-out* (LOSO) cross-validation. There are *25 available subjects* (4 out of the original 29 were removed because their EEG recordings were identical); each fold leaves one subject out, trains the model on the remaining *24 subjects*, and evaluates on the held-out subject.\
Because lapses are the minority class (only the *longest 10%* of reaction times are labeled as lapses), the training set is imbalanced between the positive (lapse) and negative (non-lapse) trials. To deal with this potential imbalance, we train with *focal loss*, which down-weights the well-classified majority trials so that the model focuses on the harder, minority lapse trials.\
Since the classes are imbalanced, accuracy would be a misleading metric. We therefore use the *F1 score* to evaluate the model's performance.

=== Result
- *Multiple Windows*:\
  To increase the amount of training data, we first collected *multiple windows per trial*: *4 windows of 2 s each*, ending at *100 ms, 200 ms, 300 ms and 400 ms before the stimulus* in attempt to increase the number of training samples. This augmentation *did not help* the model -- it actually *decreased* the F1 score: as we reduced the number of windows from 4 to 2 to 1, the F1 score increased.
  #table(
    columns: (1fr, 1fr),
    align: center,
    table.header([*Windows*], [*F1 score*]),
    [4], [$0.578 ± 0.059$],
    [2], [$0.584 ± 0.061$],
    [1], [$0.591 ± 0.059$],
  )
- *Focal Loss*:\
  We limited the *ratio of positive and negative samples* in the training set and adjusted the focal loss accordingly. *Changing the sample ratio produced no improvement*: for a single window, the F1 score stays around *0.59*.

#bibliography("../../refs.bib", title: "References")
