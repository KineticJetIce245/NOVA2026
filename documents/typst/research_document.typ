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

== Training EEGNet Using the COG-BCI Dataset

In order to obtain a clean

#bibliography("../../refs.bib", title: "References")
