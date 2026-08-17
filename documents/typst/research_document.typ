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

= Mind Wandering Detection
== Processing Data from COG-BCI Dataset
The *COG-BCI database* #cite(<hinss_2022_6874129>) is a multi-session, multi-task EEG dataset designed for research on *passive brain-computer interfaces* (pBCI). Each subject performed the same battery of cognitive tasks across three sessions while *63 EEG channels* were recorded at *500 Hz* with a *common-average reference*. For every condition, the dataset also provides trial-level *behavioral logs*, which makes it possible to align neural activity with overt performance.

=== Dataset Structure
The local copy follows a BIDS-like layout, organized per subject (`sub-XX`) and session (`ses-S1`--`ses-S3`):
- `eeg/` -- raw EEGLAB files (`.set` header + `.fdt` binary data) for each condition;
- `behavioral/` -- trial-level behavioral logs (`.mat`) mirroring the same conditions;
- `chanlocs/` -- channel names and 3D coordinates.
Each session contains the same task battery: the N-back task (0/1/2-back), the Flanker task, the Multi-Attribute Task Battery (MATB), the Psychomotor Vigilance Task (PVT), and resting-state blocks (eyes open / eyes closed) at the start and end of the session.

=== Psychomotor Vigilance Task
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

The behavioral log `PVT.mat` describes the same 90 trials at the performance level:

#table(
  columns: (auto, auto, 1fr),
  align: (left, left, left),
  table.header([Field], [Value], [Meaning]),
  [`reaction_times`], [`1×90 double`], [Reaction time (s) of each of the 90 trials],
  [`error_trial`], [`90×1 double`], [Binary flag marking erroneous trials (anticipations / misses)],
  [`reaction_time_error_trial`],
  [`1×90 double`],
  [Reaction time for error trials; large negative sentinel values where no valid RT exists],

  [`isi_time`], [`90×1 double`], [Inter-stimulus interval preceding each trial],
  [`simin` / `simax`], [2 / 10], [Design bounds of the ISI in seconds],
  [`ntrials`], [90], [Total number of trials],
  [`triggers`], [`5×2 table`], [Counts of the trigger codes used in the task],
)

Taken together, `PVT.set` / `PVT.fdt` provide the continuous neural signal while `PVT.mat` provides the per-trial behavioral ground truth; aligning the two (via `event` and `isi_time`) yields trials labeled by reaction time, which can be thresholded into attentive vs. mind-wandering episodes.

#bibliography("../../refs.bib", title: "References")
