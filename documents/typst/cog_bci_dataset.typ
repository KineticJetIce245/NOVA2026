#set page(
  margin: 2.5cm,
  numbering: "1",
)

#let codes(body) = {
  align(center, block(
    inset: 0.5em,
    fill: rgb("#f5f5f5"),
    stroke: 0.5pt,
    radius: 4pt,
    width: 90%,

    align(left, body),
  ))
}

#set par(justify: true)

#align(center)[
  #text(size: 20pt, weight: "bold")[COG-BCI Dataset]
]

= Overview
The *COG-BCI database* (#cite(<hinss_2022_6874129>)) is a multi-session, multi-task EEG dataset designed for research on *passive brain-computer interfaces* (pBCI). It provides simultaneous EEG recordings together with behavioral data across several well-established cognitive tasks, making it suited for studying cognitive workload, sustained attention, and vigilance in ecologically valid experimental settings.

In our local copy, the dataset is organized per subject and session, following a BIDS-like layout:
#codes(
  ```
  datasets/COG-BCI/
  └── sub-01/
      ├── ses-S1/
      │   ├── behavioral/   # trial-level behavioral logs (.mat)
      │   ├── chanlocs/     # channel locations (text)
      │   └── eeg/          # raw EEG (.set / .fdt)
      ├── ses-S2/
      └── ses-S3/
  ```,
)

Each session contains the *same set of tasks*, with one EEGLAB pair (`.set` / `.fdt`) per condition.

= Task battery

#table(
  columns: (1fr, 1.4fr, 1.6fr),
  align: center + horizon,
  table.header([Task], [EEG files], [Measured construct]),
  [N-back (0/1/2-back)], [`zeroBACK` `oneBACK` `twoBACK`], [Working-memory load at three difficulty levels],
  [Flanker], [`Flanker`], [Inhibitory control / response conflict],
  [MATB], [`MATBeasy` `MATBmed` `MATBdiff`], [Multi-attribute task battery — sustained attention (easy / medium / difficult)],
  [PVT], [`PVT`], [Psychomotor vigilance / fatigue],
  [Resting state], [`RS_Beg_EC` `RS_Beg_EO` `RS_End_Ec` `RS_End_EO`], [Eyes-closed / eyes-open baselines at session start and end],
)

The behavioral logs mirror these conditions (`0-Back.mat`, `1-Back.mat`, `2-Back.mat`, `Flanker.mat`, `MATB_Easy.mat`, `MATB_Med.mat`, `MATB_Diff.mat`, `PVT.mat`), which enables joint analysis of neural and behavioral responses.

= Recording setup
The montage (`chanlocs/get_chanlocs.txt`) corresponds to a *64-channel EEG cap* with scalp electrodes named after the 10-10 system (e.g. `Fp1`, `Fz`, `F3`, `POz`, `O1`, `O2`), extended with:
- `ECG1` — an ECG/EOG reference channel.
- `nas`, `lhj`, `rhj` — anatomical landmarks (nasion and left/right pre-auricular points) used for coordinate registration, not signal.

Channel coordinates are provided as 3D head-surface positions in millimeters, which is directly compatible with MNE's forward-modeling and plotting machinery.

= Reading with MNE
Raw EEG is stored in EEGLAB format (`.set` + `.fdt`), readable with MNE via:
#codes(
  ```python
  import mne
  from nova2026.config import DATASET  # points to a .set file

  raw = mne.io.read_raw_eeglab(DATASET, preload=True)
  print(raw.info)         # channels, sampling rate, events
  print(raw.annotations)  # trial markers
  ```,
)

See `src/nova2026/data/mne_reader.py` for the existing example in this repository.

= Bibliography
#bibliography("../../refs.bib", title: "References")
