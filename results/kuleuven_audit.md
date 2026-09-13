# KU Leuven converted-trial audit and frozen feature contract

Generated: `2026-09-13T01:50:57-04:00` by `python -B -m scripts.auditory.audit_kuleuven`.
Scope: 16 subjects, 320 trials, 15.15 GiB on disk

This is step 4 of `final_connection.md`. It measures what the conversion of
step 2 actually wrote, before step 5 trains anything on it, and it freezes
the contract step 5 must treat as immutable. Nothing here writes into
`datasets/`; the machine-readable record of every trial is the
`results/kuleuven_audit_20260913-015057.json` written in the same run.


## 1. Verdict

| expectation | result | detail |
| --- | --- | --- |
| twenty_trials_per_subject | PASS | trial counts per subject: [20] |
| durations_in_plausible_range | PASS | 4 distinct durations inside [60, 600] s: [123.992188, 388.992188, 390.992188, 398.992188] |
| both_label_values_per_subject | PASS | subjects missing a label value: [] |
| labels_constant_within_trial | PASS | 0 trial(s) with more than one label value [] |
| rep_folds_into_its_base_group | PASS | 192 trial(s) present a rep_ candidate; canonical group keys: ['part1_track1_dry.wav|part1_track2_dry.wav', 'part2_track1_dry.wav|part2_track2_dry.wav', 'part3_track1_dry.wav|part3_track2_dry.wav', 'part4_track1_dry.wav|part4_track2_dry.wav'] |
| unit_exponent_parameterized | PASS | default exponent -6 -> Repair._to_uv 1.0; exponent 0 -> 1000000.0 |

Per-trial invariant checks: **all hold**.

## 2. Per subject

| subject | trials | durations (s) | eeg Hz | audio Hz | labels 0/1 | audio-EEG (s) | groups | stored groups | peak |EEG| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 5852.5 |
| S2 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 1039.0 |
| S3 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 885.0 |
| S4 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 4224.9 |
| S5 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 3706.0 |
| S6 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 771.6 |
| S7 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 2463.0 |
| S8 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 1144.8 |
| S9 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 3037.5 |
| S10 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 5415.1 |
| S11 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 1665.2 |
| S12 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 4359.5 |
| S13 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 2091.2 |
| S14 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 10867.6 |
| S15 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 1157.1 |
| S16 | 20 | [123.992188, 388.992188, 390.992188, 398.992188] | [128.0] | [44100.0] | 16/4 | 1.008 to 72.934 | 4 | 8 | 1516.2 |

`labels 0/1` counts trials whose label is candidate A / candidate B;
`groups` counts the groups after folding `rep_*` into its base, `stored
groups` counts what the converted `group` field actually carries. `peak
|EEG|` is the largest absolute sample among the subject's trials, which is
what identifies the recorded unit: hundreds means microvolts.


## 3. Cross-subject measurements

- Distinct sample rates: `[128.0]`; distinct candidate rates: `[44100.0]`.
- Duration histogram: `123.992` s -> 192, `388.992` s -> 64, `390.992` s -> 32, `398.992` s -> 32
- Trials whose candidates are not strictly longer than the EEG: 0.
- Largest audio-minus-EEG difference: 72.933776 s on `S16/trial_007.npz` (398.992188 s EEG, 471.925964 s audio).
- Smallest audio-minus-EEG difference: 1.007812 s on `S1/trial_008.npz` (123.992188 s EEG, 125.000000 s audio).
- Non-finite values: EEG 0, timestamps 0, candidates 0 trial(s).
- Label balance: 256 trials label A and 64 label B, but over time A 48894.0 s and B 25087.5 s. Always answering A therefore scores 66.1% of the known time, so that -- not 50% -- is the null a window-level accuracy must beat; trial-count balance and time balance must not be confused.

## 4. Group map (the key step 5 splits on)

Merge rule: `rep_partN_trackM_dry.wav` and `partN_trackM_dry.wav` are one
story, so the group key is the sorted pair of candidate stories with the
`rep_` prefix folded away. Both halves of a pair are always presented
together, so a held-out group removes both of that part's stories; the pair
is the finest unit a leakage-free split can hold out.

| group (canonical, S1 example) | trials | trial indices |
| --- | --- | --- |
| part1_track1_dry.wav|part1_track2_dry.wav | 5 | 0, 4, 8, 12, 16 |
| part2_track1_dry.wav|part2_track2_dry.wav | 5 | 1, 5, 9, 13, 17 |
| part3_track1_dry.wav|part3_track2_dry.wav | 5 | 2, 6, 10, 14, 18 |
| part4_track1_dry.wav|part4_track2_dry.wav | 5 | 3, 7, 11, 15, 19 |

Every subject carries the same canonical groups; the per-subject counts and
trial indices are in the machine-readable report. Stories are shared across
subjects, so a held-out story leaves every subject at once.

| candidate story | subjects presenting it | trial appearances |
| --- | --- | --- |
| part1_track1_dry.wav | 16 | 80 |
| part1_track2_dry.wav | 16 | 80 |
| part2_track1_dry.wav | 16 | 80 |
| part2_track2_dry.wav | 16 | 80 |
| part3_track1_dry.wav | 16 | 80 |
| part3_track2_dry.wav | 16 | 80 |
| part4_track1_dry.wav | 16 | 80 |
| part4_track2_dry.wav | 16 | 80 |

**Warning for step 5**: `nova2026.auditory.evaluation.check_split` and
`assert_held_out` compare `trial.group.split("|")` tokens literally. The
converted `group` field still carries the `rep_` prefix, so both guards see
`{part1_track1_dry.wav, part1_track2_dry.wav}` and
`{rep_part1_track1_dry.wav, rep_part1_track2_dry.wav}` as disjoint and will
accept a split that trains and validates on the same 125 s of audio.
Canonicalise every group token with `canonical_story` (or build splits from
`group_key`) before calling either guard.

## 5. Channel subsets

### KU Leuven, 64 channels (baseline comparable with the paper)

Column order is the standard BioSemi64 order recorded in `datasets/AAD-KULeuven/metadata.json`; `channel_names[47] == "Cz"`.

### Shared subset, 20 channels (live contract)

| channel | KU Leuven column |
| --- | --- |
| Fp1 | 0 |
| Fp2 | 33 |
| F7 | 6 |
| F3 | 4 |
| Fz | 37 |
| F4 | 39 |
| F8 | 41 |
| T7 | 14 |
| C3 | 12 |
| C4 | 49 |
| T8 | 51 |
| Cz | 47 |
| P7 | 22 |
| P3 | 20 |
| Pz | 30 |
| P4 | 57 |
| P8 | 59 |
| Oz | 28 |
| O1 | 26 |
| O2 | 63 |

- Names read back at those indices: `['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2']`
- Match against the `metadata.json` order: **PASS**
- Indices unique: True
- Live-only electrodes excluded from decoding: `['F9', 'F10', 'M1', 'M2']`
- Live cap order (24 channels): `['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10', 'M1', 'T7', 'C3', 'C4', 'T8', 'M2', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2']`
- Live cap channel set equals the 20 shared + 4 live-only, without repeats: True

## 6. The `rep_` identity, re-measured

| base file | repeated file | base s | rep s | compared | max abs diff |
| --- | --- | --- | --- | --- | --- |
| part1_track1_dry.wav | rep_part1_track1_dry.wav | 394.000 | 125.000 | 5512500 | 0.0 |
| part1_track2_dry.wav | rep_part1_track2_dry.wav | 394.000 | 125.000 | 5512500 | 0.0 |
| part2_track1_dry.wav | rep_part2_track1_dry.wav | 395.326 | 125.000 | 5512500 | 0.0 |
| part2_track2_dry.wav | rep_part2_track2_dry.wav | 395.326 | 125.000 | 5512500 | 0.0 |
| part3_track1_dry.wav | rep_part3_track1_dry.wav | 394.045 | 125.000 | 5512500 | 0.0 |
| part3_track2_dry.wav | rep_part3_track2_dry.wav | 394.045 | 125.000 | 5512500 | 0.0 |
| part4_track1_dry.wav | rep_part4_track1_dry.wav | 471.926 | 125.000 | 5512500 | 0.0 |
| part4_track2_dry.wav | rep_part4_track2_dry.wav | 471.926 | 125.000 | 5512500 | 0.0 |

A difference of 0 over the compared prefix is what makes the two files one
group. Their SHA256 values differ because the files differ in length, which
is exactly the signal a naive grouping rule mistakes for two stories.


## 7. Frozen feature contract

### 7.1 Audio-envelope contract

- `band`: `[1.0, 9.0]`
- `envelope_method`: `'rectify-causal-lowpass8-20-bandpass3-linear-grid-v1'`
- `lag_seconds`: `0.4`
- `sample_rate`: `64`
- `lag_samples`: `26` (from `lag_seconds`)
- `MIN_MARGIN`: `0.5`

### 7.2 Precomputed envelopes, `datasets/audio/<stem>.npz`

- Arrays: `['envelope', 'metadata', 'timestamps']`; `envelope` [25216, 1] float32, `timestamps` float64
- Metadata keys: `['audio_rate', 'band', 'envelope_method', 'generated_at', 'generator_version', 'sample_rate', 'source_path', 'source_rate', 'source_samples', 'source_sha256']`
- Method `rectify-causal-lowpass8-20-bandpass3-linear-grid-v1`, generator version `1`, 64 Hz, band `[1.0, 9.0]`, source rate 44100

### 7.3 Processing chain (compared against trained models)

- `bandpass`: `[1.0, 9.0]`
- `eeg_channels`: `['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']`
- `filter_order`: `3`
- `input_reference`: `"No reference channel is shipped. RawData.Channels is 1..64 with no reference label, the subject structs carry no montage or electrode-position field, and the distribution contains no channel-location file (the only entries under datasets/AAD-KULeuven are S1.mat..S16.mat, stimuli/, stimuli.zip, preprocess_data.m, README.txt.txt and the Zenodo record JSON). The published preprocess_data.m re-references by subtracting EegData(:,48) from every channel (params.rereference = 'Cz'), which fixes column 48 (1-based) as Cz; the standard BioSemi64 order also places Cz at index 48, so the two agree. That subtraction belongs to the authors' preprocessing and not to the shipped files, so the EegData this loader reads is an unreferenced / unknown-reference recording. The auditory chain therefore carries trial.reference into its contract verbatim and applies its own band-pass (1-9 Hz, order 3, see src/nova2026/auditory/streaming.py) rather than assuming any reference."`
- `input_sfreq`: `np.float64(128.0)`
- `output_sfreq`: `64`
- `resample_quality`: `'QQ'`
- `stage`: `'auditory-current-streaming-v2'`
- `step_seconds`: `1.0`
- `units`: `'uV'`
- `upstream_processing`: `"Shipped by the dataset authors: each trial was high-pass filtered at 0.5 Hz and downsampled to 128 Hz, and FileHeader.SampleRate is 128 in every trial, confirming the shipped rate. The recorded rate is stated inconsistently in README.txt.txt (8196 Hz in the recording description, 8192 Hz in the trial description); the shipped files cannot settle which is correct, so neither figure is relied on. Envelope source: both README.txt.txt and preprocess_data.m compute envelopes from the DRY stimulus, so each trials[i].stimuli name is mapped to its dry counterpart (a trailing _hrtf becomes _dry, any rep_ prefix is preserved, an existing _dry name is left unchanged). Channel naming is a DERIVATION, not data shipped with the dataset: no channel-location file is included, so column j of EegData is named by the standard BioSemi64 order (mne.channels.make_standard_montage('biosemi64').ch_names), which is consistent with the authors' Cz re-reference of column 48."`
- `window_seconds`: `5.0`

`source_unit_exponent` is **not** a contract key: it selects the units of the endpoint safety check and never rescales a sample, so adding it would invalidate every trained decoder. The default for this chain is `-6`, giving `Repair._to_uv = 1.0`.

### 7.4 Unit convention

`source_unit_exponent` is the power of ten of the source unit relative to
volts: `0` = volts, `-6` = microvolts. `Repair` multiplies endpoints by
`10 ** (exponent + 6)` for its saturation and amplitude checks and uses the
result for nothing else.

| field | value |
| --- | --- |
| meaning | power of ten of the source unit relative to volts |
| volts | 0 |
| microvolts | -6 |
| kuleuven_trials | -6 |
| ant_rig | 0 |
| repair_to_uv_at_kuleuven | 1.0 |
| repair_to_uv_at_ant | 1000000.0 |

KU Leuven trials are microvolts, so `-6` is the correct exponent for them and
`_to_uv` is `1.0`: the check reads the recorded values unchanged. Feeding
volts under `-6` would read 0.0833 V as 0.0833 uV and pass a saturated signal
through the check, which is why the live path must declare its own exponent.

### 7.5 Audio truncation rule

Candidate audio is kept up to and including the last EEG sample:

```python
usable = floor((len(eeg) - 1) * audio_rate / sample_rate) + 1
audio = audio[:usable]
```

Equivalently: keep every candidate sample whose time on the trial clock is at
or before `timestamps[-1]`, drop the rest. The candidates are longer than the
EEG in all 320 trials (by 1.008 to
72.934 s), so the rule always drops something, and a
window reaching past the last EEG sample would have neither a label nor an
envelope to align against.

## 8. What step 5 must treat as immutable

1. The `AuditoryConfig` values and the envelope method in 7.1, and the
   envelope record layout in 7.2: a change invalidates every precomputed
   envelope and every trained decoder.
2. Every field of the processing-chain contract in 7.3, including `stage`,
   `units: "uV"`, the band and the filter order. New run policy belongs in
   `stream_config` arguments and never in the contract dict.
3. The 64-channel order and the 20 shared indices in section 5, which are the
   only reason a 64-channel and a 20-channel decoder can be compared at all.
4. The unit convention in 7.4: KU Leuven trials are microvolts and the
   chain's default `source_unit_exponent` is `-6`; the ANT rig's volts are
   `0` and must be declared as such at the call site.
5. The truncation rule in 7.5.
6. The group key in section 4: `rep_*` folds into its base before any
   train/validation/test split is drawn.


## 9. Per-trial detail

Columns: trial file, EEG samples x channels, EEG duration in seconds,
candidate array shape, candidate rate, audio-minus-EEG in seconds, the
distinct label values inside the trial, and the canonical group.

| trial | eeg | duration (s) | audio | audio Hz | audio-EEG (s) | labels | group key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S1/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S1/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S1/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S1/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S1/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S1/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S1/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S1/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S1/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S1/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S1/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S1/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S1/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S1/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S1/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S1/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S1/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S1/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S1/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S2/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S2/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S2/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S2/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S2/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S2/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S2/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S2/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S2/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S2/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S2/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S2/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S2/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S2/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S2/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S2/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S2/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S2/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S2/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S2/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S3/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S3/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S3/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S3/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S3/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S3/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S3/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S3/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S3/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S3/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S3/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S3/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S3/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S3/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S3/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S3/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S3/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S3/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S3/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S3/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S4/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S4/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S4/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S4/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S4/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S4/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S4/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S4/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S4/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S4/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S4/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S4/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S4/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S4/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S4/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S4/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S4/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S4/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S4/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S4/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S5/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S5/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S5/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S5/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S5/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S5/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S5/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S5/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S5/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S5/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S5/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S5/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S5/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S5/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S5/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S5/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S5/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S5/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S5/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S5/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S6/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S6/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S6/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S6/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S6/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S6/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S6/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S6/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S6/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S6/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S6/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S6/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S6/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S6/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S6/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S6/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S6/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S6/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S6/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S6/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S7/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S7/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S7/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S7/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S7/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S7/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S7/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S7/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S7/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S7/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S7/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S7/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S7/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S7/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S7/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S7/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S7/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S7/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S7/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S7/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S8/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S8/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S8/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S8/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S8/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S8/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S8/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S8/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S8/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S8/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S8/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S8/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S8/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S8/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S8/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S8/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S8/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S8/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S8/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S8/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S9/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S9/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S9/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S9/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S9/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S9/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S9/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S9/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S9/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S9/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S9/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S9/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S9/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S9/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S9/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S9/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S9/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S9/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S9/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S9/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S10/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S10/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S10/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S10/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S10/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S10/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S10/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S10/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S10/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S10/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S10/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S10/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S10/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S10/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S10/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S10/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S10/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S10/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S10/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S10/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S11/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S11/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S11/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S11/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S11/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S11/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S11/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S11/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S11/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S11/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S11/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S11/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S11/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S11/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S11/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S11/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S11/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S11/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S11/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S11/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S12/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S12/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S12/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S12/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S12/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S12/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S12/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S12/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S12/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S12/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S12/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S12/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S12/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S12/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S12/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S12/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S12/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S12/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S12/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S12/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S13/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S13/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S13/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S13/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S13/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S13/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S13/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S13/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S13/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S13/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S13/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S13/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S13/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S13/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S13/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S13/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S13/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S13/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S13/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S13/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S14/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S14/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S14/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S14/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S14/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S14/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S14/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S14/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S14/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S14/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S14/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S14/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S14/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S14/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S14/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S14/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S14/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S14/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S14/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S14/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S15/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S15/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S15/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S15/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S15/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S15/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S15/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S15/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S15/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S15/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S15/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S15/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S15/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S15/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S15/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S15/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S15/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S15/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S15/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S15/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S16/trial_000.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S16/trial_001.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S16/trial_002.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S16/trial_003.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S16/trial_004.npz | 49792x64 | 388.992188 | 17375400x2 | 44100 | 5.007812 | [1] | part1_track1_dry.wav|part1_track2_dry.wav |
| S16/trial_005.npz | 50048x64 | 390.992188 | 17433863x2 | 44100 | 4.333504 | [1] | part2_track1_dry.wav|part2_track2_dry.wav |
| S16/trial_006.npz | 49792x64 | 388.992188 | 17377370x2 | 44100 | 5.052484 | [1] | part3_track1_dry.wav|part3_track2_dry.wav |
| S16/trial_007.npz | 51072x64 | 398.992188 | 20811935x2 | 44100 | 72.933776 | [1] | part4_track1_dry.wav|part4_track2_dry.wav |
| S16/trial_008.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S16/trial_009.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S16/trial_010.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S16/trial_011.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S16/trial_012.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S16/trial_013.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S16/trial_014.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S16/trial_015.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |
| S16/trial_016.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part1_track1_dry.wav|part1_track2_dry.wav |
| S16/trial_017.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part2_track1_dry.wav|part2_track2_dry.wav |
| S16/trial_018.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part3_track1_dry.wav|part3_track2_dry.wav |
| S16/trial_019.npz | 15872x64 | 123.992188 | 5512500x2 | 44100 | 1.007812 | [0] | part4_track1_dry.wav|part4_track2_dry.wav |

## 10. Reproduce

```powershell
python -B -m scripts.auditory.audit_kuleuven
```

Machine-readable record: `results/kuleuven_audit_20260913-015057.json`.

