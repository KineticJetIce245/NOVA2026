# ANT Neuro test recording — reference report

Purpose: every fact below was measured on this machine from the files themselves, so a
later reader (human or model) can rely on it instead of re-deriving it. Nothing here is
inferred from a montage, a datasheet, or a paper unless the line says so. Statements that
come from the operator rather than from the files are marked **operator statement** /
**operator testimony** (§1, §4) and must not be treated as measurements.

Recording date: 2026-09-12. Operator session: `Lacroix_Flo2`.

---

## 1. Where the files are

| Path | What it is |
| --- | --- |
| `tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_19-34-06.cnt` | session 1 EEG (4.8 MB) |
| `tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_19-41-11.cnt` | session 2 EEG (5.1 MB) |
| `tmp/antneurodata/audio/*.evt` | matching event-marker files, one per session |
| `tmp/antneurodata/audio_files/experiment/dichotic_15min.wav` | the stimulus the operator reports as played (**operator statement**, not derivable from the files): measured 48 000 Hz, **2 channels**, 900.00 s (172.8 MB) |
| `…/experiment/left_mono.wav`, `right_mono.wav` | one channel each, 48 000 Hz mono, 900.00 s (86.4 MB each) |
| `…/experiment/left_cut.wav`, `right_cut.wav` | 48 000 Hz mono, 900.00 s |
| `…/experiment/left_raw.wav`, `right_raw.wav` | 48 000 Hz mono, **1130.43 s / 1091.51 s** — source material, longer than the stimulus |
| `…/experiment/check_left.wav`, `check_right.wav` | 48 000 Hz mono, 900.00 s — channel-check copies |

`tmp/` is git-ignored, so none of this is committed.

The playback description — which file was played and the session start offsets in §4 — is
**operator testimony**; those two start offsets cannot be recovered from the recorded files.
Every other number in this report was measured on this machine from the files themselves.

## 2. How to read the EEG

`mne.io.read_raw_cnt` is **the wrong reader** and fails with a misleading message:
`Event table offset from header (1701667150) is larger than file size (4758888)`.
It is a NeuroScan reader; the file is ANT Neuro. MNE says so in the same warning.

The correct reader is `mne.io.read_raw_ant`, which needs the third-party package
`antio`. MNE's own minimum is `antio>=0.5.0` (`mne/io/ant/ant.py`); this repository pins
**`antio>=0.7.1`** in `pyproject.toml` as the optional extra `eeg-ant`, and 0.7.1 is what
was used here (LGPL-3.0, maintained by the MNE core team).

```python
import mne
raw = mne.io.read_raw_ant("tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_19-34-06.cnt",
                          preload=False, verbose="ERROR")
```

`read_raw_ant` also parses the sibling `.evt` file into `raw.annotations`; the marker names
and their timings below came from there, cross-checked against a direct binary parse of the
`.evt` file.

## 3. Measured EEG properties

| Property | Session 1 (`19-34-06`) | Session 2 (`19-41-11`) |
| --- | --- | --- |
| Sampling rate | 500.0 Hz | 500.0 Hz |
| Channels | 24 | 24 |
| Duration | 310.19 s | 328.63 s |
| Units | `raw.get_data()` is in **volts** (MNE convention): peak 0.08333333594 V = 83 333.336 µV. The file itself stores integer samples (int32) with a calibration of **0.0078125 µV per LSB**; `antio` returns µV and MNE multiplies those µV by 1e-6 | same |

Channel names, in file column order (read from `raw.ch_names`, identical in both sessions):

```
Fp1 Fp2 F9 F7 F3 Fz F4 F8 F10 M1 T7 C3 C4 T8 M2 Cz P7 P3 Pz P4 P8 Oz O1 O2
```

**The 0.0833 V peak is a railed channel, not signal.** In session 2 `F8` is at the rail for
**all** 164 317 samples (min = max = mean = 83 333.336 µV, std 0.00); in session 1 it is there
for 155 089 of 155 093 samples (99.997 %), i.e. the whole session, with only 4 samples below
that band (minimum 77 619.922 µV). `F3` is saturated for **39.81 %** of session 1 (61 742
samples) and 0.00 % of session 2. Across the two sessions the per-channel DC mean runs from
−42 298 µV (session 2 `Fp1`) to +83 333 µV (`F8`). Amplitude, variance or band power computed
on those two channels therefore measures a rail, not EEG, and the peak is the same over the
first 5 s and over the whole session because the rail is essentially always on. Do not read
0.0833 V as genuine EEG amplitude.

The reference **is** declared by the files. Both `.cnt` headers carry it: every channel line
of `[Basic Channel Data]` reads `Fp1 1 0.0078125 uV REF:CPz` … `O2 1 0.0078125 uV REF:CPz`;
`mne.io.read_raw_ant` logs `All 24 EEG channels are referenced to CPz.`; and `antio` returns
`ref=CPz` for all 24 channels. `CPz` is **not** one of the 24 recorded electrodes, so the
reference is an implicit one, not a channel in the file. Only the **ground** is unknown: the
header has no `[Ground]` field. Cross-dataset consequence: the KU Leuven AAD dataset is
re-referenced to `Cz` (`datasets/AAD-KULeuven/preprocess_data.m` sets
`params.rereference = 'Cz'` and subtracts column 48), this rig to `CPz`, so a 20-channel
contract shared by the two is re-referenced differently on each side — a real mismatch to
settle before a decoder trained on one is applied to the other. Twenty of the 24 electrode
names also exist in the standard BioSemi64 order used by the KU Leuven AAD dataset; the four
that do not are `F9 F10 M1 M2`.

## 4. Markers and the audio timeline

Event codes, as reported by `read_raw_ant` into `raw.annotations`:

| Code | Name | Meaning used in this experiment |
| --- | --- | --- |
| 1004 | `Start` | session/recording start anchor |
| 1001 | `Left side` | cue: attend the **left** candidate |
| 1006 | `Custom Annotation` | cue: attend the **right** candidate |
| 1007 | `Saying-YES` | not an attention cue |
| — | `impedance` | device impedance event, not a cue |

Marker counts: session 1 has 26 (1× Start, 12× Left side, 12× Custom Annotation, 1 impedance);
session 2 has 28 (2× Start, 12× Left side, 12× Custom Annotation, 1× Saying-YES, 1 impedance).
`Left side` and `Custom Annotation` alternate strictly, 12 each.

Switch intervals between consecutive cues, measured from those onsets:

| Set | n | min | max | mean |
| --- | --- | --- | --- | --- |
| Session 1 | 23 | 8.004 s | 15.348 s | **12.546 s** |
| Session 2 as recorded (all 12 `1006` retained) | 23 | 10.922 s | 16.692 s | **12.947 s** |
| Both sessions, all 12 `1006` retained | 46 | 8.004 s | 16.692 s | **12.746 s** |

An "8–17 s, mean ≈ 12.6 s" summary holds only for session 1; session 2's mean is 12.947 s and
the pooled mean is 12.746 s. Discarding the disputed `1006` (below) would push the pooled mean
to 13.030 s and the pooled maximum to 25.670 s.

**The first cue differs between the two sessions**: session 1's first cue is `1001` (left) at
EEG t = 17.746 s, session 2's first is `1006` (right) at EEG t = 19.790 s. Any timeline
builder that assumes "first cue = left" is wrong for session 2.

**Audio alignment (given by the operator, not inferred):**

- Session 1: playback starts at **0 s** of the stimulus.
- Session 2: playback starts at **4:27 = 267.0 s** of the stimulus.
- Within a session the `Start` marker is the anchor, so
  `stimulus position = start_offset + (EEG time − Start marker time)`.
  Session 1 has `Start` at EEG t = 5.006 s; session 2 at EEG t = 12.838 s.
  Session 2 therefore covers stimulus 279.8–595.3 s. Both sessions together cover about
  595 s of the 900 s stimulus.

**Markers to exclude** (operator-confirmed, with one measured caveat):

| Session | EEG t | Marker | Why |
| --- | --- | --- | --- |
| 2 | 148.354 s | `1007/Saying-YES` | not an attention cue; 2.000 s (1000 samples) long, the only multi-sample event |
| 2 | 326.692 s | second `1004/Start` | accidental press while stopping the recording |

The operator also reported the `1006/Custom Annotation` at 148.402 s — 0.048 s after the
`1007` — as an accidental press. The file-level evidence does not support dropping it, so it
is **retained and flagged ambiguous**:

- Session 2 contains exactly **12** `1006` including that one, i.e. as many as `1001`. Dropping it leaves 12 `1001` but only **11** `1006` — the 12/12 balance is lost.
- Dropping it breaks the strict alternation: `1001@135.554` would be followed by `1001@161.224`, a **25.670 s** left stretch. With the `1006` retained, the two intervals `135.554→148.402` = 12.848 s and `148.402→161.224` = 12.822 s are both mid-range for this recording (10.922–16.692 s).
- The `1007/Saying-YES` event runs 2.000 s = 1000 samples from 148.354 s and therefore **overlaps** the disputed `1006`; its 2 s span is the anomaly, not the cue.

The evidence therefore points to the `1006` at 148.402 s being the real right cue and the
`1007` being the extra event. After excluding only the `1007` and the second `1004/Start`,
session 2 has exactly 12 `1001` and 12 `1006` cues, strictly alternating.

## 5. Which audio file is the reference

Measured by bit-comparison against the two channels of `dichotic_15min.wav`:

| File | Compared with left channel | Compared with right channel |
| --- | --- | --- |
| `left_mono.wav` | **bit-identical** (max abs diff 0) | unrelated (corr ≈ −0.0005) |
| `right_mono.wav` | unrelated | **bit-identical** |
| `check_left.wav` | bit-identical | — |
| `check_right.wav` | — | bit-identical |
| `left_cut.wav` | **not** bit-identical: corr = 0.9999999696 with the left channel, a pure ~1.4791× gain (least-squares slope 1.4791085218, max residual 0.740 counts, i.e. int16 rounding only), max abs diff 10 308 | — |
| `right_cut.wav` | — | bit-identical |
| `left_raw.wav` / `right_raw.wav` | **not played** — longer than the stimulus (1130.43 / 1091.51 s) | — |

The two channels of `dichotic_15min.wav` are essentially uncorrelated (corr ≈ −0.0005),
so this is a genuine two-source dichotic presentation, not a level or delay variant.

**Consequence for analysis:** reference envelopes must be computed from `left_mono.wav` and
`right_mono.wav` (equivalently, from the two channels of `dichotic_15min.wav`). Using
`left_raw.wav` / `right_raw.wav` would align the decoder against material the participant
never heard.

## 6. What this recording does and does not demonstrate

- It **is** a real recording from the rig the live demo will use, with real attention cues,
  so it exercises the acquisition format, the marker path and the channel contract.
- It is a **dichotic (one source per ear)** design. The intended final demonstration plays
  the same mixture to both ears and asks which stream the participant attends, which is a
  harder decoding problem with no ear-of-arrival cue. Accuracy measured on this recording
  therefore does not transfer as a prediction for the same-mixture design.
- No hearing-benefit, comprehension or clinical claim can be made from it.

## 7. Reproduce

```powershell
# EEG reference, properties, every marker onset and the cue-interval statistics, both sessions
.venv\Scripts\python.exe -c "import numpy as np, mne; from antio import read_cnt; fs=['tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_%s.cnt'%t for t in ('19-34-06','19-41-11')]; cs=[read_cnt(f) for f in fs]; print('header channel line:',[[l for l in open(f,'rb').read(1000).decode('latin-1').splitlines() if 'REF:' in l][0] for f in fs], ' [Ground] present:', [b'[Ground]' in open(f,'rb').read(1000) for f in fs]); print('antio unit/ref:',[(sorted({c.get_channel(i,encoding='latin-1')[1] for i in range(c.get_channel_count())}),sorted({c.get_channel(i,encoding='latin-1')[2] for i in range(c.get_channel_count())})) for c in cs]); rs=[(t,mne.io.read_raw_ant(f,preload=False,verbose='ERROR')) for t,f in zip(('19-34-06','19-41-11'),fs)]; [print('===',t,r.info['sfreq'],'Hz',len(r.ch_names),'ch %.3f s'%(r.n_times/r.info['sfreq']),r.ch_names) for t,r in rs]; [print(t,'%9.3f dur=%4.2f %s'%(o,d,s)) for t,r in rs for o,d,s in zip(r.annotations.onset,r.annotations.duration,r.annotations.description)]; [print(t,'cue intervals n=%d min=%.3f max=%.3f mean=%.3f'%(np.diff(x).size,np.diff(x).min(),np.diff(x).max(),np.diff(x).mean())) for t,r in rs for x in [[o for o,s in zip(r.annotations.onset,r.annotations.description) if s[:4] in ('1001','1006')]]]; [print('pooled intervals n=%d min=%.3f max=%.3f mean=%.3f'%(y.size,y.min(),y.max(),y.mean())) for y in [np.concatenate([np.diff([o for o,s in zip(r.annotations.onset,r.annotations.description) if s[:4] in ('1001','1006')]) for t,r in rs])]]"
```

```powershell
# amplitude, DC level and channel saturation, both sessions
.venv\Scripts\python.exe -c "import numpy as np, mne; P='tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_%s.cnt'; D=[(t,(lambda r:(r,r.get_data()*1e6))(mne.io.read_raw_ant(P%t,preload=True,verbose='ERROR'))) for t in ('19-34-06','19-41-11')]; [print(t,'first-5s peak=%.6f uV (%.11g V)  whole-session peak=%.6f uV'%(np.abs(d[:,:2500]).max(),np.abs(d[:,:2500]).max()/1e6,np.abs(d).max()),'| per-channel DC mean %.1f .. %.1f uV'%(d.mean(axis=1).min(),d.mean(axis=1).max()),'| F8 min/max/mean/std=%.3f/%.3f/%.3f/%.3f uV railed=%.2f%%'%(d[c.index('F8')].min(),d[c.index('F8')].max(),d[c.index('F8')].mean(),d[c.index('F8')].std(),100*((d[c.index('F8')]>=0.999*np.abs(d).max()).sum())/d.shape[1]),'| F3 railed=%.2f%%'%(100*((d[c.index('F3')]>=0.999*np.abs(d).max()).sum())/d.shape[1])) for t,(r,d) in D for c in [r.ch_names]]"
```

```powershell
# every audio file against both channels of dichotic_15min.wav, plus the left_cut gain fit
.venv\Scripts\python.exe -c "import numpy as np; from scipy.io import wavfile as w; E='tmp/antneurodata/audio_files/experiment/'; g=lambda n: w.read(E+n)[1].astype(np.int32); r,d=w.read(E+'dichotic_15min.wav'); L=d[:,0].astype(np.int32); R=d[:,1].astype(np.int32); del d; print('dichotic_15min.wav  sr=%d  ch=2  samples/ch=%d  dur=%.2f s'%(r,L.size,L.size/r)); print('corr(L,R) = %.12f'%np.corrcoef(L.astype(np.float64),R.astype(np.float64))[0,1]); [print('%-16s max|f-L|=%7d  max|f-R|=%7d'%(n,np.abs(g(n)-L).max(),np.abs(g(n)-R).max())) for n in ('left_mono.wav','right_mono.wav','check_left.wav','check_right.wav','left_cut.wav','right_cut.wav')]; lc=g('left_cut.wav').astype(np.float64); lf=L.astype(np.float64); s=(lc*lf).sum()/(lf*lf).sum(); print('left_cut: corr=%.12f gain=%.10f max residual=%.3f counts'%(np.corrcoef(lc,lf)[0,1], s, np.abs(lc-s*lf).max())); [print('%-16s samples=%d  dur=%.2f s (not played)'%(n,g(n).size,g(n).size/r)) for n in ('left_raw.wav','right_raw.wav')]"
```
