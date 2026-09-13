# ANT live-route run - 20260913-112656

Session `antneuro_19-34-06` (155093 samples, 310.184 s, ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2'] channels) published on LSL at 500 Hz; the live pre-flight accepted it and the session ran for 130 s at 1x.

## What accepted the source, and what it had to reject

- declared by the outlet: 20 channels, 500 Hz, units ['microvolts'], type 'EEG'
- accepted: 20 electrodes by name, exponent -6 (microvolts), no column dropped

| counter-check | refused with |
| --- | --- |
| volts-instead-of-microvolts | Source voltage units for Fp1 do not match the configured exponent (0). |
| wrong-rate | The source sampling rate (500 Hz) does not match the configured rate (1000 Hz). |
| wrong-channel-name | Source is missing required channels: ['XFp1', 'XFp2', 'XF7', 'XF3', 'XFz', 'XF4', 'XF8', 'XT7', 'XC3', 'XC4', 'XT8', 'XCz', 'XP7', 'XP3', 'XPz', 'XP4', 'XP8', 'XOz', 'XO1', 'XO2'] (source has ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2')). |

## Channel policy: a declared railed electrode, not a wider limit

- excluded electrodes: ['F8', 'F3']; `Repair` saturation guard 75000 uV (chain default), amplitude guard 500 uV - the guards themselves are unchanged for every channel
- measured on the published window: F8 peak 83333.3 uV railed 100.0 %; F3 peak 83333.3 uV railed 90.9 %
- every other electrode in the window peaks below 50% railed, so the guard still judges them: a railed channel that is NOT declared excluded still stops the run
- excluded electrodes keep their columns: excluded electrodes keep their columns, are still repaired and still counted in the bad-channel census; no column is dropped

## Decisions against the session's own marker labels

- status `'stopped'`, decisions {}
- session failure null
- decision census: 0 decided (A 0, B 0), 495 uncertain, 8 unavailable, 0 evidence_gap, of 121 windows (None counted by the session)
- decided windows None of None windows; None of them fell in the operator's unknown band and are excluded
- accuracy on labelled windows None, majority-class rate on the same windows None (this is the null the recording's own balance sets: A 69396 vs B 65050 labelled samples), balanced None
- per-class recall on decided frames: null
- false selection changes: None of None consecutive committed pairs (rate None); the labels themselves switch None times between the same pairs

## Transport diagnostics (the live path's own counters)

- `max_lag` 1.147318 s over 649 blocks / 64900 samples
- `gaps` 0, largest 0.0 s
- source clock as delivered: 499.999999 Hz (nominal 500)
- peak deviation from a perfect grid: 377.2254375 s; re-locks 0, suspicious steps 0
- pre-chain rate conversion: 128.0 (16070 samples); chain resampler startup delay 0.0234375 s (QQ)
- recovery events 0, repaired samples 0, bad-channel census {'C3': 121, 'C4': 121, 'Cz': 121, 'F3': 121, 'F7': 121, 'F8': 121, 'Fp1': 121, 'Fp2': 121, 'Fz': 121, 'O1': 121, 'O2': 121, 'P3': 121, 'P4': 115, 'P7': 121, 'P8': 121, 'Pz': 121, 'T7': 121, 'T8': 121, 'F4': 112, 'Oz': 66}

## Frontend gate (Node-side evidence, not a browser)

- gain gate open: False at None s; attenuated gain frames 0 of 503
- the vendored frontend's own decoders rejected 0 of 2044 packets

## What this proves, and what it does not

Proves: the recorded ANT session drives the live route - LSL transport, 20-channel pre-flight by name, 500 Hz, microvolts after the import's conversion - and produces decisions, packets and gain frames on this machine, with the transport's own timing counters recorded above. A railed electrode, declared with `--exclude-channels` and measured railed on the published window, cannot stop the run while the endpoint guard keeps its own value for every other channel.

Does not prove: that an amplifier works (no amplifier was present); anything about a live participant (the input is a recording); the audio-to-EEG loopback offset the plan requires before a human study (`residual_offset_seconds`, tolerance +-30 ms, sections 3.11/3.17-4) - there is no audio device in this path, so the only loopback here is LSL delivery; and nothing about the model's accuracy on this rig, because the recording is a different rig with a different reference (CPz against the model's Cz) and two railed electrodes (F8 throughout, F3 for 39.8 % of session 1).

## Reading this record: what stopped the run, and what the exclusions did

`Repair`'s endpoint guard is per-channel and was never the whole obstacle. Two
separate faults had to be cleared to get a full session out of this recording:

1. **the pre-chain rate declaration.** The 500 -> 128 Hz adapter changed the
   data the chain saw without moving `input_sfreq` with it, so `Repair` read one
   128 Hz step as 3.9 samples at 500 Hz, synthesised three missing rows per
   sample and rejected every window as `interpolated` (7608 phantom repairs, one
   recovery, `EEG quality faults persisted beyond the allowed duration (15s)`
   at 26.4 s). Declaring the fed rate is what cleared the session: it now runs
   the whole 130 s with 0 repairs, 0 recoveries and 0 invalid windows.
2. **the railed electrodes.** F8 (100 % of the published window) and F3 (90.9 %)
   are declared with `--exclude-channels` and measured railed before the session
   exists. `Repair`'s saturation guard stays at the chain's own 75 000 uV and its
   amplitude guard at 500 uV for all twenty channels; the excluded pair keeps
   its columns, is still repaired and is still counted (both appear in the
   census, 121 of 121 windows).

## V4 (ANT side): why there is no accuracy number here

Every one of the 121 windows was rejected as an artifact, so the controller had
no admissible evidence and never committed: 495 `uncertain` and 8 `unavailable`
frames, 0 decided. The mechanism is measured, not guessed: the quality monitor
faults 13 of 20 electrodes for `amplitude` on this stretch, because the
recording's own level drifts far past the 500 uV excursion limit - median
excursion from the anchor level is 1.5 mV on Fp1, 2.0 mV on F4, and 41.5 mV peak
on the kept montage. `_verdict` treats any faulted channel as an artifact, so
`signal_quality` is 0 and no frame can commit.

That is the honest result: **V4's ANT side remains unachieved**, for a reason
that is about this recording and the published criterion, not about a threshold
that could be nudged. The label balance the null comes from is stated in the
record (A 69 396 vs B 65 050 labelled samples, 51.6 % A), and the model is
trained on KU Leuven data at 128 Hz with a Cz reference while this rig is
CPz-referenced, so even a committing run would be expected to score near that
null.
