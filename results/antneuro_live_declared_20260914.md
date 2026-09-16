# ANT live-route run - 20260913-121032

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

- excluded electrodes: ['F8', 'F3']; `Repair` saturation guard 75000 uV (chain default), amplitude guard 20000 uV (explicit --amplitude-limit-uv; the chain's own default stays 500 uV for every caller that declares nothing) - the guards themselves are unchanged for every channel
- measured on the published window: F8 peak 83333.3 uV railed 100.0 %; F3 peak 83333.3 uV railed 90.9 %
- every other electrode in the window peaks below 50% railed, so the guard still judges them: a railed channel that is NOT declared excluded still stops the run
- excluded electrodes keep their columns: excluded electrodes keep their columns, are still repaired and still counted in the bad-channel census; no column is dropped

## The declared amplitude limit, and the measurement it came from

- anchor: first row of the delivered stream (QualityMonitor's own anchor); excursion measured over the 65000 samples of the published window
- at the chain's own default 500 uV, 20 of 20 electrodes fault (C3, C4, Cz, F3, F4, F7, F8, Fp1, Fp2, Fz, O1, O2, Oz, P3, P4, P7, P8, Pz, T7, T8) - the amplitude fault that makes every window an artifact, so no decision is ever committed
- declared for this run: 20000 uV (explicit --amplitude-limit-uv); 2 of 20 electrodes fault at it (F3, F8), and the largest excursion on an electrode it still admits is 19521.5 uV
- what the declared value reaches: QualityMonitor's 'amplitude' fault (which decides whether a window is an artifact) and Repair's endpoint-jump check; one declared value with one owner, so the two verdicts cannot disagree
- what a declared limit does NOT do: it does not make the check selective. The amplitude fault asks only whether an electrode stayed within the declared distance of its own anchor level; a recording whose *normal* movement exceeds the old limit cannot be separated from an artifact by this criterion at any value, so `signal_quality` for this session means 'nothing moved further than N uV from its anchor', not 'this window is artifact-free'. Saturation, flatline and the channel census are untouched and still judge the window.

## Decisions against the session's own marker labels

- status `'stopped'`, decisions {}
- session failure null
- decision census: 0 decided (A 0, B 0), 495 uncertain, 8 unavailable, 0 evidence_gap, of 121 windows (None counted by the session)
- decided windows None of None windows; None of them fell in the operator's unknown band and are excluded
- accuracy on labelled windows None, majority-class rate on the same windows None (this is the null the recording's own balance sets: A 69396 vs B 65050 labelled samples), balanced None
- per-class recall on decided frames: null
- false selection changes: None of None consecutive committed pairs (rate None); the labels themselves switch None times between the same pairs

## Transport diagnostics (the live path's own counters)

- `max_lag` 1.016972 s over 649 blocks / 64900 samples
- `gaps` 0, largest 0.0 s
- source clock as delivered: 500.000025 Hz (nominal 500)
- peak deviation from a perfect grid: 377.2254375 s; re-locks 0, suspicious steps 0
- pre-chain rate conversion: 128.0 (16070 samples); chain resampler startup delay 0.0234375 s (QQ)
- recovery events 0, repaired samples 0, bad-channel census {'F3': 121, 'F8': 121}

## Frontend gate (Node-side evidence, not a browser)

- gain gate open: False at None s; attenuated gain frames 0 of 503
- the vendored frontend's own decoders rejected 0 of 2042 packets

## What this proves, and what it does not

Proves: the recorded ANT session drives the live route - LSL transport, 20-channel pre-flight by name, 500 Hz, microvolts after the import's conversion - and produces decisions, packets and gain frames on this machine, with the transport's own timing counters recorded above. A railed electrode, declared with `--exclude-channels` and measured railed on the published window, cannot stop the run while the endpoint guard keeps its own value for every other channel.

Does not prove: that an amplifier works (no amplifier was present); anything about a live participant (the input is a recording); the audio-to-EEG loopback offset the plan requires before a human study (`residual_offset_seconds`, tolerance +-30 ms, sections 3.11/3.17-4) - there is no audio device in this path, so the only loopback here is LSL delivery; and nothing about the model's accuracy on this rig, because the recording is a different rig with a different reference (CPz against the model's Cz) and two railed electrodes (F8 throughout, F3 for 39.8 % of session 1).
