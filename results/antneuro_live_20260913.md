# ANT live-route run - 20260913-110547

Session `antneuro_19-34-06` (155093 samples, 310.184 s, ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2'] channels) published on LSL at 500 Hz; the live pre-flight accepted it and the session ran for 130 s at 1x.

## What accepted the source, and what it had to reject

- declared by the outlet: 20 channels, 500 Hz, units ['microvolts'], type 'EEG'
- accepted: 20 electrodes by name, exponent -6 (microvolts), no column dropped

| counter-check | refused with |
| --- | --- |
| volts-instead-of-microvolts | Source voltage units for Fp1 do not match the configured exponent (0). |
| wrong-rate | The source sampling rate (500 Hz) does not match the configured rate (1000 Hz). |
| wrong-channel-name | Source is missing required channels: ['XFp1', 'XFp2', 'XF7', 'XF3', 'XFz', 'XF4', 'XF8', 'XT7', 'XC3', 'XC4', 'XT8', 'XCz', 'XP7', 'XP3', 'XPz', 'XP4', 'XP8', 'XOz', 'XO1', 'XO2'] (source has ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'Cz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz', 'O1', 'O2')). |

## Transport diagnostics (the live path's own counters)

- `max_lag` 1.089082 s over 137 blocks / 13700 samples
- `gaps` 0, largest 0.0 s
- source clock as delivered: 500.000125 Hz (nominal 500)
- peak deviation from a perfect grid: 0.0 s; re-locks 0, suspicious steps 0
- pre-chain rate conversion: 128.0 (3382 samples); chain resampler startup delay 0.014 s (QQ)
- recovery events 1, repaired samples 7608, bad-channel census {'F3': 9, 'F8': 9, 'Cz': 5, 'F4': 4}

## Decisions against the session's own marker labels

- status `'error'`, decisions {}
- session failure null
- decided windows None of None windows; None of them fell in the operator's unknown band and are excluded
- accuracy on labelled windows None, majority-class rate on the same windows None, balanced None

## Frontend gate (Node-side evidence, not a browser)

- gain gate open: False at None s; attenuated gain frames 0 of 80
- the vendored frontend's own decoders rejected 0 of 352 packets

## What this proves, and what it does not

Proves: the recorded ANT session drives the live route - LSL transport, 20-channel pre-flight by name, 500 Hz, microvolts after the import's conversion - and produces decisions, packets and gain frames on this machine, with the transport's own timing counters recorded above.

Does not prove: that an amplifier works (no amplifier was present); anything about a live participant (the input is a recording); the audio-to-EEG loopback offset the plan requires before a human study (`residual_offset_seconds`, tolerance +-30 ms, sections 3.11/3.17-4) - there is no audio device in this path, so the only loopback here is LSL delivery; and nothing about the model's accuracy on this rig, because the recording is a different rig with a different reference (CPz against the model's Cz) and two railed electrodes (F8 throughout, F3 for 39.8 % of session 1).
