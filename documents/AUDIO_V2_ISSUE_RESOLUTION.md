# Audio v2 issue resolution - 12 September 2026

Scope: the existing `audio_v2` branch only. Work is in a separate checkout so the
user's `stream` checkout and its uncommitted work are preserved. No other branch
is merged into, committed on, reset, or pushed.

Reference: `NOVA2026_Current_Issues.pdf`, all 22 pages. Document suggestions were
treated as findings to verify, not as instructions overriding the requested scope.

## Upstream comparison and selected implementation

Reviewed [NovaAAD at 8d2f8ef](https://github.com/veiyyu/novaAAD/tree/8d2f8ef), including
`novaAAD3/nova/aad_adapter.py`. Useful ideas retained: channel selection by name,
envelopes evaluated on EEG timestamps, fixed lag-tail exclusion, expiry on the audio
thread, and consecutive-decision dwell. They are integrated into one existing
`nova2026.auditory` package. The three upstream package copies, old acquisition API,
unimplemented adapter runner, weak confidence settings, and accuracy claims were
not imported. No changes were made to the external repository.

Read the current remote `stream` tip, `d364eb9a49afc2d058cf9f1922759181eafffbbd`.
Its streaming files match `b267737`; its additional visual-detection work is outside
this task. Copied the current streaming package, tests, demonstration and guide
into `audio_v2`, including auto-quality, explicit QQ and uptime fixes. Remote-tracking
branch refs were not updated by fetching the exact commit object.

## Issue register

These resolutions apply to the supported audio_v2 implementation. Upstream-only
files are replaced by the named local equivalent, rather than copied and patched.

| Report issue | Resolution on audio_v2 |
|---|---|
| ISS-01, ISS-17 | Confidence is rechecked for every new result; weak evidence clears selection. Default margin 0.5 and three consecutive confident windows. Neutral preserves both talkers. |
| ISS-02 | `scripts.auditory.live` uses StreamSession/Acquire and source timestamps. WAV is paced; zero-decision runs fail. No requested-row accounting. |
| ISS-03, ISS-04 | DAC sample-position mapping, timing logs and 30 ms discontinuity checks. Device playback requires a matching measured timing profile with residual <=30 ms. No claim that lags absorb offsets. Physical loopback measurement remains an external requirement. |
| ISS-05 | All auditory processing uses current streaming primitives; live acquisition uses StreamSession. |
| ISS-06 | `TimestampedAudio.align` implements the envelope provider; runnable `scripts.auditory.live` replaces the upstream TODO bridge. |
| ISS-07 | Evaluation randomizes actual candidate audio columns and matching truth before processing; it does not advertise post-score swapping as blinding. |
| ISS-08 | Reports nonoverlapping window counts, abstention and binomial intervals. Zero output has no forced tie decision. No fixed 0.58 null assertion. |
| ISS-09 | Unsupported upstream calibration/generic accuracy recommendations are explicitly excluded. No unreproduced figure is used as local evidence. |
| ISS-10 | Local commands use .npz. Saving a .npy model is explicitly rejected; paths are not silently renamed. |
| ISS-11 | `train --base` loads through RidgeDecoder.load, checks contracts and preserves base feature coordinates for its prior. |
| ISS-12, ISS-15 | Required labels come from the model; connected labels/count/types/units are checked. Extra columns are selected by name; missing/duplicate labels fail. No 56/57/64 assumption. |
| ISS-13 | Current CircularBuffer emits every completed hop; regression includes 700-row chunks against a 64-row hop. |
| ISS-14 | Full preprocessing contract, actual EEG width, required feature metadata, normalization dimensions and weight shape are checked. No fabricated legacy bundle metadata. |
| ISS-16 | Controller gains requires finite source time. |
| ISS-18 | Live and replay-playback construction fail on incompatible contracts before audio starts. The generic adapter requires a matching contract. |
| ISS-19 | Null tests span seeds and assert abstention; chunk tests include chunks larger than the hop. |
| ISS-20 | One maintained auditory package; upstream duplicate packages remain reference material only. |
| ISS-21 | Default is five seconds. Training exposes --window/--history; live --window must match the stored model. |
| ISS-22 | RidgeDecoder.design excludes unavailable lag-tail rows for all scorers. |
| ISS-23 | Live EEG, envelope timestamps, emission and controller expiry use local LSL time. PortAudio's clock is explicitly mapped via callback timing; monotonic is only used for replay scheduling/timeouts. |
| ISS-24 | Float WAVs above unit peak are normalized after finite-value validation. |
| ISS-25 | Split checks reject any shared stimulus identifier, including one story shared between different pairs. Curated identifiers remain necessary when filenames conceal repeated excerpts. |
| ISS-26 | Current pipeline port plus the D1-D11 fixes below. |

## Audio_v2 defect checklist

- D1: strict automatic resampling with a 3 s age limit, 1 s reserve and explicit
  QQ permission after the causal bandpass. Emission-age regression stays under expiry.
- D2: persistent-fault limit is at least twice the window plus settling, minimum
  15 s. Single-sample and quarter-second artifacts reject windows and recover.
- D3: reject actual channel-width mismatch before broadcasting or arithmetic.
- D4: EEG failure sends an invalid result and lets independent playback finish;
  audio errors take precedence and preserve the processing error as their cause;
  blocked output has a bounded shutdown deadline.
- D5: held-out enforcement is in the replay library, shared with evaluation;
  warm-start development provenance is also retained.
- D6: actual render end is passed into duration metrics; a delayed correct switch
  is not counted as a false selection change.
- D7: stronger confidence plus consecutive evidence; null and zero probes abstain.
- D8: nonpositive/nonfinite chunk durations fail immediately.
- D9: replay continues across the EEG span and emits audio-unavailable diagnostics
  when audio is truncated.
- D10: duplicate evidence timestamps cannot reapply or flip a decision.
- D11: missing feature keys, nonfinite imported arrays, and zero availability
  timestamps are handled explicitly.

The old review's direct test of a legacy `StreamingResampler(quality="LQ")` is
superseded by the supported auditory processor's strict auto-quality test. Explicit
LQ in the legacy module is unchanged; auditory runners no longer use that module.
Tests which recorded the old incorrect behavior are not correctness requirements.

## Confidence finding

The report recommends starting near 0.22 on its longer-window upstream example.
On this branch's five-second windows, ten independent Gaussian-noise probes
produced peak margins from 0.233 to 0.666; clean synthetic scores were near 1.0
against the attended candidate. Thus 0.22 was insufficient here. The conservative
0.5 default plus three consecutive decisions passed the added null regressions.
This is a software guard, not a calibrated probability or a universal physiological
threshold. Choose clinical/research settings on separate development data, never
the held-out trial whose performance is reported.

## Verification and limits

All 251 tests pass on Windows/Python 3.14: 181 current streaming tests, 38 auditory
tests, and 32 legacy streaming tests. This includes real LSL loopback recording.
Auditory regression coverage includes transient faults,
late/duplicate evidence, noise, timing maps, metadata corruption, held-out grouping,
warm starts, duration accounting, worker error precedence and a blocked device.
The live test publishes known EEG through real LSL with timestamps anchored to
the paced audio clock and verifies decoded candidates and evidence age.

The demo and evaluation CLIs completed, producing a new decoder, mixed WAV,
52 controller/fault comparisons and seeded null-control summaries. The clean
synthetic replay reported 23.99225 seconds accounted for, zero wrong-suppression
seconds and 0.6225 coverage; initial warmup/dwell accounts for neutral time.
`git diff --check` passed, and the imported streaming package/tests/demo compare
identically to the current remote tip's versions.

Real amplifier/acoustic timing, microphone input, real KU Leuven/AASD performance,
and human benefit were not measured. Prerecorded two-candidate playback is the
supported live audio source. Hardware playback requires an operator-measured,
calibration-matched profile; writing a JSON profile does not establish that a
physical measurement was performed. Upstream generic/personalized performance
claims remain unverified and are not reported as reproduced.
