# Where the auditory-attention demo stands

A plain-language account of what we built, what it actually does today, what it
cannot do yet, and the realistic ways forward. Written 2026-09-14 on branch
`integration`.

Every number below was measured in this project. Where a number comes from a paper
rather than our own runs, the sentence says so.

---

## 1. The short version

**The pipeline runs. You can watch it work.** One command plays a recording of
someone listening to two voices, feeds their EEG through the real processing chain,
decides second by second which voice they are attending to, shows that decision in
the browser interface, and lowers the volume of the other voice. On our data it
opens the volume control at 9 seconds and holds a decision on 284 of 496 volume
frames.

**The accuracy is weak.** On 5-second windows the decoder is right about 62 % of
the time when it commits to an answer; the best possible score in the calibration
we ran was 69 %. On 60-second windows the same kind of decoder reaches about 88 %,
but we have not trained a 60-second model yet. A person would not want to wear this
for an hour in its current state.

**Nothing has been tested on a live person.** No amplifier has been connected. All
results come from replaying recordings.

**And on the user's own recorded session, the system currently produces no
decisions at all.** The data flows through correctly — right electrodes, right
sample rate, right units, no dropped packets — but the decoder refuses to score any
window, because that recording was made with a different reference electrode (CPz)
than the one the decoder was trained with (Cz). Section 5 explains why that matters
and what it would take to fix.

**The end goal** is a person wearing the headset, listening to two things, and the
system making the thing they pay attention to louder while the other one fades.
Section 7 lays out what that needs.

---

## 2. What the system is, in plain terms

A person wearing EEG electrodes listens to two audio streams at once. The system
tries to decide, continuously, which of the two they are listening to.

It works on one simple observation: **when you listen to speech, your brain's
electrical activity follows the rhythm of that speech.** The loudness of speech goes
up and down a few times per second, and the EEG signal wobbles in step with it,
slightly delayed. If we extract the "loudness over time" curve from each of the two
audio streams, and compare each curve against the EEG, the curve belonging to the
stream the person is actually attending to matches a little better.

So the whole method is: build a comparison score for stream A, build one for stream
B, and whichever score is higher is our guess. That is all a "decoder" is here — a
learned weighting of the EEG channels and their recent history, chosen so that its
output correlates with speech loudness.

Three details matter and all three bit us at some point:

1. **The comparison must line up in time.** If the audio and the EEG are offset by
   even 100 milliseconds, the scores degrade badly. Section 4.3 quantifies this.
2. **The decoder is trained on a particular setup.** It learns spatial weights for
   specific electrode positions and a specific reference. Move it to a different
   cap or a different reference and the learned weights no longer mean the same
   thing.
3. **The scores are not probabilities.** A score of 0.22 versus −0.41 means "A fits
   better than B", nothing more. We therefore require a minimum gap between the two
   scores before we let the system act on a decision. That gap is a tunable number,
   and it turned out to matter enormously (Section 3.3).

---

## 3. What we measured

### 3.1 Decoder accuracy

Trained and tested on the KU Leuven public dataset (16 people, 20 recordings each,
64 electrodes). "Balanced" means the average of the hit rate on each of the two
answers — it is the honest measure here, because in this dataset 66 % of the time
the correct answer is the same one, so a system that always says that answer scores
66 % while knowing nothing.

| Test setup | Raw accuracy | Balanced accuracy |
| --- | --- | --- |
| 64 electrodes, stories held out | 0.617 | **0.620** |
| 64 electrodes, people held out (16 folds) | 0.609 | **0.612** |
| 20 electrodes (the subset the real headset has), stories held out | 0.594 | **0.597** |
| 20 electrodes, people held out | 0.586 | **0.589** |

Two things to read from this table. First, **raw accuracy is below the 66 % that
comes free** — so any report quoting raw accuracy is misleading, including any we
produced before we noticed. Second, **using only the 20 electrodes the real headset
has costs about 2 points of balanced accuracy.** That is the measured price of
moving from the dataset's cap to the actual one.

### 3.2 Longer windows are much better

Same data, same electrodes, changing only how much EEG the decoder looks at before
deciding:

| Window | 5 s | 10 s | 30 s | 60 s |
| --- | --- | --- | --- | --- |
| Balanced accuracy (64 electrodes) | 0.637 | 0.698 | 0.796 | **0.875** |

This is the single largest available improvement, and it costs only latency: a
60-second window means the first decision arrives about a minute after playback
starts. We do not have a 60-second model — the decoder is tied to its window length
at training time, so one would have to be trained. We have the data and the training
code; it is a matter of running it.

### 3.3 How strict to be before acting

The system can stay quiet when the two scores are close. How close is "too close" is
a free choice, and it trades coverage against accuracy:

| Minimum score gap | Frames where it speaks up | Accuracy when speaking | Balanced accuracy when speaking |
| --- | --- | --- | --- |
| 0.05 | 52 % | 0.684 | 0.688 |
| 0.15 | 21 % | 0.757 | 0.761 |
| 0.50 (our original setting) | 0.15 % | — | — |

The 0.50 setting was never calibrated; it was a conservative guess, and it meant the
system stayed silent 99.85 % of the time. On the demo trial this showed up as only
12 of 496 volume frames doing anything. Changing it to 0.05 took that to 284 frames
and moved the first decision from 40 seconds to 8.6 seconds. The underlying accuracy
did not change at all — only how often the system is willing to act.

### 3.4 Timing is the most expensive thing to get wrong

We shifted the audio deliberately against the EEG and measured what it costs:

| Misalignment | Cost in balanced accuracy |
| --- | --- |
| 100 ms | 0.119 (audio early) / 0.150 (audio late) |

For comparison, **replacing all 44 extra electrodes with the 20 the headset has
costs 0.022.** Being 100 ms out of step is five to seven times more damaging than
throwing away two thirds of the electrodes. The decoder's tolerance is roughly
±100 ms; beyond about ±250 ms it is guessing.

One measured component of that budget: the resampling inside our processing chain
introduces a fixed 14 ms startup delay, and nothing in either the replay path or the
live path currently compensates for it.

### 3.5 Deliberate fault testing

We ran 23 scenarios: one clean, fifteen with something broken (EEG stream stops,
data arrives late, timestamps jump, the same data arrives twice, a channel dies,
the connection drops, the browser reconnects, the audio clock drifts, malformed
network messages, a missing precomputed file, mismatched audio lengths), and seven
adversarial inputs (duplicate blocks, non-monotonic timestamps, wrong sample rate,
wrong channel count, wrong units, huge gaps, saturated signals).

Results: **nothing crashed, every failure produced a visible symptom rather than a
silent wrong answer, and no scenario contaminated another.** Five of the scenarios
revealed that a safeguard we assumed existed does not: duplicate data is refused but
not counted, there is no clock-drift fitting anywhere so the "synchronised" field is
always blank, network stalls are not counted, and two kinds of mismatch degrade
window by window instead of refusing to start.

### 3.6 The real recording

The user's own session (two runs, 24 attention switches each, 500 Hz, 24 electrodes)
imports cleanly: the marker table is exported for human checking, labels survive on
87 % of the time after excluding the transition buffers, and the live path accepts
the stream — 20 electrodes matched by name, 500 Hz, microvolts, zero gaps, clock
accurate to 0.01 parts per million, no message rejected.

But **zero decisions come out.** Every window fails inside the decoder with
"preprocessing contract does not match the model". Two of the mismatching items are
descriptions rather than settings: the recording says its reference is CPz, the
model says Cz. Section 5.1 explains this.

---

## 4. The code, and what state it is in

### 4.1 What exists

| Area | Where | Notes |
| --- | --- | --- |
| Real-time EEG processing (repair, quality checks, causal filtering, resampling, windowing, recovery) | `src/nova2026/streaming/` | Pre-existing. We did not change its contract |
| Attention decoding, decision logic, audio mixing | `src/nova2026/auditory/` | Extended with `session.py`, `sources.py`, `producer.py`, `render.py` |
| One session object shared by replay, live, and network modes | `src/nova2026/auditory/session.py` | Deliberately has no web-framework dependency |
| Network layer between Python and the browser | `src/nova2026/transport/` | 68 contract tests |
| Browser interface (React) | `apps/attune-ui/` | Copied unchanged from the teammate's repository, with the source commit recorded |
| Small stand-in for the teammate's Python backend, used only by three of the interface's tests | `apps/backend/` | Clearly marked as a test stand-in |
| Run a full demo | `python -B -m scripts.auditory_ui.demo` | One command: serves the interface, plays the recording, runs the session |
| Run the fault matrix headlessly | `python -B -m scripts.auditory_ui.demorun` | 23 scenarios, writes a report |
| Train a decoder | `python -B -m scripts.auditory.train_kuleuven` | Two contracts: 64 and 20 electrodes |
| Calibrate the decision threshold | `python -B -m scripts.auditory.margin_calibration` | Produces the table in 3.3 |
| Measure the cost of misalignment | `python -B -m scripts.auditory.shift_sweep` | Produces 3.4 |
| Import the real recording | `python -B -m scripts.auditory.antneuro` | Marker table plus trials |
| Drive the live path with the real recording | `python -B -m scripts.getlive.ant_live` | Used for section 3.6 |
| The plan and every decision | `final_connection.md` | 742 lines, decisions D-01 to D-44 |
| Claims and non-claims | `VALIDATION.md` | 339 lines |
| The processing rules that must not move | `results/kuleuven_audit.md`, sections 7–8 | Changing these invalidates every trained model |
| Raw evidence for every number | `results/` | 38 files, each from a real run |

Roughly 880 automated tests pass across the repository, and 53 in the browser
interface.

### 4.2 The path a signal takes

```
two audio files  ──►  loudness curves (one per file) ─┐
                                                      ├──► comparison ──► score A, score B
EEG electrodes  ──►  clean, filter, resample  ──► windows ─┘                       │
                                                                                    ▼
                                        decision (A / B / not sure)  ◄── minimum gap test
                                                    │
                          ┌─────────────────────────┴──────────────────────┐
                          ▼                                                ▼
        volume control sent to the browser                     quality and timing checks
        (lower the unattended stream)                          (refuse when unsure)
```

Everything to the left of "volume control" runs in Python. The browser only draws
what it is told and applies the volume change; it never computes anything about
attention itself.

### 4.3 Decisions we made that are easy to get wrong

- **The playback position is whatever the browser reports.** Python never invents a
  media timestamp; it echoes the browser's own value back, because our interface
  refuses to act on a position it did not report itself.
- **Reference loudness curves are precomputed and checked.** If the loudness curve
  for a chosen audio file is missing, the session refuses to start rather than
  computing one on the fly.
- **Ground-truth labels never enter the decoder.** There is a test that flips every
  label and asserts that not a single decision changes.
- **When unsure, the system says so.** A window that fails a quality check, arrives
  late, or lacks evidence produces an explicit "not sure" rather than a guess. This
  is why the fault matrix has no silent failures — and also why the system is quiet
  more often than one might like (Section 3.3).
- **The volume only ever goes down.** The interface refuses any instruction that
  would amplify a stream above its original level.

### 4.4 Code we removed

A repository-wide cleanup deleted the superseded COG-BCI route and several `legacy`
directories (26 files, about 4,400 lines), unified document names, and renamed 119
scratch files. It is all recoverable from git history.

---

## 5. Limitations

### 5.1 The decoder cannot be moved between setups

This is the biggest practical obstacle and it is not a bug. The decoder learns how
much each electrode matters and how the channels relate to each other. Those weights
depend on where the electrodes are and what the signal is referenced to. Our
training data uses a Cz reference; the real headset uses CPz. The learned spatial
weights are therefore being applied to a signal that was measured differently.

Two consequences:

- Cross-setup use is currently forbidden outright — our own check refuses to run
  when any contract field differs, and two of those fields are descriptive text
  ("which reference", "how the data was produced") that can never match across rigs.
- Even if we let it through by relaxing the check, **the underlying mismatch would
  remain**. An honest fix is to train on data from the same rig, not to weaken the
  check.

### 5.2 The effect is weak

62 % balanced accuracy on 5-second windows means roughly three correct decisions out
of five. Deployed naively, a listener would notice the wrong stream being favoured
several times a minute. The 60-second-window number (88 %) is much more usable but
has not been trained, and it means waiting a minute for the first decision.

### 5.3 Nothing has been tested on a person

No amplifier, no participant, no live run. Everything reported here is a recording
played back, including the "live" sections in 3.6, which use the live code path but
a recorded file as the source.

### 5.4 The browser has never been opened

All interface evidence comes from running the interface's own code under Node and
inspecting the output. The layout, the controls, real audio playback, and the actual
latency of the sound card are unverified. A five-minute manual check in a real
browser would close this.

### 5.5 Timing has never been measured end to end

Given how expensive misalignment is (3.4), this is the most important open
measurement. What is missing is the delay between "a sound leaves the headphones"
and "that moment appears in the EEG", measured with a loopback cable or a
microphone. Without it we cannot say how well the system is aligned on real
hardware. The current design budgets ±30 ms for the residual after calibration.

### 5.6 The quality checks do not fit every recording

Our electrode-quality limits were tuned on the dataset's clean recordings. The real
recording drifts by 1.5–2.0 millivolts from its starting level — normal for that
hardware, but far above the 500 µV limit — so with the default settings almost every
window was rejected as an artifact. We made the limit a declared, recorded setting
and the pipeline then ran to the end. But honesty requires saying what that costs:
with a limit high enough to admit this recording, the check can no longer tell
"drift" from "a real spike". Two related detectors still work independently: the
absolute saturation check (which is what identified the two dead electrodes) and a
flat-line check.

### 5.7 Two electrodes of the real headset are unusable

In the recorded session, F8 is stuck at the amplifier's maximum for the entire
recording and F3 for 40 % of it. We can exclude them (and they remain recorded in
the output), but that is 2 of 20 channels lost, and the cause is worth investigating
on the hardware side — a saturating electrode is usually a contact or gel problem,
not a brain signal.

### 5.8 Four safeguards we thought existed do not

Found by the fault matrix: duplicate data blocks are rejected but not counted; there
is no clock-drift fitting so the "synchronised" status is always blank; network
stalls are not counted; and a sample-rate or audio-length mismatch degrades quietly
window by window instead of refusing to start. None of these produce wrong answers
today, but each one is a place where a future problem would be hard to see.

### 5.9 Thresholds that were guessed and never calibrated

The minimum score gap (now calibrated, 3.3) and the electrode-quality limit (now
declarable, 5.6) were both originally guesses. The gain applied to the unattended
stream (6 dB) is still a guess. So is the 0.4-second ramp used when changing volume.

---

## 6. What the end goal actually requires

The target: a person listening to two sources of audio, with the system making the
attended one louder and the other one quieter, continuously and correctly enough to
be usable.

### 6.1 Two ways to present the audio

**Ear-separated (what we have working).** One source in the left ear, the other in
the right. This is what the KU Leuven dataset uses, what the recorded session used,
and what our decoder was trained and measured on. The volume control works naturally:
each ear has its own volume, so "make the attended one louder" is one number per ear.
It is also somewhat easier for the decoder, because the two sources are physically
separated.

**One mixture in both ears.** Both sources are mixed together and played to both
ears; the listener concentrates on one. This is closer to a real hearing-aid
scenario and is what the user ultimately wants. It is harder: there is no
left-versus-right cue for the decoder to pick up, so the same model would score
lower. It also needs a different audio path, because our browser code attenuates
ears, not sources. Two ways to get there:

- *Interim, no interface changes:* send a left channel weighted toward source A and
  a right channel weighted toward source B. The existing per-ear volume then behaves
  roughly like "favour A / favour B". This is a genuine mixing decision and should be
  measured, not assumed.
- *Proper:* mix the two sources inside the browser with separate gain per source.
  This needs a change to the interface code, which we have so far kept untouched.

### 6.2 Steps that would improve the result, in order of value

1. **Train a 60-second-window decoder.** Measured effect: balanced accuracy from
   0.64 to 0.88 in the same evaluation. Cost: one training run; a longer wait for
   the first decision. This is the cheapest large win available.
2. **Record more sessions on the same rig and retrain on them.** This fixes the
   reference mismatch (5.1) and adapts the decoder to the actual cap and amplifier.
   Two or three more sessions with the same setup would be enough to train and
   test honestly.
3. **Measure the audio-to-EEG delay once, with a loopback.** Until this is done,
   nobody can say whether the system is aligned, and section 3.4 says alignment is
   the most expensive thing to get wrong.
4. **A short per-person calibration.** Even a few minutes of labelled data from the
   person about to use the system, used to adjust the decoder, is the standard way
   this problem is made practical. Alternatively an unsupervised method that adapts
   without labels — correlation-based approaches that look for the structure of the
   attended stream rather than a fixed set of weights are known to transfer between
   people better than the linear method we use, and are worth trying next.
5. **A manual override.** A button that lets the listener say "no, the other one".
   This is not a research contribution, but it converts an accuracy problem into a
   tolerable one, and it is what any real product in this space does.

### 6.3 The realistic implementation route

The infrastructure for a live system already exists and is tested: the acquisition
path, the bounded task queue, the recorder, and the timing model. What a live
deployment adds is:

- a **connector for the amplifier** (partly written, never exercised with hardware),
- the **loopback measurement** above, stored as a calibration profile and checked at
  start-up,
- a **decision about gain**: how much to attenuate, how fast to ramp, how long to
  wait before switching,
- **safety behaviour**: what the system does when it is unsure — ours currently
  returns to equal volumes, which is the conservative choice,
- and **a way to keep the listener in control** (6.2, item 5).

### 6.4 What we would not do

We would not quote the dataset numbers as an expectation for a live person, because
the two setups differ in reference and cap. We would not present raw accuracy on
this dataset, because the majority answer alone scores 66 %. And we would not claim
a hearing benefit: the measurements here are about a decision, not about whether
anyone hears better.

---

## 7. If someone picks this up tomorrow

Read `final_connection.md` section 5 for the step-by-step plan with evidence, and
`VALIDATION.md` for the claim boundaries. Then:

- to see it work: `python -B -m scripts.auditory_ui.demo`
- to see the fault behaviour: `python -B -m scripts.auditory_ui.demorun`
- to reproduce any number in section 3: the corresponding script is named in 4.1,
  and its output is in `results/`.

The repository is on branch `integration`, every commit is pushed, and the plan file
records not only what was decided but which alternatives were rejected and why.
