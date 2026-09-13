# Final report — the EEG auditory-attention demo

Written by the main agent on 2026-09-14, on branch `integration`. Everything below
is traceable to a commit or a file in this repository; where something is *not*
established, it says so in the same sentence.

---

## 1. What was asked, and what exists now

**Asked**: a demo that reads a person's EEG and shows, in real time, which of two
simultaneous audio streams they are attending to — displayed in the teammate's
`attune-ui`, with the unattended stream attenuated. The user's final goal is a live
human; the no-hardware demo must run completely, and the final verification must be
**one clean run plus many faulted runs** on real data.

**Exists now**:

| Piece | Where | State |
| --- | --- | --- |
| Real-time EEG chain (repair, quality, causal filter, resample, ring buffer, recovery) | `src/nova2026/streaming/` | reused, unchanged contract |
| Speech-attention decoder + controller + mixer | `src/nova2026/auditory/` | extended: `session.py`, `sources.py`, `producer.py`, `render.py` |
| One session shared by replay, live and transport paths | `src/nova2026/auditory/session.py` | no FastAPI import; the transport depends on it, not the reverse |
| Packet transport behind one same-origin app | `src/nova2026/transport/` | 68 contract tests |
| The frontend, verbatim from the teammate | `apps/attune-ui/` + `apps/backend/` (test double) | 53 tests green |
| One-command demo | `python -B -m scripts.auditory_ui.demo` | exit 0, V1–V6 |
| Headless perturbation matrix | `python -B -m scripts.auditory_ui.demorun` | 23 cases, exit 0 |
| Real recorded rig through the live route | `python -B -m scripts.getlive.ant_live` | transport-verified; decode blocked (see §4) |
| Plan, decisions, contract, validation | `final_connection.md`, `results/kuleuven_audit.md`, `VALIDATION.md` | 742 / 585 / 339 lines |

56 commits on `integration`, all pushed to `origin`. Test suite at the last
whole-repo run: **880 tests, 6 suites, 0 skipped, green**; frontend 53/53.

---

## 2. The numbers, each with its scope

| Measurement | Value | Scope it is valid in |
| --- | --- | --- |
| Decoder, 64 channels, held-out story | balanced **0.6195**, accuracy 0.6170 | KU Leuven, 5 s windows. **Raw accuracy is below the null** (0.6586 per window, 0.6609 by time), so only the balanced number means anything |
| Decoder, 64 channels, leave-one-subject-out | balanced **0.6115** | same |
| Decoder, 20 channels (the live rig's subset) | balanced **0.5974** / 0.5891 | same; the 2.2-point drop from 64 channels is the measured cost of the live contract |
| Window length | 0.637 → 0.698 → 0.796 → **0.875** balanced at 5/10/30/60 s | 30–60 s points use 4 subjects and **are untested at non-zero offset** |
| Offset cost | **0.1191 early / 0.1496 late** balanced accuracy per 100 ms | peak at −25 ms, half-depth width 104 ms, chance by ±250 ms |
| Margin calibration | at `0.05`: coverage **0.524**, accuracy 0.684, balanced **0.688**, 4/4 stories above chance | measured on refitted per-fold models; the deployed model is never scored |
| Margin in use | demo defaults to the calibrated `0.05`; `MIN_MARGIN = 0.5` unchanged for every other caller | — |
| Demo effect, same trial and model | margin 0.5 → 12 gain frames; **0.05 → 284 of 496**, first decision 40 s → **8.6 s** | one KU Leuven trial, replay |
| Live route acceptance | 20 electrodes by name, 500 Hz, µV, `gaps=0`, 0.01 ppm, grid deviation 0.0 s, `rejected=0` | the operator's own recorded ANT session |
| Perturbation matrix | 23 cases: **crash 0/23, explicit 23/23, contamination 0/23** | 18 PASS, 5 FINDING (see §4) |

**The single most important relation**: 100 ms of misalignment costs 5–7× the
entire 64-to-20-channel gap. The alignment budget dominates the channel budget, and
the frontend's `0.75 s` gate bounds *report staleness*, not alignment.

---

## 3. How the work was run

**My role**: define each step, dispatch it to one subagent, verify independently
(read-only checks, re-run the evidence command), register the decision in
`final_connection.md` §7, commit and push. **I do not write implementation code or
tests** — that rule (§9) was set by the user after I violated it once.

**23 subagents**, one per step plus repairs: baseline, KU Leuven conversion,
envelope precompute, data audit + contract freeze, decoder training, offset sweep
(two attempts), transport port, frontend vendoring, session + producer, media
timeline, margin calibration, demo, demorun + perturbation matrix, ANT import,
ANT-through-LSL, railed-electrode exclusion, quality limit, report verification,
report correction, doc cleanup, ANT live unblocking. Every one reported real
commands and real output; several falsified their own results before I could.

**Budget discipline**: §6.2 caps each step and forbids retrying the same failing
command more than twice; §6.4 requires each new test to be falsifiable by mutation;
D-18 forbids two subagents touching one file; D-23/D-26 cap a deliverable at three
subagents. Every overrun is recorded in the decision log rather than hidden —
several steps ran at 2× their call budget, and that is in the record.

---

## 4. What is NOT established

1. **No live participant.** Every number is replay or a recording pushed through
   the live wiring. No amplifier was connected.
2. **V4's ANT side is unachieved.** The recorded rig runs through the live route
   but emits **zero decisions**. Causes were peeled off one at a time, each
   measured: the 500 Hz vs 128 Hz contract mismatch (an explicit adapter stage),
   two railed electrodes (per-channel exclusion), a quality limit calibrated for a
   different recording (a declarable, recorded threshold), and finally a
   **provenance-text equality check** — the recording's `.cnt` header declares
   **CPz**, the model records **Cz**, and no honest run can make those equal.
3. **No browser ever rendered the UI.** All frontend evidence is Node executing the
   vendored `protocol.js` / `state.js` / `decoders.js` / `dashboard` modules. There
   is no screenshot, no layout check, no real `AudioContext` latency.
4. **The audio-to-EEG loopback offset is unmeasured** (`residual_offset_seconds`,
   ±30 ms budget). No audio device exists on this path. One component is measured:
   the chain's own resampler startup delay, **0.014 s**, which neither path
   compensates.
5. **Four mechanisms the plan assumed are missing** (V9's findings): duplicated
   blocks are refused by the chain but not counted at acquisition; there is **no
   clock fit anywhere**, so `sync` stays `unobserved`; the transport counts no
   stalls; and contract mismatches (candidate length, sample rate) **degrade per
   window instead of refusing at startup** — "it started" does not mean "the source
   matches the model".
6. **The quality limit's cost**: on the ANT recording, declaring 20 000 µV leaves
   the amplitude criterion with **zero selectivity** — the recording's normal drift
   already exceeded the old limit, so `signal_quality == 1.0` only means "nothing
   moved more than 20 mV".
7. **No 30/60 s model.** The grid's best point (0.05 @ 60 s: coverage 0.384,
   balanced 0.948) needs a model trained at that length; window length lives in the
   decoder contract, so it is not a run-policy field.
8. **Deliberately unclaimed**: hearing or comprehension benefit, cross-subject
   generalisation beyond the LOSO numbers, latency compensation, acoustic output
   measurement, and that the ANT playback offsets are anything but operator
   testimony.

---

## 5. My own errors, all of them caught by subagents and recorded

| What I asserted | How it was wrong | Recorded as |
| --- | --- | --- |
| The old KU Leuven loader would raise on duplicate file names | It never raised; the real defect was **silently using the hrtf rendering** as the reference envelope | `final_connection.md` §1 fact 2 |
| `source_unit_exponent = -6` was a wrong dimension assumption; it should be `0` | Backwards: `_to_uv = 10**(e+6)`, so `-6` is exactly right and `0` would have introduced the bug | §1 fact 10, §3.11 |
| The ANT report's "reference electrode unknown" | Both `.cnt` headers declare `REF:CPz` | D-24 / corrected report |
| Three spurious markers, all excluded | Excluding the third breaks the 12/12 alternation and opens a 25.67 s mislabelled stretch; the evidence says it is the real cue | D-32 |
| The sync guard requires `sync.status == "observed"` | **That mechanism does not exist**; the guard could never be satisfied | D-36 |
| — and I wrote a test double by hand, before the rule forbidding it existed | Kept in §9 as the counter-example | §9 item 4 |

Three of these were errors in the *plan* that a subagent disproved by measurement;
two were errors in the *evidence report I wrote*; one was a process violation.

---

## 6. What to do next

1. **Decide the contract gate** (in flight): gate the comparison on the
   processing-deciding keys and record the provenance that differs, or opt in per
   run, or retrain on data from the same rig. Only the third makes the mismatch
   disappear — and it needs more than one recording.
2. **Record more data on this rig** — same CPz reference, same electrode
   positions — then train on it. Reference mismatch is not fixable in code.
3. **Book the amplifier** for a live run: everything except the hardware is wired
   and verified. The remaining measurement is the audio loopback offset.
4. Optionally close V9's two counting gaps (duplicates at acquisition, stalls at
   the transport) — observability, not correctness.

---

## 7. Where to look

| You want | Read |
| --- | --- |
| The plan and every decision | `final_connection.md` (§5 steps, §7 D-01…D-44) |
| What is claimed and what is not | `VALIDATION.md` |
| The frozen processing contract | `results/kuleuven_audit.md` §7–8 |
| The wire contract | `documents/auditory_ui_protocol.md` |
| Every number's raw evidence | `results/` (38 committed files) |
| The real rig's measured properties | `results/antneuro_testset_report.md` |
