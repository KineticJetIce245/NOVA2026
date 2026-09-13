# A prompt to hand to an agent on the Mac

This file exists so the demo can be brought up on a machine that has none of this
session's context. Everything below the line is meant to be copied whole and given
to an agent running on the Mac, with the repository path filled in first.

It is written for the **no-hardware** demo: the EEG comes from a replayed KU Leuven
recording, so no amplifier is needed. The live-amplifier route is a different entry
point (`scripts/getlive/live_demo.sh`) and a different set of prerequisites.

Background the operator may want first: `documents/where_the_demo_stands.md` says
what works, what does not, and what has never been tested on a person.

---

You are on macOS. The repository is at `<FILL IN THE PATH>` — a checkout of the
`integration` branch, possibly on an exFAT external drive. Work in the repository
root. Report what you actually observe; do not fill gaps with plausible values.

## What this is

A visible EEG auditory-attention demo. A backend replays a real recorded EEG
(KU Leuven AAD dataset) through the real processing chain at 1× speed, decides once
per 0.25 s which of two speech streams the listener is attending to, and a browser
page shows that decision while turning the other stream down by 6 dB. The page also
draws two EEG traces: the raw signal, and the 1–9 Hz band-passed signal the decoder
actually uses.

**No hardware is required.** The EEG is a recording being replayed.

## Your task

Bring this demo up on this Mac and report honestly what it does.

### 1. Get the current code

```sh
git rev-parse --abbrev-ref HEAD      # expect: integration
git pull
```

Then confirm the code contains the EEG-trace feature — this is a hard gate, because
an older checkout will run but the page will never draw the traces:

```sh
grep -c eeg_display src/nova2026/auditory/producer.py
```

If that prints `0`, the checkout is older than the feature. **Stop and tell me** —
do not continue and do not try to reconstruct it.

### 2. Create the Python environment

`.venv` is machine-specific and must not be copied between machines. The drive may
carry a broken stub (a symlink pointing at a path on another Mac); if `.venv` exists
and is not a working environment, `rm -rf .venv` first.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

Python 3.12 or newer is required.

### 3. Confirm the assets are present

These are excluded by `.gitignore`, so `git pull` never brings them. Check each:

```sh
ls -l models/auditory_kuleuven.npz
ls datasets/audio/*.npz | wc -l          # expect 18
ls -l datasets/AAD-KULeuven/converted/S1/trial_004.npz
```

- `models/auditory_kuleuven.npz` — the 64-channel decoder
- `datasets/audio/*.npz` — the speech envelopes; **a session refuses to start without them**
- `datasets/AAD-KULeuven/converted/S1/trial_004.npz` — a 389-second trial

Any of them missing: stop and tell me which.

### 4. Rebuild the frontend — do not skip this

`apps/attune-ui/dist/` is git-ignored **and is a build artifact of one specific
commit**. The repository's own sync manifest is explicit about the consequence:
after a pull you must rebuild, or the page you are served silently disagrees with
the `apps/attune-ui/src/**` you are reading, and no test fails.

```sh
npm --prefix apps/attune-ui ci
npm --prefix apps/attune-ui run build
```

### 5. Run it

```sh
sh scripts/auditory_ui/serve.sh --no-browser --seconds 60 --serve-seconds 300
```

Use `sh <script>` rather than `./<script>`: exFAT does not store the executable bit,
so the direct form can fail with "permission denied".

The script prints a `http://127.0.0.1:<port>/` URL **before the replay starts**, so
you can open it immediately. Open it and click **Play**.

### 6. Verify, then report

Answer all five, quoting the exact text you see on the page:

1. Is there sound? **Only one** of the two streams should be attenuated by 6 dB; the
   page states this in words (for example "Source B turned down to −6 dB").
2. Does the EEG panel draw **two** traces (upper = raw, lower = band-passed), or
   does it still read "Awaiting EEG display data"?
3. What does "CURRENT FOCUS" show — Source A / Source B / Uncertain / Unavailable?
4. Any red warnings? In particular `REJECTED PACKETS` or
   `Playback synchronization unavailable. Stop, then Play to reconnect.`
5. After the replay ends, is the page still served, and does it say
   `Session: stopped` with "Values are historical"?

## Known issues — report them, do not try to fix them

- **The media-control slot is exclusive, and this is the likeliest reason for no
  sound.** Exactly one media controller is allowed at a time, and the page's Play
  button sends its readiness message *before* it starts the audio. So if the demo's
  own stand-in client holds the slot, the page's message is refused and **the page
  produces no sound at all** — it shows "Playback synchronization unavailable. Stop,
  then Play to reconnect." The stand-in is supposed to hold off and ask the server
  who owns the slot before sending anything, so the page should get its turn; that
  reasoning is tested but **no human has yet confirmed it in a real browser**, which
  is part of why you are running this. Quote the log line that says who won
  (`media slot: ...` / `media owner: ...`). If there is no sound, re-run and click
  Play promptly, and report whether that changed the outcome.
- **The volume control opens about 9 seconds in** — a 2-second warm-up plus the first
  5-second window plus the decision. No attenuation during the first seconds is
  expected, not a fault.
- **"Uncertain" is a legitimate result.** The decision rule needs the score difference
  to exceed a ±0.05 band; abstaining when it does not is the designed behaviour.
- **A `REJECTED PACKETS` alarm is usually benign here.** On connect the server
  deliberately replays its retained snapshot, and that replay has historically been
  miscounted as rejected packets. Report the number and the time it appeared; do not
  treat it as data loss without checking.
- **The decision is made on a 5-second window of replayed EEG.** It is a real
  computation on real recorded data, but the person listening is not the person whose
  EEG was recorded, and no participant has ever been tested live.

## Do not

- Do not start a second demo instance: two instances compete for one rendered media
  file and for the media slot.
- Do not run `git commit`, `git checkout`, `git revert` or `git stash` — I need to see
  the working tree's real state.
- Do not install anything into the system Python; work inside `.venv`.
- Do not run the whole test suite. Individual suites cost 5–8 minutes each and other
  work is in flight; run only single modules by name if you need to.

## Report format

1. `git rev-parse HEAD` and `git status --porcelain`
2. The actual output of steps 1–5, with the raw error text for anything that failed
3. Answers to the five questions in step 6, quoting what you saw
4. Clearly separate **observed** from **inferred**. Do not invent a number, a log line
   or a screenshot description
5. Anything you need me to decide
