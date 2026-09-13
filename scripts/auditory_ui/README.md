# `scripts/auditory_ui/` — the attention demo's assembly and evidence commands

One folder per presentation mode (plan D-21). This folder holds the current mode:
a replay of recorded EEG, with the two candidates scored against their own
precomputed envelopes. The mode-agnostic parts — the session, the producer, the
transport — live in `src/nova2026/`.

## What is here

| Path | What it does |
| --- | --- |
| `session.py` | Starts the real FastAPI app on loopback, starts a session, subscribes to `/ws/live` with a real WebSocket client, asserts what arrived, and writes a run record. This is the step-8 evidence command. |
| `decode_packets.mjs` | Replays a recorded packet stream through the vendored frontend's own validator, state machine and React components, and renders the dashboard to static HTML. |

## Running a session

```powershell
# A generated fixture: no dataset, no models directory, ~10 s.
.venv\Scripts\python.exe -B -m scripts.auditory_ui.session --synthetic `
  --seconds 20 --speed 4 --out results/auditory_session_synthetic.json

# A real KU Leuven trial, replayed at 1x. trial_008 is one of the short trials
# (~125 s); most S1 trials are 389-399 s (trial_004: 388.992 s, measured).
.venv\Scripts\python.exe -B -m scripts.auditory_ui.session `
  --trial datasets/AAD-KULeuven/converted/S1/trial_008.npz `
  --model models/auditory_kuleuven.npz --speed 1 `
  --out results/auditory_session_trial008.json
```

The command prints one line per assertion and exits non-zero if any of them
fails. Its outputs are:

* `results/auditory_session_<name>.json` — the run record: the trial, the
  **run policy**, the model contract, the session summary, a bounded packet
  excerpt, every assertion, and the post-hoc comparison with the trial's label.
* `output/auditory_ui/packets_<name>.jsonl` — every packet exactly as it arrived
  over the socket.
* `output/auditory_ui/<name>.log` — the console output of the run.

The label comparison in the record is **plumbing, not accuracy**: it says which
way the published decisions went, and it runs after the stream is closed so that
nothing in the decision path can see a label.

## Rendering what the frontend shows

```powershell
node scripts/auditory_ui/decode_packets.mjs `
  output/auditory_ui/packets_trial008.jsonl results/auditory_frontend_trial008.html
```

This repository has no browser in its test path, so this script is the substitute
for a screenshot: it feeds the recorded packets through
`apps/attune-ui/src/protocol.js`, `state.js`, `decoders.js` and `Dashboard.js`
itself — the same modules the browser runs — counts what the client rejected, and
renders the dashboard to HTML with `react-dom/server`. It prints one line per
check. The HTML is the evidence artifact; it is static markup, not a screenshot,
and it shows the card as it looked at the moment of the last A/B decision.

## Starting the page yourself (Windows)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\auditory_ui\serve.ps1
```

`serve.ps1` is a flag-remembering wrapper around the demo above: it serves
`apps/attune-ui/dist`, opens the printed loopback URL in the default browser and
reads that URL back out of the demo's own output (the port is ephemeral). Useful
switches: `-Rebuild` (run `npm run build` in `apps/attune-ui` first — `dist/` is
git-ignored and goes stale as `src/` changes), `-Trial`, `-Seconds`, `-Margin`,
`-NoBrowser`, `-NoKeepAlive`.

It restarts the demo when it exits, which was a workaround for a defect that is
now fixed: `drive()` used to end with an unconditional `stop.set()`, so
`--serve-seconds` and `--browser` printed "still serving" and then exited at the
end of the replay. The drive phase now ends on an event of its own, so the demo
serves out its stay-open window on the same port, and the restart is no longer
what keeps a page up.

## Run policy

The streaming chain's own default (`check_channels=True, max_bad_channels=0`)
halts a KU Leuven trial when electrode faults persist past the recovery budget
(plan section 3.17 item 1), so the demo states its policy instead of inheriting
one: `check_channels=False`, `max_bad_channels=0`, recorded in the run record and
carried into the `audio_sources` packet so the stream itself says which policy
produced it. Pass `--strict-policy` to run with the chain default.

## Security posture

The server binds `127.0.0.1` only and refuses any other host. No filesystem path
is accepted over HTTP; the media path, if any, is server-side configuration. The
console output is packet types, counts and assertion verdicts — never packet
payloads. The packet stream is written to `output/auditory_ui/` as the run's
evidence: it holds decision words, correlations, gains and candidate file names,
and nothing that identifies a participant.
