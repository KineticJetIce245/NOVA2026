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
`-ServeSeconds` (12 h by default), `-NoBrowser`, `-KeepAlive`.

It starts the demo **once** and waits for it. The demo serves its own stay-open
window (`-ServeSeconds` → `--serve-seconds`) on the port it printed, so nothing
has to restart it. `-KeepAlive` restores the older behaviour of restarting the
demo when it exits, which was a workaround for a defect that is now fixed:
`drive()` used to end with an unconditional `stop.set()`, so `--serve-seconds` and
`--browser` printed "still serving" and then exited at the end of the replay. The
drive phase now ends on an event of its own, so the demo serves out its window on
the same port — which makes a restart a way to replace a working page with a new
URL, and is why it is opt-in. `scripts/auditory/tests/test_demo_serve.py` locks
both directions (the replay must not stop the server; a stop request still must).

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

## Who owns the media slot — `--media-owner` (one slot, so it is decided, not raced)

The transport allows **exactly one** media controller at a time
(`MediaTimeline.control` refuses every other `client_id` with a 409, and only that
owner's own `stopped` releases it). Two clients want it: the browser page, and
this demo's own `SimulatedMediaClient` stand-in. The stand-in used to prepare the
instant `/api/session/start` answered, so it won by construction and the page
lost — and the page's loss was silent: `mediaController.play()` sends `prepare`
*before* `element.play()`, so a refused `prepare` means the element is never
played at all. That is the defect recorded as **D-52**, and it is why "I can't
hear anything" was true on some runs and false on others with no code change.

`--media-owner` states the intent instead of hoping for the quicker client:

| mode | who claims the slot | use it for |
| --- | --- | --- |
| `demo` | the stand-in, immediately — the behaviour of every run before this flag | unattended runs, exactly as before |
| `standby` | the page is offered the slot for `--standby-seconds` (default 10); the stand-in takes it only if nothing claimed it | a URL you open yourself |
| `page` | the page only; the stand-in never sends a command | a window certain to be watched |
| `auto` | `--open-browser` → `page`; otherwise `standby` | the default |

The mode, the window and the outcome are printed and written into the run record
(`media_owner`), so "who owned the slot" is never a guess after the fact.

**What it costs.** Only `page` is free. `standby` spends up to `--standby-seconds`
of a run in which nothing claims the slot, and then there is **no attenuation at
all** — the gain gate needs a genuinely reporting controller, and this process is
forbidden from inventing a playback position (decision D-02). That is why the
default is not `demo`: an unattended run pays ten seconds, while a person who
opens the URL by hand otherwise loses the entire demo. `standby` is not a timing
heuristic about how fast somebody clicks — the fact it reads is the transport's own
record of a *completed* handshake (`MediaTimeline.claimed`), which no probe can be
overtaken on. What the window decides is only how long the slot stays reserved for
a page that has not arrived yet; a page arriving after it is refused exactly as it
is today, and that refusal is recorded.

`--open-browser` therefore also means "the page owns the slot": this process opens
the page itself, so there is no race to lose and no window to wait out.

**Known limit, in the frontend, not fixed here.** When the page *is* refused it
prints "Playback synchronization unavailable. Stop, then Play to reconnect." The
retry that suggests does not work: `stopped` is refused for the same ownership
reason, so the operator is told to do something that cannot succeed. The fix
belongs in `apps/attune-ui/src/mediaController.js` (report the refusal as an
ownership conflict and name the recovery, or re-`prepare` only after a `stop` the
frontend itself has confirmed), and that file is owned by another agent.

`scripts/auditory/tests/test_media_ownership.py` locks all of it against a real
`MediaTimeline` over a real socket; `mutate_media_ownership.py` in the same
directory restores each half of the defect and shows which test goes red.
