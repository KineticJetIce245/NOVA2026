# The wire contract between this backend and the vendored `attune-ui`

This document specifies what travels between `src/nova2026/transport/` and the front end in
`apps/attune-ui/`, so that a reader can tell which side owns which value without opening both
trees. It describes the implementation as it stands at step 12; the front end is a vendored copy of
a teammate's client (source commit and per-file hashes in `apps/attune-ui/PROVENANCE.md`), so the
contract is not ours to change unilaterally.

Companion documents: [`media_timeline_contract.md`](media_timeline_contract.md) explains *why* the
timeline works this way and what `0.75 s` does and does not mean;
[`../VALIDATION.md`](../VALIDATION.md) states what has been measured on this path and what has not.

## 0. Two rules that shape everything below

1. **Decisions are made in Python only.** The front end validates, decodes and displays; it never
   re-computes, smooths or infers a measurement (`apps/attune-ui/src/decoders.js`, first line).
2. **The backend transcribes playback time, it does not author it.** `media_id`,
   `media_revision` and `media_time_s` are copied from what the browser reported; the browser
   re-validates them field by field before any gain may be applied (decision D-02).

## 1. The packet envelope (version 1)

Built and validated by `src/nova2026/transport/protocol.py`; re-validated independently by
`apps/attune-ui/src/protocol.js` before the client renders anything. The transport is deliberately
the stricter of the two sides: whatever it accepts must also survive the client's validator, or a
frame is dropped *after* the server counted it as delivered.

| Field | Type | Rule |
| --- | --- | --- |
| `version` | int | exactly `1`, the only version accepted or built |
| `type` | string | nonempty, at most 128 characters |
| `source` | string | nonempty, at most 128 characters; the producing stream |
| `session_id` | string | nonempty, at most 128 characters |
| `sequence` | int | assigned by the publisher, nonnegative, at most `2**53 - 1`; the only ordering the stream has |
| `timestamp` | number | **session-relative** seconds, finite, nonnegative, at most `2**53 - 1`; never wall-clock time |
| `payload` | object | JSON object; the envelope does not interpret it |

Whole-packet limits: JSON nesting depth at most **32** (the root envelope included), encoded size at
most **256 KiB** (262144 bytes on both sides), numbers finite (no `NaN`/`Infinity`), object keys
strings only. A validated packet is JSON round-tripped, so it is detached from the producer's
objects and later mutation cannot rewrite what was already published; `numpy` scalars must be
converted at the producer boundary, because the strict type check is what keeps the two validators
from disagreeing. Unknown types and extra fields are preserved, not stripped.

## 2. Snapshot, then incremental - and the 1013 close

* `GET /api/state` returns one atomic snapshot:
  `{ "sequence": <high-water>, "session_id": <id|null>, "packets": [<=128 packets] }`.
  The client's `validateSnapshot` requires: at most 128 packets, every packet's `session_id` equal
  to the snapshot's, strictly increasing sequences all `<= sequence`, and **at most one packet per
  `(source, type)` key** - the snapshot carries the latest packet of each live stream, not history.
* `/ws/live` then delivers every event with a sequence strictly greater than the snapshot's
  high-water mark. That is what closes the subscribe race: no gap is possible between snapshot and
  stream.
* The client's `acceptPacket` (`state.js`) refuses, and counts into `state.rejected`:
  a packet whose `sequence` is **not greater** than the last accepted one (duplicate or
  out-of-order), or a packet whose `timestamp` regresses on the same `(source, type)` stream. A
  packet from a different `session_id` resets the stream list rather than being rejected.
* There is **no unbounded per-client queue**. A socket that overruns the **256-packet ring**
  produces a `LaggedSubscriber`, and the server closes it with close code **1013**
  (`CLOSE_LAGGED` in `src/nova2026/transport/server.py`). A send that does not complete within the
  5 s send timeout is closed the same way. The client reconnects and takes a fresh snapshot.
  Losing history is explicit on purpose: a silent gap in the sequence would be worse, because the
  client counts protocol violations and refuses to guess.

`sequence` is process-wide and never resets inside one server process; a backend restart therefore
starts again from a low number, and the client accepts that **only** through a fresh snapshot, not
through the event stream (case C9 of the perturbation matrix).

## 3. Packet types and the payload fields the front end reads

`decoders.js` holds one display adapter per type. It coerces and range-checks; it never derives a
value. A field the adapter cannot validate becomes `null`, and a payload that fails a whole-payload
condition (for example `audio_sources` without exactly two distinct ids `A` and `B`) becomes an
empty result rather than a guess.

| `type` | Payload fields used | Rules the adapter enforces | Published today |
| --- | --- | --- | --- |
| `attention` | `decision` (`A`/`B`/`uncertain`/`unavailable`), `attended`, `correlation_a`, `correlation_b`, `media_id`, `media_revision`, `media_time_s` | when `decision` is present it wins over the legacy `attended` field; `attended` is exposed only as `A`/`B`; correlations are raw and are never smoothed | per frame |
| `gain` | `a_db`, `b_db`, `media_id`, `media_revision`, `media_time_s` | the media reference is copied only when the packet carries `media_id` | per frame |
| `signal_quality` | `quality`, `artifact` | numbers/booleans coerced; `bad_channels` is carried in the payload and named by the producer for display | per frame |
| `media` | `media_id`, `title`, `media_time_s`, `duration_s`, `playback_state`, `revision`, `server_reference_s`, `sync_status` | `playback_state` is one of `playing`/`paused`/`stopped`, otherwise `null` | every 250 ms while a session runs |
| `prediction` | `status` (`ok`/`invalid`/`unavailable`/`error`), `provider_id`, `provider_name`, `provider_version`, `task`, `timestamp`, `window_id`, `window_start`, `window_end`, `outputs[].{name,value,semantic_type,label}`, `reasons[]`, `metadata` | only object-shaped outputs survive; reasons are filtered to strings | when a decision is `A` or `B` |
| `sync` | `status`, `offset_ms`, `drift_warning`, `timeline`, `fixed_latency_ms` | `status` is free text and is *not* constrained by the adapter | at start and when the reason set changes |
| `session` | `status`, `session_id`, `source` | **no adapter**: `decodePacket` passes the payload through and flags `known: false`; the dashboard and the gain gate read `values.status === 'running'` | lifecycle, `source = "server"` |
| `audio_sources` | `sources[]` with `id` (`A`/`B`), `label`, `input_type`, `reference` | exactly two sources with distinct ids, else `[]` | once at start |
| `eeg_display` | `sample_rate`, `channels[]`, `samples[][]` | sample rate must be positive, channels strings, samples finite numbers | **no** - nothing publishes it today |
| `vigilance` | `score`, `metric`, `lapse_score` | scores must lie in `[0, 1]` | **no** |
| `feedback` | `status`, `action_type`, `message`, `severity`, plus `simulated` and `metadata.development_only` | anything not marked `simulated: true` **and** `metadata.development_only: true` decodes to `suppressed`; this is how real data can never be shown as a demonstration | **no** |

The demo and evidence runs publish eight types: `session`, `audio_sources`, `attention`, `gain`,
`signal_quality`, `media`, `sync`, `prediction`
(`results/demo_run_20260914.md`: 2123 packets, `rejected = 0`).

## 4. The media handshake

Front-end side: `apps/attune-ui/src/mediaController.js`. Backend side:
`src/nova2026/transport/media.py`, served at `GET /api/media` (asset identity: `media_id`, `title`,
`kind`, `url`), `GET /api/media/file` (the bytes) and **`POST /api/media/control`** (the report
channel, reused from the original UI, so the contract did not have to change).

Request body:

| Field | Meaning |
| --- | --- |
| `session_id`, `media_id` | must match this timeline, or the timeline raises |
| `client_id`, `request_id` | client identity and a monotonic request counter |
| `action` | exactly one of `prepare`, `playing`, `paused`, `stopped`, `report` |
| `media_time_s` | the position the client read; `0` for `prepare` and `stopped` |
| `duration_s` | the element's duration |

Order of a playback: `prepare` (rewind to 0, then report it) -> `playing` -> `report` every 250 ms
while prepared, connected and not stale -> `paused` on interruption -> `stopped` at the end.
Position is accepted only if it advances plausibly: `PLAYING_ADVANCE_ALLOWANCE = 0.5 s`,
`IDLE_ADVANCE_ALLOWANCE = 0.5 s`, `PAUSED_ADVANCE_ALLOWANCE = 0.1 s`, and a backward jump is
tolerated only within `BACKWARD_TOLERANCE = 0.02 s`. Anything else is refused, and **the refusal is
sticky** - the timeline stays invalid until a fresh `prepare`, so a bad report cannot be followed by
good ones that quietly restore a gain.

Acknowledgement, and what the client does with it: the response carries `session_id`, `media_id`,
an integer `revision` and `sync_status`. The client rejects the acknowledgement (and then pauses,
returns to neutral gain and marks playback as errored) unless all of these hold:

* `result.session_id` and `result.media_id` are the ones it asked about;
* the round trip took at most **1500 ms** (`AbortSignal.timeout(1500)`);
* `Number.isInteger(result.revision)`;
* `result.sync_status === (action === 'stopped' ? 'desynchronized' : 'observed')`.

Afterwards `lastAck` must stay within 1500 ms for the client to consider itself `ready`; the same
1500 ms is `FRESH_SECONDS` on the server, where a timeline is valid only while a report has arrived
inside that window. `MediaTimeline.media_reference()` returns `None` - not a stale value - when the
report is older than that, when nothing was prepared, or when the timeline was invalidated; the
producer reads it **once per frame** and stamps the same reference onto the attention and the gain
packet, so the two can never disagree by one 250 ms step.

`MediaBroadcaster` publishes one `media` packet every **250 ms** (`DEFAULT_INTERVAL = 0.25`), only
while a session is `running`, with `source = "server"`, carrying the last reported position,
`revision` and `sync_status` (`observed` while valid and fresh, `desynchronized` otherwise).

Measured on the replay path, `S1/trial_008` at 1x (step 9,
`results/auditory_media_20260914-trial008.json`): 339 control commands, all acknowledged `observed`,
**one revision (2) for the whole session**, and `|delta t|` over 361 gain packets of max 0.300 s /
mean 0.149 s - a phase offset between two 250 ms cadences, not accumulating drift.

## 5. The gain gate

`mediaAudio.js::mediaFocusReady` is one boolean expression. `scripts/auditory_ui/gate_evidence.mjs`
enumerates it as the **fourteen clauses** below and evaluates them against the state a browser would
hold; every step-9 and step-10 record reports 14/14 true at the moment the gate opened.

| # | Clause | Reads |
| --- | --- | --- |
| 1 | `connection.isConnected` | `connection === 'connected' && !stale && !error` |
| 2 | `session.statusRunning` | newest `session` packet |
| 3 | `playback.ready` | client: prepared and an acknowledgement inside 1500 ms, no error |
| 4 | `playback.stateIsPlayingOrPaused` | client playback state |
| 5 | `media.syncStatusObserved` | newest `media` packet |
| 6 | `media.playbackStateMatches` | `media` vs client |
| 7 | `media.revisionMatches` | `media` vs client |
| 8 | `media.mediaIdMatches` | `media` vs client |
| 9 | `sync.statusIsUsable` | newest `sync` packet: not `desynchronized` and not `invalid` |
| 10 | `attention.sessionIdMatches` | `attention` vs client state |
| 11 | `attention.mediaIdMatches` | `attention` vs client |
| 12 | `attention.mediaRevisionMatches` | `attention` vs client |
| 13 | `attention.timeWithin750ms` | `abs(attention.media_time_s - playback.time) <= 0.75` |
| 14 | `attention.decisionIsAOrB` | `decision` when present, else `attended` |

Clause 9 is worth reading twice: it accepts `unobserved`. The gate does **not** require a clock fit;
it only rejects a timeline the server has explicitly declared bad. On the replay path the session's
`sync` packet says `unobserved` for the whole run (see section 7), and the gate still opens.

`playbackGains(state, playback, 'attune')` then applies five further conditions before any
attenuation leaves neutral: the newest `gain` packet's `media_time_s` must equal the newest
`attention` packet's, both must belong to the current `session_id`, `media_id` and `revision` must
match the client's, the time must be finite and within 0.75 s of the playback clock, and every dB
value must be in `[-80, 0]`. The last one is the "attenuate only, never amplify" rule enforced at
the boundary: an unexpected positive gain fails to neutral `[1, 1]` rather than being applied. The
values are converted with `dbToLinear` = `10 ** (db / 20)`.

Evidence: at the first opening the condition set held 14/14 and the applied gains were
`[1.0, 0.5011872336272722]` = 0 dB / -6.0 dB, bit-identical to `dbToLinear` of the packet's own
`a_db`/`b_db` (`results/auditory_media_gate_20260914-trial008.json`). Under margin 0.5 the gate
opened at 40.0 s with 12 of 361 gain frames attenuated; under the calibrated margin 0.05 (the
operating point, D-29) it opened at 9.0 s with 284 of 496.

## 6. Where each side lives

| Piece | Backend | Front end |
| --- | --- | --- |
| envelope construction/validation | `src/nova2026/transport/protocol.py` | `apps/attune-ui/src/protocol.js` |
| ordering, duplicate rejection, stream bookkeeping | `src/nova2026/transport/publisher.py` (ring + latest snapshot) | `apps/attune-ui/src/state.js` |
| payload decoding for display | - | `apps/attune-ui/src/decoders.js` |
| playback timeline state machine | `src/nova2026/transport/media.py` | `apps/attune-ui/src/mediaController.js` |
| gain gate and Web Audio application | `src/nova2026/auditory/producer.py` (publishes `a_db`/`b_db`) | `apps/attune-ui/src/mediaAudio.js` |
| HTTP + WebSocket + SPA mount | `src/nova2026/transport/server.py` | - |

## 7. `0.75 s` bounds report staleness, not alignment

The clause `abs(attention.media_time_s - playback.time) <= 0.75` and its twin inside
`playbackGains` are **not** an alignment specification, and this document says so explicitly so
that no later reader quotes them as one:

* step 5.5 measured the decoder's tolerance to a timing offset (5 s window, held-out stories): the
  correlation peak is at -25 ms, the half-depth width is 104 ms, and by +/-250 ms the decoder is
  back to chance. Per 100 ms of residual misalignment the balanced accuracy falls by 0.1191 (early)
  or 0.1496 (late) - five to seven times the entire cost of dropping from 64 to 20 channels (0.022)
  (`results/aad_shift_sweep_20260913-040112.md`);
* against that, **0.75 s is roughly seven times the decoder's whole tolerance**. It guards against
  a *stale report* - a stalled tab, a report that stopped arriving, a seek nobody acknowledged -
  and it is deliberately loose because the front end cannot know the offset and must not guess one;
* a run whose residual offset really were 0.75 s would have a decoder contribution of zero, i.e.
  chance-level accuracy, while the UI displayed "synchronized";
* the **alignment budget is +/-100 ms**. The loopback measurement that would establish it
  (`residual_offset_seconds`, tolerance +/-30 ms) belongs to step 11 and has not been taken;
* **there is no clock fit anywhere in this implementation.** The session's `sync` packet is
  `unobserved` with `offset_ms = null` and `drift_warning = null` on the replay path, and the
  perturbation case C11 showed the same after injecting 1.25x drift (`|delta t|` up to 2.63 s):
  drift is visible only as a difference between reported positions. `sync_status: "observed"` in a
  *media* packet is a different statement - it means "a report arrived inside 1500 ms" - and it is
  not evidence that the clocks agree. Neither field means the two clocks have been fitted.

The full reasoning is in [`media_timeline_contract.md`](media_timeline_contract.md) section 2, and
the measurement is in `results/aad_shift_sweep_20260913-040112.md` (the earlier sweep,
`...-033723.*`, is retracted and must not be cited).

## 8. What this document does not cover

* The pairing of EEG time to media time on the acquisition side (that is
  `documents/timebase_design.md` and `src/nova2026/auditory/session.py`).
* The rationale for the timeline rules, the freshness window and the 250 ms cadence
  (`documents/media_timeline_contract.md`).
* What has actually been proven on this path: no browser has executed the front end, no audio
  device has been measured, and the UI evidence is Node running the vendored modules
  (`VALIDATION.md` sections 1 and 3).
* The perturbation matrix that exercises this contract adversarially - protocol-violating packets,
  disconnect and reconnect, backend restart, clock drift, playback stalls
  (`results/perturbation_20260913-062335.md`, cases C8, C9, C11, C12, C13).
