# The media timeline: what `0.75 s` does and does not mean

Step 9 wired the browser's playback path into the transport. This note records the
two facts a later reader needs before quoting any number from that path, and the
one measurement that is *not* an alignment specification.

## 1. The frontend owns playback time, and the backend only transcribes it

`apps/attune-ui/src/mediaController.js` sends `POST /api/media/control` with the
position it read from the `<audio>` element, and `apps/attune-ui/src/mediaAudio.js`
validates the answer field by field before any gain may be applied. The backend
therefore has **transcription rights, not authorship** over `media_id`,
`media_revision` and `media_time_s` (decision D-02 in `final_connection.md`).

The implementation that matches that rule:

| Piece | File | What it guarantees |
| --- | --- | --- |
| `MediaTimeline.control` | `src/nova2026/transport/media.py` | Accepts only plausible progress, refuses the rest, and makes the refusal sticky |
| `MediaTimeline.media_reference` | same | Returns the last reported position and the acknowledged revision, or `None` when a report is older than 1.5 s or the timeline was invalidated |
| `MediaBroadcaster` | same | Publishes that snapshot as a `media` packet every 250 ms, only while a session runs, so the dashboard has a timeline to read |
| `AttentionProducer._emit_frame` | `src/nova2026/auditory/producer.py` | Reads the reference **once per frame** and stamps it onto both the attention and the gain packet |

The last row is not tidiness. `mediaFocusReady` compares the attention packet's
position with the playback clock, and `playbackGains` additionally demands that
the gain packet's position be *equal* to the attention packet's. Two reads of a
timeline that the controller is still reporting into can differ by one 250 ms step,
and the gain path would then fall back to neutral while every individual field
still looked correct.

## 2. `|Δt| ≤ 0.75 s` bounds staleness, not alignment

`final_connection.md` section 3.17 item 4 measured the decoder's tolerance to a
timing offset: with a 5 s window on the held-out stories, the correlation peak is at
−25 ms, the half-depth width is 104 ms, and at ±250 ms the decoder is back to
chance. Per 100 ms of residual misalignment the balanced accuracy falls by
0.1191 (early) or 0.1496 (late) — five to seven times the entire cost of dropping
from 64 channels to the 20 channels a live cap can carry (0.022).

Against that, `0.75 s` is roughly **seven times the decoder's whole tolerance**. It
is a guard against a *stale report* — a tab that stalled, a report that stopped
arriving, a seek nobody acknowledged — and it is deliberately loose, because the
frontend cannot know the offset and must not guess one. It must never be quoted as
an alignment budget:

* the alignment budget is **±100 ms**, and step 11's loopback measurement
  (`residual_offset_seconds`, tolerance ±30 ms) is the stricter and correct target;
* a run whose residual offset really were 0.75 s would have a decoder contribution
  of zero, i.e. chance-level accuracy, while the UI reported "synchronized";
* the `sync.status` field a session publishes is `unobserved` on the replay path,
  because no browser clock has been fitted there. It is not a claim that the clocks
  agree; it is the absence of a measurement, and step 12 says so in
  [`../VALIDATION.md`](../VALIDATION.md) §4 and in
  [`auditory_ui_protocol.md`](auditory_ui_protocol.md) §7.

## 3. What step 9 measured on the replay path

`S1/trial_008`, 90 s at 1× (commit for step 9; `results/auditory_media_20260914-trial008.json`):

* handshake: `prepare` → `playing` → `report` every 250 ms; 339 commands, all
  acknowledged `observed`, one revision (2) for the whole session;
* `|Δt|` over 361 gain packets: max **0.300 s**, mean **0.149 s**, min 0.000 s,
  all within the frontend's 0.75 s clause. The residual is the phase offset between
  the producer's 250 ms frame cadence and the browser's own 250 ms report cadence,
  not drift: it never accumulates;
* the frontend's own `mediaFocusReady` held for the first time at t = 40 s, with all
  fourteen clauses true, and `playbackGains(..., 'attune')` then returned
  `[1.0, 0.5012]` — candidate B attenuated by 6 dB, candidate A untouched.

Not measured anywhere on this path: the audio device's block timing and drift
(`TimestampedAudio.diagnostics()` needs a real DAC), the loopback offset from
playback to EEG, and the acoustic latency. Those belong to steps 11 and 12, and the
run record says `block_timing.available = false` rather than filling the field with
a number nobody took.
