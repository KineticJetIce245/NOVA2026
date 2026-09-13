# Phase 3 result boundary

Scientific code produces results; adapters serialize them; the existing session
worker assigns session identity, sequence, and publishes to REST/WebSocket.
No adapter imports scientific modules or opens EEG/audio sources.

Contract additions (envelope remains version 1):
- `attention`: existing attended A/B/null and correlations remain. New `decision`
  is A, B, uncertain, or unavailable. Uncertain is an upstream decision, never a
  correlation threshold invented here. Non-A/B decisions have attended=null.
- `vigilance`: metric=vigilance uses existing score. metric=lapse_probability
  uses lapse_score, with score=null. For the existing NOVA checkpoint this is
  the softmax score for `slowest_session_decile` (its PVT label), not a new
  physiological vigilance measurement. Lapse probability is never inverted or
  mislabeled as vigilance. Unavailable values are null.
- `sync`: status (caller-defined nonempty label; default unknown), offset_ms and
  drift_warning. Unknown offset/warning default to null, not zero/false. No
  physical synchronization is inferred from matching result timestamps.
- `eeg_display`: caller-prepared channel/sample display data only. No filtering,
  resampling, acquisition or analysis happens here.

All result timestamps must be supplied explicitly as session-relative seconds.
`CombinedDecision.end_time_s` may refer to a recording clock; conversion to the
session clock belongs to the integrating coordinator. Transport timestamps do
not prove physical EEG/audio synchronization. Source names identify algorithm
or metadata producers, not a required device. Numeric scalar serialization and
shape/range validation do not change scientific algorithms.

## Exact publishing entry point

A teammate calls **`ResultAdapter.publish(result)`**. `ResultProducer` supplies
that adapter to a coordinator callback and plugs into the existing app factory:

```python
from backend.app.server import create_app
from backend.adapters.results import ResultProducer
from backend.adapters.contracts import AttentionResult, SyncResult
from backend.adapters.legacy import combined_results, aad_result, lapse_result


def run_results(adapter, stop):
    # Example publishes unavailable results only; no hardware/model is started.
    adapter.publish(AttentionResult(
        timestamp=0.0, source='my-aad', decision='unavailable'))
    adapter.publish(SyncResult(timestamp=0.0, source='my-sync'))
    stop.wait()


app = create_app(producer_factory=lambda: ResultProducer(run_results))
```

This is an example for a future integration module, not a new default app or
an automatically connected model. Launch that module with Uvicorn and the same
port 8001 when it exists. Session start invokes the coordinator in the existing
background worker; stop signals its event; failures use the existing session
error path. Constructors must remain lightweight. Do not call run_results from
a request handler. Keep each session's adapter within that worker's lifetime.

Inside an actual coordinator, after existing code has produced a result:

```python
# Existing CombinedPipeline.decide(...) return value or decision.to_dict():
for result in combined_results(decision, timestamp=session_time_seconds):
    adapter.publish(result)

# Existing novaAAD mixer output (0=A, 1=B), without a combined pipeline:
adapter.publish(aad_result(mixer.attended, correlations,
                           timestamp=session_time_seconds))

# Existing NOVA LapsePredictor.predict(...) output:
adapter.publish(lapse_result(lapse_score, timestamp=session_time_seconds))

# Explicit upstream uncertain state (no new threshold here):
adapter.publish(AttentionResult(session_time_seconds, 'my-aad', 'uncertain'))
```

For a custom producer already implementing `run(publish, stop)`, construct
`ResultAdapter(publish)` directly. The callback generates the real session ID,
sequence, and envelope through Sessions/Publisher. Do not supply your own
sequence or use the adapter's temporary validation envelope as a real packet.
Multiple results are independent events, not an atomic transaction. Invalid
results raise before publication; callers can explicitly publish unavailable
when appropriate. Error recovery/quality decisions belong to the coordinator.

## Independent boundaries

| Concern | Contract / owner | Future implementations |
|---|---|---|
| EEG input | `backend.sources.base.EEGSource`: start/read(timeout)/stop | replay EEG; live LSL EEG |
| Audio input | `backend.sources.base.AudioSource`: start/read(timeout)/stop | prerecorded audio; microphone; headset audio |
| Attention | `contracts.AttentionAlgorithm.evaluate(eeg, audio)` or legacy result converter | existing novaAAD; CombinedPipeline output |
| Vigilance | `contracts.VigilanceAlgorithm.evaluate(eeg)` or `lapse_result` | existing NOVA predictor |
| Synchronization | `contracts.Synchronization.metadata(timestamp)` / `SyncResult` | separately validated clock/alignment provider |
| Transport | `ResultProducer` → `ResultAdapter` → existing Sessions/Publisher | unchanged REST and WebSocket |

Algorithm protocols describe an optional wrapper, not a requirement to rename
existing scientific methods. Source read values deliberately remain unspecified:
channel order, units, preprocessing, sample rate and clock domain must be explicit
in the eventual acquisition/model integration contract. This phase neither
assumes they are compatible nor implements alignment/resampling. Keep source
references in the coordinator, not the result adapter or web transport. Always
stop owned sources in a finally block; reads must be bounded to honor session
cancellation. Never send raw acquisition buffers as eeg_display accidentally.

The same React REST/WebSocket client accepts packets for all these combinations.
Only its decoder recognizes the additive decision and lapse metric fields; no
transport change or visual redesign is required. Simulation flags must agree
between the ResultProducer session and supplied records; the adapter does not
invent provenance. No datasets/models were loaded to test these boundaries.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/attune-backbone-venv/bin/python -m unittest discover -s backend/tests -v
npm test --prefix frontend
npm run build --prefix frontend
```
