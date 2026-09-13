# Development-only feedback

`PredictionResult -> FeedbackPolicy -> FeedbackAction -> Publisher/WS -> FeedbackPanel`

`backend/app/feedback.py` defines the generic `FeedbackPolicy` protocol,
`FeedbackAction` record and `DevelopmentFeedbackPolicy`. The session worker's
publish callback rehydrates an existing prediction packet into PredictionResult,
evaluates the policy, and publishes a separate `feedback` packet. Provider/model
code and the deterministic mock generator do not contain feedback decisions.
The original prediction packet is preserved. Evaluations/publications are serialized
per session. A new session gets a fresh policy; no global cooldown leaks across sessions.

## FeedbackAction contract

- `status`: none, active, suppressed
- `action_type`: visual, audio, haptic, control, none
- `message`: display text
- `severity`: info, warning, critical
- `source_provider`, `source_task`: original result identity (unknown if malformed)
- `timestamp`: nonnegative session-relative event time
- `reason`: machine-readable policy outcome
- `metadata`: policy identifier, development_only, scientific_interpretation=false,
  cooldown_seconds, cooldown_remaining_seconds, last_active_timestamp
- `simulated`: true for every action from this development policy

Transport uses the unchanged version-1 envelope, type=feedback and
source=development-feedback; session/sequence remain publisher-owned. Audio,
haptic and control are contract vocabulary only. No actuators or audio playback
are implemented. Policy output is not an attention/vigilance assessment.

## Deterministic policy

Only status=ok, provider_id=mock, task=transport_demo, metadata.mock=true and
metadata.scientific_interpretation=false are eligible. Exactly one finite numeric
`development_value` output with semantic_type=development is required. Result
and event timestamps must agree. Booleans, strings, missing/nonfinite values,
malformed results and invalid/unavailable/error statuses never activate feedback.
Nonmock providers always produce suppression; this is not a real-feedback policy.

- Value < 8: none / info.
- 8 <= value < 24: active / visual / warning, if cooldown permits.
- Value >= 24: active / visual / critical, if cooldown permits.

These numbers exercise UI states only. They have no scientific interpretation.
Cooldown is five seconds of monotonic **session event time**, not a wall-clock
actuation guarantee for accelerated replay. Default mock pacing advances event
time with its 250ms loop. Suppressed/low/invalid results do not reset last-active
time. At exactly five seconds a new event is eligible; a severity increase does
not bypass cooldown. Duplicate/regressing timestamps suppress activation; emitted
feedback time is clamped to the policy's latest event time to avoid a regressing
feedback stream. Latest feedback replaces prior feedback in the state snapshot;
active is an event, not a latched instruction. Subsequent packets can show cooldown.

## Frontend

FeedbackPanel is always present, showing a SIMULATED / DEVELOPMENT ONLY badge,
status, message, severity, timestamp, reason and cooldown remaining as of the
latest event (not a browser-computed timer). Active feedback has a contrasting
border/background and alert semantics. Stale/disconnected or stopped/error sessions
do not highlight cached feedback as active. No thresholds live in React.
Unknown/unmarked feedback is suppressed by display validation. Existing prediction
and other result displays remain available. No manual bypass/control was added:
Start mock session already produces all needed test ranges (warning at about 2s,
stronger feedback at about 7s after cooldown).

Restart the backend process to load this Python change, then start a mock session.
Vite reloads the frontend changes. No hardware or external services are needed.

## Verification

From `/Users/yutong/Documents/New project/NOVA`:

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/attune-backbone-venv/bin/python -m unittest discover -s backend/tests -v
npm test --prefix frontend
npm run build --prefix frontend
```

Tests cover ranges, invalid/unavailable/error/nonmock results, bad numeric values,
exact cooldown edge, stale time, session reset, packet/WS delivery and rendered
active/none/suppressed/stale/development labels. No physical response is tested.

## Real integration remains separate

Review target semantics and provenance, choose validated policy parameters,
clock/expiry rules, per-provider arbitration, user controls and actuator lifecycle
before enabling real feedback. Integrate through FeedbackPolicy rather than moving
thresholds into a model or frontend. No NOVA2026/novaAAD model connection was added.
