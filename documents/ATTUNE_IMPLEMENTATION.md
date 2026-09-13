# ATTUNE implementation and reproduction

The complete application is in **NOVA2026 on its ordinary `audio_flo` branch**.
`backend/` contains FastAPI and the NOVA adapter; `frontend/` contains React/Vite.
The current UI styling is preserved. Neither attune-ui nor novaAAD is a runtime
dependency or publishing destination. No recordings or model weights are committed.

## Implemented wiring

| Handoff item | Implementation |
| --- | --- |
| S1 | Live estimate callback isolates consumer exceptions and counts them; cooperative session stop |
| S2 | Incremental envelope provider, bounded feature and timestamp histories, original candidate blocks |
| S3 | UI `backend/adapters/nova_live.py`, ResultProducer and environment-configured server |
| S4 | Median/MAD normalization, two-sided null quantile, model metadata, fire rate/precision; existing dwell and expiry |
| S5 | Live TimeBase grid/off, XDF grid placement, calibration/XDF tools in this same checkout |
| S6 | Source-clock epoch ledger, first-DAC-block JSON markers, optional nonblocking marker receiver and cue mask |
| S7 | 2 Hz prepared EEG display, software sync diagnostics, actual mixer gains, channel quality and presentation |
| S8 | Offline/training tolerant-channel flags; faults remain visible |
| S9 | Dichotic and diotic stereo mixers, mono replay API preserved, arm recorded in trial/model/report/UI |
| S10 | Mono-pair/stereo ingest, candidate independence guard, unknown live labels |
| S11 | eego24 order, operator-declared CPz reference/Fpz ground, positional labels forbidden in real playback/XDF |
| S12 | ANT reader extra, explicit recording anchors/exclusions, joint stimulus slicing and disjoint segment groups |
| W1/W3a/W4/W5 | 5 s trained model, saved onset offsets and CLI override, lazy torch imports and in-repository backend dependencies |

Participant playback additionally requires calibration in the selected arm, a
null-calibrated gate, and the same validated measured timing profile as training.
Tiny diagnostic datasets still fit reconstruction weights but get `gate_unavailable`;
the dashboard and physical playback refuse those uncalibrated bundles.

## Local real data findings

The supplied files differ from two handoff claims. Session 2 has **12 left / 11
right** cues after removing the specified accidental markers, not 12/12. The
ambiguous 148.354–161.224 s interval is masked unknown. F8 is flat in session 1;
training requires `--allow-flat-channels F8`, stores the disabled channel, and
ignores it consistently after model serialization. The montage remains 24 channels.
Unexpected flat channels still fail training.

Raw data is converted from volts to microvolts and centered per session. Large
excursions remain (session 1 absolute 95th percentile about 1,525 µV); the demo
uses an explicit record-only artifact policy. It is not evidence of clean cap data.
Physical live sources must supply the declared upstream processing independently;
offline session centering is not a causal live high-pass implementation.

Training uses session 1 stimulus [0,267); development validation uses session 2
[306,582). Both discard the shared interval. This tests disjoint segments of the
same pair and participant, not transfer to another pair/person/cap.

Selected ridge alpha: 100. Development validation: 10/26 correct (38.46%).
The null gate used 53 windows, a 1% false-fire **target**, and fired on 1/26
validation windows (3.85%), with that one decision correct. The validation set
also selected alpha. Neither this precision nor the tail target establishes a
population guarantee. A three-window dwell may mean no real-data gain commitment.
Unit tests separately verify the controller's commitment and exact 6 dB ducking.

## Reproduce on Windows

From NOVA2026 with the project virtual environment and UI backend requirements:

```powershell
$env:PYTHONPATH = 'C:/coding/git/NOVA2026/src;C:/coding/git/NOVA2026'
.venv/Scripts/python.exe -m pip install -e '.[eeg-ant,audio,calibration]' -r backend/requirements.txt
.venv/Scripts/python.exe -m scripts.attune.convert_ant --out datasets/attune
.venv/Scripts/python.exe -m scripts.auditory.train --train datasets/attune/Lacroix_Flo2_2026-09-12_19-34-06.npz --validation datasets/attune/Lacroix_Flo2_2026-09-12_19-41-11.npz --model models/attune_eego24.npz --history 5 --no-channel-check --allow-flat-channels F8
.venv/Scripts/python.exe -m scripts.attune.serve --trial datasets/attune/Lacroix_Flo2_2026-09-12_19-41-11.npz --model models/attune_eego24.npz
```

Before launching, run `npm ci --prefix frontend` and `npm run build --prefix frontend`.
Open http://127.0.0.1:8001 and start a session. FastAPI serves the built dashboard,
API and WebSocket together. For frontend development only, `npm run dev --prefix
frontend` starts Vite on port 5173 and proxies to the same backend on port 8001. Each session starts
a separate, finite PlayerLSL process with 32 ms chunks and an unused real-data tail.
Stop cleans up the player, acquisition and audio worker. Reports and stereo WAVs
are under `results/attune-server/runs`. WAV output is paced but inaudible.
Use `--presentation diotic` to exercise the other stereo software route; the EEG
is still from the dichotic recording and is labelled recorded data in the UI.

For a physical amplifier, launch the UI backend with `ATTUNE_NOVA_TRIAL`,
`ATTUNE_NOVA_MODEL`, `ATTUNE_NOVA_STREAM`, `ATTUNE_NOVA_OUTPUT=play`, and
`ATTUNE_NOVA_TIMING_PROFILE` set to the measured profile. Optional
`ATTUNE_NOVA_MARKERS=AAD_Markers` enables cue-aware window rejection.
Do not set `ATTUNE_NOVA_PLAYER_RAW` for a physical source.

## Automated verification

```powershell
npm ci --prefix frontend
npm run build --prefix frontend
.venv/Scripts/python.exe -m pytest tests scripts/auditory/tests backend/tests -q
.venv/Scripts/python.exe -m unittest discover -s backend/tests -q
$env:ATTUNE_PYTHON = 'C:/coding/git/NOVA2026/.venv/Scripts/python.exe'
npm test --prefix frontend
.venv/Scripts/python.exe -m scripts.attune.test_end_to_end --trial datasets/attune/Lacroix_Flo2_2026-09-12_19-41-11.npz --model models/attune_eego24.npz --seconds 240 --out results/attune-240
.venv/Scripts/python.exe -m scripts.attune.test_end_to_end --trial datasets/attune/Lacroix_Flo2_2026-09-12_19-41-11.npz --model models/attune_eego24.npz --seconds 35 --presentation diotic --stop-after 8 --managed-player --out results/attune-managed
```

Use a fresh output directory per end-to-end test. It asserts finite stereo output,
real scores, 1 Hz evidence, packet provenance/order, REST state, WebSocket delivery,
zero gaps/recovery/consumer faults, and passes actual network packets into the
shipped JavaScript decoder. The recorded-EEG test checks transport; it does not
claim source playback onset is synchronized enough to evaluate classification.

## Requires the rig / new data

W2 physical loopback residual, participant playback, new diotic calibration and
the handoff's participant-level definition of done remain **unverified**. No zero
offset or acoustic latency has been fabricated. Sync offset/drift-warning remain
null. W3 cross-cap/base-model transfer is unverified; incompatible priors fail.
The available real-data model does not demonstrate reliable attention decoding
or hearing benefit. A new well-controlled calibration with longer blocks and at
least two distinct audio pairs is still needed before a participant demo.

The UI source was incorporated from the supplied attune-ui checkout at commit
`c331022`, retaining its design and tests. Calibration and XDF tools are implemented
directly in `scripts/attune/`; no separate engine snapshot is required. Only
the `audio_flo` branch in NOVA2026 is published. The earlier sibling-repository commits were never
pushed and are not required to install, test or run this branch.
