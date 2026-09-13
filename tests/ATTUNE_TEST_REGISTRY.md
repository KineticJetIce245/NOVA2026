# ATTUNE test registry

All tests run from the project environment with `PYTHONPATH=src;.` on Windows.

| Test | Coverage |
| --- | --- |
| test_mixers_reject_unsafe_input (15 parameter cases) | Unsafe audio/gain dimensions, bounds and nonfinite values in all mixers |
| test_stereo_routing_ducking_and_mono_equivalence | Dichotic A/left B/right, 6 dB steady attenuation, bit-identical diotic/mono |
| test_online_equivalence_causality_and_bounded_ring | Over 60 windows identical to offline reference, causal edge, bounded history, malformed blocks |
| test_gate_null_quantile_centering_and_dwell_expiry | Two-sided null tail, centering, signed selection, three-window dwell and expiry |
| test_gate_rejects_inadequate_null (4 cases) | Empty, degenerate, nonfinite and short calibration |
| test_gate_precision_is_unavailable_without_fires | No invented precision when abstaining |
| test_candidate_pcm_layout_and_unknown_truth | Pair/stereo PCM equality, channel layout, unknown labels, collapsed candidate rejection |
| test_ledger_masks_boundaries_and_refuses_overlap | Cue transition masking and nonoverlapping epochs |
| test_montage_profile_operator_declaration | eego24 labels, CPz/Fpz declaration and missing-channel failure |
| test_data_package_does_not_import_torch | Lightweight AAD package import |
| test_flat_channel_requires_explicit_policy_and_survives_save | Named flat channel opt-in, fixed montage and zero influence after reload |
| test_replay_policy_and_offset_forwarding | Offline quality flags and audio onset offset reach the shared chain |
| test_onset_comes_from_first_dac_callback_and_both_arms | DAC source-clock marker, not device-open time, both calibration arms |
| test_xdf_requires_labels_and_audio_anchor | Missing channel labels/session:start-only recordings fail closed |
| test_xdf_preserves_samples_unknown_truth_and_offset_roundtrip | Grid sample count, unknown labels, persisted sub-sample onset |
| LiveAudioTests.test_timestamped_lsl_to_paced_audio | Actual LSL loop with a 1 ms consumer callback |
| LiveAudioTests.test_faulting_consumer_cannot_stop_acquisition | Actual LSL loop continues through every consumer exception |
| scripts.attune.test_end_to_end | Real CNT-derived data, separate finite PlayerLSL, 1 Hz estimates, HTTP/WS order/provenance, stereo WAV and actual JS decoder |

Unit cases are in `tests/test_attune.py`, `tests/test_attune_calibration.py`;
live callback cases are in `scripts/auditory/tests/test_live_lsl.py`.
Real-data fixtures remain local in datasets/experiment and datasets/attune.
See `documents/ATTUNE_IMPLEMENTATION.md` for commands and limitations.


## One-repository integration

`tests/test_attune_repository.py` checks isolated imports without PYTHONPATH,
production UI assets/API/WebSocket on one server, and absence of sibling-repository
launcher dependencies. All imported backend cases are in backend/tests/ (60 tests);
all frontend cases are in frontend/tests/ (55 tests), run with npm test --prefix frontend.
These include NOVA record validation, producer lifecycle, simulated=false provenance,
null timing fields and ear mapping, previously tested in the supplied UI checkout.
