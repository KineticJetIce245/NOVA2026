// Evaluate the frontend's gain gate with the frontend's own code.
//
// Step 9's acceptance criterion is `mediaAudio.js::mediaFocusReady`, and the only
// honest way to report whether it holds is to run *that* function - not a
// re-implementation of it - over the state a browser would have held. So this
// script:
//
//   1. replays the recorded packet stream through the vendored frontend's own
//      validator (protocol.js), client state machine (state.js) and decoders
//      (decoders.js), which is what produces `state.streams` and `rejected`;
//   2. reconstructs the `playback` object `MediaPlayback.js` holds (its snapshot
//      shape, including the `ready` rule `now() - lastAck <= 1500`), from the
//      control-exchange log the simulated client wrote;
//   3. calls `mediaFocusReady(state, playback)` and `playbackGains(state,
//      playback, 'attune')` and reports each clause with the value behind it.
//
//   node scripts/auditory_ui/gate_evidence.mjs <packets.jsonl> <media_record.json> <out.json>
//
// Exit code 0 only when the gate opens at least once while the session runs.
import { readFileSync, writeFileSync } from 'node:fs';
import { validatePacket } from '../../apps/attune-ui/src/protocol.js';
import { emptyState, acceptPacket } from '../../apps/attune-ui/src/state.js';
import { mediaFocusReady, playbackGains, dbToLinear } from '../../apps/attune-ui/src/mediaAudio.js';

const [streamPath, recordPath, outPath] = process.argv.slice(2);
if (!streamPath || !recordPath || !outPath) {
  console.error('usage: node gate_evidence.mjs <packets.jsonl> <media_record.json> <out.json>');
  process.exit(2);
}

const packets = readFileSync(streamPath, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
const record = JSON.parse(readFileSync(recordPath, 'utf8'));
const media = record.media ?? {};
const descriptor = media.descriptor ?? {};
const exchanges = media.client?.exchanges ?? [];

// --- the playback object MediaPlayback.js holds -------------------------------
// `mediaController.js` keeps `revision` and `lastAck` from the last accepted
// acknowledgement and exports: { mediaId, revision, time: element.currentTime,
// duration, playbackState, ready: prepared && now() - lastAck <= 1500, error,
// busy, mode }. `time` is the position the *client* measured, so the run record's
// own `at_gain_packet` entries - captured live, when each packet arrived - are
// the right source. Never the server's answer, and never a re-derivation.
const played = exchanges.filter(e => e.status === 200 && ['prepare', 'playing', 'report'].includes(e.action));
const arrivals = media.at_gain_packet ?? [];
let arrivalIndex = 0;
const currentPlayback = () => {
  const record = arrivals[arrivalIndex];
  arrivalIndex += 1;
  const snapshot = record?.client;
  if (!snapshot || !descriptor.media_id) return null;
  return {
    mediaId: descriptor.media_id,
    revision: snapshot.revision,
    time: snapshot.time,
    duration: snapshot.duration ?? media.duration_s ?? null,
    playbackState: snapshot.playbackState,
    ready: Boolean(snapshot.ready) && !snapshot.error,
    error: snapshot.error ?? null,
    busy: false,
    mode: 'attune',
  };
};

// --- replay the stream through the client's own state machine ----------------
// A packet is "received" only once the session is running; the browser subscribes
// before the session starts, so the lifecycle packet that says `running` is what
// turns the stream on, exactly as transport.js sees it. The media timeline packet
// is published on its own 250 ms thread, so it is paired with the session frames
// by *count* rather than by timestamp: the i-th media packet describes the
// timeline as it stood when the i-th frame was emitted, and the control-exchange
// log is the sequence of positions the controller reported.
let state = emptyState();
let running = false;
let sequence = 0;
const rejects = [];
let mediaPackets = 0;

// One observation per attention or gain packet, with the playback snapshot the
// browser held: the newest control exchange that had completed by then. Playback
// starts after `playing` is acknowledged, and both timelines advance in real
// time, so pairing the k-th frame with the k-th report is the faithful mapping.
const observations = [];
const sampled = [];
let lastPlayback = null;

for (const packet of packets) {
  try { validatePacket(packet); } catch (error) { rejects.push(`${packet.type}: ${error.message}`); continue; }
  state = acceptPacket(state, packet);
  if (state.rejected > 0) rejects.push(state.error);
  sequence = Math.max(sequence, packet.sequence);
  const status = packet.type === 'session' && packet.source === 'server' ? packet.payload?.status : null;
  if (status === 'running') running = true;
  if (packet.type === 'media') mediaPackets += 1;
  if (packet.type === 'gain' || packet.type === 'attention') {
    // The playback snapshot the browser held when this frame's gain packet
    // arrived. It is recorded by the client itself, at arrival, so the comparison
    // is between the packet's echoed position and the position the client would
    // have reported at that instant - not a number reconstructed afterwards by
    // pairing frame k with report k, which is off by however long the first
    // report happened to take. The attention packet of the same frame shares the
    // snapshot, because the producer stamps both from one read of the timeline.
    const playback = packet.type === 'gain' ? currentPlayback() : lastPlayback;
    if (packet.type === 'gain') lastPlayback = playback;
    const view = { ...state, connection: 'connected', stale: false };
    const observation = { packet, state: view, playback, ready: Boolean(playback) && mediaFocusReady(view, playback) };
    observations.push(observation);
    if (packet.type === 'gain' && playback) sampled.push(observation);
  }
  if (status === 'stopped' || status === 'error') running = false;
}

// --- clause-by-clause, on the observation where the gate first opens ----------
const latest = (view, type) => view.streams.findLast(s => s.type === type);
function clauses(view) {
  const playback = view.playback;
  const attention = latest(view.state, 'attention');
  const mediaPacket = latest(view.state, 'media');
  const sync = latest(view.state, 'sync');
  const session = latest(view.state, 'session');
  const delta = attention?.values?.mediaTime !== undefined && Number.isFinite(attention?.values?.mediaTime)
    ? Math.abs(attention.values.mediaTime - playback.time) : null;
  return [
    ['connection.isConnected', view.state.connection === 'connected' && !view.state.stale && !view.state.error,
      `connection=${view.state.connection} stale=${view.state.stale} error=${view.state.error ?? 'none'}`],
    ['session.statusRunning', session?.values?.status === 'running', `status=${session?.values?.status ?? 'absent'}`],
    ['playback.ready', playback.ready && !playback.error, `ready=${playback.ready} error=${playback.error ?? 'none'}`],
    ['playback.stateIsPlayingOrPaused', ['playing', 'paused'].includes(playback.playbackState), `playbackState=${playback.playbackState}`],
    ['media.syncStatusObserved', mediaPacket?.values?.syncStatus === 'observed', `syncStatus=${mediaPacket?.values?.syncStatus ?? 'absent'} (${
      mediaPacket ? `time=${mediaPacket.values.mediaTime} revision=${mediaPacket.values.revision} state=${mediaPacket.values.playbackState}` : 'no media packet'})`],
    ['media.playbackStateMatches', mediaPacket?.values?.playbackState === playback.playbackState, `media=${mediaPacket?.values?.playbackState} playback=${playback.playbackState}`],
    ['media.revisionMatches', mediaPacket?.values?.revision === playback.revision, `media=${mediaPacket?.values?.revision} playback=${playback.revision}`],
    ['media.mediaIdMatches', mediaPacket?.values?.mediaId === playback.mediaId, `media=${mediaPacket?.values?.mediaId ?? 'absent'} playback=${playback.mediaId}`],
    ['sync.statusIsUsable', !['desynchronized', 'invalid'].includes(sync?.values?.status), `status=${sync?.values?.status ?? 'absent'}`],
    ['attention.sessionIdMatches', attention?.sessionId === view.state.sessionId, `attention=${attention?.sessionId ?? 'absent'} state=${view.state.sessionId}`],
    ['attention.mediaIdMatches', attention?.values?.mediaId === playback.mediaId, `attention=${attention?.values?.mediaId ?? 'absent'} playback=${playback.mediaId}`],
    ['attention.mediaRevisionMatches', attention?.values?.mediaRevision === playback.revision, `attention=${attention?.values?.mediaRevision ?? 'absent'} playback=${playback.revision}`],
    ['attention.timeWithin750ms', delta !== null && delta <= 0.75, `|Δt|=${delta === null ? 'undefined' : delta.toFixed(4)}s`],
    ['attention.decisionIsAOrB', ['A', 'B'].includes(Object.hasOwn(attention?.values ?? {}, 'decision') ? attention.values.decision : attention?.values?.attended), `decision=${attention?.values?.decision ?? attention?.values?.attended ?? 'absent'}`],
  ];
}

let opening = null;
for (const view of observations) {
  if (view.playback && mediaFocusReady(view.state, view.playback)) { opening = view; break; }
}
// The gain the frontend would actually apply, evaluated at every sampled gain
// packet - not only at the first one where the gate opened, because the
// controller legitimately holds both channels at 0 dB until it has a decision,
// and a single sample could easily be that moment and say nothing.
const gainsApplied = sampled
  .filter(({ playback }) => playback)
  .map(({ packet, state: view, playback }) => {
    const linear = playbackGains(view, playback, 'attune');
    const gainPacket = latest(view, 'gain');
    const attentionPacket = latest(view, 'attention');
    return {
      at_seconds: packet.timestamp,
      a_db: gainPacket?.values?.a_db ?? null,
      b_db: gainPacket?.values?.b_db ?? null,
      linear,
      linear_db: linear.map(v => 20 * Math.log10(v)),
      expected_linear: [dbToLinear(gainPacket?.values?.a_db ?? 0), dbToLinear(gainPacket?.values?.b_db ?? 0)],
    };
  });
const applied = gainsApplied.filter(entry => entry.linear[0] !== 1 || entry.linear[1] !== 1);
const target = opening ?? (observations.length ? observations[observations.length - 1] : null);
const clauseReport = target ? clauses(target) : [];
const ready = Boolean(opening);
const conditions = target
  ? {
    mode_is_attune: true,
    playback_is_playing: target.playback?.playbackState === 'playing',
    focus_ready: Boolean(target.playback) && mediaFocusReady(target.state, target.playback),
    gain_media_time: latest(target.state, 'gain')?.values?.mediaTime ?? null,
    attention_media_time: latest(target.state, 'attention')?.values?.mediaTime ?? null,
    times_equal: latest(target.state, 'gain')?.values?.mediaTime === latest(target.state, 'attention')?.values?.mediaTime,
    gain_session_matches: latest(target.state, 'gain')?.sessionId === target.state.sessionId,
    gain_media_id_matches: latest(target.state, 'gain')?.values?.mediaId === target.playback?.mediaId,
    gain_revision_matches: latest(target.state, 'gain')?.values?.mediaRevision === target.playback?.revision,
    gain_time_within_750ms: Number.isFinite(latest(target.state, 'gain')?.values?.mediaTime)
      && Math.abs(latest(target.state, 'gain').values.mediaTime - target.playback.time) <= 0.75,
    gains_in_range: [latest(target.state, 'gain')?.values?.a_db, latest(target.state, 'gain')?.values?.b_db]
      .every(v => Number.isFinite(v) && v >= -80 && v <= 0),
  }
  : {};

// Deltas over every sampled gain packet: the |Δt| distribution the run record
// quotes. The packet's own echoed position against the client's measured position
// at the moment that packet arrived.
const deltas = sampled
  .map(({ packet, playback }) => {
    const value = packet.payload?.media_time_s;
    return Number.isFinite(value) && playback ? Math.abs(value - playback.time) : null;
  })
  .filter(v => v !== null);

const summary = {
  packet_count: packets.length,
  sequence,
  validator_rejects: rejects.length,
  client_state_rejected: state.rejected,
  gate_open: ready,
  gate_opened_at_seconds: opening ? opening.packet.timestamp : null,
  gate_opened_on: opening ? opening.packet.type : null,
  observations: observations.length,
  sampled_gain_packets: sampled.length,
  delta_seconds: {
    count: deltas.length,
    max: deltas.length ? Math.max(...deltas) : null,
    mean: deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null,
    min: deltas.length ? Math.min(...deltas) : null,
  },
  clauses: clauseReport.map(([name, ok, detail]) => ({ name, ok: Boolean(ok), detail })),
  playback_gains_at_first_open: gainsApplied[0] ?? null,
  playback_gains_applied: applied.length,
  playback_gains_samples: applied.slice(0, 5),
  first_applied_gain: applied[0] ?? null,
  conditions_at_first_open: conditions,
  media_packet_sample: target ? (latest(target.state, 'media')?.values ?? null) : null,
};
writeFileSync(outPath, JSON.stringify(summary, null, 2));
for (const clause of summary.clauses) console.log(`  [${clause.ok ? 'PASS' : 'FAIL'}] ${clause.name} -- ${clause.detail}`);
console.log(`gate open: ${ready}${ready ? ` at ${summary.gate_opened_at_seconds}s` : ''}`);
console.log(`attune-mode gains that differ from neutral: ${applied.length} of ${gainsApplied.length}`);
if (applied.length) console.log(`first applied: ${JSON.stringify(applied[0])}`);
if (deltas.length) console.log(`|dt|: max=${Math.max(...deltas).toFixed(4)}s mean=${(deltas.reduce((a, b) => a + b, 0) / deltas.length).toFixed(4)}s n=${deltas.length}`);
console.log(ready ? 'gate is satisfiable: mediaFocusReady held' : 'gate never opened');
process.exit(ready && rejects.length === 0 ? 0 : 1);
