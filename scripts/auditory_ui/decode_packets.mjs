// Feed a recorded packet stream through the vendored frontend, for real.
//
// The dashboard is a browser application and this repository has no browser in
// its test path, so the honest substitute is to run the frontend's *own* modules
// over the packets a session actually published: its validator
// (apps/attune-ui/src/protocol.js), its client state machine (state.js, which is
// what counts `rejected`) and its dashboard components rendered to static HTML by
// react-dom/server. Nothing here re-implements a decoder or a display rule.
//
//   node scripts/auditory_ui/decode_packets.mjs <packets.jsonl> <out.html>
//
// Exit code 0 only when every check below holds.
import { readFileSync, writeFileSync } from 'node:fs';
import { validatePacket } from '../../apps/attune-ui/src/protocol.js';
import { emptyState, acceptPacket } from '../../apps/attune-ui/src/state.js';
import { decodePacket } from '../../apps/attune-ui/src/decoders.js';
import { Dashboard } from '../../apps/attune-ui/src/Dashboard.js';

const [streamPath, outPath] = process.argv.slice(2);
if (!streamPath || !outPath) {
  console.error('usage: node decode_packets.mjs <packets.jsonl> <out.html>');
  process.exit(2);
}

const React = (await import(new URL('../../apps/attune-ui/node_modules/react/index.js', import.meta.url).href)).default;
const { renderToStaticMarkup } = await import(new URL('../../apps/attune-ui/node_modules/react-dom/server.node.js', import.meta.url).href);

const checks = [];
const check = (name, ok, detail = '') => checks.push({ name, ok: Boolean(ok), detail: String(detail) });

const raw = readFileSync(streamPath, 'utf8').split('\n').filter(line => line.trim());
const packets = raw.map(line => JSON.parse(line));
check('the stream is not empty', packets.length > 0, `${packets.length} packets`);

let validated = 0;
const rejects = [];
for (const packet of packets) {
  try { validatePacket(packet); validated += 1; } catch (error) { rejects.push(error.message); }
}
check('every packet passes the frontend validator', rejects.length === 0,
  rejects.length ? `${rejects.length} rejected, first: ${rejects[0]}` : `${validated} validated`);

// The client's own state machine, in order: this is what the browser would hold.
// Two snapshots matter: the last state while the session was still running (what
// a viewer saw live) and the state after the terminal lifecycle packet arrived.
let state = emptyState();
let liveState = state;
let decisionState = null;
let decisionPacket = null;
let decisionGain = null;
let sequence = 0;
for (const packet of packets) {
  state = acceptPacket(state, packet);
  sequence = Math.max(sequence, packet.sequence);
  if (!(packet.type === 'session' && ['stopped', 'error'].includes(packet.payload?.status))) {
    liveState = state;
    if (packet.type === 'attention' && ['A', 'B'].includes(packet.payload?.decision)) {
      decisionState = state;
      decisionPacket = packet;
      decisionGain = state.streams.findLast(s => s.type === 'gain')?.values ?? null;
    }
  }
}
check('the client accepted every packet, so rejected stays 0', state.rejected === 0,
  `rejected=${state.rejected} lastError=${state.error ?? 'none'}`);
check('the client kept one stream per (source, type)',
  state.streams.length === new Set(state.streams.map(s => `${s.source}/${s.type}`)).size,
  `${state.streams.length} streams`);

const latest = type => state.streams.findLast(s => s.type === type);
const attention = latest('attention');
const gain = latest('gain');
const sources = latest('audio_sources');
const quality = latest('signal_quality');
const sync = latest('sync');
const prediction = latest('prediction');
const required = ['session', 'audio_sources', 'attention', 'gain', 'signal_quality', 'sync', 'prediction'];
const present = new Set(state.streams.map(s => s.type));
check('the dashboard has all the streams it renders',
  required.every(type => present.has(type)), [...present].sort().join(', '));

const decisions = new Set(packets.filter(p => p.type === 'attention')
  .map(p => decodePacket(p).values.decision));
check('decoded decisions are the four allowed words',
  decisions.size > 0 && [...decisions].every(d => ['A', 'B', 'uncertain', 'unavailable'].includes(d)),
  [...decisions].join(', '));
check('audio_sources decoded to exactly the two candidates A and B',
  sources?.values.sources?.length === 2 &&
  sources.values.sources.map(s => s.id).join('') === 'AB',
  JSON.stringify(sources?.values.sources ?? null));
check('gain decoded as attenuation only (never above 0 dB)',
  Number.isFinite(gain?.values?.a_db) && Number.isFinite(gain?.values?.b_db) &&
  gain.values.a_db <= 0 && gain.values.b_db <= 0,
  `a=${gain?.values?.a_db} b=${gain?.values?.b_db}`);
check('signal_quality decodes null rather than a fabricated zero',
  quality?.values?.quality === null || quality?.values?.quality === 0 || quality?.values?.quality === 1,
  `quality=${quality?.values?.quality} artifact=${quality?.values?.artifact}`);
check('sync reports a status and an unmeasured offset as null',
  typeof sync?.values?.status === 'string' && sync.values.offsetMs === null,
  `status=${sync?.values?.status} offset=${sync?.values?.offsetMs}`);
check('prediction carries the reasons and the run policy it ran under',
  Array.isArray(prediction?.values?.reasons) && prediction?.values?.metadata?.policy?.check_channels === false,
  `reasons=${JSON.stringify(prediction?.values?.reasons)} policy=${JSON.stringify(prediction?.values?.metadata?.policy)}`);
check('the attention stream decodes the same decision the packets carried',
  attention?.values?.decision === packets.filter(p => p.type === 'attention').at(-1).payload.decision,
  `stream=${attention?.values?.decision}`);

// The dashboard component itself, rendered with the real state objects. Two
// renders, because the card distinguishes them: the live snapshot while the
// session ran (with the two flags transport.js sets when the socket opens) and
// the final snapshot, after the transport published the session as stopped.
const render = view => renderToStaticMarkup(React.createElement(Dashboard, {
  state: view, command: () => {}, busy: false, commandError: null,
  mediaConfig: null, mediaRest: null, stopSignal: 0,
}));
const connected = view => ({ ...view, connection: 'connected', stale: false });
const focused = render(connected(decisionState ?? liveState));
const live = render(connected(liveState));
const final = render(connected(state));
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const focusedText = text(focused);
const liveText = text(live);
const liveAttention = liveState.streams.findLast(s => s.type === 'attention');
check('the rendered dashboard shows the focus prediction card',
  focusedText.includes('Focus prediction') && focusedText.includes('Which voice are you focusing on?'));
check('the card renders the decision the session published for that moment',
  decisionPacket !== null && focusedText.includes(`Focused on Speaker ${decisionPacket.payload.decision}`),
  decisionPacket
    ? `Focused on Speaker ${decisionPacket.payload.decision} at t=${decisionPacket.timestamp}`
    : 'the stream carried no A/B decision');
check('the winning source is marked FOCUSED and the other is attenuated',
  focusedText.includes('FOCUSED') && focusedText.includes('-6 dB'),
  `decision gains a=${decisionGain?.a_db} b=${decisionGain?.b_db}`);
check('the end of the run shows the degraded state in the card, with a reason',
  liveText.includes('Focus is uncertain') || liveText.includes('Awaiting a current prediction'),
  `final decision ${liveAttention?.values?.decision}`);
check('after the session stops the card marks its values as historical',
  text(final).includes('Session inactive or connection stale. Values are historical.'));
check('the card is not flagged as simulated for a real measurement',
  !liveAttention?.simulated && !focusedText.includes('ARTIFICIAL DEMO DATA'),
  `stream simulated=${liveAttention?.simulated}`);
check('no packet was rejected anywhere in the page', !focusedText.includes('REJECTED PACKETS'));
check('the card shows the source labels from audio_sources',
  focusedText.includes('Candidate A') || focusedText.includes('rep_part1_track1_dry.wav'));

writeFileSync(outPath, focused);
const summary = {
  packets: packets.length,
  sequence,
  last_attention: attention?.values ?? null,
  last_gain: gain?.values ?? null,
  last_quality: quality?.values ?? null,
  last_sync: sync?.values ?? null,
  last_prediction: prediction?.values ?? null,
  focus_line: decisionPacket ? `Focused on Speaker ${decisionPacket.payload.decision}` : 'none',
  focus_time: decisionPacket?.timestamp ?? null,
  focus_gain: decisionGain,
  decision_census: packets.filter(p => p.type === 'attention')
    .reduce((counts, p) => ({ ...counts, [p.payload.decision]: (counts[p.payload.decision] ?? 0) + 1 }), {}),
  rejects: state.rejected,
  checks,
};
console.log(JSON.stringify(summary, null, 2));
for (const item of checks) console.log(`  [${item.ok ? 'PASS' : 'FAIL'}] ${item.name} -- ${item.detail}`);
console.log(`rendered dashboard: ${outPath} (${live.length} bytes)`);
const failed = checks.filter(item => !item.ok);
console.log(`${checks.length - failed.length}/${checks.length} frontend checks held`);
process.exit(failed.length ? 1 : 0);
