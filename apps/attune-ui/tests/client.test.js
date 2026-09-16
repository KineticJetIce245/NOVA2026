import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { validatePacket, validateSnapshot } from '../src/protocol.js';
import { decodePacket } from '../src/decoders.js';
import { emptyState, acceptPacket, fromSnapshot } from '../src/state.js';
import { createRestClient } from '../src/rest.js';
import { createTransport, websocketUrl } from '../src/transport.js';
import { Dashboard } from '../src/Dashboard.js';
const packet = (sequence = 1, type = 'attention', payload = {}, extra = {}) => ({ version: 1, sequence,
  type, source: 'mock', session_id: 's1', timestamp: sequence / 4, payload, ...extra });
const snapshot = (packets = [], sequence = packets.at(-1)?.sequence ?? 0, session_id = packets[0]?.session_id ?? null) => ({ packets, sequence, session_id });
const tick = () => new Promise(resolve => setImmediate(resolve));
function harness(restOverride) {
  const timers = new Map(), sockets = [], states = [];
  let timerId = 0, currentSnapshot = snapshot(), calls = 0;
  const rest = { state: async () => { calls++; return currentSnapshot; }, ...restOverride };
  const transport = createTransport({ rest, url: 'ws://local/ws/live', onState: state => states.push(state),
    schedule: (fn, delay) => { timers.set(++timerId, { fn, delay }); return timerId; }, cancel: id => timers.delete(id),
    socketFactory: () => { const socket = { close() { this.closed = true; } }; sockets.push(socket); return socket; } });
  return { transport, timers, sockets, states, setSnapshot: s => { currentSnapshot = s; }, calls: () => calls,
    runTimer() { const [id, timer] = timers.entries().next().value; timers.delete(id); timer.fn(); return timer.delay; } };
}

test('F02-T01 attention decoder preserves decisions and correlations', () => {
  const d = decodePacket(packet(1, 'attention', { attended: 'B', correlation_a: 0, correlation_b: -.3, simulated: true }));
  assert.deepEqual(d.values, { attended: 'B', correlationA: 0, correlationB: -.3, reasons: [] });
  assert.equal(d.simulated, true); assert.equal(d.known, true);
});
test('F02-T02 vigilance decoder preserves valid zero and rejects unavailable scores', () => {
  assert.equal(decodePacket(packet(1, 'vigilance', { score: 0 })).values.score, 0);
  for (const score of [undefined, null, '0.8', -1, 2]) assert.equal(decodePacket(packet(1, 'vigilance', { score })).values.score, null);
});
test('F02-T03 sync decoder preserves false zero and unknown values', () => {
  assert.deepEqual(decodePacket(packet(1, 'sync', { status: 'unknown', offset_ms: 0, drift_warning: false })).values,
    { status: 'unknown', offsetMs: 0, driftWarning: false, timeline: null, fixedLatencyMs: null });
});
test('F02-T04 EEG decoder passes display samples without computation', () => {
  const payload = { sample_rate: 32, channels: ['one'], samples: [[0, -1, .3]] };
  assert.deepEqual(decodePacket(packet(1, 'eeg_display', payload)).values,
    { sampleRate: 32, channels: ['one'], samples: [[0, -1, .3]] });
  assert.equal(decodePacket(packet(1, 'eeg_display', { samples: [['bad']] })).values.samples, null);
});
test('F02-T05 unknown types and registry extensions preserve opaque payloads', () => {
  const p = packet(1, '__proto__', { future: { measurement: null }, enabled: false });
  assert.deepEqual(decodePacket(p).values, p.payload); assert.equal(decodePacket(p).known, false);
  assert.deepEqual(decodePacket(p, new Map([['__proto__', () => ({ custom: true })]])).values, { custom: true });
});
test('F02-T06 missing optional fields remain unavailable in all decoders', () => {
  for (const type of ['attention', 'vigilance', 'sync', 'eeg_display']) {
    const d = decodePacket(packet(1, type));
    assert.equal(d.simulated, null);
    assert.ok(Object.values(d.values).every(v => v === null || (Array.isArray(v) && v.length === 0)));
  }
});
test('F02-T07 malformed envelopes and nonfinite JSON are rejected', () => {
  for (const input of ['{', 'null', '[]', {}, packet(1, 'attention', []), packet(true), packet(-1), packet(2 ** 53),
    packet(1, '', {}), packet(1, 'x', {}, { version: 2 }), packet(1, 'x', {}, { timestamp: NaN }),
    packet(1, 'x', {}, { timestamp: -1 }), packet(1, 'x', {}, { source: null }), packet(1, 'x', { n: Infinity })]) {
    assert.throws(() => validatePacket(input));
  }
  const original = packet(1, 'future', { nested: { x: null } });
  const copy = validatePacket(original); copy.payload.nested.x = 1;
  assert.equal(original.payload.nested.x, null);
});
test('F02-T08 packet size depth and identifier bounds are enforced', () => {
  assert.throws(() => validatePacket(packet(1, 'x', { text: 'x'.repeat(262144) })));
  assert.throws(() => validatePacket(packet(1, 'x'.repeat(129))));
  let deep = {}; for (let i = 0; i < 34; i++) deep = { deep };
  assert.throws(() => validatePacket(packet(1, 'x', deep)));
});
test('F02-T09 duplicate and out-of-order sequences cannot replace state', () => {
  const state = acceptPacket(emptyState(), packet(4, 'attention', { attended: 'A' }));
  for (const sequence of [4, 3]) {
    const next = acceptPacket(state, packet(sequence, 'attention', { attended: 'B' }));
    assert.equal(next.streams[0].values.attended, 'A'); assert.equal(next.sequence, 4); assert.equal(next.rejected, 1);
  }
  assert.equal(acceptPacket(state, packet(9)).sequence, 9); // Latest-state replay legitimately skips sequences.
});
test('F02-T10 timestamps are ordered per stream and failures are atomic', () => {
  const state = acceptPacket(emptyState(), packet(5));
  const next = acceptPacket(state, packet(6, 'attention', {}, { timestamp: 0 }));
  assert.equal(next.sequence, 5); assert.deepEqual(next.streams, state.streams);
  assert.equal(acceptPacket(state, packet(6, 'sync', {}, { timestamp: 0 })).sequence, 6);
});
test('F02-T11 new sessions clear old streams and snapshots permit process reset', () => {
  const state = acceptPacket(emptyState(), packet(20));
  const next = acceptPacket(state, packet(21, 'sync', {}, { session_id: 's2', timestamp: 0 }));
  assert.equal(next.streams.length, 1); assert.equal(next.sessionId, 's2');
  const reset = fromSnapshot(snapshot([packet(1, 'vigilance', {}, { session_id: 'restart' })]));
  assert.equal(reset.sequence, 1); assert.equal(reset.streams.length, 1);
});
test('F02-T12 malformed inconsistent snapshots are rejected', () => {
  for (const s of [null, {}, snapshot([packet(2)], 1), snapshot([packet(2), packet(1)], 2),
    snapshot([packet(1)], 1, 'other'), snapshot([packet(1), packet(2)], 2), snapshot([], -1)]) assert.throws(() => validateSnapshot(s));
});
test('F02-T13 latest state bounds source streams', () => {
  let state = emptyState();
  for (let i = 1; i <= 140; i++) state = acceptPacket(state, packet(i, `future-${i}`));
  assert.equal(state.streams.length, 128); assert.equal(state.streams[0].sequence, 13);
});
test('F02-T14 REST methods paths and HTTP failures', async () => {
  const calls = [], signal = new AbortController().signal;
  const rest = createRestClient(async (url, options) => { calls.push([url, options]); return { ok: true, json: async () => ({ ok: true }) }; });
  await rest.health(signal); await rest.state(signal); await rest.start(signal); await rest.stop(signal);
  assert.deepEqual(calls.map(([url, o]) => [url, o.method]), [['/api/health', 'GET'], ['/api/state', 'GET'], ['/api/session/start', 'POST'], ['/api/session/stop', 'POST']]);
  assert.ok(calls.every(([, o]) => o.signal === signal && o.cache === 'no-store'));
  await assert.rejects(createRestClient(async () => ({ ok: false, status: 503 })).start(), /503/);
});
test('F02-T15 reconnect resnapshots and ignores previous socket callbacks', async () => {
  const h = harness(); h.setSnapshot(snapshot([packet(5)])); h.transport.start(); await tick();
  const old = h.sockets[0]; old.onopen(); const oldMessage = old.onmessage;
  old.onclose(); assert.equal(h.transport.getState().stale, true);
  h.setSnapshot(snapshot([packet(1, 'sync', {}, { session_id: 'restart' })]));
  assert.equal(h.runTimer(), 500); await tick(); h.sockets[1].onopen();
  oldMessage({ data: JSON.stringify(packet(99)) });
  assert.equal(h.transport.getState().sequence, 1); assert.equal(h.transport.getState().sessionId, 'restart');
  h.sockets[1].onmessage({ data: JSON.stringify(packet(2, 'vigilance', { score: .5 }, { session_id: 'restart' })) });
  assert.equal(h.transport.getState().sequence, 2); assert.equal(h.calls(), 2);
  h.transport.stop(); assert.equal(h.timers.size, 0); assert.ok(h.sockets.every(s => s.closed));
});
test('F02-T16 failed REST retries with bounded backoff and stop cancels', async () => {
  const h = harness({ state: async () => { throw Error('offline'); } });
  h.transport.start(); await tick();
  for (const delay of [500, 1000, 2000, 4000, 8000, 8000]) { assert.equal(h.runTimer(), delay); await tick(); }
  assert.equal(h.transport.getState().connection, 'reconnecting'); assert.equal(h.sockets.length, 0);
  h.transport.stop(); assert.equal(h.timers.size, 0);
});
test('F02-T17 delayed snapshots and timeouts cannot revive stopped clients', async () => {
  let resolve;
  const h = harness({ state: () => new Promise(r => { resolve = r; }) });
  h.transport.start(); assert.equal(h.runTimer(), 10000);
  assert.equal(h.transport.getState().connection, 'reconnecting');
  h.transport.stop(); resolve(snapshot()); await tick(); assert.equal(h.sockets.length, 0);
  assert.equal(h.transport.getState().connection, 'disconnected');
});
test('F02-T18 socket errors malformed frames and secure URLs', async () => {
  assert.equal(websocketUrl({ protocol: 'https:', host: 'example:443' }), 'wss://example:443/ws/live');
  const h = harness(); h.transport.start(); h.transport.start(); await tick(); assert.equal(h.sockets.length, 1);
  h.sockets[0].onopen(); h.sockets[0].onmessage({ data: '{' }); h.sockets[0].onmessage({ data: new ArrayBuffer(2) });
  assert.equal(h.transport.getState().rejected, 2); assert.equal(h.transport.getState().streams.length, 0);
  h.sockets[0].onerror(); assert.equal(h.transport.getState().stale, true); h.transport.stop();
});
test('F02-T19 Python mock packets decode across language boundary', () => {
  const code = 'import json\nfrom backend.adapters.mock import results\nfrom backend.app.protocol import make_packet\nprint(json.dumps([make_packet(k,t,i+1,s,"fixture",p) for i,(k,t,s,p) in enumerate(results(0))]))';
  const packets = JSON.parse(execFileSync(process.env.ATTUNE_PYTHON || 'python3', ['-B', '-c', code], { cwd: new URL('../../', import.meta.url), encoding: 'utf8' }));
  let state = emptyState(); for (const p of packets) state = acceptPacket(state, p);
  assert.equal(state.rejected, 0); assert.equal(state.streams.length, 8);
  assert.deepEqual(state.streams.filter(s => s.known).map(s => s.type), ['audio_sources', 'attention', 'vigilance', 'signal_quality', 'sync', 'eeg_display', 'gain', 'prediction']);
  assert.equal(state.streams.find(s => s.type === 'signal_quality').values.quality, null);
});
test('F02-T20 debug dashboard renders status unavailable values and escaped payloads', () => {
  const state = acceptPacket(emptyState(), packet(1, 'future', { measurement: null, text: '<script>bad()</script>' }));
  const html = renderToStaticMarkup(React.createElement(Dashboard, { state, command() {} }));
  for (const label of ['ATTUNE', 'STALE', 'Unknown packet fallback', 'null', 'Start mock session', 'disabled']) assert.ok(html.includes(label));
  assert.ok(!html.includes('<script>')); assert.ok(html.includes('&lt;script&gt;'));
});
test('F02-T21 restart between REST snapshot and socket forces a fresh snapshot', async () => {
  const h = harness(); h.setSnapshot(snapshot([packet(50)])); h.transport.start(); await tick();
  h.sockets[0].onopen();
  h.sockets[0].onmessage({ data: JSON.stringify(packet(1, 'attention', {}, { session_id: 'restart' })) });
  assert.equal(h.transport.getState().connection, 'reconnecting'); assert.equal(h.transport.getState().stale, true);
  h.setSnapshot(snapshot([packet(1, 'attention', {}, { session_id: 'restart' })]));
  h.runTimer(); await tick(); h.sockets[1].onopen();
  assert.equal(h.transport.getState().sessionId, 'restart'); assert.equal(h.transport.getState().sequence, 1);
  h.transport.stop();
});
test('F02-T22 replay duplicates do not replace newer snapshot measurements', async () => {
  const h = harness(); h.setSnapshot(snapshot([packet(8, 'attention', { attended: 'B' })]));
  h.transport.start(); await tick(); h.sockets[0].onopen();
  h.sockets[0].onmessage({ data: JSON.stringify(packet(7, 'attention', { attended: 'A' })) });
  assert.equal(h.transport.getState().streams[0].values.attended, 'B');
  h.sockets[0].onmessage({ data: JSON.stringify(packet(9, 'attention', { attended: 'A' })) });
  assert.equal(h.transport.getState().streams[0].values.attended, 'A'); h.transport.stop();
});
