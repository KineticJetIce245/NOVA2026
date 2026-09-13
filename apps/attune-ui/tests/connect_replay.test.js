// The connect handshake is REST-then-socket, and the server replays its retained
// snapshot to every new socket on purpose (src/nova2026/transport/server.py closes
// the subscribe race that way). REST sets the high-water mark first, so every
// replayed packet is at or below `state.sequence` and `acceptPacket`'s duplicate
// guard used to count all of them: a healthy page showed "REJECTED PACKETS: 3 --
// measurements may be missing or stale" for a legitimate protocol pattern.
//
// These tests pin both halves of the fix: the replay is absorbed, and the guard
// that catches genuine mid-stream duplicates and out-of-order packets still
// counts them.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createTransport } from '../src/transport.js';
const packet = (sequence = 1, type = 'attention', payload = {}, extra = {}) => ({ version: 1, sequence,
  type, source: 'mock', session_id: 's1', timestamp: sequence / 4, payload, ...extra });
const snapshot = (packets = [], sequence = packets.at(-1)?.sequence ?? 0, session_id = packets[0]?.session_id ?? null) => ({ packets, sequence, session_id });
const tick = () => new Promise(resolve => setImmediate(resolve));
// Same shape as tests/client.test.js: a fake REST client, fake sockets and fake
// timers, so the handshake is exercised without a DOM, a network or a clock.
function harness(restOverride) {
  const timers = new Map(), sockets = [], states = [];
  let timerId = 0, currentSnapshot = snapshot(), calls = 0;
  const rest = { state: async () => { calls++; return currentSnapshot; }, ...restOverride };
  const transport = createTransport({ rest, url: 'ws://local/ws/live', onState: state => states.push(state),
    schedule: (fn, delay) => { timers.set(++timerId, { fn, delay }); return timerId; }, cancel: id => timers.delete(id),
    socketFactory: () => { const socket = { close() { this.closed = true; } }; sockets.push(socket); return socket; } });
  return { transport, timers, sockets, states, setSnapshot: s => { currentSnapshot = s; }, calls: () => calls };
}
// The server's replay: the retained per-stream snapshot, re-sent verbatim, in
// order. This is what server.py writes to a socket the moment it is accepted.
const replay = (socket, packets) => { for (const p of packets) socket.onmessage({ data: JSON.stringify(p) }); };

test('F02-T23 the connect-time snapshot replay is not counted as a rejection', async () => {
  const h = harness();
  const retained = [packet(1, 'attention', { attended: 'A' }), packet(2, 'vigilance', { score: .5 }), packet(3, 'sync', { status: 'locked' })];
  h.setSnapshot(snapshot(retained));
  h.transport.start(); await tick(); h.sockets[0].onopen();
  replay(h.sockets[0], retained);
  const state = h.transport.getState();
  assert.equal(state.rejected, 0);
  assert.equal(state.error, null);
  assert.equal(state.sequence, 3);
  assert.equal(state.streams.length, 3);
  assert.equal(state.connection, 'connected');
  h.transport.stop();
});

test('F02-T24 a genuine duplicate mid-stream is still counted as rejected', async () => {
  const h = harness();
  const retained = [packet(1, 'attention', { attended: 'A' }), packet(2, 'vigilance', { score: .5 }), packet(3, 'sync', { status: 'locked' })];
  h.setSnapshot(snapshot(retained));
  h.transport.start(); await tick(); h.sockets[0].onopen();
  replay(h.sockets[0], retained);
  assert.equal(h.transport.getState().rejected, 0);
  // The stream moves past the replayed high-water mark: the replay window is shut.
  h.sockets[0].onmessage({ data: JSON.stringify(packet(4, 'attention', { attended: 'B' })) });
  assert.equal(h.transport.getState().sequence, 4);
  // A duplicate of the newest packet: repeats state already shown, still a reject.
  h.sockets[0].onmessage({ data: JSON.stringify(packet(4, 'attention', { attended: 'B' })) });
  assert.equal(h.transport.getState().rejected, 1);
  // An out-of-order packet at or below the old replay floor, arriving mid-stream:
  // the window is shut, so this is counted too, not silently absorbed.
  h.sockets[0].onmessage({ data: JSON.stringify(packet(3, 'sync', { status: 'locked' })) });
  assert.equal(h.transport.getState().rejected, 2);
  assert.equal(h.transport.getState().sequence, 4);
  assert.equal(h.transport.getState().streams.find(s => s.type === 'attention').values.attended, 'B');
  h.transport.stop();
});

test('F02-T25 with no replay the stream behaves exactly as before', async () => {
  // Nothing retained server-side: REST hands over an empty snapshot, so the socket
  // is the only source and no packet is inside a replay window.
  const h = harness();
  h.setSnapshot(snapshot());
  h.transport.start(); await tick(); h.sockets[0].onopen();
  h.sockets[0].onmessage({ data: JSON.stringify(packet(1, 'attention', { attended: 'A' })) });
  h.sockets[0].onmessage({ data: JSON.stringify(packet(3, 'attention', { attended: 'B' })) });
  assert.equal(h.transport.getState().rejected, 0);
  assert.equal(h.transport.getState().sequence, 3);
  h.sockets[0].onmessage({ data: JSON.stringify(packet(2, 'attention', { attended: 'A' })) });
  h.sockets[0].onmessage({ data: JSON.stringify(packet(3, 'vigilance', { score: .5 })) });
  h.sockets[0].onmessage({ data: '{' });
  assert.equal(h.transport.getState().rejected, 3);
  assert.equal(h.transport.getState().sequence, 3);
  assert.equal(h.transport.getState().streams.find(s => s.type === 'attention').values.attended, 'B');
  h.transport.stop();
});
