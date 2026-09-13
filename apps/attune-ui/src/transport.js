import { emptyState, acceptPacket, fromSnapshot } from './state.js';
import { createRestClient } from './rest.js';
import { validatePacket } from './protocol.js';
export function websocketUrl(location = globalThis.location) {
  return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/live`;
}
export function createTransport({ rest = createRestClient(), socketFactory = url => new WebSocket(url),
  url = websocketUrl(), schedule = setTimeout, cancel = clearTimeout, onState = () => {} } = {}) {
  let state = emptyState(), generation = 0, active = false, socket, retry, deadline, controller, failures = 0;
  const emit = next => { state = next; onState(state); };
  function cleanup() {
    cancel(retry); cancel(deadline); controller?.abort();
    if (socket) { socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null; socket.close(); socket = null; }
  }
  async function connect() {
    if (!active) return;
    cleanup();
    const token = ++generation;
    const current = () => active && token === generation;
    // Connect-time replay bookkeeping, rebuilt for each connection below.
    let replayFloor = -1, replaying = false;
    controller = new AbortController();
    emit({ ...state, connection: failures ? 'reconnecting' : 'connecting', stale: true });
    function fail(message) {
      if (!current()) return;
      ++generation; cleanup();
      emit({ ...state, connection: 'reconnecting', stale: true, error: message });
      retry = schedule(connect, Math.min(500 * 2 ** Math.min(failures++, 4), 8000));
    }
    deadline = schedule(() => fail('Connection timed out'), 10000);
    try {
      const snapshot = fromSnapshot(await rest.state(controller.signal));
      if (!current()) return;
      // REST is the reset boundary: backend process restarts may reset sequences.
      emit({ ...snapshot, connection: 'connecting', stale: true });
      // The server replays its retained snapshot to every newly connected socket,
      // on purpose, to close the subscribe race (src/nova2026/transport/server.py).
      // So a new socket's first packets are exactly the state REST just handed us.
      // They are absorbed at the socket boundary rather than handed to
      // acceptPacket: its `sequence <= state.sequence` guard is right and stays
      // untouched, but one replay of what we already show is not a failure and
      // must not raise the page's red "measurements may be missing" alarm. The
      // window closes on the first packet past the snapshot's high-water mark, so
      // a duplicate arriving later, mid-stream, is still counted by that guard.
      replayFloor = snapshot.sequence;
      replaying = true;
      socket = socketFactory(url);
      socket.onopen = () => {
        if (!current()) return;
        cancel(deadline);
        emit({ ...state, connection: 'connected', stale: false, error: null });
      };
      socket.onmessage = event => {
        if (!current()) return;
        if (typeof event.data !== 'string') { emit({ ...state, rejected: state.rejected + 1, error: 'Expected JSON text' }); return; }
        let packet;
        try { packet = validatePacket(event.data); }
        catch (error) { emit({ ...state, rejected: state.rejected + 1, error: error.message }); return; }
        if (state.sessionId !== null && packet.session_id !== state.sessionId && packet.sequence <= state.sequence) {
          fail('Backend epoch changed; refreshing snapshot'); return;
        }
        // Connect-time replay of the snapshot REST just applied (see the note where
        // `replaying` is set): already-displayed state, not an event and not a loss.
        if (replaying && packet.sequence <= replayFloor) return;
        replaying = false;
        const next = acceptPacket(state, packet);
        // Snapshot replay duplicates are harmless; only new valid packets reset backoff.
        if (next.sequence > state.sequence) failures = 0;
        emit(next);
      };
      socket.onclose = () => fail('Stream disconnected');
      socket.onerror = () => fail('Stream unavailable');
    } catch (error) { fail(error.message); }
  }
  return {
    start() { if (!active) { active = true; failures = 0; void connect(); } },
    stop() { active = false; ++generation; cleanup(); emit({ ...state, connection: 'disconnected', stale: true }); },
    getState: () => state,
  };
}
