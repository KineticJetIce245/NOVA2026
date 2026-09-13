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
