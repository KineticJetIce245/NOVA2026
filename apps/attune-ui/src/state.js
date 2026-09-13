import { validatePacket, validateSnapshot } from './protocol.js';
import { decodePacket } from './decoders.js';
import { appendFrame } from './decisionStory.js';
export function emptyState() {
  return { connection: 'idle', stale: true, sessionId: null, sequence: -1, streams: [], history: [], rejected: 0, error: null };
}
export function acceptPacket(state, input) {
  try {
    const p = validatePacket(input);
    if (p.sequence <= state.sequence) throw Error('Duplicate or out-of-order sequence');
    const sameSession = p.session_id === state.sessionId;
    const streams = sameSession ? state.streams : [];
    const history = sameSession ? state.history : [];
    const previous = streams.find(s => s.source === p.source && s.type === p.type);
    if (previous && p.timestamp < previous.timestamp) throw Error('Regressing stream timestamp');
    const decoded = decodePacket(p);
    const next = streams.filter(s => s.source !== p.source || s.type !== p.type);
    next.push(decoded);
    // Display ledger: one bounded entry per received attention/gain packet, in
    // arrival order, so the page can show the session instead of its last value.
    return { ...state, sessionId: p.session_id, sequence: p.sequence, streams: next.slice(-128),
      history: appendFrame(history, p.type, decoded), error: null };
  } catch (error) {
    return { ...state, rejected: state.rejected + 1, error: error.message };
  }
}
export function fromSnapshot(input) {
  const snapshot = validateSnapshot(input);
  let state = emptyState();
  for (const packet of snapshot.packets) state = acceptPacket(state, packet);
  return { ...state, sessionId: snapshot.session_id, sequence: snapshot.sequence };
}
