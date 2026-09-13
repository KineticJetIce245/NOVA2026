import { validatePacket, validateSnapshot } from './protocol.js';
import { decodePacket } from './decoders.js';
export function emptyState() {
  return { connection: 'idle', stale: true, sessionId: null, sequence: -1, streams: [], rejected: 0, error: null };
}
export function acceptPacket(state, input) {
  try {
    const p = validatePacket(input);
    if (p.sequence <= state.sequence) throw Error('Duplicate or out-of-order sequence');
    const sameSession = p.session_id === state.sessionId;
    const streams = sameSession ? state.streams : [];
    const previous = streams.find(s => s.source === p.source && s.type === p.type);
    if (previous && p.timestamp < previous.timestamp) throw Error('Regressing stream timestamp');
    const next = streams.filter(s => s.source !== p.source || s.type !== p.type);
    next.push(decodePacket(p));
    return { ...state, sessionId: p.session_id, sequence: p.sequence, streams: next.slice(-128), error: null };
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
