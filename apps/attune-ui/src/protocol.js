export const MAX_PACKET_BYTES = 262144;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
function jsonValue(value, depth = 0) {
  if (depth > 32) throw Error('JSON nesting too deep');
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (Array.isArray(value) || object(value)) {
    for (const child of Object.values(value)) jsonValue(child, depth + 1);
    return;
  }
  throw Error('Invalid JSON value');
}
export function validatePacket(input) {
  if (typeof input === 'string' && new TextEncoder().encode(input).length > MAX_PACKET_BYTES) throw Error('Packet too large');
  const p = typeof input === 'string' ? JSON.parse(input) : input;
  if (!object(p) || p.version !== 1) throw Error('Unsupported envelope');
  for (const key of ['type', 'source', 'session_id']) {
    if (typeof p[key] !== 'string' || [...p[key]].length < 1 || [...p[key]].length > 128) throw Error(`Invalid ${key}`);
  }
  if (!Number.isSafeInteger(p.sequence) || p.sequence < 0) throw Error('Invalid sequence');
  if (typeof p.timestamp !== 'number' || !Number.isFinite(p.timestamp) || p.timestamp < 0 || p.timestamp > Number.MAX_SAFE_INTEGER) throw Error('Invalid timestamp');
  if (!object(p.payload)) throw Error('Invalid payload');
  jsonValue(p);
  const encoded = JSON.stringify(p);
  if (new TextEncoder().encode(encoded).length > MAX_PACKET_BYTES) throw Error('Packet too large');
  return JSON.parse(encoded);
}
export function validateSnapshot(value) {
  if (!object(value) || !Number.isSafeInteger(value.sequence) || value.sequence < 0 ||
      !(value.session_id === null || (typeof value.session_id === 'string' && value.session_id.length > 0)) ||
      !Array.isArray(value.packets) || value.packets.length > 128) throw Error('Invalid snapshot');
  const packets = value.packets.map(validatePacket);
  let last = -1;
  const keys = new Set();
  for (const p of packets) {
    const key = JSON.stringify([p.source, p.type]);
    if (p.session_id !== value.session_id || p.sequence <= last || p.sequence > value.sequence || keys.has(key)) throw Error('Inconsistent snapshot');
    last = p.sequence; keys.add(key);
  }
  return { sequence: value.sequence, session_id: value.session_id, packets };
}
