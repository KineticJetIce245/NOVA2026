// Display adaptation only: no inference, filtering, smoothing or signal analysis.
const number = v => typeof v === 'number' && Number.isFinite(v) ? v : null;
const text = v => typeof v === 'string' ? v : null;
const boolean = v => typeof v === 'boolean' ? v : null;
const mediaReference = p => Object.hasOwn(p, 'media_id') ? { mediaId: text(p.media_id), mediaRevision: number(p.media_revision), mediaTime: number(p.media_time_s) } : {};
export const decoders = new Map([
  ['media', p => ({ mediaId: text(p.media_id), title: text(p.title), mediaTime: number(p.media_time_s), duration: number(p.duration_s),
    playbackState: ['playing', 'paused', 'stopped'].includes(p.playback_state) ? p.playback_state : null,
    revision: number(p.revision), serverReference: number(p.server_reference_s), syncStatus: text(p.sync_status) })],
  ['audio_sources', p => ({ sources: Array.isArray(p.sources) && p.sources.length === 2 &&
    p.sources.every(s => s && typeof s === 'object' && ['A', 'B'].includes(s.id)) &&
    new Set(p.sources.map(s => s.id)).size === 2 ? p.sources.map(s => ({
      id: s.id, label: text(s.label), inputType: text(s.input_type), reference: text(s.reference),
    })) : [] })],
  ['gain', p => ({ a_db: number(p.a_db), b_db: number(p.b_db), ...mediaReference(p) })],
  ['signal_quality', p => ({ quality: number(p.quality), artifact: boolean(p.artifact) })],
  ['feedback', p => ({
    status: p.simulated === true && p.metadata?.development_only === true && ['none', 'active', 'suppressed'].includes(p.status) ? p.status : 'suppressed',
    actionType: text(p.action_type), message: text(p.message), severity: ['info', 'warning', 'critical'].includes(p.severity) ? p.severity : 'info',
    sourceProvider: text(p.source_provider), sourceTask: text(p.source_task), reason: text(p.reason),
    cooldownRemaining: number(p.metadata?.cooldown_remaining_seconds),
  })],
  ['prediction', p => ({
    status: ['ok', 'invalid', 'unavailable', 'error'].includes(p.status) ? p.status : null,
    providerId: text(p.provider_id), providerName: text(p.provider_name),
    providerVersion: text(p.provider_version), task: text(p.task),
    timestamp: number(p.timestamp), windowId: text(p.window_id),
    windowStart: number(p.window_start), windowEnd: number(p.window_end),
    outputs: Array.isArray(p.outputs) ? p.outputs.filter(output => output !== null && typeof output === 'object' && !Array.isArray(output)).map(output => ({
      name: text(output.name), value: Object.hasOwn(output, 'value') ? output.value : null,
      semanticType: text(output.semantic_type), label: text(output.label),
    })) : [], reasons: Array.isArray(p.reasons) ? p.reasons.filter(v => typeof v === 'string') : [],
    metadata: p.metadata !== null && typeof p.metadata === 'object' && !Array.isArray(p.metadata) ? p.metadata : {},
  })],
  ['attention', p => {
    const explicit = Object.hasOwn(p, 'decision');
    const decision = ['A', 'B', 'uncertain', 'unavailable'].includes(p.decision) ? p.decision : null;
    const attended = explicit ? decision : p.attended;
    return { ...mediaReference(p), attended: ['A', 'B'].includes(attended) ? attended : null,
      correlationA: number(p.correlation_a), correlationB: number(p.correlation_b),
      reasons: Array.isArray(p.reasons) ? p.reasons.filter(reason => typeof reason === 'string') : [],
      ...(explicit ? { decision } : {}) };
  }],
  ['vigilance', p => ({
    score: p.metric === 'lapse_probability' ? null : number(p.score) !== null && p.score >= 0 && p.score <= 1 ? p.score : null,
    ...(Object.hasOwn(p, 'metric') ? { metric: ['vigilance', 'lapse_probability'].includes(p.metric) ? p.metric : null } : {}),
    ...(Object.hasOwn(p, 'lapse_score') ? { lapseScore: number(p.lapse_score) !== null && p.lapse_score >= 0 && p.lapse_score <= 1 ? p.lapse_score : null } : {}),
  })],
  ['sync', p => ({ status: text(p.status), offsetMs: number(p.offset_ms), driftWarning: boolean(p.drift_warning),
    timeline: text(p.timeline), fixedLatencyMs: number(p.fixed_latency_ms) })],
  ['eeg_display', p => {
    // Frozen two-trace packet: index 0 of `samples` is RAW EEG, index 0 of
    // `filtered_samples` is the same channel after the packet's own causal
    // 1-9 Hz band-pass. Both rates may differ; both traces may be absent.
    // Values pass through unchanged: a wrong-typed, non-finite or absent field
    // is null, never coerced and never zero. Extended fields exist only when the
    // packet declares them, so older three-field packets keep their shape.
    const trace = v => Array.isArray(v) && v.every(row => Array.isArray(row) && row.every(cell => number(cell) !== null)) ? v : null;
    const positive = v => number(v) !== null && v > 0 ? v : null;
    const declared = (key, name, decode) => Object.hasOwn(p, key) ? { [name]: decode(p[key]) } : {};
    return {
      sampleRate: positive(p.sample_rate),
      channels: Array.isArray(p.channels) && p.channels.every(v => typeof v === 'string') ? p.channels : null,
      samples: trace(p.samples),
      ...declared('filtered_samples', 'filteredSamples', trace),
      ...declared('filtered_sample_rate', 'filteredSampleRate', positive),
      ...declared('original_units', 'originalUnits', text),
      ...declared('filtered_units', 'filteredUnits', text),
      ...declared('original_scale_hint_uv', 'originalScaleHintUv', positive),
      ...declared('filtered_scale_hint_uv', 'filteredScaleHintUv', positive),
      ...declared('window_seconds', 'windowSeconds', positive),
      ...declared('lag_seconds', 'lagSeconds', number),
      ...declared('channel_source', 'channelSource', text),
    };
  }],
]);
export function decodePacket(packet, registry = decoders) {
  const decode = registry.get(packet.type);
  return { type: packet.type, source: packet.source, sessionId: packet.session_id,
    sequence: packet.sequence, timestamp: packet.timestamp, simulated: boolean(packet.payload.simulated),
    known: Boolean(decode), values: decode ? decode(packet.payload) : packet.payload };
}
