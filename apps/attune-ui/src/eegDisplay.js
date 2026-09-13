// Pure display geometry for the two-trace EEG panel.
// Display adaptation only: the values published by the backend are drawn as they
// arrived. No filtering, smoothing, inference, resampling or silent rescaling.
// The one transformation that does happen (clipping at the packet's own scale
// hint) is reported in the caption, and each trace is mapped on its own x axis
// from its own sample rate.
export const TRACE_BOX = Object.freeze({
  width: 600, height: 160, dividerY: 80, halfHeight: 34,
  rawCenterY: 40, filteredCenterY: 120, minHalfRange: 1e-3,
});
const finite = v => typeof v === 'number' && Number.isFinite(v);
const positive = v => finite(v) && v > 0 ? v : null;
const round = n => Math.round(n * 100) / 100;
const MICROVOLT = new Set(['uv', 'µv', 'μv', 'microvolt', 'microvolts', 'uv (microvolts)']);

// index 0 is the trace the decoder consumes for `samples`, and the band-passed
// trace for `filtered_samples`. A non-finite value makes the whole trace
// Unavailable: it is never drawn as zero and never silently replaced.
export function traceValues(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const values = rows[0];
  if (!Array.isArray(values) || values.length === 0) return null;
  return values.every(finite) ? values : null;
}
export function peakAbs(values) {
  if (!Array.isArray(values) || values.length === 0 || !values.every(finite)) return null;
  return values.reduce((peak, v) => Math.max(peak, Math.abs(v)), 0);
}
// Symmetric half-range actually drawn: the trace's own peak, floored against
// divide-by-zero and capped at the packet's scale hint for that trace.
export function drawnHalfRange(peak, hintUv) {
  if (!finite(peak)) return null;
  const data = Math.max(peak, TRACE_BOX.minHalfRange);
  const hint = positive(hintUv);
  return hint === null ? data : Math.min(data, hint);
}
export function traceWindowSeconds(values, sampleRate, windowSeconds) {
  const declared = positive(windowSeconds);
  if (declared !== null) return declared;
  const rate = positive(sampleRate);
  return rate !== null && Array.isArray(values) && values.length > 0 ? values.length / rate : null;
}
// x = sample_index / own_sample_rate, normalised by the shared window, so two
// traces at different rates still share one time grid.
export function tracePoints({ values, sampleRate, windowSeconds, centerY, halfRange }) {
  const rate = positive(sampleRate), window = positive(windowSeconds), drawn = positive(halfRange);
  if (!Array.isArray(values) || values.length === 0 || drawn === null) return null;
  const timed = rate !== null && window !== null;
  return values.map((v, index) => {
    // Own rate for the timed mapping; bare index spacing (no seconds claimed)
    // when the packet declares no rate for this trace.
    const x = timed ? (index / rate) / window * TRACE_BOX.width
      : values.length > 1 ? index / (values.length - 1) * TRACE_BOX.width : 0;
    const y = centerY - Math.max(-1, Math.min(1, v / drawn)) * TRACE_BOX.halfHeight;
    return `${round(x)},${round(y)}`;
  }).join(' ');
}
function trace({ values, sampleRate, units, hintUv, centerY, windowSeconds }) {
  const rate = positive(sampleRate);
  const hint = positive(hintUv);
  const peak = peakAbs(values);
  const halfRange = drawnHalfRange(peak, hint);
  const points = values === null ? null : tracePoints({ values, sampleRate: rate, windowSeconds, centerY, halfRange });
  return { available: points !== null, values, sampleRate: rate, timeMapped: rate !== null && positive(windowSeconds) !== null,
    windowSeconds: positive(windowSeconds), units: typeof units === 'string' ? units : null,
    hintUv: hint, peak, halfRange, points, clipped: peak !== null && hint !== null && peak > hint };
}
export function eegDisplayModel(stream) {
  const v = stream?.values ?? {};
  const rawValues = traceValues(v.samples);
  const filteredValues = traceValues(v.filteredSamples);
  const rawRate = positive(v.sampleRate);
  const filteredRate = positive(v.filteredSampleRate);
  // One grid for both bands. Declared window first; otherwise derived per trace
  // from its own rate and the longer of the two is used, and said to be derived.
  const declared = positive(v.windowSeconds);
  const derived = [traceWindowSeconds(rawValues, rawRate, null), traceWindowSeconds(filteredValues, filteredRate, null)].filter(finite);
  const windowSeconds = declared ?? (derived.length > 0 ? Math.max(...derived) : null);
  const common = { windowSeconds };
  const raw = trace({ values: rawValues, sampleRate: rawRate, units: v.originalUnits, hintUv: v.originalScaleHintUv, centerY: TRACE_BOX.rawCenterY, ...common });
  const filtered = trace({ values: filteredValues, sampleRate: filteredRate, units: v.filteredUnits, hintUv: v.filteredScaleHintUv, centerY: TRACE_BOX.filteredCenterY, ...common });
  const status = raw.available && filtered.available ? 'both' : raw.available ? 'raw-only' : filtered.available ? 'filtered-only' : 'awaiting';
  return { status, raw, filtered, windowSeconds, windowDeclared: declared !== null,
    lagSeconds: finite(v.lagSeconds) ? v.lagSeconds : null,
    channel: Array.isArray(v.channels) && typeof v.channels[0] === 'string' ? v.channels[0] : null,
    channelSource: typeof v.channelSource === 'string' ? v.channelSource : null };
}
export const formatUv = n => finite(n) ? (Math.round(n * 10) / 10).toFixed(1) : 'Unavailable';
export function traceCallout(t) {
  if (!t.available) return t.values === null && t.peak === null
    ? 'Unavailable — no finite samples for this trace'
    : 'Unavailable — sample rate or window missing, so this trace cannot be placed on the time grid';
  const parts = [`drawn ±${formatUv(t.halfRange)} ${t.units ?? 'units (not declared by the packet)'}`];
  if (t.hintUv === null) parts.push('no packet scale hint, drawn uncapped at the data peak');
  else {
    parts.push(`packet hint ±${formatUv(t.hintUv)} uV`);
    parts.push(t.clipped ? 'clipped at the hint' : 'not clipped');
  }
  if (t.units !== null && !MICROVOLT.has(t.units.trim().toLowerCase())) parts.push(`note: the hint is expressed in uV while this trace is in ${t.units} — the cap is applied as supplied, without unit conversion`);
  return parts.join(' · ');
}
export function captionLines(model) {
  const rate = t => t.sampleRate === null
    ? 'sample rate not declared, so this trace is drawn at even sample-index spacing and the time grid does not apply to it'
    : `${t.sampleRate} Hz`;
  const lag = model.lagSeconds === null ? 'window position against the session clock not declared'
    : model.lagSeconds === 0 ? 'window end aligned with the session clock (declared lag 0 s), not corrected'
    : model.lagSeconds > 0 ? `window ends ${formatUv(model.lagSeconds)} s behind the session clock, not corrected`
    : `window ends ${formatUv(Math.abs(model.lagSeconds))} s ahead of the session clock, not corrected`;
  const grid = model.raw.timeMapped && model.filtered.timeMapped ? 'one shared time grid' : 'time grid drawn from the declared window only';
  return [
    `Raw EEG — what the electrodes picked up · ${traceCallout(model.raw)} · original unit label ${model.raw.units ?? 'Unavailable'} · ${rate(model.raw)}.`,
    `Band-passed 1-9 Hz (the packet's own causal filter, not applied here) — the part the decoder actually uses · ${traceCallout(model.filtered)} · filtered unit label ${model.filtered.units ?? 'Unavailable'} · ${rate(model.filtered)}.`,
    `Channel ${model.channel ?? 'Unavailable'}${model.channelSource ? ` (${model.channelSource})` : ''} · window ${formatUv(model.windowSeconds)} s${model.windowDeclared || model.windowSeconds === null ? '' : ' (derived from samples and rate, not declared by the packet)'} · ${grid} · ${lag}.`,
    EEG_CAVEAT,
  ];
}
// The honest caveat: this is display data forwarded for the panel.
export const EEG_CAVEAT = 'Display data only — these traces are what the display received, not a measurement of the viewer.';
