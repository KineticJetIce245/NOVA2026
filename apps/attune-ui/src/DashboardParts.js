import React from 'react';
import { eegDisplayModel, captionLines, TRACE_BOX } from './eegDisplay.js';
const h = React.createElement;
// Legend, draw order and colour key for the two stacked traces.
export const TRACES = [['raw', 'Raw EEG — what the electrodes picked up'], ['filtered', 'Band-passed 1-9 Hz — the part the decoder actually uses']];
const value = (v, suffix = '') => typeof v === 'number' && Number.isFinite(v) ? `${v}${suffix}` : 'Unavailable';
export function StatusPill({ children }) { return h('span', { className: 'status-pill' }, children); }
export function Header({ state, session, mediaMode = false }) {
  return h('header', { className: 'dashboard-header' },
    h('div', null, h('h1', null, 'ATTUNE'), h('p', null, 'Neuro-Adaptive Hearing')),
    h('div', { className: 'header-status', role: 'status' },
      h(StatusPill, null, mediaMode && state.connection === 'connected' && !state.stale && session === 'running' ? 'LIVE' : `Connection: ${state.connection}`),
      h(StatusPill, null, `Session: ${session ?? 'awaiting'}`),
      mediaMode && state.streams.some(stream => stream.simulated === true) && h(StatusPill, null, 'SIMULATED'),
      state.stale && h(StatusPill, null, 'STALE / unavailable live connection')));
}
export function VoiceSource({ speaker, selected, correlation, metadata, gain, mediaMode = false }) {
  if (mediaMode) return h('div', { className: `voice-source session-source${selected ? ' selected' : ''}`, 'aria-label': `Source ${speaker}${selected ? ', focused' : ''}` },
    h('h3', null, `Source ${speaker}`),
    h('div', { className: 'source-score' }, h('strong', null, value(correlation)), h('span', null, 'Correlation')),
    h('p', { className: 'source-gain', 'aria-label': `Source ${speaker} gain` }, value(gain, ' dB')),
    h('div', { className: 'focus-badge-slot' }, selected && h('strong', { className: 'focused-badge' }, 'FOCUSED')));

  return h('div', { className: `voice-source${selected ? ' selected' : ''}`, 'aria-label': `${mediaMode ? 'Source' : 'Speaker'} ${speaker}${selected ? ', focused' : ''}` },
    h('div', { className: 'voice-orbit', 'aria-hidden': true }, h('span', null, speaker)),
    h('h3', null, `Source ${speaker}`),
    selected && h('strong', { className: 'focused-badge' }, 'FOCUSED'),
    !mediaMode && h('p', null, metadata?.label ?? `Audio Source ${speaker}`),
    !mediaMode && h('p', null, `Input type: ${metadata?.inputType ?? 'Unavailable'}`),
    !mediaMode && metadata?.reference && h('p', { className: 'source-reference' }, metadata.reference),
    h('p', null, `Gain: ${value(gain, ' dB')}`),
    h('div', { className: 'correlation' }, h('span', null, 'Correlation'), h('strong', null, value(correlation))));
}
export function PredictionCard({ stream, inactive, sources = [], gain, mediaMode = false }) {
  const data = stream?.values ?? {};
  const decision = inactive ? 'unavailable' : (Object.hasOwn(data, 'decision') ? data.decision : data.attended) ?? 'unavailable';
  if (mediaMode) return h('section', { className: 'session-prediction', 'aria-label': 'Focus prediction' },
    h('h2', { className: 'comparison-heading' }, 'Source Comparison'),
    h('div', { className: 'voice-grid' }, ...['A', 'B'].map(speaker => h(VoiceSource, {
      key: speaker, speaker, mediaMode: true, selected: decision === speaker,
      correlation: data[`correlation${speaker}`], gain: gain?.[`${speaker.toLowerCase()}_db`],
    }))),
    h('div', { className: 'current-focus', role: 'status', 'aria-live': 'polite' },
      h('span', null, 'CURRENT FOCUS'),
      h('strong', null, ['A', 'B'].includes(decision) ? `Source ${decision}` : decision === 'uncertain' ? 'Uncertain' : 'Unavailable')));
  const label = ['A', 'B'].includes(decision) ? `Focused on ${mediaMode ? 'Source' : 'Speaker'} ${decision}` : decision === 'uncertain' ? 'Focus is uncertain' : 'Awaiting a current prediction';
  return h('section', { className: 'card focus-card', 'aria-label': 'Focus prediction' },
    h('div', { className: 'card-heading' }, h('h2', null, 'Focus prediction'), h(StatusPill, null, stream?.simulated ? 'SIMULATED' : 'Source status not declared')),
    h('p', { className: 'focus-title' }, 'Which voice are you focusing on?'),
    h('p', { className: 'focus-result', role: 'status', 'aria-live': 'polite' }, label),
    h('div', { className: 'voice-grid' }, ...['A', 'B'].map(speaker => h(VoiceSource, { key: speaker, speaker, mediaMode, selected: decision === speaker, correlation: data[`correlation${speaker}`], metadata: sources?.find(source => source.id === speaker), gain: gain?.[`${speaker.toLowerCase()}_db`] }))),
    h('div', { className: 'focus-summary' }, h('span', null, `Prediction state: ${decision}`), h('span', null, 'Confidence: Unavailable')),
    inactive && h('p', { className: 'muted' }, 'Session inactive or connection stale. Values are historical.'));
}
export function StatusCard({ title, rows }) {
  return h('section', { className: 'card' }, h('h2', null, title), h('dl', { className: 'metric-list' }, ...rows.map(([label, content]) => h('div', { key: label }, h('dt', null, label), h('dd', null, content ?? 'Unavailable')))));
}
export function LiveSignalPanel({ stream, inactive, compact = false }) {
  const model = eegDisplayModel(stream);
  const lines = captionLines(model);
  // One vertical gridline per second of the declared window, spanning both bands.
  const grid = typeof model.windowSeconds === 'number' && Number.isFinite(model.windowSeconds) && model.windowSeconds > 0
    ? Array.from({ length: Math.max(0, Math.ceil(model.windowSeconds) - 1) }, (unused, index) => (index + 1) / model.windowSeconds * TRACE_BOX.width)
      .filter(x => x > 0 && x < TRACE_BOX.width)
      .map(x => h('line', { key: `grid-${x}`, className: 'eeg-grid', x1: x, y1: 0, x2: x, y2: TRACE_BOX.height }))
    : [];
  return h('section', { className: 'card signal-card' },
    h('div', { className: 'card-heading' }, h('h2', null, compact ? 'EEG' : 'Live signal / history'), h(StatusPill, null, inactive ? 'Historical / inactive' : stream?.simulated ? 'SIMULATED display' : 'Display data')),
    model.status === 'awaiting' ? h('div', { className: 'signal-empty' }, 'Awaiting EEG display data') : h(React.Fragment, null,
      h('div', { className: 'eeg-legend timeline-legend' }, ...TRACES.map(([kind, legend]) => h('span', { key: kind, className: 'legend-item' },
        h('span', { className: `legend-swatch eeg-swatch-${kind}`, 'aria-hidden': true }), legend))),
      h('svg', { viewBox: `0 0 ${TRACE_BOX.width} ${TRACE_BOX.height}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': 'EEG display samples, first channel — raw above, band-passed below', className: 'signal-trace eeg-trace' },
        ...grid,
        h('line', { className: 'eeg-divider', x1: 0, y1: TRACE_BOX.dividerY, x2: TRACE_BOX.width, y2: TRACE_BOX.dividerY }),
        model.raw.points && h('polyline', { className: 'eeg-raw', points: model.raw.points }),
        model.filtered.points && h('polyline', { className: 'eeg-filtered', points: model.filtered.points })),
      h('div', { className: 'eeg-note' }, ...lines.map((line, index) => h('p', { key: `line-${index}`,
        className: `eeg-note-line${index < 2 ? ` eeg-callout-${TRACES[index][0]}${model[TRACES[index][0]].clipped ? ' eeg-scale-clipped' : ''}` : index === 3 ? ' eeg-caveat' : ''}` }, line)))),
    model.status === 'awaiting' && h('p', { className: 'muted' }, 'A space for incoming signal samples. No measurements available yet.'));
}
export function SessionControls({ state, command, busy, commandError, mediaMode = false }) {
  return h('section', { className: 'card session-controls', 'aria-label': 'Session controls' },
    h('div', null, h('h2', null, 'Your session'), h('p', null, mediaMode ? 'Shared media · artificial attention · no EEG hardware' : 'Development stream · no hardware or audio playback')),
    h('div', { className: 'button-group' }, h('button', { className: 'primary', disabled: busy || state.connection !== 'connected', onClick: () => command('start') }, 'Start mock session'), h('button', { disabled: busy || state.connection !== 'connected', onClick: () => command('stop') }, 'Stop session')),
    commandError && h('p', { role: 'alert' }, commandError));
}
export { value as displayValue };

export function SecondaryMetrics({ vigilance, sync, quality, inactive = false }) {
  const score = vigilance?.score;
  const percentage = Number.isFinite(score) && score >= 0 && score <= 1 ? `${Math.round(score * 100)}%` : 'Unavailable';
  const rows = [['Vigilance', percentage], ['Synchronization', sync?.status ?? 'Unknown'], ['Signal Quality', value(quality?.quality)]];
  return h('section', { className: 'secondary-metrics', 'aria-label': 'Secondary monitoring' },
    h('dl', null, ...rows.map(([label, content]) => h('div', { key: label },
      h('dt', null, label), h('dd', null, inactive ? 'Unavailable' : content)))));
}
