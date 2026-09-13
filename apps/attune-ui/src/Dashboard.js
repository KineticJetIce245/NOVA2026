import React from 'react';
import { MediaPlayback } from './MediaPlayback.js';
import { FeedbackPanel } from './FeedbackPanel.js';
import { Header, PredictionCard, StatusCard, SecondaryMetrics, LiveSignalPanel, SessionControls, displayValue } from './DashboardParts.js';
const h = React.createElement;
export function Dashboard({ state, command, busy = false, commandError = null, mediaConfig = null, mediaRest, stopSignal = 0 }) {
  const inactive = (!mediaConfig && state.streams.some(s => s.type === 'media')) || state.connection !== 'connected' || state.stale || state.streams.some(s => s.type === 'session' && ['stopped', 'error'].includes(s.values.status));
  const latest = type => state.streams.findLast(s => s.type === type);
  const session = latest('session')?.values.status;
  const vigilance = latest('vigilance')?.values;
  const gain = latest('gain')?.values;
  return h('main', { className: `dashboard${mediaConfig ? ' audio-session' : ''}` },
    h(Header, { state, session, mediaMode: Boolean(mediaConfig) }),
    h('p', { className: 'development-notice' }, 'RESEARCH PROTOTYPE · DEVELOPMENT ONLY'),
    state.streams.some(stream => stream.simulated === true) && h('aside', { className: 'card demo-banner', role: 'status' },
      h('strong', null, 'ARTIFICIAL DEMO DATA'),
      h('p', null, 'The interface is exercising the real ATTUNE transport path. These values are not participant measurements.')),
    mediaConfig ? h(MediaPlayback, { config: mediaConfig, rest: mediaRest, state, stopSignal },
      ({ inactive: mediaInactive }) => h(PredictionCard, { stream: latest('attention'), inactive: inactive || mediaInactive, gain, mediaMode: true })) :
      h(PredictionCard, { stream: latest('attention'), inactive, sources: latest('audio_sources')?.values.sources, gain }),
    mediaConfig && h(SecondaryMetrics, { vigilance, sync: latest('sync')?.values, quality: latest('signal_quality')?.values, inactive }),
    h(mediaConfig ? 'details' : 'div', { className: mediaConfig ? 'additional-status' : undefined },
      mediaConfig && h('summary', null, `Connection: ${state.connection} · Session: ${session ?? 'Waiting'} · More status`),
      h('div', { className: 'status-grid' },
      h(StatusCard, { title: 'System status', rows: [
        ['Backend', state.connection], ['EEG source', latest('eeg_display')?.source],
        ['EEG hardware status', 'Unavailable'], ['Sync offset', displayValue(latest('sync')?.values.offsetMs, ' ms')], ['Synchronization', latest('sync')?.values.status], ['Session', session],
      ] }),
      h(StatusCard, { title: 'Live status', rows: [
        ['Vigilance score', displayValue(vigilance?.score)], ['Lapse score', displayValue(vigilance?.lapseScore)],
        ['Gain A', displayValue(gain?.a_db, ' dB')], ['Gain B', displayValue(gain?.b_db, ' dB')],
        ['Signal quality', displayValue(latest('signal_quality')?.values.quality)],
        ['Artifact', latest('signal_quality')?.values.artifact === true ? 'Detected' : latest('signal_quality')?.values.artifact === false ? 'Not detected' : 'Unavailable'],
      ] }))),
    inactive && h('p', { className: 'muted' }, 'Displayed stream values are historical / unavailable live measurements.'),
    h(LiveSignalPanel, { stream: latest('eeg_display'), inactive, compact: Boolean(mediaConfig) }),
    h(SessionControls, { state, command, busy, commandError, mediaMode: Boolean(mediaConfig) }),
    h('p', null, `Session Time: ${latest('attention')?.timestamp?.toFixed(2) ?? 'Unavailable'} s`),
    state.rejected > 0 && h('p', { className: 'rejected-warning', role: 'alert' },
      h('strong', null, `REJECTED PACKETS: ${state.rejected}`),
      ` — ${state.rejected} received packet${state.rejected === 1 ? '' : 's'} failed envelope validation or latest-state acceptance and were dropped without display. Measurements may be missing or stale. Last reason: ${state.error ?? 'not reported'}`),
    h('p', null, `Session: ${state.sessionId ?? 'none'} · Sequence: ${state.sequence} · Rejected packets: ${state.rejected}`),
    state.error && h('p', null, `Transport note: ${state.error}`),
    h('p', null, 'null = unavailable; unknown packet types are retained without interpretation. Timestamps are session-relative seconds.'),
    state.streams.length === 0 && h('p', null, 'Waiting for result packets.'),
    h(FeedbackPanel, { stream: state.streams.findLast(s => s.type === 'feedback'), stale: state.stale, inactive: state.streams.some(s => s.type === 'session' && ['stopped', 'error'].includes(s.values.status)) }),
    h('details', { className: 'card diagnostics' }, h('summary', null, 'Stream diagnostics · decoded results'), ...state.streams.filter(stream => stream.type !== 'feedback').map(stream => h('section', { key: JSON.stringify([stream.source, stream.type]) },
      h('h2', null, stream.type === 'prediction' ? 'Prediction result' : `${stream.type} / ${stream.source}`),
      stream.type === 'prediction' ? h(React.Fragment, null,
        stream.simulated === true && stream.values.metadata.mock === true && h('strong', {className: 'development-badge'}, 'SIMULATED · DEVELOPMENT ONLY · NO SCIENTIFIC INTERPRETATION'),
        h('p', {role: 'status'}, inactive ? 'Historical prediction — not current' : 'Current session prediction'),
        h('p', null, `Provider: ${stream.values.providerId ?? stream.source} · Task: ${stream.values.task ?? 'unavailable'} · Status: ${stream.values.status ?? 'unavailable'}`),
        h('p', null, stream.values.outputs.map(output => `${output.name ?? 'output'} · Value: ${String(output.value)} · Semantic type: ${output.semanticType ?? 'unavailable'}`).join(' · ') || stream.values.reasons.join(', ')),
        h('pre', null, JSON.stringify(stream.values, null, 2))) : h(React.Fragment, null,
        h('p', null, `${stream.known ? 'Registered decoder' : 'Unknown packet fallback'} · ${stream.simulated === true ? 'SIMULATED' : 'Simulation status unavailable / not declared'} · t=${stream.timestamp} · seq=${stream.sequence}`),
        h('pre', null, JSON.stringify(stream.values, null, 2)))))),
  );
}
