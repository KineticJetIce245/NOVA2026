import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { decodePacket } from '../src/decoders.js';
import { LiveSignalPanel } from '../src/DashboardParts.js';
import { Dashboard } from '../src/Dashboard.js';
import { emptyState, acceptPacket } from '../src/state.js';
import { peakAbs, traceValues, drawnHalfRange, tracePoints, eegDisplayModel, captionLines } from '../src/eegDisplay.js';
const packet = (payload, sequence = 1) => ({ version: 1, sequence, type: 'eeg_display', source: 'nova-aad', session_id: 's1', timestamp: 1, payload });
// The frozen two-trace packet: raw and band-passed at different rates.
const column = values => values.map(value => [value]);
// On the wire `samples` is samples x channels: one row per time sample, one entry
// per channel (tests/streaming/test_eeg_display.py pins len(row) == len(channels)
// for every row, and the row count as the number of points). The display tap names
// a single electrode, so every row holds exactly one value.
const payload = () => ({ sample_rate: 16, channels: ['Cz'], samples: column([0, 60, -60, 30]), filtered_samples: column([0, 8, -4, 2]), filtered_sample_rate: 8,
  original_units: 'uV', filtered_units: 'uV', original_scale_hint_uv: 60, filtered_scale_hint_uv: 20, window_seconds: 5, lag_seconds: 1.5, channel_source: 'highest variance channel' });
const decoded = (override = {}) => decodePacket(packet({ ...payload(), ...override })).values;
const render = (values, props = {}) => renderToStaticMarkup(React.createElement(LiveSignalPanel, { stream: { timestamp: 2, values }, ...props }));
const renderStream = stream => renderToStaticMarkup(React.createElement(LiveSignalPanel, { stream }));
const tracesOf = html => (html.match(/<polyline[^>]*>/g) ?? []).map(tag => ({
  className: (tag.match(/class="([^"]*)"/) ?? [])[1] ?? null,
  points: (tag.match(/points="([^"]*)"/) ?? [])[1] ?? null }));
const ys = points => points.split(' ').map(pair => Number(pair.split(',')[1]));

test('EEG-F01 eeg_display decodes every frozen two-trace field unchanged', () => {
  assert.deepEqual(decoded(), { sampleRate: 16, channels: ['Cz'], samples: column([0, 60, -60, 30]), filteredSamples: column([0, 8, -4, 2]), filteredSampleRate: 8,
    originalUnits: 'uV', filteredUnits: 'uV', originalScaleHintUv: 60, filteredScaleHintUv: 20, windowSeconds: 5, lagSeconds: 1.5,
    channelSource: 'highest variance channel' });
  // Older three-field packets keep the older shape: no invented keys, no zeroes.
  assert.deepEqual(decodePacket(packet({ sample_rate: 32, channels: ['one'], samples: column([0, -1, .3]) })).values,
    { sampleRate: 32, channels: ['one'], samples: column([0, -1, .3]) });
});

test('EEG-F02 wrong-typed fields decode to null per field, without coercion or throwing', () => {
  const names = { sample_rate: 'sampleRate', channels: 'channels', samples: 'samples', filtered_samples: 'filteredSamples',
    filtered_sample_rate: 'filteredSampleRate', original_units: 'originalUnits', filtered_units: 'filteredUnits',
    original_scale_hint_uv: 'originalScaleHintUv', filtered_scale_hint_uv: 'filteredScaleHintUv', window_seconds: 'windowSeconds',
    lag_seconds: 'lagSeconds', channel_source: 'channelSource' };
  const bad = [['sample_rate', '16'], ['sample_rate', 0], ['channels', [1]], ['samples', [['bad']]], ['samples', [[0], [NaN]]],
    ['samples', [[0], [Infinity]]], ['filtered_samples', [['bad']]], ['filtered_samples', [[0], [NaN]]], ['filtered_sample_rate', '8'],
    ['filtered_sample_rate', 0], ['original_units', 7], ['filtered_units', null], ['original_scale_hint_uv', '60'],
    ['filtered_scale_hint_uv', -20], ['window_seconds', '5'], ['window_seconds', 0], ['lag_seconds', NaN], ['channel_source', 3]];
  for (const [key, value] of bad) {
    const values = decoded({ [key]: value });
    assert.equal(values[names[key]], null, `${key}=${String(value)} must decode to null, never a coerced value`);
    if (key !== 'sample_rate') assert.equal(values.sampleRate, 16);
    if (key !== 'channels') assert.deepEqual(values.channels, ['Cz']);
    if (key !== 'samples') assert.deepEqual(values.samples, column([0, 60, -60, 30]));
    if (key !== 'filtered_samples') assert.deepEqual(values.filteredSamples, column([0, 8, -4, 2]));
  }
});

test('EEG-F03 half-range is the trace peak, floored and capped at that trace own hint', () => {
  assert.deepEqual(traceValues(column([0, 1, -2])), [0, 1, -2]);   // samples x channels: channel 0 down the rows
  assert.deepEqual(traceValues([[7, 9], [8, 9]]), [7, 8]);         // two channels: still channel 0, never row 0
  assert.equal(traceValues(column([0, NaN])), null);  // non-finite => Unavailable, never drawn as zero
  assert.equal(traceValues(column([0, null])), null);
  assert.equal(traceValues([]), null);
  assert.equal(traceValues([[]]), null);
  assert.equal(traceValues(null), null);
  assert.equal(peakAbs([0, 5, -9]), 9);
  assert.equal(peakAbs([0, NaN]), null);
  assert.equal(drawnHalfRange(9, 60), 9);             // data below the hint: the data sets the scale
  assert.equal(drawnHalfRange(90, 60), 60);           // data above the hint: clipped at the hint
  assert.equal(drawnHalfRange(90, null), 90);         // no hint published: uncapped, and said so
  assert.equal(drawnHalfRange(0, 60), 1e-3);          // flat trace: floor, no divide-by-zero
  assert.equal(drawnHalfRange(NaN, 60), null);
  assert.equal(tracePoints({ values: [1], sampleRate: null, windowSeconds: null, centerY: 40, halfRange: 1 }), '0,6');
  assert.equal(tracePoints({ values: [1], sampleRate: 16, windowSeconds: 5, centerY: 40, halfRange: null }), null);
});

test('EEG-F04 two traces in one svg: raw on top, band-passed below, both spanning the declared window', () => {
  const traces = tracesOf(render(decoded()));
  assert.equal(traces.length, 2, 'both the raw and the band-passed trace must be drawn');
  assert.deepEqual(traces.map(trace => trace.className), ['eeg-raw', 'eeg-filtered'], 'draw order is raw first, band-passed second');
  const [raw, filtered] = traces.map(trace => trace.points);
  // Both traces span the declared 5 s window across the full width. The packet's
  // points are a decimated trace covering `window_seconds` (the producer decimates
  // both traces to one shared point count), so `sample_rate` is the rate the chain
  // was fed, not the spacing of these points: x is the window, not index/rate.
  assert.equal(raw, '0,40 200,6 400,74 600,23');
  assert.equal(filtered, '0,120 200,86 400,137 600,111.5');
  // swap guard: each trace stays inside its own band and carries its own data
  assert.ok(ys(raw).every(y => y <= 74), 'the raw trace must stay inside the top band');
  assert.ok(ys(filtered).every(y => y >= 86), 'the band-passed trace must stay inside the bottom band');
});

test('EEG-F05 the drawn scale is the trace half-range, never a fixed illustrative ±1', () => {
  const traces = tracesOf(render(decoded({ samples: column([0, 9, -3]), original_scale_hint_uv: 60, filtered_samples: column([0, 2]), filtered_scale_hint_uv: 20 })));
  assert.equal(traces[0].points, '0,40 300,6 600,51.33');                      // ±9 uV drawn inside a ±60 uV hint
  assert.notEqual(traces[0].points, '0,40 300,6 600,40');                      // what a hard-coded ±1 clamp would draw
  assert.equal(traces[1].points, '0,120 600,86');                              // ±2 uV drawn inside a ±20 uV hint
  assert.ok(!traces[0].points.includes('51.33,') && !traces[0].points.includes(',51.34'));
  const html = render(decoded({ samples: column([0, 9, -3]), original_scale_hint_uv: 60 }));
  assert.ok(html.includes('drawn ±9.0 uV'), 'the actual half-range drawn must be stated');
  assert.ok(html.includes('packet hint ±60.0 uV'), 'the hint must be stated');
  assert.ok(html.includes('not clipped'));
});

test('EEG-F06 data above the hint clips visibly at the band edge and says so', () => {
  const html = render(decoded({ samples: column([0, 90, -90]), original_scale_hint_uv: 60, filtered_samples: column([0, 5]), filtered_scale_hint_uv: 20 }));
  const [raw] = tracesOf(html);
  assert.equal(raw.points, '0,40 300,6 600,74');                                // pinned to both edges of the raw band
  assert.match(html, /eeg-scale-clipped/);
  assert.ok(html.includes('drawn ±60.0 uV'), 'the drawn half-range is the capped one');
  assert.ok(html.includes('packet hint ±60.0 uV'));
  assert.ok(html.includes('clipped at the hint'));
});

test('EEG-F07 captions state half-ranges, hints, both unit labels, channel, window and lag', () => {
  const html = render(decoded());
  for (const text of ['drawn ±60.0 uV', 'drawn ±8.0 uV', 'packet hint ±60.0 uV', 'packet hint ±20.0 uV',
    'original unit label uV', 'filtered unit label uV', '16 Hz', '8 Hz', 'Channel Cz', 'highest variance channel',
    'window 5.0 s', 'one shared time grid', 'window ends 1.5 s behind the session clock, not corrected',
    'what the electrodes picked up', 'the part the decoder actually uses', 'not a measurement of the viewer', 'Display data']) {
    assert.ok(html.includes(text), `caption must state: ${text}`);
  }
  assert.ok(!html.includes('Display scale ±1'));
  // A hint expressed in uV against data in another unit is flagged, never silently converted.
  const counts = captionLines(eegDisplayModel({ values: decoded({ original_units: 'counts' }) })).join(' ');
  assert.ok(counts.includes('the hint is expressed in uV while this trace is in counts'));
});

test('EEG-F08 a raw-only packet draws the raw trace and marks the other Unavailable', () => {
  const html = render(decoded({ filtered_samples: null, filtered_units: null, filtered_scale_hint_uv: null }));
  const traces = tracesOf(html);
  assert.equal(traces.length, 1);
  assert.equal(traces[0].className, 'eeg-raw');
  assert.ok(!html.includes('Awaiting EEG display data'));
  assert.ok(html.includes('Band-passed 1-9 Hz'));
  assert.ok(html.includes('Unavailable — no finite samples for this trace'));
  const filteredOnly = tracesOf(render(decoded({ samples: null, original_units: null, original_scale_hint_uv: null })));
  assert.deepEqual(filteredOnly.map(trace => trace.className), ['eeg-filtered']);
});

test('EEG-F09 no usable trace keeps the awaiting state and draws nothing', () => {
  for (const stream of [undefined, { timestamp: 0, values: {} }, { timestamp: 0, values: decoded({ samples: null, filtered_samples: null }) },
    { timestamp: 0, values: decoded({ samples: column([0, null]), filtered_samples: column([NaN, 1]) }) }]) {
    const html = renderStream(stream);
    assert.ok(html.includes('Awaiting EEG display data'));
    assert.ok(html.includes('No measurements available yet'));
    assert.equal(tracesOf(html).length, 0);
    assert.ok(!html.includes('<svg'));
  }
});

test('EEG-F10 a trace without a declared rate is index-spaced and never claims seconds', () => {
  const html = renderStream({ timestamp: 0, values: { samples: column([0, 1, -1]), channels: ['illustration'] } });
  const traces = tracesOf(html);
  assert.equal(traces.length, 1);
  assert.equal(traces[0].points, '0,40 300,6 600,74');                          // even index spacing, no time claim
  assert.ok(html.includes('sample rate not declared, so this trace is drawn at even sample-index spacing'));
  assert.ok(html.includes('window Unavailable s'));
  assert.ok(html.includes('Channel illustration'));
});

test('EEG-F11 a full 5 s packet at two different rates lands on one shared grid in the dashboard', () => {
  const raw = Array.from({ length: 80 }, (unused, i) => Math.sin(i / 4) * 70);        // ~80 points at 16 Hz
  const filtered = Array.from({ length: 80 }, (unused, i) => Math.sin(i / 4) * 6);    // same channel, 32 Hz
  const payloads = [
    { version: 1, sequence: 1, type: 'media', source: 'nova-aad', session_id: 's1', timestamp: 0.25,
      payload: { media_id: 'm1', title: 'KU Leuven trial', duration_s: 389, playback_state: 'playing' } },
    { version: 1, sequence: 2, type: 'eeg_display', source: 'nova-aad', session_id: 's1', timestamp: 0.5,
      payload: { ...payload(), samples: column(raw), filtered_samples: column(filtered), filtered_sample_rate: 32 } },
  ];
  const state = payloads.reduce((accumulated, next) => acceptPacket(accumulated, next), emptyState());
  assert.equal(state.rejected, 0);
  const html = renderToStaticMarkup(React.createElement(Dashboard, { state, command() {} }));
  const traces = tracesOf(html);
  assert.deepEqual(traces.map(trace => trace.className), ['eeg-raw', 'eeg-filtered']);
  for (const trace of traces) {
    assert.equal(trace.points.split(' ').length, 80, 'every received sample of the window must be drawn');
    assert.ok(!trace.points.includes('NaN'));
  }
  const [rawPoints, filteredPoints] = traces.map(trace => trace.points.split(' ').map(pair => pair.split(',').map(Number)));
  // One shared grid: both traces span the declared window over the full box, so
  // index i lands on the same x in both even though the packet declares two rates.
  assert.equal(rawPoints[0][0], 0);
  assert.equal(rawPoints[79][0], 600);
  assert.deepEqual(rawPoints.map(pair => pair[0]), filteredPoints.map(pair => pair[0]));
  assert.equal((html.match(/class="eeg-grid"/g) ?? []).length, 4); // one gridline per second, shared by both bands
  const rawPeak = Math.max(...raw.map(value => Math.abs(value)));
  assert.ok(html.includes(`drawn ±${Math.min(rawPeak, 60).toFixed(1)} uV`), 'the capped half-range is stated');
  assert.equal(rawPeak > 60, html.includes('clipped at the hint'));
  assert.ok(html.includes('not clipped') && html.includes('Channel Cz'));
  assert.ok(!html.includes('Awaiting EEG display data'));
});
