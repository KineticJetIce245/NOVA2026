import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Dashboard } from '../src/Dashboard.js';
import { PredictionCard, LiveSignalPanel, SessionControls } from '../src/DashboardParts.js';
import { emptyState } from '../src/state.js';
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));
test('VIS-F01 only the explicit focused speaker is selected', () => {
  for (const decision of ['A', 'B']) {
    const html = render(PredictionCard, { stream: { values: { decision, attended: 'A', correlationA: 0 } } });
    assert.equal((html.match(/voice-source selected/g) ?? []).length, 1);
    assert.match(html, new RegExp(`Speaker ${decision}, focused`));
    assert.match(html, /<strong>0<\/strong>/);
  }
});
test('VIS-F02 uncertain unavailable and inactive never select a speaker', () => {
  for (const decision of ['uncertain', 'unavailable', null]) {
    assert.doesNotMatch(render(PredictionCard, { stream: { values: { decision, attended: 'A' } } }), /voice-source selected/);
  }
  assert.doesNotMatch(render(PredictionCard, { inactive: true, stream: { values: { decision: 'B' } } }), /voice-source selected/);
});
test('VIS-F03 absent measurements stay unavailable and empty signal has no trace', () => {
  const html = render(Dashboard, { state: emptyState(), command() {} });
  for (const text of ['ATTUNE', 'System status', 'Live status', 'Confidence: Unavailable', 'Awaiting EEG display data', 'DEVELOPMENT ONLY']) assert.ok(html.includes(text));
  assert.doesNotMatch(html, /<polyline/);
});
test('VIS-F04 existing EEG samples render as a labeled historical display', () => {
  const html = render(LiveSignalPanel, { inactive: true, stream: { timestamp: 0, values: { samples: [[0, 1, -1]], channels: ['illustration'] } } });
  assert.match(html, /<polyline/);
  assert.match(html, /EEG display samples, first channel/);
  assert.match(html, /Historical \/ inactive/);
});
test('VIS-F05 controls preserve start stop commands and connection gating', () => {
  const calls = [];
  const props = { state: { connection: 'connected' }, command: name => calls.push(name), busy: false };
  const tree = SessionControls(props);
  const buttons = tree.props.children[1].props.children;
  buttons.forEach(button => button.props.onClick());
  assert.deepEqual(calls, ['start', 'stop']);
  assert.doesNotMatch(render(SessionControls, props), /disabled/);
  assert.match(render(SessionControls, { ...props, busy: true }), /disabled/);
  assert.match(render(SessionControls, { ...props, state: { connection: 'reconnecting' } }), /disabled/);
});
