import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Dashboard } from '../src/Dashboard.js';
import { decodePacket } from '../src/decoders.js';
const stream = status => decodePacket({ type: 'feedback', source: 'development-feedback', timestamp: 5,
  payload: { status, simulated: true, action_type: status === 'active' ? 'visual' : 'none',
    message: status === 'active' ? 'Development test: warning feedback.' : 'No development feedback.',
    severity: 'warning', reason: status === 'suppressed' ? 'cooldown' : 'test',
    metadata: { development_only: true, cooldown_remaining_seconds: 5 } } });
const render = (streams = [], stale = false) => renderToStaticMarkup(React.createElement(Dashboard, {
  state: { streams, stale, connection: 'connected', sequence: 1, rejected: 0 }, command() {} }));
test('FB-F01 active feedback is obvious and development labeled', () => {
  const html = render([stream('active')]);
  for (const text of ['Feedback', 'feedback-active', 'role="alert"', 'SIMULATED', 'DEVELOPMENT ONLY', 'warning feedback', 'Timestamp: 5']) assert.ok(html.includes(text));
});
test('FB-F02 none and empty feedback render safely', () => {
  for (const streams of [[], [stream('none')]]) {
    const html = render(streams); assert.ok(html.includes('Feedback status: none')); assert.ok(!html.includes('feedback-active'));
  }
});
test('FB-F03 cooldown suppression is visible', () => {
  const html = render([stream('suppressed')]);
  assert.ok(html.includes('suppressed')); assert.ok(html.includes('cooldown')); assert.ok(html.includes('5 s')); assert.ok(!html.includes('feedback-active'));
});
test('FB-F04 stale or stopped sessions cannot highlight cached active feedback', () => {
  assert.ok(!render([stream('active')], true).includes('feedback-active'));
  assert.ok(!render([stream('active'), { type: 'session', source: 'server', values: { status: 'stopped' } }]).includes('feedback-active'));
});
test('FB-F05 undeclared feedback cannot appear active', () => {
  assert.equal(decodePacket({ type: 'feedback', payload: { status: 'active' } }).values.status, 'suppressed');
});
