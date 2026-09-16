import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { acceptPacket, emptyState } from '../src/state.js';
import { Dashboard } from '../src/Dashboard.js';
const render = state => renderToStaticMarkup(React.createElement(Dashboard, { state, command() {} }));
test('MR-F01 default mock generation decodes and renders provider output with development labels', () => {
  const code = `import json
from backend.adapters.mock import results
from backend.app.protocol import make_packet
print(json.dumps([make_packet(k,t,i+1,s,'test',p) for i,(k,t,s,p) in enumerate(results(0))]))`;
  const packets = JSON.parse(execFileSync(process.env.ATTUNE_PYTHON || 'python3', ['-B','-c',code], { cwd: new URL('../../', import.meta.url), encoding: 'utf8' }));
  let state = { ...emptyState(), stale: false, connection: 'connected' };
  for (const packet of packets) state = acceptPacket(state, packet);
  const prediction = state.streams.find(s => s.type === 'prediction');
  assert.equal(prediction.values.providerId, 'mock');
  assert.equal(prediction.simulated, true);
  assert.equal(prediction.values.outputs[0].semanticType, 'development');
  assert.equal(prediction.values.outputs[0].value, 1);
  const html = render(state);
  for (const label of ['Prediction result','SIMULATED · DEVELOPMENT ONLY','NO SCIENTIFIC INTERPRETATION','Provider: mock','Task: transport_demo','Status: ok','development_value','Value: 1','Semantic type: development']) assert.ok(html.includes(label),label);
  state = acceptPacket(state, {version:1, type:'session', timestamp:1, sequence:100, source:'server',session_id:'test',payload:{status:'stopped'}});
  assert.ok(render(state).includes('Historical prediction — not current'));
  assert.ok(!render(state).includes('Current session prediction'));
  assert.ok(!render(state).includes('feedback-active'));
});
