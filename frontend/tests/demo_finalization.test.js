import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { decodePacket } from '../src/decoders.js';
import { Dashboard } from '../src/Dashboard.js';
import { emptyState } from '../src/state.js';
const decode = (type, payload) => decodePacket({type, payload, source: 'demo', timestamp: 0, sequence: 1, session_id: 'demo'});
const render = (streams, extra = {}) => renderToStaticMarkup(React.createElement(Dashboard, {
  state: {...emptyState(), connection: 'connected', stale: false, streams, ...extra}, command() {},
}));
test('FINAL-F01 explicit A/B and every inactive state gate focus', () => {
  for (const decision of ['A', 'B', 'uncertain', 'unavailable']) {
    const streams = [decode('attention', {decision, correlation_a: 0, correlation_b: .42})];
    const html = render(streams);
    assert.equal((html.match(/voice-source selected/g) ?? []).length, ['A','B'].includes(decision) ? 1 : 0);
    if (['A','B'].includes(decision)) assert.match(html, new RegExp(`Speaker ${decision}, focused`));
    for (const extra of [{stale:true}, {connection:'disconnected'}, {connection:'reconnecting'}])
      assert.doesNotMatch(render(streams, extra), /voice-source selected/);
    for (const status of ['stopped', 'error'])
      assert.doesNotMatch(render([...streams, decode('session', {status})]), /voice-source selected/);
  }
});
test('FINAL-F02 source metadata, gains and artificial banner use decoded packets', () => {
  const streams = [decode('audio_sources', {simulated:true, sources:[
    {id:'A', label:'Recorded Interview', input_type:'recording', reference:'clip-a.wav'},
    {id:'B', label:'YouTube Speech', input_type:'youtube', reference:'video-example'},
  ]}), decode('attention', {decision:'B', correlation_a:0, correlation_b:.42}),
  decode('gain', {a_db:0,b_db:-6})];
  const html = render(streams);
  for (const text of ['Recorded Interview','recording','clip-a.wav','YouTube Speech','youtube','ARTIFICIAL DEMO DATA','FOCUSED','Correlation','0 dB','-6 dB']) assert.ok(html.includes(text), text);
  assert.doesNotMatch(html, /confidence percentage|42%|<audio|<iframe/i);
  assert.doesNotMatch(render([decode('attention', {decision:'A', simulated:false})]), /ARTIFICIAL DEMO DATA/);
});
test('FINAL-F03 malformed optional fields are safe, zero and opaque fallback survive', () => {
  for (const sources of [null, {}, [null], [{id:'A'}, {id:'A'}], [null, {id:'B'}]]) {
    assert.deepEqual(decode('audio_sources', {sources}).values.sources, []);
    assert.match(render([decode('audio_sources', {sources})]), /Unavailable/);
  }
  const gain = decode('gain', {a_db:0,b_db:'bad'});
  assert.deepEqual(gain.values, {a_db:0,b_db:null});
  assert.deepEqual(decode('signal_quality', {quality:0,artifact:false}).values, {quality:0,artifact:false});
  assert.deepEqual(decode('signal_quality', {quality:null,artifact:'bad'}).values, {quality:null,artifact:null});
  assert.equal(decode('future', {custom:0}).values.custom, 0);
  assert.doesNotThrow(() => render([decode('prediction', {outputs:[null, 0, {}, 'bad']})]));
});
