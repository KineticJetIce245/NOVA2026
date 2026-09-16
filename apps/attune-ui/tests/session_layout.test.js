import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { PredictionCard, SecondaryMetrics, LiveSignalPanel } from '../src/DashboardParts.js';
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));
test('LAYOUT-F01 source pair precedes current focus with backend values', () => {
  const html = render(PredictionCard, {mediaMode:true,stream:{values:{decision:'A',correlationA:.42,correlationB:.19}},gain:{a_db:0,b_db:-8}});
  assert.ok(html.indexOf('voice-grid') < html.indexOf('CURRENT FOCUS'));
  for (const value of ['0.42','0.19','0 dB','-8 dB','FOCUSED','Source A','Source B']) assert.ok(html.includes(value));
  assert.equal((html.match(/session-source selected/g)??[]).length,1);
  assert.doesNotMatch(html,/voice-orbit|Which voice|Confidence/);
});
test('LAYOUT-F02 inactive and uncertain layout never retains an active source', () => {
  for (const props of [{inactive:true,stream:{values:{decision:'A'}}},{stream:{values:{decision:'uncertain'}}},{}]) {
    const html=render(PredictionCard,{mediaMode:true,...props});
    assert.doesNotMatch(html,/session-source selected|FOCUSED/);
    assert.match(html,/Unavailable|Uncertain/);
  }
});
test('LAYOUT-F03 secondary metrics format supplied score and preserve unknowns', () => {
  const html=render(SecondaryMetrics,{vigilance:{score:.72},sync:{status:'unknown'},quality:{quality:null}});
  for (const value of ['Vigilance','72%','Synchronization','unknown','Signal Quality','Unavailable']) assert.ok(html.includes(value));
  assert.doesNotMatch(html,/Synced|Good/);
  assert.match(render(SecondaryMetrics,{vigilance:{score:0}}),/0%/);
  assert.doesNotMatch(render(SecondaryMetrics,{vigilance:{score:.72},inactive:true}),/72%/);
});

test('LAYOUT-F04 compact EEG retains supplied display waveform and historical label', () => {
  const html=render(LiveSignalPanel,{compact:true,inactive:true,stream:{timestamp:0,simulated:true,values:{samples:[[0,.2,0]],channels:['illustration']}}});
  assert.match(html, /<h2>EEG<\/h2>/);
  assert.match(html, /<polyline/);
  assert.match(html, /Historical \/ inactive/);
  assert.doesNotMatch(render(LiveSignalPanel,{compact:true}), /<polyline/);
});
