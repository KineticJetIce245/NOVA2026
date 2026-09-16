import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { formatMediaTime, mediaProgress, dbToLinear, playbackGains, mediaFocusReady, mediaLinked, createStereoAudio } from '../src/mediaAudio.js';
import { createMediaController } from '../src/mediaController.js';
import { decodePacket } from '../src/decoders.js';
import { MediaPlayback } from '../src/MediaPlayback.js';
import { PredictionCard } from '../src/DashboardParts.js';
const packet = (type, payload) => decodePacket({type, payload, source:'test', session_id:'session', timestamp:9, sequence:1});
const playback = {ready:true, mediaId:'media', revision:2, time:6.2, playbackState:'playing'};
const state = () => ({connection:'connected', stale:false, sessionId:'session', streams:[
  packet('session', {status:'running'}),
  packet('media', {media_id:'media',revision:2,playback_state:'playing',sync_status:'observed'}),
  packet('attention', {decision:'B',media_id:'media',media_revision:2,media_time_s:6.2}),
  packet('gain', {a_db:-6,b_db:0,media_id:'media',media_revision:2,media_time_s:6.2}),
]});
test('MEDIA-F01 time formatting progress and decibel conversion', () => {
  for (const [value,text] of [[0,'00:00.00'],[18.423,'00:18.42'],[65.2,'01:05.20'],[null,'--:--.--']]) assert.equal(formatMediaTime(value),text);
  assert.equal(mediaProgress(6,12),.5); assert.equal(mediaProgress(6,null),0); assert.equal(mediaProgress(20,10),1);
  assert.equal(dbToLinear(0),1); assert.ok(Math.abs(dbToLinear(-6)-.501187)<.000001);
});
test('MEDIA-F02 backend gains only when current and all failures neutral', () => {
  const good = state();
  assert.deepEqual(playbackGains(good,playback,'attune'),[dbToLinear(-6),1]);
  assert.deepEqual(playbackGains(good,playback,'original'),[1,1]);
  for (const change of [{stale:true},{connection:'disconnected'},{error:'Invalid packet'}]) {
    assert.deepEqual(playbackGains({...good,...change},playback,'attune'),[1,1]);
    assert.equal(mediaFocusReady({...good,...change},playback),false);
  }
  // `time` here is this page's own clock. A copy that claims to be AHEAD of it is
  // refused, because a copy that merely lags cannot explain being in the future.
  for (const change of [{ready:false},{error:'desynchronized'},{playbackState:'stopped'},{time:5},{revision:3},{mediaId:'other'}])
    assert.deepEqual(playbackGains(good,{...playback,...change},'attune'),[1,1]);
  for (const decision of ['uncertain','unavailable',null]) {
    const s=state();s.streams[2].values.decision=decision;
    assert.deepEqual(playbackGains(s,playback,'attune'),[1,1]);
  }
  for (const value of [null, Infinity, 6, -100]) {
    const s=state();s.streams[3].values.a_db=value;
    assert.deepEqual(playbackGains(s,playback,'attune'),[1,1]);
  }
  const s=state();s.streams[0].values.status='stopped';assert.equal(mediaFocusReady(s,playback),false);
  const paused=state();paused.streams[1].values.playbackState='paused';
  assert.equal(mediaFocusReady(paused,{...playback,playbackState:'paused'}),true);
  assert.deepEqual(playbackGains(paused,{...playback,playbackState:'paused'},'attune'),[1,1]);
});
test('MEDIA-F03 two independent audio gains smooth automation and cleanup', async () => {
  const nodes=[], connections=[], ramps=[];
  const node = kind => { const n={kind,connect(...args){connections.push([kind,...args]);},disconnect(){n.closed=true;},gain:{setTargetAtTime(...args){ramps.push(args);}}};nodes.push(n);return n;};
  class Context {currentTime=3;createMediaElementSource(){return node('source');}createChannelSplitter(n){assert.equal(n,2);return node('split');}createChannelMerger(n){assert.equal(n,2);return node('merge');}createGain(){return node('gain');}resume(){return Promise.resolve();}close(){return Promise.resolve();}}
  let paused=false;const audio=createStereoAudio({pause(){paused=true;}},Context);
  audio.apply([.5,1]);assert.deepEqual(ramps,[[.5,3,.1]]);
  audio.apply([.5,1]);assert.equal(ramps.length,1);
  audio.apply([1,1]);assert.deepEqual(ramps[1],[1,3,.1]);
  assert.equal(connections.filter(c=>c[0]==='split').length,2);
  await audio.close();assert.ok(paused);assert.ok(nodes.every(n=>n.closed));
});
test('MEDIA-F04 media title controls progress and neutral source-only markup', () => {
  const html=renderToStaticMarkup(React.createElement(MediaPlayback,{config:{title:'Shared Conversation',kind:'video',url:'/api/media/file'},state:state(),rest:{}},
    ({inactive})=>React.createElement(PredictionCard,{inactive,mediaMode:true,stream:state().streams[2]})));
  for(const label of ['Shared Conversation','Media Time','progress','Source A','Source B','Original Mix','ATTUNE','Play','Pause','Stop']) assert.ok(html.includes(label),label);
  assert.doesNotMatch(html,/Left|Right|Male|Female|Channel [01]|voice-source selected|controls=""/);
});
function harness() {
  let time=0;const calls=[], applied=[];
  const element={currentTime:0,duration:90,paused:true,pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}};
  const rest={async mediaControl(body){calls.push(body);return {session_id:body.session_id,media_id:body.media_id,revision:2,sync_status:body.action==='stopped'?'desynchronized':'observed'};}};
  const controller=createMediaController({element,config:{media_id:'media'},rest,clientId:'browser',now:()=>time,
    audioFactory:()=>({resume:async()=>{},apply:values=>applied.push(values),close:async()=>{}})});
  controller.update(state());
  return {controller,element,rest,calls,applied,setTime:v=>{time=v;}};
}
test('MEDIA-F05 controller handshake play pause freeze resume stop', async () => {
  const h=harness();await h.controller.play();assert.deepEqual(h.calls.map(c=>c.action),['prepare','playing']);
  h.element.currentTime=6.2;await h.controller.pause();h.setTime(30000);await h.controller.tick();
  assert.equal(h.calls.at(-1).media_time_s,6.2);assert.equal(h.controller.snapshot().playbackState,'paused');
  await h.controller.play();assert.equal(h.element.currentTime,6.2);
  await h.controller.stop();assert.equal(h.element.currentTime,0);assert.equal(h.controller.snapshot().playbackState,'stopped');
  assert.equal(h.calls.at(-1).action,'stopped');assert.deepEqual(h.applied.at(-1),[1,1]);h.controller.dispose();
});
test('MEDIA-F06 rejected autoplay and failed reporting never emphasize', async () => {
  const h=harness();h.element.play=async()=>{throw Error('blocked');};await h.controller.play();
  assert.ok(h.controller.snapshot().error);assert.deepEqual(h.applied.at(-1),[1,1]);h.controller.dispose();
  const other=harness();await other.controller.play();other.rest.mediaControl=async()=>{throw Error('offline');};await other.controller.tick();
  assert.ok(other.element.paused);assert.equal(other.controller.snapshot().ready,false);assert.deepEqual(other.applied.at(-1),[1,1]);other.controller.dispose();
});
test('MEDIA-F07 delayed prepare cannot revive stopped playback', async () => {
  const h=harness();let release;
  const normal=h.rest.mediaControl;
  h.rest.mediaControl=body=>body.action==='prepare'?new Promise(resolve=>{release=()=>normal(body).then(resolve);}):normal(body);
  const playing=h.controller.play();
  for(let i=0;i<8 && !release;i++) await Promise.resolve();
  assert.ok(release);
  const stopped=h.controller.stop();release();await Promise.all([playing,stopped]);
  assert.equal(h.element.paused,true);assert.equal(h.controller.snapshot().playbackState,'stopped');
  assert.equal(h.controller.snapshot().ready,false);assert.equal(h.calls.at(-1).action,'stopped');h.controller.dispose();
});
test('MEDIA-F08 media decoder missing values and revision mismatch are safe', () => {
  const empty=packet('media',{}).values;
  assert.equal(empty.mediaTime,null);assert.equal(empty.playbackState,null);
  const s=state();s.streams[3].values.mediaRevision=99;
  assert.deepEqual(playbackGains(s,playback,'attune'),[1,1]);
  s.streams[1].values.syncStatus='desynchronized';assert.equal(mediaFocusReady(s,playback),false);
});
test('MEDIA-F09 mixed result frames and unexpected seeking fail neutral', async () => {
  const s=state();s.streams[3].values.mediaTime=6;
  assert.deepEqual(playbackGains(s,playback,'attune'),[1,1]);
  const h=harness();await h.controller.play();h.element.currentTime=45;h.controller.seeking();
  assert.ok(h.controller.snapshot().error);assert.deepEqual(h.applied.at(-1),[1,1]);
  await h.controller.stop();h.controller.seeking();h.controller.seeked();
  assert.equal(h.controller.snapshot().playbackState,'stopped');h.controller.dispose();
});

test('MEDIA-F10 a sampled copy that lags behind is held, not shown as no data', () => {
  // Measured on the live route: the backend's copy of the playback position
  // advances in ~6.5 s steps while this page's clock runs continuously, so between
  // updates the copy is behind by up to that much. Holding the last decision is the
  // point; falling back to "No data" several times a minute was the visible defect.
  for (const lag of [0, 1.8, 6.5, 7.9]) {
    const s = state();
    s.streams[2].values.mediaTime = playback.time - lag;
    s.streams[3].values.mediaTime = playback.time - lag;
    assert.equal(mediaFocusReady(s, playback), true, `${lag}s behind must still be held`);
    assert.deepEqual(playbackGains(s, playback, 'attune'), [dbToLinear(-6), 1],
      `${lag}s behind must still attenuate the unattended source`);
  }
  // Past the transport's update cadence it is no longer a live copy.
  for (const lag of [8.1, 30]) {
    const s = state();
    s.streams[2].values.mediaTime = playback.time - lag;
    s.streams[3].values.mediaTime = playback.time - lag;
    assert.equal(mediaFocusReady(s, playback), false, `${lag}s behind is not a live copy`);
    assert.deepEqual(playbackGains(s, playback, 'attune'), [1, 1]);
  }
  // Ahead of this page is never explained by a copy that lags.
  for (const lead of [0.76, 1, 5]) {
    const s = state();
    s.streams[2].values.mediaTime = playback.time + lead;
    s.streams[3].values.mediaTime = playback.time + lead;
    assert.equal(mediaFocusReady(s, playback), false, `a copy ${lead}s ahead is refused`);
    assert.deepEqual(playbackGains(s, playback, 'attune'), [1, 1]);
  }
  // And the boundary itself is inclusive on both sides.
  const edge = state();
  edge.streams[2].values.mediaTime = playback.time - 8;
  edge.streams[3].values.mediaTime = playback.time - 8;
  assert.equal(mediaFocusReady(edge, playback), true);
});

test('MEDIA-F11 an abstention is a live page with no decision, not a dead player', () => {
  // The two questions are different. `mediaFocusReady` is the GAIN gate and must
  // demand a decision -- there is nothing to attenuate while the system abstains.
  // The page's own liveness must not: conflating them rendered an abstention as
  // "No data · Playback is not live" on a run whose playback was perfectly live,
  // which is what the synthetic-ANT demo showed almost continuously.
  const abstaining = state();
  abstaining.streams[2].values.decision = 'uncertain';
  assert.equal(mediaFocusReady(abstaining, playback), false, 'no decision, no gain');
  assert.deepEqual(playbackGains(abstaining, playback, 'attune'), [1, 1]);
  assert.equal(mediaLinked(abstaining, playback), true, 'but the page is live');

  for (const decision of ['unavailable', null]) {
    const s = state();
    s.streams[2].values.decision = decision;
    assert.equal(mediaLinked(s, playback), true, `${decision} is still a live page`);
    assert.equal(mediaFocusReady(s, playback), false);
  }

  // Everything that really does mean "not live" still says so.
  for (const change of [{ stale: true }, { connection: 'disconnected' }, { error: 'Invalid packet' }])
    assert.equal(mediaLinked({ ...abstaining, ...change }, playback), false, JSON.stringify(change));
  for (const change of [{ ready: false }, { error: 'desynchronized' }, { playbackState: 'stopped' },
    { revision: 3 }, { mediaId: 'other' }, { time: 60 }])
    assert.equal(mediaLinked(abstaining, { ...playback, ...change }), false, JSON.stringify(change));
  const stopped = state();
  stopped.streams[0].values.status = 'stopped';
  assert.equal(mediaLinked(stopped, playback), false, 'a stopped session is not live');
});
