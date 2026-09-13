import test from 'node:test';
import assert from 'node:assert/strict';
import {decodePacket} from '../src/decoders.js';
const decode = (type, payload) => decodePacket({type, payload, source:'nova_aad', timestamp:1,
  sequence:2, session_id:'real'});
test('NOVA presentation preserves left/right mapping and rejects unknown mode', () => {
  assert.deepEqual(decode('presentation', {mode:'dichotic',left:'A',right:'B',simulated:false}).values,
    {mode:'dichotic',left:'A',right:'B'});
  assert.equal(decode('presentation', {mode:'mono'}).values.mode, null);
});
test('Real NOVA packets retain false simulation and unknown measurements', () => {
  const sync = decode('sync', {simulated:false,status:'software_clock_observed',offset_ms:null,drift_warning:null});
  assert.equal(sync.simulated,false);
  assert.equal(sync.values.offsetMs,null);
  assert.equal(sync.values.driftWarning,null);
  const p=decode('prediction',{simulated:false,status:'invalid',outputs:[],reasons:['artifact'],metadata:{simulated:false}});
  assert.equal(p.simulated,false);
  assert.equal(p.values.status,'invalid');
});
