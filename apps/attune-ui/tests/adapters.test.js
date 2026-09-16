import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { decodePacket } from '../src/decoders.js';
import { acceptPacket, emptyState } from '../src/state.js';
const packet = payload => ({ type: 'attention', payload });
test('A03-F01 explicit attention states survive decoding', () => {
  for (const decision of ['A', 'B', 'uncertain', 'unavailable']) {
    const values = decodePacket(packet({ decision, attended: 'A' })).values;
    assert.equal(values.decision, decision);
    assert.equal(values.attended, ['A', 'B'].includes(decision) ? decision : null);
  }
  assert.equal(decodePacket(packet({ decision: 'bad', attended: 'A' })).values.attended, null);
});
test('A03-F02 lapse probability is not displayed as vigilance', () => {
  const values = decodePacket({ type: 'vigilance', payload: { metric: 'lapse_probability', lapse_score: .8, score: .2 } }).values;
  assert.equal(values.lapseScore, .8); assert.equal(values.score, null);
  assert.equal(values.metric, 'lapse_probability');
});
test('A03-F03 real Python adapters flow through unchanged frontend transport state', () => {
  const code = `import json
from backend.adapters.results import ResultAdapter
from backend.adapters.contracts import AttentionResult, SyncResult
from backend.adapters.legacy import lapse_result
from backend.app.publisher import Publisher
p=Publisher()
p.begin('adapter-fixture')
a=ResultAdapter(lambda k,t,s,v:p.publish(k,t,s,'adapter-fixture',v))
for i,d in enumerate(['A','B','uncertain','unavailable']):
 a.publish(AttentionResult(i,'aad',d))
a.publish(lapse_result(.8,timestamp=4))
a.publish(SyncResult(4,'sync'))
print(json.dumps(p.events_after(0)))`;
  const packets = JSON.parse(execFileSync(process.env.ATTUNE_PYTHON || 'python3', ['-B', '-c', code], { cwd: new URL('../../', import.meta.url), encoding: 'utf8' }));
  let state = emptyState();
  for (const p of packets) {
    state = acceptPacket(state, p);
    if (p.type === 'attention') assert.equal(state.streams.find(s => s.type === 'attention').values.decision, p.payload.decision);
  }
  assert.equal(state.rejected, 0);
  assert.equal(state.streams.find(s => s.type === 'sync').values.offsetMs, null);
  assert.equal(state.streams.find(s => s.type === 'sync').values.driftWarning, null);
  assert.equal(state.streams.find(s => s.type === 'vigilance').values.lapseScore, .8);
});
test('A04-F01 generic prediction decoder preserves multiple semantic outputs', () => {
  const values = decodePacket({type: 'prediction', payload: {
    status: 'ok', provider_id: 'nova_aad', provider_name: 'novaAAD', task: 'auditory_attention',
    outputs: [{name: 'speaker_a_correlation', value: .18, semantic_type: 'correlation'},
      {name: 'attended_speaker', value: 'B', semantic_type: 'class'}], reasons: [], metadata: {},
  }}).values;
  assert.equal(values.providerId, 'nova_aad');
  assert.deepEqual(values.outputs.map(output => output.semanticType), ['correlation', 'class']);
});
