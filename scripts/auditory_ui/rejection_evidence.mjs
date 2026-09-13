// Two frontend truths the perturbation matrix asks for, from the frontend's own code.
//
// The repository has no browser in its test path, so the honest substitute is to
// run the vendored frontend's *own* modules over real runs:
//
//   node scripts/auditory_ui/rejection_evidence.mjs reject <packets.jsonl> <out.json>
//   node scripts/auditory_ui/rejection_evidence.mjs reset  <reset_input.json> <out.json>
//
// `reject` (case C13) feeds a session's recorded stream through
// `protocol.js::validatePacket` and `state.js::acceptPacket`, then offers the same
// state machine a fixed set of illegal packets - a wrong envelope version, a
// negative sequence, JSON nested past the depth limit, and a payload larger than
// `MAX_PACKET_BYTES` - and reports how many of them `state.rejected` counted. It
// also renders the real `Dashboard` component over the resulting state, because
// the plan says the `rejected` warning must be *visible*, and the visible form of
// it is what the component prints.
//
// `reset` (case C9) replays a session's stream, applies the fresh snapshot a
// restarted backend serves, and then replays the new backend's stream, which
// starts its sequence over. It reports whether the client accepted that, and - for
// contrast - what would have happened without the snapshot, so the claim "a
// reset sequence is accepted from a fresh snapshot" is shown together with the
// condition it depends on.
//
// Exit code 0 only when the mode's own invariants held.
import { readFileSync, writeFileSync } from 'node:fs';
import { validatePacket, MAX_PACKET_BYTES } from '../../apps/attune-ui/src/protocol.js';
import * as frontendState from '../../apps/attune-ui/src/state.js';
import { emptyState, acceptPacket } from '../../apps/attune-ui/src/state.js';
import { Dashboard } from '../../apps/attune-ui/src/Dashboard.js';

const [mode, inputPath, outPath] = process.argv.slice(2);
if (!mode || !inputPath || !outPath || !['reject', 'reset'].includes(mode)) {
  console.error('usage: node rejection_evidence.mjs <reject|reset> <input> <out.json>');
  process.exit(2);
}

const renderDashboard = async (view) => {
  const React = (await import(new URL('../../apps/attune-ui/node_modules/react/index.js', import.meta.url).href)).default;
  const { renderToStaticMarkup } = await import(new URL('../../apps/attune-ui/node_modules/react-dom/server.node.js', import.meta.url).href);
  const html = renderToStaticMarkup(React.createElement(Dashboard, {
    state: { ...view, connection: 'connected', stale: false },
    command: () => {}, busy: false, commandError: null, mediaConfig: null, mediaRest: null, stopSignal: 0,
  }));
  return html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
};

function feed(state, packets) {
  const errors = [];
  let rejectedBefore = state.rejected;
  for (const packet of packets) {
    state = acceptPacket(state, packet);
    if (state.rejected > rejectedBefore) {
      errors.push(state.error);
      rejectedBefore = state.rejected;
    }
  }
  return { state, errors };
}

if (mode === 'reject') {
  const packets = readFileSync(inputPath, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
  const top = (type) => packets.filter(p => p.type === type).at(-1) ?? packets.at(-1);
  const model = top('attention');
  const illegal = [
    { kind: 'wrong envelope version', packet: { ...model, version: 2 } },
    { kind: 'negative sequence', packet: { ...model, sequence: -1 } },
    {
      kind: 'json nesting too deep',
      packet: {
        ...model,
        payload: (() => { let node = { leaf: true }; for (let i = 0; i < 40; i += 1) node = { child: node }; return node; })(),
      },
    },
    {
      kind: 'packet too large',
      packet: { ...model, payload: { ...model.payload, filler: 'x'.repeat(MAX_PACKET_BYTES + 1024) } },
    },
  ];

  let state = emptyState();
  const recorded = feed(state, packets);
  const recordedByType = {};
  for (const packet of packets) recordedByType[packet.type] = (recordedByType[packet.type] ?? 0) + 1;

  const adversarial = { injected: illegal.length, rejected: 0, accepted: 0, kinds: [] };
  state = recorded.state;
  for (const entry of illegal) {
    const before = state.rejected;
    state = acceptPacket(state, entry.packet);
    const counted = state.rejected > before;
    if (counted) adversarial.rejected += 1; else adversarial.accepted += 1;
    adversarial.kinds.push({ kind: entry.kind, counted, error: state.error });
  }

  let rendered = null;
  let renderError = null;
  try {
    rendered = await renderDashboard(state);
  } catch (error) {
    renderError = error.message;
  }
  const visible = Boolean(rendered) && rendered.includes('REJECTED PACKETS');

  const report = {
    mode,
    packet_count: packets.length,
    recorded: {
      accepted: packets.length - recorded.state.rejected,
      rejected: recorded.state.rejected,
      by_type: recordedByType,
      errors: recorded.errors.slice(0, 4),
    },
    adversarial,
    client_state_rejected: state.rejected,
    rendered_with_rejected: visible,
    rendered_marker: visible ? 'REJECTED PACKETS' : null,
    render_error: renderError,
    apply_snapshot_available: typeof frontendState.applySnapshot === 'function',
  };
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  const ok =
    adversarial.rejected === adversarial.injected &&
    recorded.state.rejected === 0 &&
    report.rendered_with_rejected;
  process.exit(ok ? 0 : 1);
}

if (mode === 'reset') {
  const input = JSON.parse(readFileSync(inputPath, 'utf8'));
  const streamA = input.stream_a ?? [];
  const streamB = input.stream_b ?? [];
  const snapshotB = input.snapshot_b ?? null;
  // `state.js` exports the snapshot entry point as `fromSnapshot`; the fallback
  // name is kept so a renamed import fails loudly as `apply_snapshot_available:
  // false` rather than silently skipping the reset.
  const applySnapshot = frontendState.fromSnapshot ?? frontendState.applySnapshot;

  const first = feed(emptyState(), streamA);
  const stateAfterA = first.state;

  // What the client would do if the restarted backend's stream arrived with no
  // fresh snapshot: the cursor is still the old, higher one.
  const naive = feed(emptyState(), streamA);
  const naiveAfterB = feed(naive.state, streamB);

  let afterReset = stateAfterA;
  let snapshotError = null;
  if (typeof applySnapshot === 'function' && snapshotB) {
    try {
      // `fromSnapshot(snapshot)` rebuilds the client state from the snapshot
      // alone - it is the transport's own entry point for a fresh connection.
      afterReset = applySnapshot(snapshotB);
    } catch (error) {
      snapshotError = error.message;
      afterReset = stateAfterA;
    }
  }
  const second = feed(afterReset, streamB);

  const report = {
    mode,
    stream_a_packets: streamA.length,
    stream_b_packets: streamB.length,
    sequence_after_a: stateAfterA.sequence,
    snapshot_sequence: snapshotB?.sequence ?? null,
    sequence_b_first: streamB[0]?.sequence ?? null,
    sequence_b_last: streamB.at(-1)?.sequence ?? null,
    apply_snapshot_available: typeof applySnapshot === 'function',
    snapshot_error: snapshotError,
    accepted_after_reset: streamB.length - (second.state.rejected - afterReset.rejected),
    duplicates_from_snapshot: streamB.filter(p => p.sequence <= (snapshotB?.sequence ?? -1)).length,
    rejected: second.state.rejected,
    sequence_after: second.state.sequence,
    errors: second.errors.slice(0, 4),
    without_snapshot: {
      rejected: naiveAfterB.state.rejected,
      errors: naiveAfterB.errors.slice(0, 2),
    },
  };
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  const ok =
    report.accepted_after_reset > 0 &&
    report.rejected === 0 &&
    report.without_snapshot.rejected > 0;
  process.exit(ok ? 0 : 1);
}
