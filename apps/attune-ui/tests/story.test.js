// Story presentation tests. The packet fixture is the real recorded run
// (output/auditory_ui/run_now_packets.jsonl), so the counts asserted here are
// the run's actual counts, not invented ones. Protocol behaviour is covered by
// the other suites; these tests cover what the page puts on screen.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { DecisionStory, DifferenceTrace, SourceComparison, appendFrame, gainConsequence, storyLayout } from '../src/decisionStory.js';
import { emptyState, acceptPacket } from '../src/state.js';
import { decodePacket } from '../src/decoders.js';

const FIXTURE = fileURLToPath(new URL('../../../output/auditory_ui/run_now_packets.jsonl', import.meta.url));
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));
const marks = html => (html.match(/class="timeline-mark/g) ?? []).length;
// React escapes the en/em dashes the copy uses; normalise before matching text.
const plain = html => html.replaceAll('&#x2212;', '−').replaceAll('&#x2014;', '—').replaceAll('&#x2013;', '–');
const envelope = (type, payload, sequence) => ({ version: 1, type, source: 'nova-aad', session_id: 's-1', sequence, timestamp: sequence / 4, payload });

function replay(packets) {
  let state = emptyState();
  for (const packet of packets) state = acceptPacket(state, packet);
  return state;
}
const historyOf = packets => replay(packets).history;

// The packet log is a demo artifact and is not committed, so a clean clone has
// no copy of it. Its absence must not take the whole module down: the tests that
// need it skip with the reason below and the rest still run and still assert.
const MISSING_RUN = existsSync(FIXTURE) ? false
  : `no recorded run at ${FIXTURE} — create it with: python -B -m scripts.auditory_ui.demo`;
const recorded = MISSING_RUN ? [] : readFileSync(FIXTURE, 'utf8').trim().split('\n').map(line => JSON.parse(line));
// The decoded media stream only carries a duration once the browser has
// reported one; the trial itself is 125 s. The page reads whichever it has.
const trialDuration = 125;
const replayed = replay(recorded);
const recordedAttention = recorded.filter(p => p.type === 'attention').map(p => p.payload);
const count = decision => recordedAttention.filter(p => p.decision === decision).length;

test('STORY-F01 the display ledger keeps every received frame, in order', { skip: MISSING_RUN }, () => {
  const attention = recordedAttention.length, gains = recorded.filter(p => p.type === 'gain').length;
  const frames = replayed.history.filter(frame => frame.kind === 'attention');
  assert.equal(frames.length, attention);
  assert.equal(replayed.history.filter(frame => frame.kind === 'gain').length, gains);
  assert.ok(replayed.history.every((frame, i) => i === 0 || frame.sequence > replayed.history[i - 1].sequence));
  // Gain merges into the frame the backend stamped with the same position.
  const merged = replayed.history.find(frame => frame.kind === 'gain' && frame.aDb !== null);
  assert.equal(typeof merged.mediaTime, 'number');
  assert.equal(appendFrame(replayed.history, 'gain', decodePacket(recorded.find(p => p.type === 'gain'))), replayed.history);
});

test('STORY-F02 the timeline draws exactly the frames that arrived', { skip: MISSING_RUN }, () => {
  assert.ok(trialDuration > 0, 'the recorded run must declare a media duration');
  const positioned = replay([...recorded, envelope('media', { duration_s: trialDuration, media_time_s: 60, playback_state: 'playing', revision: 2, media_id: 'm' }, 99999)]);
  const stream = { type: 'attention', values: { mediaTime: 60 } };
  const html = plain(render(DecisionStory, { history: replayed.history,
    state: { ...positioned, streams: [...positioned.streams, stream] } }));
  assert.equal(marks(html), recordedAttention.length);
  const layout = storyLayout(replayed.history, { currentTime: 60, duration: trialDuration });
  assert.equal(layout.coverage.frames, recordedAttention.length);
  assert.equal(layout.coverage.byDecision.A, count('A'));
  assert.equal(layout.coverage.byDecision.B, count('B'));
  assert.equal(layout.coverage.abstained, count('uncertain'));
  assert.equal(layout.coverage.absent, count('unavailable'));
  assert.equal(layout.coverage.decided, count('A') + count('B'));
  assert.match(html, new RegExp(`${layout.coverage.frames} attention frames received · attributed ${layout.coverage.decided} · abstained ${layout.coverage.abstained} · no data ${layout.coverage.absent}`));
  assert.match(html, /no interpolation/i);
  // The playhead is the position the page was told, not a guess: 60 s of 124 s.
  assert.match(html, new RegExp(`left:${60 / trialDuration * 100}%`));
});

test('STORY-F03 abstention is stated as plainly as a decision', () => {
  const abstain = [envelope('attention', { decision: 'uncertain', correlation_a: .21, correlation_b: .19, reasons: ['no_decision'], media_time_s: 9, media_id: 'm', media_revision: 2 }, 1)];
  const abstaining = plain(render(DecisionStory, { history: historyOf(abstain), state: replay(abstain) }));
  assert.match(abstaining, /The system is abstaining/);
  assert.match(abstaining, /neither source is attenuated/);
  assert.match(abstaining, /reason: no_decision/);
  assert.doesNotMatch(abstaining, /FOCUSED/);
  const decide = [envelope('attention', { decision: 'A', correlation_a: .34, correlation_b: .01, reasons: [], media_time_s: 12, media_id: 'm', media_revision: 2 }, 1),
    envelope('gain', { a_db: 0, b_db: -6, media_time_s: 12, media_id: 'm', media_revision: 2 }, 2)];
  const decided = replay(decide);
  const html = plain(render(DecisionStory, { history: decided.history, state: decided }));
  assert.match(html, /Attended source<\/p><p class="verdict-value">Source A/);
  assert.match(html, /Source B turned down to -6 dB/);
  assert.match(html, /the other source is unchanged/);
  assert.equal(gainConsequence({ aDb: 0, bDb: 0 }).attenuated, false);
  assert.equal(gainConsequence({ aDb: null, bDb: null }).note, null);
});

test('STORY-F04 the two scores share one axis and the difference is named', () => {
  const packets = [envelope('attention', { decision: 'A', correlation_a: .2, correlation_b: -.1, media_time_s: 3, media_id: 'm', media_revision: 2 }, 1)];
  const state = replay(packets);
  const html = plain(render(SourceComparison, { history: state.history, state }));
  assert.equal((html.match(/shared scale ±0\.200/g) ?? []).length, 1);
  assert.match(html, /Difference A − B/);
  assert.match(html, /0\.3000 — at or above the 0\.05 margin, which is why the system committed to a source/);
  assert.match(html, /Source A scores higher/);
  assert.match(html, /not probabilities/);
  // Scores stay available as text, no longer as the headline.
  assert.match(html, /\+0\.2000/);
  assert.match(html, /−0\.1000/);
  const below = replay([envelope('attention', { decision: 'uncertain', correlation_a: .1, correlation_b: .09, media_time_s: 3, media_id: 'm', media_revision: 2 }, 1)]);
  const htmlBelow = plain(render(SourceComparison, { history: below.history, state: below }));
  assert.match(htmlBelow, /below the 0\.05 margin, which is why the system abstained/);
  // A run whose largest score is 0.1 gets a ±0.100 axis, not a padded one.
  assert.match(htmlBelow, /shared scale ±0\.100/);
});

test('STORY-F05 the difference trace is raw, unscaled data with the margin band', { skip: MISSING_RUN }, () => {
  const layout = storyLayout(replayed.history, { currentTime: 60, duration: 125 });
  const html = render(DifferenceTrace, { layout });
  assert.equal((html.match(/<circle/g) ?? []).length, layout.points.length);
  const polyline = html.match(/points="([^"]+)"/);
  assert.equal(polyline[1].split(' ').length, layout.points.length);
  assert.match(html, /Raw score difference \(A minus B\)/);
  assert.match(html, /Nothing is smoothed/);
  // One x-coordinate per scored frame: no resampling, no gaps invented.
  const xs = polyline[1].split(' ').map(pair => Number(pair.split(',')[0]));
  assert.deepEqual(xs, [...xs].sort((a, b) => a - b));
  assert.match(html, /class="trace-margin"/);
});

test('STORY-F06 empty and pre-session states claim nothing', t => {
  const empty = plain(render(DecisionStory, { history: [], state: emptyState() }));
  assert.match(empty, /<p class="verdict-value">No data<\/p>/);
  assert.equal(marks(empty), 0);
  assert.match(empty, /0 attention frames received/);
  assert.match(render(DifferenceTrace, { layout: storyLayout([], {}) }), /for every scored window/);
  // Only this last pre-session check replays the recording; the empty-state
  // assertions above run on their own packets and have already been made.
  if (MISSING_RUN) return t.skip(MISSING_RUN);
  const notLive = plain(render(DecisionStory, { history: replayed.history, state: replayed, inactive: true }));
  assert.match(notLive, /Playback is not live/);
});
