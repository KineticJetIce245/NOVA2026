// The decision story: what the decoder decided, over time, on what scale, and
// what it costs the listener. Everything here is read from packets the page
// received. Nothing is smoothed, interpolated, extended or simulated; frames
// that never arrived stay absent.
import React from 'react';
const h = React.createElement;
export const DECISION_ORDER = ['A', 'B', 'uncertain', 'unavailable'];
export const DECISION_TONE = Object.freeze({ A: 'decided', B: 'decided', uncertain: 'abstained', unavailable: 'absent' });
const DECISION_TEXT = Object.freeze({ A: 'Source A', B: 'Source B', uncertain: 'Not sure', unavailable: 'No data' });
export const MAX_FRAMES = 8192;
const FALLBACK_SCALE = 0.2;
const MARGIN = 0.05;

const finite = v => typeof v === 'number' && Number.isFinite(v);
// Four decimals: enough to separate the 0.05 margin, short enough to compare.
export const formatScore = v => finite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(4)}` : 'Unavailable';
export const formatDb = v => finite(v) ? `${v > 0 ? '+' : ''}${v} dB` : 'Unavailable';
export const decisionPhrase = decision => DECISION_TEXT[decision] ?? 'No data';

/**
 * Append one received attention/gain packet to the display ledger.
 * Returns the same array when a packet adds nothing, so React can skip work.
 */
export function appendFrame(history, type, packet) {
  if (type !== 'attention' && type !== 'gain') return history;
  const values = packet?.values ?? {};
  const index = history.findIndex(frame => frame.sequence === packet.sequence);
  if (index >= 0 && history[index].kind === type) return history;
  const previous = index >= 0 ? history[index] : null;
  const frame = { ...previous, sequence: packet.sequence };
  if (type === 'attention') {
    frame.kind = 'attention';
    frame.mediaTime = finite(values.mediaTime) ? values.mediaTime : previous?.mediaTime ?? null;
    frame.decision = DECISION_ORDER.includes(values.decision) ? values.decision
      : ['A', 'B'].includes(values.attended) ? values.attended : 'unavailable';
    frame.correlationA = finite(values.correlationA) ? values.correlationA : null;
    frame.correlationB = finite(values.correlationB) ? values.correlationB : null;
    frame.reasons = Array.isArray(values.reasons) ? values.reasons.filter(reason => typeof reason === 'string') : [];
  } else {
    frame.kind = 'gain';
    frame.aDb = finite(values.a_db) ? values.a_db : null;
    frame.bDb = finite(values.b_db) ? values.b_db : null;
    // A merged frame keeps the timestamp of the attention packet it belongs to.
    if (!finite(frame.mediaTime)) frame.mediaTime = finite(values.mediaTime) ? values.mediaTime : null;
    if (!frame.decision) frame.decision = null;
  }
  const next = index >= 0 ? history.with(index, frame) : [...history, frame];
  next.sort((left, right) => left.sequence - right.sequence);
  return next.length > MAX_FRAMES ? next.slice(next.length - MAX_FRAMES) : next;
}

/** Derive the timeline, the raw difference series and the coverage counts. */
export function storyLayout(history = [], { currentTime = null, duration = null } = {}) {
  const frames = (Array.isArray(history) ? history : []).filter(frame => frame.kind === 'attention');
  const scored = frames.filter(frame => finite(frame.correlationA) && finite(frame.correlationB));
  // The difference is the quantity the decision rule compares: kept raw.
  const points = scored.map(frame => ({ sequence: frame.sequence, mediaTime: frame.mediaTime,
    difference: frame.correlationA - frame.correlationB }));
  const observed = frames.filter(frame => finite(frame.mediaTime)).map(frame => frame.mediaTime);
  const lastTime = observed.length ? Math.max(...observed) : null;
  const viewEnd = Math.max(finite(duration) && duration > 0 ? duration : 0, lastTime ?? 0, 1);
  const playhead = finite(currentTime) ? Math.min(Math.max(currentTime, 0), viewEnd) : lastTime;
  const count = decision => frames.filter(frame => frame.decision === decision).length;
  const coverage = { frames: frames.length, withEvidence: scored.length, decided: count('A') + count('B'),
    abstained: count('uncertain'), absent: count('unavailable'),
    byDecision: { A: count('A'), B: count('B'), uncertain: count('uncertain'), unavailable: count('unavailable') } };
  // Score magnitudes only: a shared axis for the two bars and the trace.
  const magnitudes = scored.flatMap(frame => [Math.abs(frame.correlationA), Math.abs(frame.correlationB)]);
  return { frames, scored, points, coverage, viewEnd, playhead, present: frames.length > 0,
    axisMax: Math.max(MARGIN, ...magnitudes) };
}

/** Turn the latest gain packet into the sentence a listener would understand. */
export function gainConsequence(gain) {
  const rows = [['A', finite(gain?.aDb) ? gain.aDb : null], ['B', finite(gain?.bDb) ? gain.bDb : null]];
  const lowered = rows.filter(([, db]) => finite(db) && db < 0);
  return { lowered: lowered.map(([id, db]) => ({ id, db })), attenuated: lowered.length > 0,
    note: lowered.length ? `${lowered.map(([id, db]) => `${decisionPhrase(id)} turned down to ${formatDb(db)}`).join(' · ')}; the other source is unchanged` : null };
}

// The decision strip is capped at this many seconds.
//
// The media can be fifteen minutes long -- the ANT demo's candidates are 900 s --
// and a strip drawn across the whole of it compresses a minute of decisions into
// 7 % of the width: the marks pile into a stub at the left edge and the reader
// learns nothing from them. The strip therefore shows one window at a time, so its
// length is bounded whatever the source is.
//
// The window SLIDES: it ends at the newest reported position and is one cap long.
// Advancing it in whole-cap steps instead was measured at 8-17 % full for most of
// every cycle (a 130 s source drew 41 of its 521 marks, across the left sixth of
// the strip) -- the same stub, moved rather than fixed. Sliding keeps the strip
// full from the first cap onwards, and the frames that fall off the left are the
// refresh.
export const TIMELINE_WINDOW_SECONDS = 60;

/** The window the strip is showing, and the frames that belong to it. */
export function timelineWindow(frames = [], { currentTime = null } = {}) {
  const times = (frames ?? []).map(frame => frame.mediaTime).filter(finite);
  // Anchored on the NEWEST FRAME, never on the playhead. The backend's position
  // copy lags this page's clock (measured at up to ~6.5 s), so anchoring on the
  // playhead pushed the window past the data the moment a boundary was crossed and
  // the strip drew nothing at all -- the exact opposite of the point.
  const newest = Math.max(0, ...(times.length ? times : [0]));
  const end = Math.max(TIMELINE_WINDOW_SECONDS, newest);
  const start = end - TIMELINE_WINDOW_SECONDS;
  return { start, end,
    shown: (frames ?? []).filter(frame => finite(frame.mediaTime) && frame.mediaTime >= start && frame.mediaTime <= end),
    position: finite(currentTime) ? Math.min(Math.max(currentTime, start), end) : null };
}

const percent = value => `${value}%`;
const svgProps = { viewBox: '0 0 1000 100', preserveAspectRatio: 'none', role: 'img' };

function StoryTimeline({ layout }) {
  const { frames, playhead } = layout;
  const window = timelineWindow(frames, { currentTime: playhead });
  const { start, end, shown } = window;
  const x = time => Math.min(100, Math.max(0, (finite(time) ? time - start : 0) / TIMELINE_WINDOW_SECONDS * 100));
  const lastReceived = shown.reduce((max, frame) => Math.max(max, frame.mediaTime ?? start), start);
  // Before the first frame there is no position to point at: show the window's
  // start, not a guess.
  const position = window.position ?? start;
  // One frame per 0.25 s of a 124 s trial: ~0.16 % wide, distinct at any length.
  const markWidth = Math.max(0.1, Math.min(0.9, 100 / Math.max(1, shown.length)));
  return h('div', { className: 'timeline-block' },
    h('h3', { className: 'timeline-heading' }, 'Decision timeline · every frame of this session'),
    h('div', { className: 'timeline-track' },
      h('svg', { ...svgProps, className: `timeline-strip${layout.present ? '' : ' empty'}`,
        'aria-label': `Decisions for ${shown.length} received frames in the last ${TIMELINE_WINDOW_SECONDS} seconds of playback, ${start.toFixed(1)} to ${end.toFixed(1)} s` },
        ...shown.map(frame => h('rect', { key: frame.sequence, x: percent(x(frame.mediaTime)), y: 0, height: '100%',
          width: percent(markWidth), className: `timeline-mark decision-${frame.decision}` }))),
      layout.present ? h('span', { className: 'timeline-absent', style: { left: percent(x(lastReceived)) }, 'aria-hidden': true }) : null,
      h('span', { className: 'timeline-playhead', style: { left: percent(x(position)) } },
        h('span', { className: 'sr-only' }, `Playback position ${position.toFixed(1)} seconds`))),
    h('p', { className: 'timeline-legend' },
      ...DECISION_ORDER.map(decision => h('span', { key: decision, className: `legend-item legend-${DECISION_TONE[decision]}` },
        h('i', { className: `legend-swatch decision-${decision}`, 'aria-hidden': true }),
        `${decisionPhrase(decision)} · ${layout.coverage.byDecision[decision]}`)),
      h('span', { className: 'legend-item legend-absent' }, h('i', { className: 'legend-swatch', 'aria-hidden': true }), 'nothing received yet')),
    h('p', { className: 'timeline-note' }, `One mark per attention packet this page received in the last ${TIMELINE_WINDOW_SECONDS} s — ${start.toFixed(1)} to ${end.toFixed(1)} s — placed at the playback position the backend reported for it. The strip is capped at ${TIMELINE_WINDOW_SECONDS} s and slides, so a fifteen-minute source cannot compress the decisions into a sliver; older frames fall off the left. No interpolation: the hatched area right of the playhead has not been streamed yet, and gaps inside it are frames that never arrived.`));
}

export function DecisionStory({ history = [], state, inactive = false }) {
  const stream = state?.streams?.findLast(s => s.type === 'attention');
  const duration = state?.streams?.findLast(s => s.type === 'media')?.values.duration;
  const layout = storyLayout(history, { currentTime: stream?.values.mediaTime ?? null, duration });
  const current = history.findLast(frame => frame.kind === 'attention');
  const gain = history.findLast(frame => frame.kind === 'gain' && frame.aDb !== null);
  const decision = inactive ? 'unavailable' : current?.decision ?? 'unavailable';
  const consequence = gainConsequence(gain);
  const action = inactive ? 'Playback is not live: the timeline and the counts below are what this session has already produced.'
    : !current ? 'Waiting for the first attention packet.'
    : ['A', 'B'].includes(decision) ? (consequence.attenuated ? `Both sources play, but ${consequence.note}.` : 'The system committed to a source, and the latest gain packet leaves both channels at their original level.')
    : decision === 'uncertain' ? 'The system is abstaining: the two scores are too close to call, so neither source is attenuated.'
    : 'No decision yet at this playback position — the newest frame carries no attention data.';
  const difference = layout.scored.length ? layout.points[layout.points.length - 1].difference : null;
  const reason = current?.reasons?.length ? current.reasons.join(', ') : 'none reported';
  return h('section', { className: 'decision-story', 'aria-label': 'Decision story' },
    h('h2', { className: 'comparison-heading' }, 'What the system is doing right now'),
    h('div', { className: `story-verdict verdict-${decision}`, role: 'status', 'aria-live': 'polite' },
      h('p', { className: 'verdict-label' }, 'Attended source'),
      h('p', { className: 'verdict-value' }, decisionPhrase(decision)),
      h('p', { className: 'verdict-action' }, action)),
    h(StoryTimeline, { layout }),
    h('p', { className: 'story-coverage' }, `${layout.coverage.frames} attention frames received · attributed ${layout.coverage.decided} · abstained ${layout.coverage.abstained} · no data ${layout.coverage.absent}. Every figure on this page is a value the backend sent, drawn where and when it arrived; this page never infers attention of its own.`),
    h('p', { className: 'story-code' }, `Latest frame · decision ${decision} · reason: ${reason} · score A ${formatScore(current?.correlationA)} · score B ${formatScore(current?.correlationB)} · difference ${formatScore(difference)} · gain A ${formatDb(gain?.aDb)} · gain B ${formatDb(gain?.bDb)}`));
}

export function DifferenceTrace({ layout }) {
  const points = layout.points.map(point => ({ ...point, x: Math.max(0, Math.min(1000, (point.mediaTime ?? 0) / layout.viewEnd * 1000)) }));
  const scale = Math.max(MARGIN, ...points.map(point => Math.abs(point.difference)));
  const y = difference => 50 - Math.max(-1, Math.min(1, difference / scale)) * 42;
  const band = Math.max(2, MARGIN / scale * 42);
  return h('div', { className: 'trace-block' },
    h('h3', { className: 'timeline-heading' }, 'Which source wins, over the session'),
    h('svg', { ...svgProps, className: 'difference-trace',
      'aria-label': `Score difference between source A and source B for ${points.length} scored windows` },
      h('rect', { x: 0, y: 50 - band, width: 1000, height: band * 2, className: 'trace-margin' }),
      h('line', { x1: 0, y1: 50, x2: 1000, y2: 50, className: 'trace-zero' }),
      points.length > 1 ? h('polyline', { className: 'trace-line', fill: 'none',
        points: points.map(point => `${point.x.toFixed(1)},${y(point.difference).toFixed(1)}`).join(' ') }) : null,
      points.map(point => h('circle', { key: point.sequence, r: 2.5, cx: point.x, cy: y(point.difference),
        className: `trace-dot ${point.difference >= 0 ? 'trace-a' : 'trace-b'}` }))),
    h('p', { className: 'trace-note' }, `Raw score difference (A minus B) for every scored window — above the line favours A, below favours B. Vertical scale ±${scale.toFixed(3)}, taken from the largest difference this session has produced; the shaded band is the ±${MARGIN.toFixed(2)} margin the decision rule needs. Nothing is smoothed, so each switch is a visible change of sign.`));
}

export function SourceComparison({ history = [], state, inactive = false }) {
  const values = state?.streams?.findLast(s => s.type === 'attention')?.values ?? {};
  const correlationA = finite(values.correlationA) ? values.correlationA : null;
  const correlationB = finite(values.correlationB) ? values.correlationB : null;
  const decision = inactive ? 'unavailable' : Object.hasOwn(values, 'decision') ? values.decision
    : ['A', 'B'].includes(values.attended) ? values.attended : 'unavailable';
  const sources = state?.streams?.findLast(s => s.type === 'audio_sources')?.values.sources ?? [];
  const label = id => sources.find(source => source.id === id)?.label ?? `Source ${id}`;
  const measured = storyLayout(history, {}).axisMax;
  const scale = measured > MARGIN ? measured : FALLBACK_SCALE;
  const bar = (id, correlation) => h('div', { className: `axis-row${decision === id ? ' axis-row-selected' : ''}`, key: id },
    h('span', { className: 'axis-name' }, `Source ${id}`, h('small', null, label(id))),
    h('div', { className: 'axis-track', role: 'img',
      'aria-label': `Source ${id} score ${formatScore(correlation)} on a shared scale of plus or minus ${scale.toFixed(3)}` },
      h('span', { className: `axis-bar bar-${id}`, style: correlation === null ? { opacity: 0 } : correlation >= 0
        ? { left: '50%', width: percent(Math.min(50, Math.abs(correlation) / scale * 50)) }
        : { right: '50%', width: percent(Math.min(50, Math.abs(correlation) / scale * 50)) } })),
    h('span', { className: 'axis-value' }, formatScore(correlation)));
  const difference = correlationA !== null && correlationB !== null ? Math.abs(correlationA - correlationB) : null;
  const marginText = difference === null ? 'Not measurable — one of the two scores is missing.'
    : `${difference.toFixed(4)} — ${difference >= MARGIN ? 'at or above' : 'below'} the ${MARGIN.toFixed(2)} margin, which is why the system ${['A', 'B'].includes(decision) ? 'committed to a source' : 'abstained'}.`;
  const favouring = difference === null ? 'Unavailable' : correlationA > correlationB ? 'Source A scores higher' : correlationA < correlationB ? 'Source B scores higher' : 'The two scores are equal';
  return h('section', { className: 'source-comparison', 'aria-label': 'Source comparison' },
    h('h2', { className: 'comparison-heading' }, 'Source comparison'),
    h('div', { className: 'axis' },
      h('div', { className: 'axis-head' }, h('span', null, '0 = this voice\'s rhythm has nothing in common with the EEG'), h('span', null, `shared scale ±${scale.toFixed(3)}`)),
      bar('A', correlationA), bar('B', correlationB),
      h('p', { className: 'axis-note' }, 'Both bars use one scale, so what you see is the difference the decision uses. These scores are correlations, not probabilities: 0.2 is a weak but real match, and 1.0 would be a perfect one. The height matters far less than which bar is longer.')),
    h('dl', { className: 'story-metrics' },
      h('div', null, h('dt', null, 'Difference A − B'), h('dd', null, marginText)),
      h('div', null, h('dt', null, 'Who is ahead'), h('dd', null, favouring)),
      h('div', null, h('dt', null, 'Current decision'), h('dd', null, `${decision} · ${decisionPhrase(decision)}`))));
}
