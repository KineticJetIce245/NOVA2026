import React from 'react';
const h = React.createElement;
export function FeedbackPanel({ stream, stale = false, inactive = false }) {
  const feedback = stream?.values;
  const active = !stale && !inactive && feedback?.status === 'active';
  return h('section', { className: `feedback-panel${active ? ' feedback-active' : ''}`, 'aria-label': 'Feedback' },
    h('h2', null, 'Feedback'),
    h('strong', { className: 'development-badge' }, 'SIMULATED · DEVELOPMENT ONLY'),
    h('p', { role: active ? 'alert' : 'status' }, stale ? 'Feedback status: stale — not active' : inactive ? 'Feedback status: suppressed — session inactive' : `Feedback status: ${feedback?.status ?? 'none'}`),
    h('p', null, feedback?.message ?? 'No development feedback received.'),
    h('p', null, `Severity: ${feedback?.severity ?? 'info'} · Timestamp: ${stream?.timestamp ?? 'unavailable'}`),
    h('p', null, `Reason: ${feedback?.reason ?? 'waiting for mock results'}`),
    h('p', null, `Cooldown remaining at this event: ${feedback?.cooldownRemaining ?? 0} s`),
    h('p', null, 'Visual test only. No scientific interpretation, audio, haptic, or device action.'));
}
