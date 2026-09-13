// Playback mechanics only. Decisions and dB gains always come from backend packets.
export const dbToLinear = db => 10 ** (db / 20);
export function formatMediaTime(value) {
  if (!Number.isFinite(value) || value < 0) return '--:--.--';
  const hundredths = Math.floor((value + Number.EPSILON) * 100);
  return `${String(Math.floor(hundredths / 6000)).padStart(2, '0')}:${String(Math.floor(hundredths / 100) % 60).padStart(2, '0')}.${String(hundredths % 100).padStart(2, '0')}`;
}
export const mediaProgress = (time, duration) => Number.isFinite(time) && Number.isFinite(duration) && duration > 0 ? Math.max(0, Math.min(1, time / duration)) : 0;
export function mediaFocusReady(state, playback) {
  const latest = type => state.streams.findLast(s => s.type === type);
  const attention = latest('attention');
  const media = latest('media');
  const sync = latest('sync');
  const session = latest('session');
  return state.connection === 'connected' && !state.stale && !state.error &&
    session?.values.status === 'running' && playback.ready && !playback.error &&
    ['playing', 'paused'].includes(playback.playbackState) &&
    media?.values.syncStatus === 'observed' && media.values.playbackState === playback.playbackState &&
    media.values.revision === playback.revision && media.values.mediaId === playback.mediaId &&
    !['desynchronized', 'invalid'].includes(sync?.values.status) &&
    attention?.sessionId === state.sessionId && attention.values.mediaId === playback.mediaId &&
    attention.values.mediaRevision === playback.revision &&
    Number.isFinite(attention.values.mediaTime) && Math.abs(attention.values.mediaTime - playback.time) <= .75 &&
    ['A', 'B'].includes(Object.hasOwn(attention.values, 'decision') ? attention.values.decision : attention.values.attended);
}
export function playbackGains(state, playback, mode) {
  const neutral = [1, 1];
  if (mode !== 'attune' || playback.playbackState !== 'playing' || !mediaFocusReady(state, playback)) return neutral;
  const gain = state.streams.findLast(s => s.type === 'gain');
  const attention = state.streams.findLast(s => s.type === 'attention');
  if (gain?.values.mediaTime !== attention?.values.mediaTime || gain?.sessionId !== state.sessionId || gain.values.mediaId !== playback.mediaId ||
      gain.values.mediaRevision !== playback.revision || !Number.isFinite(gain.values.mediaTime) ||
      Math.abs(gain.values.mediaTime - playback.time) > .75) return neutral;
  const db = [gain.values.a_db, gain.values.b_db];
  // This player attenuates only; unexpected amplification fails neutral.
  return db.every(v => Number.isFinite(v) && v >= -80 && v <= 0) ? db.map(dbToLinear) : neutral;
}
export function createStereoAudio(element, Context = globalThis.AudioContext ?? globalThis.webkitAudioContext) {
  if (!Context) throw Error('Web Audio unavailable');
  const context = new Context();
  const source = context.createMediaElementSource(element);
  const splitter = context.createChannelSplitter(2);
  const merger = context.createChannelMerger(2);
  const gains = [context.createGain(), context.createGain()];
  source.connect(splitter);
  gains.forEach((gain, index) => { splitter.connect(gain, index); gain.connect(merger, 0, index); });
  merger.connect(context.destination);
  let previous = [1, 1];
  return {
    resume: () => context.resume(),
    apply(values) {
      values.forEach((value, i) => {
        if (value === previous[i]) return;
        const param = gains[i].gain;
        // 0.1 s time constant reaches ~98% of the target in 400 ms.
        param.setTargetAtTime(value, context.currentTime, .1);
      });
      previous = [...values];
    },
    async close() {
      element.pause();
      for (const node of [source, splitter, ...gains, merger]) node.disconnect();
      await context.close();
    },
  };
}
