import { createStereoAudio, playbackGains } from './mediaAudio.js';

export function createMediaController({ element, config, rest, onChange = () => {},
  audioFactory = createStereoAudio, now = () => performance.now(), clientId = globalThis.crypto.randomUUID() }) {
  let audio, state, requestId = 0, sessionId, queue = Promise.resolve(), disposed = false;
  let mode = 'original', lastAck = -Infinity, revision = null, prepared = false, error = null;
  let busy = false, playbackState = 'stopped', generation = 0, internalSeek = false;
  const snapshot = () => ({ mediaId: config.media_id, revision, time: element.currentTime || 0,
    duration: Number.isFinite(element.duration) ? element.duration : null, playbackState,
    ready: prepared && now() - lastAck <= 1500, error, busy, mode });
  function emit() { if (!disposed) onChange(snapshot()); }
  function neutral() { audio?.apply([1, 1]); }
  function fail() {
    error = 'Playback synchronization unavailable. Stop, then Play to reconnect.';
    element.pause(); playbackState = 'paused'; lastAck = -Infinity; neutral(); emit();
  }
  async function send(action) {
    if (!sessionId) throw Error('Start a session first');
    const started = now();
    const activeSession = sessionId;
    const token = generation;
    const result = await rest.mediaControl({ session_id: sessionId, media_id: config.media_id,
      client_id: clientId, request_id: ++requestId, action,
      media_time_s: action === 'prepare' || action === 'stopped' ? 0 : element.currentTime,
      duration_s: element.duration }, AbortSignal.timeout(1500));
    if (disposed || token !== generation || activeSession !== sessionId || result.session_id !== sessionId || result.media_id !== config.media_id || now() - started > 1500 ||
        !Number.isInteger(result.revision) || result.sync_status !== (action === 'stopped' ? 'desynchronized' : 'observed')) throw Error('Invalid media acknowledgement');
    revision = result.revision; lastAck = now();
    return result;
  }
  function enqueue(task) {
    const operation = queue.then(task);
    queue = operation.catch(() => {});
    return operation;
  }
  function rewind() {
    if (element.currentTime !== 0) { internalSeek = true; element.currentTime = 0; }
  }
  function resetLocal() {
    ++generation; element.pause(); rewind();
    prepared = false; playbackState = 'stopped'; lastAck = -Infinity; neutral(); emit();
  }
  return {
    snapshot,
    update(next) {
      state = next;
      const session = state.streams.findLast(s => s.type === 'session')?.values.status;
      if (sessionId !== state.sessionId || session !== 'running') {
        if (sessionId || prepared) resetLocal();
        sessionId = session === 'running' ? state.sessionId : null;
      }
      if (state.connection !== 'connected' || state.stale) { element.pause(); if (prepared) playbackState = 'paused'; lastAck = -Infinity; neutral(); }
      audio?.apply(playbackGains(state, snapshot(), mode)); emit();
    },
    setMode(value) { mode = value === 'attune' ? 'attune' : 'original'; audio?.apply(state ? playbackGains(state, snapshot(), mode) : [1, 1]); emit(); },
    async play() {
      if (busy || !sessionId || state?.connection !== 'connected' || state?.stale) return;
      busy = true; error = null; emit();
      const token = generation;
      try {
        audio ??= audioFactory(element);
        await audio.resume();
        await enqueue(async () => {
          if (token !== generation || disposed) return;
          if (!prepared) { rewind(); await send('prepare'); prepared = true; }
          if (token !== generation || disposed) return;
          await element.play();
          if (token !== generation || disposed) { element.pause(); return; }
          playbackState = 'playing';
          await send('playing');
        });
      } catch { if (token === generation && !disposed) fail(); }
      finally { busy = false; emit(); }
    },
    async pause() {
      ++generation;
      element.pause(); playbackState = 'paused'; neutral(); emit();
      if (!prepared) return;
      try { await enqueue(() => send('paused')); } catch { fail(); }
      emit();
    },
    async stop() {
      const canNotify = Boolean(sessionId) && Number.isFinite(element.duration);
      resetLocal(); error = null;
      if (canNotify) {
        busy = true; emit();
        try { await enqueue(() => send('stopped')); } catch { error = 'Stop acknowledgement unavailable; restart the session before playback.'; }
        finally { busy = false; }
      }
      emit();
    },
    async tick() {
      if (disposed) return;
      if (state) audio?.apply(playbackGains(state, snapshot(), mode));
      const token = generation;
      if (prepared && !busy && !error && state?.connection === 'connected' && !state.stale) {
        busy = true;
        try { await enqueue(() => send('report')); } catch { if (token === generation && !disposed) fail(); }
        finally { busy = false; }
      }
      emit();
    },
    interrupted() { if (prepared && playbackState === 'playing') { element.pause(); playbackState = 'paused'; neutral(); void enqueue(() => send('paused')).catch(fail); emit(); } },
    seeking() { if (!internalSeek && prepared && playbackState !== 'stopped') fail(); },
    seeked() { internalSeek = false; },
    invalid() { fail(); },
    dispose() { disposed = true; ++generation; element.pause(); neutral(); void audio?.close(); },
  };
}
