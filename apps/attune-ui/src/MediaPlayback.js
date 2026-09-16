import React, { useEffect, useRef, useState } from 'react';
import { createMediaController } from './mediaController.js';
import { formatMediaTime, mediaProgress, mediaLinked } from './mediaAudio.js';
const h = React.createElement;
export function MediaPlayback({ config, rest, state, stopSignal = 0, children }) {
  const element = useRef(null), controller = useRef(null);
  const [playback, setPlayback] = useState({ time: 0, duration: null, playbackState: 'stopped', mode: 'original', ready: false });
  useEffect(() => {
    const instance = createMediaController({ element: element.current, config, rest, onChange: setPlayback });
    controller.current = instance;
    instance.update(state);
    const timer = setInterval(() => { void instance.tick(); }, 250);
    return () => { clearInterval(timer); instance.dispose(); controller.current = null; };
  }, [config, rest]);
  useEffect(() => { controller.current?.update(state); }, [state]);
  useEffect(() => { if (stopSignal) void controller.current?.stop(); }, [stopSignal]);
  // Live-or-not is about playback, not about whether the newest window committed:
  // an abstention is a result the page must show, not a dead player.
  const inactive = !mediaLinked(state, playback);
  return h('section', { className: 'media-area', 'aria-label': 'Media playback' },
    h('div', { className: 'main-media-card' },
    h('div', { className: 'media-heading' }, h('h2', null, config.title || 'Demo Audio'),
      h('p', { className: 'media-time' }, h('span', { className: 'sr-only' }, 'Media Time '), `${formatMediaTime(playback.time)} / ${formatMediaTime(playback.duration)}`),
      h('div', { className: 'media-progress', role: 'progressbar', 'aria-label': 'Media progress', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': mediaProgress(playback.time, playback.duration) * 100, style: { '--progress': `${mediaProgress(playback.time, playback.duration) * 100}%` } }, h('span', { className: 'media-progress-fill' }), h('span', { className: 'media-progress-dot' })),
      h('p', { className: 'sr-only' }, `${playback.playbackState} · ${playback.ready ? 'Media position acknowledged' : 'Waiting for media synchronization'}`)),
    h(config.kind === 'video' ? 'video' : 'audio', { ref: element, src: config.url, preload: 'metadata', playsInline: true,
      className: 'shared-media', controls: false, disablePictureInPicture: true,
      onLoadedMetadata: () => setPlayback(current => ({ ...current, duration: element.current.duration })),
      onEnded: () => { void controller.current?.stop(); }, onWaiting: () => controller.current?.interrupted(),
      onPause: () => controller.current?.interrupted(),
      onSeeked: () => controller.current?.seeked(),
      onSeeking: () => controller.current?.seeking(),
      onError: () => controller.current?.invalid(),
      onRateChange: () => { if (element.current.playbackRate !== 1) { element.current.playbackRate = 1; controller.current?.invalid(); } } }),
    h('div', { className: 'button-group transport-buttons' },
        h('button', { className: 'primary', disabled: playback.busy || !state.sessionId || state.connection !== 'connected' || !playback.duration,
          onClick: () => { void controller.current?.play(); } }, 'Play'),
        h('button', { 'aria-label': 'Pause', title: 'Pause', disabled: playback.playbackState !== 'playing', onClick: () => { void controller.current?.pause(); } }, 'Pause'),
        h('button', { 'aria-label': 'Stop', title: 'Stop', onClick: () => { void controller.current?.stop(); } }, 'Stop'))),
    children({ inactive }),
    h('div', { className: 'playback-controls session-playback-controls audio-mode-card' },
      h('h2', null, 'Audio Mode'),
        h('div', { className: 'button-group audio-mode-buttons', role: 'group', 'aria-label': 'Audio mode' },
        ...[['original', 'Original'], ['attune', 'ATTUNE']].map(([mode, label]) => h('button', {
          key: mode, 'aria-label': mode === 'original' ? 'Original Mix' : 'ATTUNE', 'aria-pressed': playback.mode === mode, onClick: () => controller.current?.setMode(mode),
        }, label))),
      h('p', null, playback.mode === 'original' ? 'Unmodified stereo mix' : !inactive && playback.playbackState === 'playing' ? 'Adaptive focus applied' : 'Neutral mix · waiting for current focus'),
      playback.error && h('p', { role: 'alert' }, playback.error)));
}
