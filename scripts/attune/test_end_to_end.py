"""Real ANT recording -> separate PlayerLSL -> NOVA -> HTTP/WebSocket -> React decoder.

This checks software transport, not acoustic timing or held-out classification.
Run after convert_ant and training. Outputs reproducible JSON and WAV evidence.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np


def player(raw_path, name, ready):
    from mne_lsl.player import PlayerLSL
    p = PlayerLSL(raw_path, chunk_size=16, name=name, n_repeat=1)
    p.start()
    Path(ready).write_text('ready')
    try:
        while p.running:
            time.sleep(.1)
    finally:
        if p.running:
            p.stop()


async def exercise(args):
    import httpx
    import websockets
    import mne
    from nova2026.auditory.data import load_trial, save_trial
    trial = load_trial(args.trial)
    if args.seconds + 15 >= len(trial.eeg)/trial.sample_rate:
        raise ValueError('Player needs at least 15 seconds of unused real EEG tail.')
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out/'source_raw.fif'
    info = mne.create_info(list(trial.channel_names), round(trial.sample_rate), 'eeg')
    mne.io.RawArray(trial.eeg.T/1e6, info, verbose='ERROR').save(raw_path, overwrite=True, verbose='ERROR')
    trial.audio = trial.audio[:round(args.seconds*trial.audio_rate)]
    trial_path = args.out/'live_trial.npz'
    save_trial(trial, trial_path)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    name = 'attune-real-'+uuid4().hex[:8]
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root/'src'), str(root), str(args.ui)]),
           'ATTUNE_NOVA_TRIAL': str(trial_path.resolve()), 'ATTUNE_NOVA_MODEL': str(args.model.resolve()),
           'ATTUNE_NOVA_STREAM': name, 'ATTUNE_NOVA_OUTPUT': 'wav',
           'ATTUNE_NOVA_PRESENTATION': args.presentation, 'ATTUNE_NOVA_CHANNEL_POLICY': 'record-only',
           'ATTUNE_NOVA_REPORT_DIR': str((args.out/'runs').resolve())}
    if args.managed_player:
        env['ATTUNE_NOVA_PLAYER_RAW'] = str(raw_path.resolve())
    processes = []
    log = (args.out/'process.log').open('w')
    packets = []
    started = time.monotonic()
    try:
        processes.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app.server:app',
                         '--host', '127.0.0.1', '--port', str(port)], env=env, stdout=log, stderr=log))
        url = f'http://127.0.0.1:{port}'
        async with httpx.AsyncClient(base_url=url, timeout=5.) as client:
            deadline = time.monotonic()+45
            while True:
                try:
                    health = await client.get('/api/health')
                    if health.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline or processes[0].poll() is not None:
                    raise RuntimeError('Backend startup failed; see process.log')
                await asyncio.sleep(.1)
            ready = args.out/(name+'.ready')
            if not args.managed_player:
                processes.append(subprocess.Popen([sys.executable, '-m', 'scripts.attune.player',
                                  '--raw', str(raw_path.resolve()), '--name', name,
                                  '--ready', str(ready.resolve())], env=env, stdout=log, stderr=log))
                deadline = time.monotonic()+20
                while not ready.exists():
                    if time.monotonic() > deadline or processes[-1].poll() is not None:
                        raise RuntimeError('Player startup failed')
                    await asyncio.sleep(.1)
            async with websockets.connect(f'ws://127.0.0.1:{port}/ws/live') as ws:
                response = await client.post('/api/session/start')
                response.raise_for_status()
                sid = response.json()['id']
                stop_seconds = None
                async with asyncio.timeout(args.seconds+25):
                    while True:
                        packet = json.loads(await ws.recv())
                        packets.append(packet)
                        if (args.stop_after and stop_seconds is None and
                                sum(p['type'] == 'attention' for p in packets) >= args.stop_after):
                            tick = time.monotonic()
                            stopped = await client.post('/api/session/stop')
                            stop_seconds = time.monotonic()-tick
                            assert stopped.status_code == 200, stopped.text
                            assert stop_seconds < 3., stop_seconds
                        if packet['type'] == 'session' and packet['payload']['status'] in ('stopped', 'error'):
                            assert packet['payload']['status'] == 'stopped', packet
                            break
                snapshot = (await client.get('/api/state')).json()
                assert snapshot['sequence'] >= packets[-1]['sequence']
                assert (await client.get('/api/sessions/'+sid)).json()['status'] == 'stopped'
        sequences = [p['sequence'] for p in packets]
        assert all(b > a for a, b in zip(sequences, sequences[1:])), 'sequence regression'
        kinds = {p['type'] for p in packets}
        assert {'attention', 'prediction', 'eeg_display', 'sync', 'gain', 'signal_quality', 'presentation'} <= kinds, kinds
        last = {}
        for p in packets:
            key = (p['source'], p['type'])
            assert np.isfinite(p['timestamp']) and p['timestamp'] >= last.get(key, 0.)
            last[key] = p['timestamp']
            if p['source'] == 'nova_aad':
                assert p['payload']['simulated'] is False, p
        attention = [p for p in packets if p['type'] == 'attention']
        assert len(attention) >= 5
        assert any(p['payload']['correlation_a'] is not None for p in attention), 'No usable real EEG scores'
        for p in packets:
            if p['type'] == 'sync':
                assert p['payload']['offset_ms'] is None and p['payload']['drift_warning'] is None
        reports = list((args.out/'runs').glob('*/timing.json'))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report['consumer_errors'] == report['acquire_gaps'] == report['recovery_segments'] == 0, report
        ends = np.array([e['evidence_end'] for e in report['estimates']])
        assert np.max(np.abs(np.diff(ends)-1)) < .005
        assert report['presentation']['mode'] == args.presentation
        from scipy.io import wavfile
        _, audio = wavfile.read(reports[0].with_name('mixed.wav'))
        assert audio.ndim == 2 and audio.shape[1] == 2 and np.all(np.isfinite(audio))
        if args.presentation == 'diotic':
            np.testing.assert_array_equal(audio[:, 0], audio[:, 1])
        else:
            assert not np.array_equal(audio[:, 0], audio[:, 1])
        packet_path = args.out/'packets.json'
        packet_path.write_text(json.dumps(packets))
        # Pass actual network packets through the shipped JS decoder, not a Python imitation.
        decoder = (args.ui/'frontend/src/decoders.js').resolve().as_uri()
        js = f"""import fs from 'node:fs'; import assert from 'node:assert/strict';
import {{decodePacket}} from {json.dumps(decoder)};
const packets=JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
for(const p of packets.filter(p=>p.source==='nova_aad')) {{
 const d=decodePacket(p); assert.equal(d.known,true); assert.equal(d.simulated,false);
 if(p.type==='attention') assert.ok(['A','B','uncertain','unavailable'].includes(d.values.decision));
 if(p.type==='presentation') assert.equal(d.values.mode,{json.dumps(args.presentation)});
}} console.log('Real packets decoded by frontend');"""
        subprocess.run(['node', '--input-type=module', '-e', js, str(packet_path.resolve())], check=True)
        result = {'passed': True, 'source': Path(args.trial).name, 'source_is_recorded_eeg': True,
                  'acoustic_timing_verified': False, 'accuracy_evaluated': False,
                  'presentation': args.presentation, 'packets': len(packets),
                  'attention_packets': len(attention), 'stop_seconds': stop_seconds,
                  'elapsed_seconds': time.monotonic()-started,
                  'acquire_gaps': report['acquire_gaps'], 'recovery_segments': report['recovery_segments'],
                  'consumer_errors': report['consumer_errors'], 'frontend_decoder': 'passed'}
        (args.out/'evidence.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--player', type=Path)
    p.add_argument('--name')
    p.add_argument('--ready')
    p.add_argument('--trial', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--ui', type=Path, default=Path('../attune-ui').resolve())
    p.add_argument('--seconds', type=float, default=40)
    p.add_argument('--presentation', choices=('dichotic', 'diotic'), default='dichotic')
    p.add_argument('--stop-after', type=int, default=0, help='Stop after this many attention packets')
    p.add_argument('--managed-player', action='store_true', help='Exercise production per-session source lifecycle')
    p.add_argument('--out', type=Path, default=Path('results/attune-e2e'))
    args = p.parse_args()
    if args.player:
        player(args.player, args.name, args.ready)
    else:
        asyncio.run(exercise(args))


if __name__ == '__main__':
    main()
