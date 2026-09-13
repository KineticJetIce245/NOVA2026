// Explicit local integration suite; requires the Phase 1 Python environment.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createServer, mergeConfig } from 'vite';
import config from '../vite.config.js';
import { createRestClient } from '../src/rest.js';
import { createTransport } from '../src/transport.js';
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate) {
  const deadline = Date.now() + 10000;
  while (!predicate()) {
    if (Date.now() > deadline) throw Error('Timed out waiting for transport state');
    await delay(20);
  }
}
test('F02-T23 live Python to Vite proxy to frontend decoders and reconnect', { timeout: 30000 }, async () => {
  const child = spawn(process.env.ATTUNE_PYTHON || 'python3', ['-B', '-m', 'uvicorn', 'backend.app.server:app',
    '--host', '127.0.0.1', '--port', '0'], { cwd: new URL('../../', import.meta.url), stdio: ['ignore', 'ignore', 'pipe'] });
  let logs = '', port, startupError, vite, transport;
  const exited = new Promise(resolve => { child.once('exit', resolve); child.once('error', resolve); });
  child.on('error', error => { startupError = error; });
  child.stderr.on('data', data => { logs += data; port = logs.match(/Uvicorn running on http:\/\/127\.0\.0\.1:(\d+)/)?.[1]; });
  try {
    await until(() => port || startupError || child.exitCode !== null);
    if (!port) throw Error(`Backend did not start: ${startupError ?? logs}`);
    vite = await createServer(mergeConfig(config, { configFile: false, root: fileURLToPath(new URL('../', import.meta.url)),
      server: { port: 0, proxy: { '/api': { target: `http://127.0.0.1:${port}` },
        '/ws': { target: `ws://127.0.0.1:${port}`, ws: true } } } }));
    await vite.listen();
    const origin = `http://127.0.0.1:${vite.httpServer.address().port}`;
    const rest = createRestClient((path, options) => fetch(origin + path, options));
    assert.equal((await rest.health()).protocol_version, 1);
    const html = await (await fetch(origin)).text(); assert.ok(html.includes('ATTUNE'));
    let state, liveSocket;
    transport = createTransport({ rest, url: origin.replace('http:', 'ws:') + '/ws/live',
      socketFactory: url => { liveSocket = new WebSocket(url); return liveSocket; }, onState: next => { state = next; } });
    transport.start(); await until(() => state?.connection === 'connected');
    const session = await rest.start(); assert.equal(session.simulated, true);
    await until(() => ['attention', 'vigilance', 'sync', 'eeg_display'].every(type => state.streams.some(s => s.type === type && s.known)));
    assert.equal(state.streams.find(s => s.type === 'attention').values.attended, 'A');
    assert.equal(state.streams.find(s => s.type === 'sync').values.offsetMs, null);
    const sequence = state.sequence;
    liveSocket.close(); await until(() => state.connection === 'reconnecting'); assert.equal(state.stale, true);
    await until(() => state.connection === 'connected' && state.sequence > sequence);
    await rest.stop(); await until(() => state.streams.some(s => s.type === 'session' && s.values.status === 'stopped'));
    assert.equal(state.sessionId, session.id);
  } finally {
    transport?.stop();
    await vite?.close();
    child.kill('SIGTERM');
    const deadline = setTimeout(() => child.kill('SIGKILL'), 4000);
    await exited; clearTimeout(deadline);
  }
  // Uvicorn can re-raise SIGTERM after completing graceful shutdown.
  assert.ok(child.exitCode === 0 || child.signalCode === 'SIGTERM', logs);
  assert.ok(logs.includes('Application shutdown complete.'), logs);
});
