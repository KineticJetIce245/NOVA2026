# ATTUNE Phase 2 transport client

Minimal React/Vite debug UI. No scientific computation, filtering, inference,
EEG acquisition, audio playback or reproduction of the legacy UI graphics.
Python provides every measurement; null means unavailable, not zero.

## Run locally

Use Node 22.12+ (tested with Node 24.18.0), npm, and the Phase 1 Python environment.
From the repository root, install frontend dependencies once:

```bash
npm ci --prefix frontend --registry=https://registry.npmjs.org
```

Terminal 1 (replace the Python path if using a different environment):

```bash
/private/tmp/attune-backbone-venv/bin/python -m uvicorn backend.app.server:app --host 127.0.0.1 --port 8001
```

If that temporary environment no longer exists, follow `backend/README.md` to
create one and install `backend/requirements-test.txt`.

Terminal 2:

```bash
npm run dev --prefix frontend
```

Open http://127.0.0.1:5173 and click **Start mock session**. Inspect attention,
vigilance, sync, and eeg_display sections; gain, signal_quality, and session
packets intentionally use the unknown-type fallback. Stop the session with the
button. Null sync offsets/signal quality are unavailable, not measured zeros.
The connection remains open when the session stops; the session packet reports
its running/stopped/error state. Stopping/restarting the backend should mark
cached state stale, reconnect, and load its new snapshot.

Vite proxies `/api` and `/ws` to loopback port 8001. These are explicit local
development ports, separate from the legacy server's default port 8000. No
backend API/CORS/config change was needed. Browser REST and WS share the page's
origin; HTTPS selects WSS automatically. Production hosting/reverse-proxy setup
is not implemented; `dist/` alone does not supply an API or WebSocket proxy.

## Layout

- `src/protocol.js`: envelope and REST snapshot validation, JSON/size/depth limits.
- `src/decoders.js`: Map registry; four display-only decoders plus opaque fallback.
- `src/state.js`: immutable normalized latest streams keyed by source/type;
  session, sequence, timestamp checks; 128-stream bound.
- `src/rest.js`: health/state/start/stop requests; HTTP error handling.
- `src/transport.js`: snapshot bootstrap, socket lifecycle, reconnect/backoff,
  cancellation, and connection/freshness state. Injectable I/O for offline tests.
- `src/Dashboard.js`: textual packet values, metadata, status and controls.
- `src/main.jsx`: React lifecycle and command wiring.
- `src/style.css`, `index.html`: minimal presentation only.
- `vite.config.js`, `package.json`, `package-lock.json`, `.gitignore`: isolated
  frontend tooling and locked dependencies; generated output stays ignored.
- `tests/client.test.js`: deterministic unit/transport/render/fixture tests.
- `tests/live.integration.js`: explicit local process/proxy integration test.

## Delivery and unavailable data

Version 1 envelope follows `backend/app/protocol.py`. Timestamps are finite
session-relative seconds. Sequence is a process-wide safe integer, not a sample
index or clock. Sequence gaps are allowed because latest-state snapshots are
sparse. Duplicate and lower sequences are rejected without replacing data;
regressing timestamps are rejected within a source/type stream. Optional missing
or invalid known measurements normalize to null. Unknown types retain their
JSON payload, with metadata and simulation disclosure, without interpretation.
React escapes their textual output. Add decoders to the Map, not transport code.

Every connection attempt fetches `/api/state` before opening `/ws/live`. Snapshot
high-water sequence makes overlap with the server's WS snapshot safe. Each retry
replaces state from a validated snapshot, allowing process sequence resets. A
new session clears previous streams. A lower-sequence different-session packet
between REST and socket opening forces another snapshot. Generation tokens
prevent retired sockets or delayed fetches from overwriting current state.

Connection states: idle, connecting, connected, reconnecting, disconnected.
Disconnect/error retains cached data but marks it stale. Connection setup times
out after 10 seconds; retries back off from 500ms to 8s, reset by new valid data.
Stop/unmount cancels timers, aborts fetch, and closes the socket. HTTP commands
have a 10-second UI deadline and are never automatically replayed: a timeout
may mean the server applied a command whose response was lost.

Limitations: latest-state delivery, not durable event replay; server close 1013
resnapshots like other disconnects. No client heartbeat/measurement-expiry policy
is invented: a silent half-open connection depends on network/socket error
detection. Missing values are not inferred. There is no physical clock sync,
scientific confidence calculation, hardware assumption, persistence, auth,
cloud deployment or visual dashboard design. Tests do not automate a browser.

## Verify

From repository root:

```bash
npm test --prefix frontend
npm run build --prefix frontend
ATTUNE_PYTHON=/private/tmp/attune-backbone-venv/bin/python npm run test:live --prefix frontend
PYTHONDONTWRITEBYTECODE=1 /private/tmp/attune-backbone-venv/bin/python -m unittest discover -s backend/tests -v
```

Unit tests use Node's built-in runner and deterministic socket/timer doubles.
One invokes `python3 -B` to decode real mock-producer fixtures without installing
scientific dependencies. The explicit live suite needs the Phase 1 Python
packages and permission to bind loopback ports; it starts its own ephemeral-port
backend/Vite servers, exercises start/stop/reconnect and cleans them up. Set
`ATTUNE_PYTHON` to a different interpreter when needed. Its deadlines bound
network waits; mock measurements remain deterministic.

Windows note: `python3` is not on `PATH` here, so the payload tests that decode
the peer repository's Python fixtures take the interpreter from `ATTUNE_PYTHON`.
Point it at this repository's environment before running the unit tests:

```powershell
$env:ATTUNE_PYTHON='C:\Files\git\NOVA2026\.venv\Scripts\python.exe'
npm --prefix apps/attune-ui test
```

Those three fixture tests (`A03-F03`, `F02-T19`, `MR-F01`) also need the peer
`backend/` package on `sys.path`, which is not part of this vendored subtree;
they fail with `ModuleNotFoundError: No module named 'backend'` until it is.
See `PROVENANCE.md` for the full vendoring record and the two deviations.
