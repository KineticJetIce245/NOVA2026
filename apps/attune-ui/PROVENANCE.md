# Provenance — `apps/attune-ui/`

This directory is a **verbatim copy of `frontend/`** from the teammate's ATTUNE
repository, vendored into NOVA2026 for plan step 7 (`final_connection.md` §3.6,
§5). It is a pure client: no backend ships with it. Keep it structurally
identical to upstream so future upstream changes can be diffed in cheaply.

## Source

| Field | Value |
| --- | --- |
| Source repository | `https://github.com/yut31/attune-ui` (teammate's repo; read-only local clone used: `C:\Files\git\_research\attune-ui`) |
| Source branch | `main` |
| **Source commit** | `4a523956c0959a2b41a547a2d1ab3edbd161af5f` ("Update ATTUNE README") |
| Source subdirectory | `frontend/` |
| Vendored to | `apps/attune-ui/` |
| Files copied | 28 (13 `src/`, 9 `tests/`, `index.html`, `vite.config.js`, `package.json`, `package-lock.json`, `.gitignore`, `README.md`) |
| Licence | The source is a **team member's repository (an ATTUNE submodule)**. Per `final_connection.md` §1 fact 4 and §3.9 H6, no licence change or re-licensing is required or made; the upstream `README.md` and file contents are carried over unmodified. |

The clone itself was never modified while producing this copy.

## What this client is

React 19.2.0 + Vite 7.3.1 debug UI that talks to a **same-origin FastAPI peer**
at `/api/*` plus `WS /ws/live` (snapshot first via `GET /api/state`, then the
socket). `vite.config.js` proxies `/api` and `/ws` to `127.0.0.1:8001` in local
development only; production is meant to be served from `dist/` by the backend.

Python owns every measurement — the frontend never recomputes, smooths or
infers (`final_connection.md` §2.1 rule 1). The Web Audio path in
`src/mediaAudio.js` (`<audio>` → `MediaElementSource` → `ChannelSplitter(2)` →
two `GainNode`s → `ChannelMerger(2)`, `setTargetAtTime(..., 0.1)`) is the reason
this port exists. Its gains are **attenuation only**: values outside
`[-80, 0]` dB are refused and fall back to unity.

## Deliberate deviations from upstream (exactly two)

Both are display/test-plumbing only. **No protocol, decoder, state or transport
semantics were changed**, and no test assertion was altered, weakened or
deleted.

### 1. Test interpreter is selectable via `ATTUNE_PYTHON`

Upstream spawns `python3`, which is correct on macOS but does not exist on
Windows (`python3` is not on `PATH` here; `py`/`python` are). Payload python
invocations now use `process.env.ATTUNE_PYTHON || 'python3'`, i.e. the default
behaviour is unchanged when the variable is unset. This mirrors the pattern
`tests/live.integration.js` already used upstream.

| File | Line | Change |
| --- | --- | --- |
| `tests/adapters.test.js` | 34 | `execFileSync('python3', …)` → `execFileSync(process.env.ATTUNE_PYTHON \|\| 'python3', …)` |
| `tests/client.test.js` | 148 | same substitution |
| `tests/mock_runtime.test.js` | 14 | same substitution |

Each edit swaps **one argument expression on one line** — the python payload
strings and every `assert.*` in those files are byte-identical to upstream.
Verification: with `ATTUNE_PYTHON` unset the tests still try `python3`
(fails here, by design); with `ATTUNE_PYTHON=.venv\Scripts\python.exe` they run
the same payloads through the repo virtualenv.

Use:

```powershell
$env:ATTUNE_PYTHON='C:\Files\git\NOVA2026\.venv\Scripts\python.exe'
npm --prefix apps/attune-ui test
```

### 2. Rejected-packet counter is surfaced prominently

`src/state.js` and `src/transport.js` silently count protocol-violating or
stale packets in `state.rejected` (and stash the last reason in `state.error`);
upstream printed the counter only in a low-contrast footer line. For a
developer this is the difference between "measurements are stale" and "why they
are stale", so a warning row is now rendered **above** that footer line,
whenever the counter is non-zero:

| File | Change |
| --- | --- |
| `src/Dashboard.js` | added a `state.rejected > 0 &&` guarded `<p class="rejected-warning" role="alert">` row (`REJECTED PACKETS: <n>` + the last reason). The original footer line is kept unchanged. |
| `src/style.css` | added `.rejected-warning` / `.rejected-warning strong` rules (warning colours, left rule) after `.development-notice`. |

Scope discipline: this is **display-only**. `protocol.js`, `decoders.js`,
`state.js` and `transport.js` are byte-identical to upstream; no new protocol
behaviour, packet type or judgement was invented. The counter itself is
upstream's (§3.6 deviation ②).

## Reproduction and verification

Executed on Windows, Node **v26.7.0** / npm **11.19.0**, from the repository
root, with `apps/attune-ui/package-lock.json` pinned (React 19.2.0,
react-dom 19.2.0, vite 7.3.1 — see `package.json`; nothing was added, upgraded
or removed):

```powershell
npm --prefix apps/attune-ui ci        # exit 0 — added 17 packages, audited 18 in 7s
npm --prefix apps/attune-ui test      # node --test tests/*.test.js
npm --prefix apps/attune-ui run build # exit 0 — 34 modules transformed, dist/ written
```

Known state at commit time (not a version problem):

- `npm test` reports **53 tests / 50 pass / 3 fail**. The 3 failures
  (`A03-F03`, `F02-T19`, `MR-F01`) spawn a Python payload that imports the
  **peer repository's `backend/` package as test fixtures**; that package is not
  part of the `frontend/` subtree that was ported, so the imports fail with
  `ModuleNotFoundError: No module named 'backend'`.
- Those 3 tests pass unchanged (27/27 in the same three files) as soon as the
  fixtures are reachable at the path the tests resolve (`apps/backend`, i.e.
  `../../` from `apps/attune-ui/tests/`). Verified by temporarily exposing the
  clone's `backend/` there and re-running, then removing it.
- `tests/live.integration.js` is **not** part of `npm test` upstream either (it
  is `npm run test:live`); it needs the peer backend plus `uvicorn` and is not
  run here.
- `dist/` is git-ignored via `apps/attune-ui/.gitignore`; `node_modules/` likewise.
- `npm audit` reports 2 advisories in the **dev toolchain** (`vite@7.3.1`,
  `esbuild@0.27.7`). Both concern the Vite/esbuild **dev server** only; NOVA2026
  serves the built `dist/` from the Python backend and does not expose the dev
  server. Upgrading is out of scope for this step (versions are pinned by
  `final_connection.md` §3.6).

## File manifest at vendoring time (SHA256)

Hashes below are of **this** copy; all files except the five listed as deviating
match the source commit byte-for-byte (verified by hash comparison of the whole
tree after copying — 28/28 identical at copy time).

```text
.gitignore                     ea51f78e69c37ba5b07274b23829325147e58ae7f0b4ddb332200f761dd6ffb1
index.html                     f8d6737c83d51d401f7382795ab379f2a9716ef4c0b53bbdaa4baab3b6a8a9e7
package-lock.json              3b8ef2b2db080e47bdd70c92174bab6aceb39ebaecf3dea263f35431bcbdb1bf
package.json                   7215c6a68e867e0651038ec954861b3381cd058489fc2bfbd7fde70f9fcfddf5
README.md                      75b80724ebbbf7b8bef7b9a96174f3f48126a4ef2480c71ea7d920f64794ff54
vite.config.js                 f228c318018a210ad248c1d4646fcf1c8a107fc4e4870f37b70edab3b2e1c0e9
src/Dashboard.js               fdc3ee90cace109597b2d9443829f6e42e5e319f65d6bfc9270573fbc6e10d15  (deviation 2)
src/DashboardParts.js          21a0074abad8d9c792ab0563167c762a42fc426e56d08e01689abb238815b662
src/decoders.js                ae25dac3f5b454cf1c0eaea03712a4887afc19f7ef6cc228b677513310d83ec6
src/FeedbackPanel.js           c03bff9c6adbaf4dee3816b66dbd08d5e2d42b0be2f81f0aa951e9f2590c9055
src/main.jsx                   dee7065bf200c551dc747370eec2a8fbf5a877a9a8e798a4f30ee5f2592e06bd
src/mediaAudio.js              44fd389f8fe2e57e772601ab99588d9368053f743396ab61d0dc60954fb6ac11
src/mediaController.js         ef42d18671dc70891e1612c5c9e79c03bf5b5ca03da589d2ce7af553da3a5719
src/MediaPlayback.js           f67d4216ed4f92c74692252a40a502b52ce638b03a85bd8f79ff2bd2448376c8
src/protocol.js                53f0a99b22949ad977eb1d490e023c01f7a0751cb4498e5f06483266a4f72510
src/rest.js                    aa238c881f3df1cdd1f25188d3dd311b26833557f4eba44e63e88ad5b8c4fe91
src/state.js                   e58c6f8b2af85ca5e015cb52f84d1662d9727ea6be0da7f60242bc17659a0f24
src/style.css                  df287dacbfaef97bc12ff29e2a90d55e85a3553daa0c1d1245999c0eb984fb08  (deviation 2)
src/transport.js               160cd4194f6d30accf321436faa16cbd81d6d235b081a95e9382f8b66d335ee2
tests/adapters.test.js         fb21439758057464c8e84ce9abe44fda84f81e554df484c7b74ff81ed59bf363  (deviation 1)
tests/client.test.js           725fc1a15bcb19ede06f33fc50832cb62aa6b297a73718475dfa59e72f701ef4  (deviation 1)
tests/dashboard.test.js        db684174ed0dd514d5bae55b4d609ba5a189cc330750edcd172517920688a0ea
tests/demo_finalization.test.js e8bc0962f959a8689945bdaedc81848c2b0ba45918551e55ad21c95774542621
tests/feedback.test.js         8b0f28c3297db7baa2bcaba3447ad8a07a25c97cb11511b5051ac8f9727d0b28
tests/live.integration.js      ae05a7e672ea07e4d945097e32a659d70fc6f225efd08d7060613f8e7e83f848
tests/media.test.js            fe0d791f93e0d25b0016204027aa381e52a0b05cf2790866954d450d0cad64e5
tests/mock_runtime.test.js     a07f9a98f1b21bc3b953260a3c9f090d5757377216aae367ad97996d75e1539d  (deviation 1)
tests/session_layout.test.js   147d55e96c7bcac0bc43fe37db102757b56b37a82b0669c4bbd3400ad6826f2d
```

## Maintenance note

To take an upstream update: fetch the teammate's repo, diff its `frontend/`
against this directory, and expect a three-line diff in `Dashboard.js`, a
two-line diff in `style.css`, and three one-line diffs in `tests/*.test.js`
attributable to the two deviations above. Anything else in a diff is an
unintended local change.
