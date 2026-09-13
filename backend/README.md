# ATTUNE backend in NOVA2026

This FastAPI application and the NOVA engine belong to the same repository.
Install `backend/requirements-test.txt` in the project Python environment. Set
PYTHONPATH to this repository and its `src` directory when running from source.

For real recorded-EEG sessions use `python -m scripts.attune.serve --trial <trial.npz>
--model <model.npz>`. Build `frontend/` first with `npm ci --prefix frontend` and
`npm run build --prefix frontend`. Open http://127.0.0.1:8001.

The same server serves the UI, `/api/health`, `/api/state`, `/api/session/start`,
`/api/session/stop`, and `/ws/live`. Each recorded session owns a finite PlayerLSL
subprocess. Stop closes the source, acquisition and audio worker.

Without NOVA configuration, `python -m uvicorn backend.app.server:app --port 8001`
runs the explicit artificial mock. Real configuration and recording conversion
are documented in `documents/ATTUNE_IMPLEMENTATION.md`.

Tests: `python -m unittest discover -s backend/tests -q`.
