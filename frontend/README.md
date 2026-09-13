# ATTUNE frontend in NOVA2026

The React/Vite interface, backend, NOVA adapter and tests live in this repository.
No sibling checkout is required. Its original styling is preserved.

From the repository root:

```powershell
npm ci --prefix frontend
npm run build --prefix frontend
```

The backend serves the resulting dashboard directly at http://127.0.0.1:8001.
Launch recorded EEG with `python -m scripts.attune.serve --trial <trial.npz>
--model <model.npz>`; see `documents/ATTUNE_IMPLEMENTATION.md` for setup.

For development, `npm run dev --prefix frontend` starts http://127.0.0.1:5173 and
proxies API/WebSocket traffic to port 8001. Production does not require Vite.

Set ATTUNE_PYTHON to the project's Python executable, then run
`npm test --prefix frontend`. Tests use the actual backend fixtures and frontend
decoders. Real-data end-to-end tests are in `scripts/attune/test_end_to_end.py`.
