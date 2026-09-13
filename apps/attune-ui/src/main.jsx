import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Dashboard } from './Dashboard.js';
import { createRestClient } from './rest.js';
import { createTransport } from './transport.js';
import { emptyState } from './state.js';
import './style.css';
const rest = createRestClient();
function App() {
  const [state, setState] = useState(emptyState);
  const [media, setMedia] = useState(null);
  const [stopSignal, setStopSignal] = useState(0);
  useEffect(() => { const abort = new AbortController(); rest.media(abort.signal).then(setMedia).catch(() => { if (!abort.signal.aborted) setError('Media configuration unavailable. Reload after the backend is ready.'); }); return () => abort.abort(); }, []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    const transport = createTransport({ rest, onState: setState });
    transport.start();
    return () => transport.stop();
  }, []);
  async function command(name) {
    if (name === 'stop') setStopSignal(value => value + 1);
    setBusy(true); setError(null);
    try { await rest[name](AbortSignal.timeout(10000)); }
    catch (e) { setError(`${e.message}. Command outcome may be unknown; inspect session state before retrying.`); }
    finally { setBusy(false); }
  }
  return <Dashboard mediaConfig={media} mediaRest={rest} stopSignal={stopSignal} state={state} busy={busy} commandError={error} command={command} />;
}
createRoot(document.getElementById('root')).render(<App />);
