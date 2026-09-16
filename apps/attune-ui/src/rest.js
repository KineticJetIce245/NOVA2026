export function createRestClient(fetchImpl = globalThis.fetch) {
  async function request(path, method = 'GET', signal, body) {
    const response = await fetchImpl(`/api/${path}`, { method, signal, cache: 'no-store', ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}) });
    if (!response.ok) throw Error(`HTTP ${response.status} (${path})`);
    return response.json();
  }
  return {
    media: signal => request('media', 'GET', signal),
    mediaControl: (body, signal) => request('media/control', 'POST', signal, body),
    health: signal => request('health', 'GET', signal),
    state: signal => request('state', 'GET', signal),
    start: signal => request('session/start', 'POST', signal),
    stop: signal => request('session/stop', 'POST', signal),
  };
}
