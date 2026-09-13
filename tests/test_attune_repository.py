"""The complete ATTUNE application must run from the NOVA2026 checkout."""
from pathlib import Path
import json
import subprocess
import sys
import re
from fastapi.testclient import TestClient
from backend.app.server import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_application_imports_are_local_even_without_pythonpath(tmp_path):
    code = '''
import pathlib,sys,json
root=pathlib.Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'src'),str(root)]
import backend.app.server, backend.adapters.nova_live, scripts.attune.serve, nova2026.auditory.streaming
modules=[backend.app.server,backend.adapters.nova_live,scripts.attune.serve,nova2026.auditory.streaming]
assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in modules)
print(json.dumps([str(pathlib.Path(m.__file__).resolve().relative_to(root)) for m in modules]))
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(json.loads(result.stdout)) == 4


def test_production_dashboard_assets_api_and_websocket_share_one_server():
    assert (ROOT/'frontend/dist/index.html').is_file(), 'Build with npm run build --prefix frontend before integration tests'
    with TestClient(create_app()) as client:
        response = client.get('/')
        assert response.status_code == 200 and 'ATTUNE' in response.text
        assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', response.text)
        assert len(assets) >= 2
        for asset in assets:
            assert client.get(asset).status_code == 200
        assert client.get('/api/health').json()['status'] == 'ok'
        started = client.post('/api/session/start').json()
        with client.websocket_connect('/ws/live') as socket:
            packet = socket.receive_json()
            assert packet['session_id'] == started['id']
        assert client.post('/api/session/stop').status_code == 200


def test_entrypoints_do_not_require_sibling_checkouts():
    for name in ('serve.py', 'test_end_to_end.py'):
        source = (ROOT/'scripts/attune'/name).read_text()
        assert '../attune-ui' not in source
        assert 'args.ui' not in source
        assert 'novaAAD' not in source
    assert not (ROOT/'scripts/attune/vendor.py').exists()
