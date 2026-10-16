import socket
import threading
import time

import httpx
import uvicorn
from openenv.core.mcp_client import MCPToolClient
from openenv.core.env_server.mcp_types import CallToolAction

from conftest import StubEvaluator
from permit_openenv.protocol import tool_result
from permit_openenv.server import create_server


def test_server_mcp_round_trip(pack, monkeypatch, tmp_path):
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'false')
    app = create_server(pack=pack, evaluator=StubEvaluator(), verifier=StubEvaluator())
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 30
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        assert server.started
        assert httpx.get(url + '/healthz').json()['status'] == 'ok'
        for path in ('/play/', '/api/play/tasks', '/api/tasks', '/api/episodes'):
            assert httpx.get(url + path).status_code in (404, 405)
        with MCPToolClient(url).sync() as client:
            reset = client.reset(task_id='003bd328a392-s47')
            assert reset.observation.metadata['instructions']
            assert {t.name for t in client.list_tools()} == {
                'get_scenario', 'check_design', 'evaluate_design', 'submit_design'}
            step = client.step(CallToolAction(tool_name='get_scenario', arguments={}))
            data = tool_result(step.observation)
            assert data['task_id'] == '003bd328a392-s47'
            step = client.step(CallToolAction(tool_name='submit_design', arguments={'design': data['draft_design']}))
            assert step.done and step.reward == 0
        from permit_openenv.episode import STORE
        record = STORE.recent()[0].record()
        assert record['geometry']['station'] == 'KMSP'
        assert record['ended'] == 'submitted'
    finally:
        server.should_exit = True
        thread.join(15)
        sock.close()
    assert not thread.is_alive()


def test_standard_gradio_playground(pack, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'true')
    app = create_server(pack=pack, evaluator=StubEvaluator(), verifier=StubEvaluator())
    with TestClient(app) as client:
        assert client.get('/web/').status_code == 200
        config = client.get('/web/config')
        assert config.status_code == 200
        assert 'PermitForge' in config.text
        assert 'Open the replay viewer' not in config.text
        for path in ('/play/', '/api/play/tasks', '/api/tasks', '/api/episodes'):
            assert client.get(path).status_code in (404, 405)
