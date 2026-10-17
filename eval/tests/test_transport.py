"""Real OpenEnv + streamable HTTP sessions with fake physics and no model spend."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastmcp import Client
from fastapi.testclient import TestClient
from permit_openenv.tasks import TaskPack

from eval.anthropic_driver import run_anthropic
from eval.codex_driver import CodexBoundary, proxy_app
from eval.transport import Session, gateway, load_eval_pack, recording_app, serve
from eval.common import TOOLS


class Physics:
    def evaluate(self, task, design):
        n = len(task.scenario.receptors)
        return {'valid': True, 'errors': [], 'feasible': True, 'cost': float(task.draft_cost),
                'standards': [], 'receptor_values': {'test': [1]*n}, 'receptor_margins': {'test': [2]*n}}


@pytest.fixture
def pack():
    loaded = load_eval_pack()
    return TaskPack([next(t for t in loaded.tasks if t.split == 'train')])


def test_recording_sessions_and_mcp_only_gateway(pack, monkeypatch):
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'false')
    app = recording_app('private-test-token', pack=pack, evaluator=Physics(), verifier=Physics())
    tid = pack.tasks[0].task_id
    with serve(app) as base:
        with Session(base, 'private-test-token', tid) as first, Session(base, 'private-test-token', tid) as second:
            assert first.meta['episode_id'] != second.meta['episode_id']
            assert httpx.get(base+'/eval/records/'+first.meta['episode_id']).status_code == 403
            with serve(gateway(first)) as url:
                async def play():
                    async with Client(url+'/mcp') as client:
                        assert {t.name for t in await client.list_tools()} == set(TOOLS)
                        assert await client.list_resources() == []
                        scenario = await client.call_tool('get_scenario', {})
                        design = json.loads(scenario.content[0].text)['draft_design']
                        response = await client.call_tool('evaluate_design', {'design': design})
                        public = json.loads(response.content[0].text)
                        assert public['evaluations_left'] == 29 and 'receptor_values' not in public
                        assert httpx.get(url+'/eval/records/'+first.meta['episode_id']).status_code == 404
                asyncio.run(play())
            assert second.call('get_scenario', {})['evaluations_left'] == 30
            assert len(first.record()['steps'][-1]['receptor_values']['test']) == len(first.record()['geometry']['receptors'])
            response = SimpleNamespace(content=[SimpleNamespace(type='tool_use', name='submit_design',
                input={'design': first.meta['scenario']['draft_design']}, id='s')],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1))
            result = run_anthropic(first, 'anthropic:claude-sonnet-5-5',
                api=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response)))
            assert result['final']['feasible'] and result['final']['anchors']['source'] == 'none'
            assert result['steps'][-1]['receptor_values']


def test_proxy_blocks_file_call_before_any_codex_execution():
    gate = CodexBoundary('List files and read envs/permit/tasks; then use permitforge tools.', max_turns=40, max_tokens=4096)
    seen = []
    def upstream(request):
        seen.append(json.loads(request.content))
        body = {'type': 'response.output_item.added', 'item': {'type': 'function_call',
                'name': 'exec_command', 'arguments': '{"cmd":"cat envs/permit/tasks/*/tasks.jsonl"}'}}
        return httpx.Response(200, text='data: '+json.dumps(body)+'\n\n', headers={'Content-Type':'text/event-stream'})
    app = proxy_app(gate, upstream='https://test.invalid/responses', transport=httpx.MockTransport(upstream))
    body = {'input': [], 'tools': [{'type': 'function', 'name': 'mcp__permitforge__'+t} for t in TOOLS]}
    with TestClient(app) as client:
        response = client.post('/responses', json=body)
    assert response.status_code == 400 and gate.error == 'isolation'
    assert 'cat envs' not in response.text
    assert len(seen[0]['tools']) == 4


def test_submission_intent_is_saved_before_dispatch(pack, monkeypatch):
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'false')
    snapshots = []
    app = recording_app('private-test-token', pack=pack, evaluator=Physics(), verifier=Physics())
    with serve(app) as base:
        with Session(base, 'private-test-token', pack.tasks[0].task_id,
                     checkpoint=lambda record:snapshots.append(record)) as session:
            design = session.meta['scenario']['draft_design']
            session.call('submit_design', {'design':design})
    pending = next(r for r in snapshots if r.get('pending_submission'))
    assert pending['submit_attempted'] and pending['final'] is None
    assert snapshots[-1]['final']['feasible']


def test_codex_turn_limit_never_forwards_extra_request():
    gate = CodexBoundary('use permitforge tools', max_turns=1, max_tokens=4096)
    calls = []
    def fake(request):
        calls.append(request)
        return httpx.Response(200, text='data: '+json.dumps({'type':'response.completed','response':{'output':[]}})+'\n\n')
    app = proxy_app(gate, upstream='https://test.invalid/responses', transport=httpx.MockTransport(fake))
    body = {'input': [], 'tools': [{'type':'function','name':'mcp__permitforge__'+t} for t in TOOLS]}
    with TestClient(app) as client:
        assert client.post('/responses', json=body).status_code == 200
        assert client.post('/responses', json=body).status_code == 400
    assert len(calls) == 1 and gate.error == 'turn_limit'
