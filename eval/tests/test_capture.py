"""Capture-everything acceptance without provider spend."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from eval.anthropic_driver import run_anthropic
from eval.common import model_slug, records
from eval.run_eval import run_panel
from eval.transport import Session, load_eval_pack, recording_app, serve
from permit_openenv.tasks import TaskPack


class Physics:
    def evaluate(self, task, design):
        return {'valid': True, 'feasible': True, 'errors': [], 'cost': float(task.draft_cost),
                'standards': [], 'receptor_values': {'s': [1]}, 'receptor_margins': {'s': [2]}}


def test_fake_provider_capture_everything_and_secret_scan(tmp_path, monkeypatch):
    from eval.export_hf import export_runs
    secret = 'test-secret-credential-123456789'
    monkeypatch.setenv('ANTHROPIC_API_KEY', secret)
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'true')
    pack = TaskPack([next(t for t in load_eval_pack().tasks if t.split == 'train')])
    task = pack.tasks[0]
    app = recording_app(secret, pack=pack, evaluator=Physics(), verifier=Physics())
    with TestClient(app) as client:
        assert client.get('/web').status_code == 404
    calls = []
    long_text = 'full assistant text ' * 400
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError('Authorization: Bearer '+secret)
        return SimpleNamespace(id='provider-id', extra_provider_field={'authorization': secret},
            content=[{'type': 'thinking', 'thinking': 'provider returned reasoning'},
                     {'type': 'text', 'text': long_text + secret},
                     {'type': 'tool_use', 'id': 'submit', 'name': 'submit_design',
                      'input': {'design': task.view()['draft_design']}}],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20))
    run = tmp_path/'run'
    with serve(app) as base:
        def driver(model, row, checkpoint, capture):
            with Session(base, secret, row['task_id'], checkpoint=checkpoint, capture=capture) as session:
                return run_anthropic(session, model, api=SimpleNamespace(messages=SimpleNamespace(create=create)),
                                     sleep=lambda _: None)
        run_panel(run, [{'task_id': task.task_id, 'split': 'train', 'station': task.scenario.station}],
                  ['anthropic:claude-sonnet-5-5'], driver, budget_usd=10)
    episode_dir = run/model_slug('anthropic:claude-sonnet-5-5')/task.task_id/'r1'
    for name in ('episode.json', 'transcript.jsonl', 'env.log'):
        assert (episode_dir/name).stat().st_size
    assert list((episode_dir/'raw').glob('*request.json'))
    assert list((episode_dir/'raw').glob('*response.json'))
    assert list((episode_dir/'attempts').rglob('error.json'))
    assert list((episode_dir/'attempts').rglob('*request.json'))
    transcript = [json.loads(line) for line in (episode_dir/'transcript.jsonl').read_text().splitlines()]
    assert transcript[0]['role'] == 'system'
    assert [r['kind'] for r in transcript] == ['message', 'message', 'message', 'tool_call', 'tool_result']
    assert 'provider returned reasoning' in json.dumps(transcript)
    assert long_text in json.dumps(transcript)
    assert all(row['at'] for row in transcript)
    assert next(row for row in transcript if row['role'] == 'assistant')['usage']['output_tokens'] == 20
    assert 'tool received: submit_design' in (episode_dir/'env.log').read_text()
    meta = json.loads((run/'run.json').read_text())
    assert set(meta) >= {'git_commit', 'prompt_version', 'prompt_sha256', 'protocol_sha256', 'models',
                         'sampling', 'concurrency', 'budget_usd', 'started_at', 'finished_at', 'host'}
    assert meta['prompt_version'] == 2 and meta['finished_at']
    assert (run/'harness.log').stat().st_size and (run/'costs.jsonl').stat().st_size
    assert len(records([run])) == 1
    export = tmp_path/'export'
    export_runs([run], export)
    assert (export/'README.md').exists()
    for name in ('episodes', 'transcripts', 'raw'):
        assert (export/name/'train.jsonl').stat().st_size
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert secret not in file.read_text(), file


def test_disconnect_and_harness_termination_finalize_private_record(tmp_path):
    from eval.capture import EpisodeCapture
    pack = TaskPack([next(t for t in load_eval_pack().tasks if t.split == 'train')])
    app = recording_app('test-private-token', pack=pack, evaluator=Physics(), verifier=Physics())
    with serve(app) as base:
        capture = EpisodeCapture(tmp_path/'episode')
        with Session(base, 'test-private-token', pack.tasks[0].task_id, capture=capture) as session:
            episode_id = session.meta['episode_id']
            session.call('get_scenario', {})
        rec = httpx.get(base+'/eval/records/'+episode_id,
                       headers={'Authorization': 'Bearer test-private-token'}).json()
        assert rec['ended'] == 'no_submit' and rec['finished_at'] and rec['reward'] == 0
        assert 'episode finished: no_submit' in (tmp_path/'episode'/'env.log').read_text()


def test_redaction_covers_structures_headers_and_environment(monkeypatch):
    from eval.privacy import redact
    monkeypatch.setenv('EXAMPLE_ACCESS_TOKEN', 'opaque-environment-secret')
    sample = {'api_key': 'short', 'nested': {'refresh_token': 'opaque-refresh-value'},
              'text': 'Authorization: Bearer opaque-bearer-value; opaque-environment-secret',
              'url': 'https://example.invalid/?api_key=url-secret-value'}
    clean = json.dumps(redact(sample))
    for secret in ('short', 'opaque-refresh-value', 'opaque-bearer-value', 'opaque-environment-secret', 'url-secret-value'):
        assert secret not in clean


@pytest.mark.parametrize('effort', ['medium', 'high', 'xhigh'])
@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'gpt-6-astra'])
def test_codex_cli_captures_full_stream_command_and_returned_reasoning(tmp_path, monkeypatch, effort, model):
    import eval.codex_driver as driver
    from eval.capture import EpisodeCapture
    pack = TaskPack([next(t for t in load_eval_pack().tasks if t.split == 'train')])
    task = pack.tasks[0]
    calls = []
    def upstream(request):
        calls.append(json.loads(request.content))
        index = len(calls)
        call = {'id': f'fc_{index}', 'type': 'function_call', 'call_id': f'call_{index}',
                'name': 'get_scenario' if index == 1 else 'submit_design', 'namespace': 'mcp__permitforge',
                'arguments': json.dumps({} if index == 1 else {'design': task.view()['draft_design']}),
                'status': 'completed'}
        thinking = {'id': f'rs_{index}', 'type': 'reasoning',
                    'summary': [{'type': 'summary_text', 'text': 'Returned reasoning summary.'}]}
        events = [{'type': 'response.created', 'response': {'id': f'r_{index}', 'status': 'in_progress', 'output': []}},
                  {'type': 'response.output_item.done', 'output_index': 0, 'item': thinking},
                  {'type': 'response.output_item.done', 'output_index': 1, 'item': call},
                  {'type': 'response.completed', 'response': {'id': f'r_{index}', 'status': 'completed',
                     'output': [thinking, call], 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}}]
        return httpx.Response(200, text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
    original = driver.proxy_app
    monkeypatch.setattr(driver, 'proxy_app', lambda boundary, **kw: original(boundary,
        upstream='https://test.invalid/responses', transport=httpx.MockTransport(upstream), **kw))
    capture = EpisodeCapture(tmp_path/'episode')
    app = recording_app('private-test-token', pack=pack, evaluator=Physics(), verifier=Physics())
    with serve(app) as base:
        with Session(base, 'private-test-token', task.task_id, capture=capture) as session:
            result = driver.run_codex(session, 'codex:'+model, wall_clock=45, effort=effort)
    assert result['final']['feasible']
    assert calls and all(body['reasoning']['effort'] == effort for body in calls)
    assert all(body['max_output_tokens'] == 32768 for body in calls)
    assert result['settings']['effort'] == result['settings']['launch_effort'] == effort
    config = json.loads((capture.root/'raw/config.json').read_text())
    assert config['model_reasoning_effort'] == json.dumps(effort)
    for name in ('codex-events.jsonl', 'codex-stderr.log', 'command.json', 'config.json', 'model-catalog.json'):
        assert (capture.root/'raw'/name).exists()
    rows = [json.loads(line) for line in (capture.root/'transcript.jsonl').read_text().splitlines()]
    assert 'Returned reasoning summary.' in json.dumps(rows)
    assert len([r for r in rows if r['kind'] == 'tool_call']) == 2
    assert all(r['usage']['output_tokens'] == 5 for r in rows if r['role'] == 'assistant' and r['kind'] == 'message')
    assert list((capture.root/'raw').glob('codex-*-response.sse'))


def test_anthropic_wire_json_is_preserved(tmp_path):
    from eval.capture import EpisodeCapture
    from test_contract import Session as FakeSession
    session = FakeSession()
    session.capture = EpisodeCapture(tmp_path/'episode')
    body = '{ "id": "wire-id", "nullable": null, "content": [{"type":"text","text":"done"}], "usage":{"input_tokens":1,"output_tokens":2} }'
    request = httpx.Request('POST', 'https://test.invalid/messages', content=b'{"model":"test"}')
    response = httpx.Response(200, request=request, text=body)
    wire = SimpleNamespace(text=lambda: body, http_response=response, parse=lambda: SimpleNamespace(
        content=[{'type': 'text', 'text': 'done'}], usage=SimpleNamespace(input_tokens=1, output_tokens=2)))
    api = SimpleNamespace(messages=SimpleNamespace(with_raw_response=SimpleNamespace(create=lambda **kw: wire)))
    result = run_anthropic(session, 'anthropic:claude-sonnet-5-5', api=api)
    assert result['ended'] == 'no_submit'
    assert (session.capture.root/'raw/anthropic-0001-1-response.json').read_text() == body
    assert (session.capture.root/'raw/anthropic-0001-1-request.json').read_text() == request.content.decode()


def test_explicit_harness_termination_records_no_submit(tmp_path):
    from eval.transport import terminate_sessions, TERMINATING
    from eval.capture import EpisodeCapture
    pack = TaskPack([next(t for t in load_eval_pack().tasks if t.split == 'train')])
    app = recording_app('private-test-token', pack=pack, evaluator=Physics(), verifier=Physics())
    snapshots = []
    try:
        with serve(app) as base:
            with Session(base, 'private-test-token', pack.tasks[0].task_id,
                         capture=EpisodeCapture(tmp_path/'episode'), checkpoint=snapshots.append) as session:
                terminate_sessions()
                record = session.record()
                assert record['ended'] == 'no_submit' and record['finished_at'] and record['reward'] == 0
                assert snapshots[-1]['finished_at'] == record['finished_at']
    finally:
        TERMINATING.clear()


def test_partial_provider_response_survives_read_failure(tmp_path):
    from eval.capture import EpisodeCapture
    from eval.codex_driver import CodexBoundary, proxy_app
    from eval.common import TOOLS
    class Partial(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"partial text"}\n\n'
            raise httpx.ReadError('connection interrupted')
    capture = EpisodeCapture(tmp_path/'episode')
    gate = CodexBoundary('test', max_turns=2, max_tokens=100, capture=capture)
    app = proxy_app(gate, upstream='https://test.invalid/responses', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Partial())))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post('/responses', json={'input': [], 'tools': [
            {'type': 'function', 'name': 'mcp__permitforge__'+name} for name in TOOLS]})
    assert response.status_code == 502
    assert gate.error == 'infrastructure' and 'ReadError' in gate.traceback
    assert 'partial text' in (capture.root/'raw/codex-0001-1-response.sse').read_text()
    assert (capture.root/'attempts/codex-0001-1/error.json').exists()


def test_interrupted_episode_keeps_cost_record_and_terminal_attempt(tmp_path):
    def driver(model, task, checkpoint, capture):
        capture.raw_text('partial.txt', 'provider diagnostic')
        checkpoint({'usage': {'input_tokens': 9, 'output_tokens': 2, 'usd': None}})
        raise RuntimeError('harness interrupted')
    run_panel(tmp_path, [{'task_id': 'task', 'split': 'train', 'station': 'X'}], ['codex:test'], driver)
    assert json.loads((tmp_path/'run.json').read_text())['finished_at']
    cost = json.loads((tmp_path/'costs.jsonl').read_text())
    assert cost['usage']['input_tokens'] == 9
    assert (tmp_path/model_slug('codex:test')/'task/r1/attempts/episode-0001/raw/partial.txt').exists()
    assert records([tmp_path]) == []


def test_resume_uses_canonical_episode_without_legacy_snapshot(tmp_path):
    from test_contract import episode
    tasks = [{'task_id': 'task', 'split': 'train', 'station': 'X'}]
    run_panel(tmp_path, tasks, ['codex:test'], lambda model, task: episode(model, task['task_id']))
    (tmp_path/model_slug('codex:test')/'task/r1.json').unlink()
    def no_retry(model, task):
        raise AssertionError('canonical completed episode must never be replayed')
    run_panel(tmp_path, tasks, ['codex:test'], no_retry)
    assert len(records([tmp_path])) == 1
