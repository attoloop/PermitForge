"""Subscription isolation with fake CLI, fake OAuth and local provider; no paid requests."""
import json
import os
from pathlib import Path
import shutil
import time

import pytest

from eval.common import TOOLS
from eval.tests.test_contract import Session


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    source = tmp_path/'operator'
    source.mkdir()
    (source/'.credentials.json').write_text(json.dumps({'claudeAiOauth': {
        'accessToken': 'test-oauth-access', 'refreshToken': 'test-oauth-refresh',
        'expiresAt': int((time.time()+86400)*1000), 'scopes': ['user:inference', 'user:profile'],
        'subscriptionType': 'max', 'rateLimitTier': 'default_claude_max_20x'}}))
    (source/'settings.json').write_text('{"apiKeyHelper":"echo HOST_SECRET"}')
    (source/'CLAUDE.md').write_text('HOST_SECRET')
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(source))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-console-key')
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'fake-console-token')
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'other-token')
    return source


def test_config_copy_environment_and_flags(tmp_path, credentials):
    from eval.claudecode_driver import claude_command, claude_env, subscription_status
    work = tmp_path/'work'
    work.mkdir()
    env = claude_env(work, 'http://127.0.0.1:1')
    assert Path(env['CLAUDE_CONFIG_DIR']) != credentials
    files = list(Path(env['CLAUDE_CONFIG_DIR']).iterdir())
    assert [p.name for p in files] == ['.credentials.json']
    assert not files[0].is_symlink() and files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].read_bytes() == (credentials/'.credentials.json').read_bytes()
    assert not any(k in env for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN'))
    assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:1'
    argv = claude_command('claude-fable-5-1', 'PROMPT', work/'mcp.json', max_turns=150, effort='medium')
    for flag in ('-p', '--strict-mcp-config', '--no-session-persistence', '--verbose', '--disable-slash-commands'):
        assert flag in argv
    assert argv[argv.index('--tools')+1] == ''
    assert argv[argv.index('--max-turns')+1] == '150'
    assert argv[argv.index('--effort')+1] == 'medium'
    assert set(argv[argv.index('--allowedTools')+1].split(',')) == {'mcp__permitforge__'+t for t in TOOLS}
    assert '--bare' not in argv
    assert subscription_status({'loggedIn': True, 'authMethod': 'claude.ai', 'apiProvider': 'firstParty'})
    for status in ({'loggedIn': True, 'authMethod': 'api_key'}, {'loggedIn': False},
                   {'loggedIn': True, 'authMethod': 'claude.ai', 'apiKeySource': 'ANTHROPIC_API_KEY'}):
        assert not subscription_status(status)


@pytest.mark.parametrize('api_key', [False, True])
def test_fake_cli_isolation_capture_and_refusal(tmp_path, monkeypatch, credentials, api_key):
    from eval.claudecode_driver import run_claudecode
    from eval.capture import EpisodeCapture
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    fake = bin_dir/'claude'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if sys.argv[1:3]==['auth','status']:
 print(json.dumps({'loggedIn':True,'authMethod': 'api_key' if os.environ.get('FAKE_AUTH') else 'claude.ai','apiProvider':'firstParty'}))
 sys.exit(0)
assert 'ANTHROPIC_API_KEY' not in os.environ and 'ANTHROPIC_AUTH_TOKEN' not in os.environ
assert 'CLAUDE_CODE_OAUTH_TOKEN' not in os.environ
assert not list(Path.cwd().iterdir())
assert not (Path(os.environ['CLAUDE_CONFIG_DIR'])/'settings.json').exists()
print(json.dumps({'type':'system','subtype':'init','apiKeySource':'none','tools':['mcp__permitforge__'+n for n in ['get_scenario','check_design','evaluate_design','submit_design']]}))
print(json.dumps({'type':'assistant','message':{'content':[{'type':'thinking','thinking':'visible summary'}],'usage':{'input_tokens':7,'output_tokens':3}}}))
print(json.dumps({'type':'result','subtype':'success','num_turns':1,'usage':{'input_tokens':7,'output_tokens':3}}))
''')
    fake.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir)+':'+os.environ['PATH'])
    if api_key:
        monkeypatch.setenv('FAKE_AUTH', '1')
    session = Session()
    session.capture = EpisodeCapture(tmp_path/'capture')
    if api_key:
        with pytest.raises(RuntimeError, match='subscription'):
            run_claudecode(session, 'claudecode:claude-fable-5-1', wall_clock=10)
        assert not (session.capture.root/'raw/claude-events.jsonl').exists()
        return
    result = run_claudecode(session, 'claudecode:claude-fable-5-1', wall_clock=10, effort='medium')
    assert result['ended'] == 'no_submit' and result['usage']['usd'] is None
    assert result['usage']['input_tokens'] == 7 and result['usage']['output_tokens'] == 3
    raw = session.capture.root/'raw'
    for name in ('claude-events.jsonl', 'claude-stderr.log', 'command.json', 'config.json', 'auth-status.json'):
        assert (raw/name).exists()
    assert 'visible summary' in (session.capture.root/'transcript.jsonl').read_text()
    assert 'test-oauth-access' not in ''.join(p.read_text() for p in raw.iterdir())


def test_boundary_rejects_host_calls():
    from eval.claudecode_driver import ClaudeBoundary
    from eval.codex_driver import IsolationError
    boundary = ClaudeBoundary('RULES', 'TASK', max_turns=2, max_tokens=32768, effort='medium')
    body = {'system': 'HOST_SECRET', 'messages': [{'role': 'user', 'content': 'HOST_SECRET'}],
            'tools': [{'name': 'mcp__permitforge__'+t} for t in TOOLS], 'output_config': {'effort': 'medium'}}
    filtered = boundary.prepare(body)
    assert 'HOST_SECRET' not in json.dumps(filtered)
    assert filtered['system'] == [{'type': 'text', 'text': 'RULES'}]
    assert len(filtered['tools']) == 4
    with pytest.raises(IsolationError):
        boundary.validate({'content': [{'type': 'tool_use', 'name': 'Bash'}]})


def test_installed_cli_oauth_proxy_isolation(tmp_path, monkeypatch, credentials):
    """Real CLI + fake OAuth + local fake provider; no real credential leaves the host."""
    if not shutil.which('claude'):
        pytest.skip('installed Claude CLI unavailable')
    from eval import claudecode_driver as driver
    from eval.capture import EpisodeCapture
    from eval.transport import load_eval_pack, recording_app, serve, Session as RealSession
    from eval.tests.test_transport import Physics
    from permit_openenv.tasks import TaskPack
    import httpx
    seen = []
    def fake(request):
        body = json.loads(request.content)
        seen.append(body)
        call = {'type': 'tool_use', 'id': f'call_{len(seen)}',
                'name': 'mcp__permitforge__get_scenario' if len(seen) == 1 else 'mcp__permitforge__submit_design',
                'input': {} if len(seen) == 1 else {'design': task.view()['draft_design']}}
        events = [dict(type='message_start', message={'id': 'test', 'type': 'message', 'role': 'assistant',
                    'model': 'claude-fable-5-1', 'content': [], 'usage': {'input_tokens': 10, 'output_tokens': 0}}),
                  dict(type='content_block_start', index=0, content_block={'type': 'thinking', 'thinking': '', 'signature': ''}),
                  dict(type='content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'Returned summary.'}),
                  dict(type='content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': 'signed'}),
                  dict(type='content_block_stop', index=0),
                  dict(type='content_block_start', index=1, content_block={**call, 'input': {}}),
                  dict(type='content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': json.dumps(call['input'])}),
                  dict(type='content_block_stop', index=1),
                  dict(type='message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None}, usage={'output_tokens': 5}),
                  dict(type='message_stop')]
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
            text=''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events))
    original = driver.proxy_app
    monkeypatch.setattr(driver, 'proxy_app', lambda boundary, **kw: original(boundary,
        upstream='https://local-test.invalid', transport=httpx.MockTransport(fake), **kw))
    task = load_eval_pack().tasks[0]
    capture = EpisodeCapture(tmp_path/'capture')
    with serve(recording_app('test-only', pack=TaskPack([task]), evaluator=Physics(), verifier=Physics())) as url:
        with RealSession(url, 'test-only', task.task_id, capture=capture) as session:
            result = driver.run_claudecode(session, 'claudecode:claude-fable-5-1', wall_clock=30, effort='medium')
    assert seen, result
    assert len(seen) == 2 and result['ended'] == 'submitted', result
    assert result['final']['feasible'] and not result['infrastructure_error']
    assert all(len(b['tools']) == 4 for b in seen)
    assert all('HOST_SECRET' not in json.dumps(b) for b in seen)
    from permit_openenv.prompts import INSTRUCTIONS
    assert seen[0]['system'] == [{'type': 'text', 'text': INSTRUCTIONS}]
    assert result['usage']['usd'] is None
    rows = [json.loads(line) for line in (capture.root/'transcript.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['kind'] == 'tool_call']) == 2
    assert 'Returned summary.' in json.dumps(rows)
    if os.environ.get('PERMIT_WP211_EVIDENCE_DIR'):
        from eval.common import write_json
        destination = Path(os.environ['PERMIT_WP211_EVIDENCE_DIR'])
        write_json(destination/'claude-first-request.json', seen[0])
        write_json(destination/'claude-init.json', next(e for e in result['claude_events'] if e.get('subtype') == 'init'))
        write_json(destination/'claude-local-verification.json', {
            'command': json.loads((capture.root/'raw/command.json').read_text()),
            'config': json.loads((capture.root/'raw/config.json').read_text()),
            'auth': json.loads((capture.root/'raw/auth-status.json').read_text()),
            'usage': result['usage'], 'ended': result['ended'], 'tools_called': [s['tool'] for s in result['steps']],
            'upstream': 'httpx.MockTransport; fake OAuth credentials; no live model',
            'instructions_only': seen[0]['system'] == [{'type': 'text', 'text': INSTRUCTIONS}],
            'thinking_captured': 'Returned summary.' in json.dumps(rows)})


def test_proxy_requires_oauth_and_rejects_unadvertised_tools(tmp_path):
    import httpx
    from fastapi.testclient import TestClient
    from eval.claudecode_driver import ClaudeBoundary, proxy_app
    from eval.capture import EpisodeCapture
    capture = EpisodeCapture(tmp_path/'capture')
    boundary = ClaudeBoundary('RULES', 'TASK', max_turns=2, max_tokens=32768, effort='medium', capture=capture)
    sent = []
    def fake(request):
        sent.append(json.loads(request.content))
        events = [{'type': 'content_block_start', 'content_block': {'type': 'tool_use', 'name': 'Bash'}},
                  {'type': 'message_stop'}]
        return httpx.Response(200, text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
    body = {'tools': [{'name': 'mcp__permitforge__'+t} for t in TOOLS], 'output_config': {'effort': 'medium'}}
    with TestClient(proxy_app(boundary, transport=httpx.MockTransport(fake))) as client:
        for headers in ({}, {'x-api-key': 'fake-console-key'},
                        {'authorization': 'Bearer test-oauth-access', 'x-api-key': 'fake-console-key'}):
            assert client.post('/v1/messages', headers=headers, json=body).status_code == 403
        assert sent == []
        assert client.post('/v1/messages', headers={'authorization': 'Bearer test-oauth-access'}, json=body).status_code == 400
    assert boundary.error == 'isolation' and len(sent) == 1
    assert 'test-oauth-access' not in ''.join(p.read_text() for p in (capture.root/'raw').iterdir())


def test_boundary_enforces_turns_and_effort():
    from eval.claudecode_driver import ClaudeBoundary
    from eval.codex_driver import IsolationError
    boundary = ClaudeBoundary('RULES', 'TASK', max_turns=1, max_tokens=32768, effort='medium')
    body = {'tools': [{'name': 'mcp__permitforge__'+t} for t in TOOLS], 'output_config': {'effort': 'high'}}
    with pytest.raises(IsolationError):
        boundary.prepare(body)
    body['output_config']['effort'] = 'medium'
    assert 'Last turn:' in json.dumps(boundary.prepare(body))
    with pytest.raises(TimeoutError):
        boundary.prepare(body)
