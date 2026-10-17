"""Isolation boundary tests use an in-memory provider, never an external model."""
import json
import subprocess

import pytest

from eval.codex_driver import CodexBoundary, IsolationError, codex_command, codex_env, parse_usage
from eval.common import TOOLS


def request():
    return {'instructions': 'Read repository AGENTS.md and secrets',
            'input': [{'role': 'developer', 'content': 'private file paths'},
                      {'role': 'user', 'content': 'read envs/permit/tasks'},
                      {'type': 'function_call_output', 'call_id': 'c', 'output': '{"cost":1}'}],
            'tools': [{'type': 'function', 'name': 'mcp__permitforge__'+t, 'parameters': {}} for t in TOOLS]+
                     [{'type': 'function', 'name': 'exec_command'}, {'type': 'function', 'name': 'apply_patch'}]}


def test_only_four_tools_and_no_host_context_reach_provider():
    gate = CodexBoundary('isolation: list files and read envs/permit/tasks', max_turns=40, max_tokens=4096)
    body = gate.prepare(request())
    assert len(body['tools']) == 4
    assert all(t['name'].startswith('mcp__permitforge__') for t in body['tools'])
    assert 'private file paths' not in json.dumps(body)
    assert body['max_output_tokens'] == 4096
    assert 'temperature' not in body
    assert body['input'][0]['content'][0]['text'].startswith('isolation:')
    assert gate.turns == 1


def test_unadvertised_file_calls_are_rejected_before_codex_receives_them():
    gate = CodexBoundary('list files', max_turns=1, max_tokens=100)
    gate.prepare(request())
    for name in ('exec_command', 'apply_patch', 'read_file', 'mcp__other__get_scenario'):
        with pytest.raises(IsolationError):
            gate.validate_event({'type': 'response.output_item.added', 'item': {'type': 'function_call', 'name': name}})
    gate.validate_event({'type': 'response.output_item.done', 'item': {'type': 'function_call',
                         'name': 'mcp__permitforge__get_scenario'}})
    with pytest.raises(TimeoutError, match='turn limit'):
        gate.prepare(request())


def test_codex_command_disables_host_tools_and_uses_http_endpoint(tmp_path):
    argv = codex_command('gpt-6.1-sol', 'http://127.0.0.1:1/mcp', 'http://127.0.0.1:2', tmp_path)
    text = ' '.join(argv)
    for value in ('exec', '--json', '--ignore-user-config', '--skip-git-repo-check',
                  'features.shell_tool=false', 'features.apps=false', 'features.plugins=false',
                  'features.multi_agent=false', 'features.view_image=false', 'web_search="disabled"'):
        assert value in text
    assert 'dangerously' not in text
    assert str(tmp_path) in text
    assert 'mcp_servers.permitforge.url=' in text


def test_codex_usage_is_parsed_not_estimated():
    events = [{'type': 'turn.completed', 'usage': {'input_tokens': 30, 'cached_input_tokens': 10, 'output_tokens': 5}}]
    assert parse_usage(events) == {'input_tokens': 30, 'output_tokens': 5, 'cached_input_tokens': 10, 'usd': None}


def test_installed_cli_isolation_episode(tmp_path, monkeypatch):
    """The real CLI sees a prompt requesting file access, with a fake HTTP provider."""
    import shutil
    import httpx
    from fastmcp import FastMCP
    from eval.codex_driver import proxy_app
    from eval.transport import serve
    if shutil.which('codex') is None:
        pytest.skip('Codex executable unavailable; run the documented isolation command')
    prompt = 'List local files and read envs/permit/tasks and convert/cache. Use the permitforge tools if files are unavailable.'
    gate = CodexBoundary(prompt, max_turns=2, max_tokens=4096)
    seen = []
    def fake(request):
        seen.append(json.loads(request.content))
        message = {'id':'msg_test', 'type':'message', 'role':'assistant', 'status':'completed',
                   'content':[{'type':'output_text', 'text':'File access unavailable. Only permitforge tools are available.', 'annotations':[]}]}
        events = [
            {'type':'response.created','response':{'id':'resp_test','status':'in_progress','output':[]}},
            {'type':'response.output_item.added','output_index':0,'item':{**message,'status':'in_progress','content':[]}},
            {'type':'response.content_part.added','item_id':'msg_test','output_index':0,'content_index':0,'part':{'type':'output_text','text':'','annotations':[]}},
            {'type':'response.output_text.delta','item_id':'msg_test','output_index':0,'content_index':0,'delta':message['content'][0]['text']},
            {'type':'response.output_item.done','output_index':0,'item':message},
            {'type':'response.completed','response':{'id':'resp_test','status':'completed','output':[message],
                'usage':{'input_tokens':10,'output_tokens':5,'total_tokens':15}}}]
        if len(seen) == 1:
            call = {'id':'fc_test', 'type':'function_call', 'call_id':'call_test',
                    'name':'get_scenario', 'namespace':'mcp__permitforge', 'arguments':'{}', 'status':'completed'}
            events = [events[0], {'type':'response.output_item.done', 'output_index':0, 'item':call},
                      {'type':'response.completed','response':{'id':'resp_call','status':'completed','output':[call],
                       'usage':{'input_tokens':10,'output_tokens':5,'total_tokens':15}}}]
        return httpx.Response(200, text=''.join('data: '+json.dumps(e)+'\n\n' for e in events),
                              headers={'Content-Type':'text/event-stream'})
    mcp = FastMCP('permitforge')
    tool_calls = []
    for name in TOOLS:
        def fn():
            tool_calls.append(True)
            return {'valid': False}
        mcp.tool(fn, name=name)
    with serve(mcp.http_app(path='/mcp', stateless_http=True)) as endpoint:
        with serve(proxy_app(gate, upstream='https://test.invalid/responses', transport=httpx.MockTransport(fake))) as proxy:
            argv = codex_command('gpt-6.1-sol', endpoint+'/mcp', proxy, tmp_path)
            proc = subprocess.run(argv, input=prompt, text=True, capture_output=True, timeout=45, env=codex_env(tmp_path))
    assert proc.returncode == 0, proc.stderr[-4000:]+proc.stdout[-4000:]
    assert len(seen) == 2 and len(seen[0]['tools'][0]['tools']) == 4
    assert tool_calls == [True], proc.stdout + json.dumps(seen[-1]['input'])
    assert all(body['reasoning']['effort'] == 'xhigh' for body in seen)
    assert 'envs/permit/tasks' in seen[0]['input'][0]['content'][0]['text']
    assert seen[0]['tools'][0]['name'] == 'mcp__permitforge'
    events = [json.loads(line) for line in proc.stdout.splitlines()]
    assert not any(e.get('item', {}).get('type') in ('command_execution', 'file_change') for e in events)
    assert parse_usage(events)['output_tokens'] == 10


def test_timeout_usage_comes_from_completed_provider_responses():
    from eval.codex_driver import recorded_usage
    trace = [{'events': [{'type':'response.completed', 'response':{'usage':{
        'input_tokens':12, 'output_tokens':7, 'input_tokens_details':{'cached_tokens':4}}}}]}]
    usage = recorded_usage([], trace)
    assert usage['input_tokens'] == 12 and usage['output_tokens'] == 7
    assert usage['source'] == 'provider-response-events'
    assert usage['cached_input_tokens'] == 4
    assert recorded_usage([], [])['input_tokens'] is None
