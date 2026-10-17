"""Isolated subscription driver: python -m eval.run_eval --models codex:gpt-6.1-sol ... ."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any

from fastapi import Request
from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION

from .common import TOOLS, now
from .transport import gateway, serve

UPSTREAM = 'https://chatgpt.com/backend-api/codex/responses'


class IsolationError(ValueError):
    pass


def qualified(item: dict) -> str:
    return item.get('namespace', '')+'__'+item['name'] if item.get('namespace') else item.get('name', '')


class CodexBoundary:
    """Only public tools and episode conversation cross the provider boundary.

    Filtering the request alone is insufficient: an unadvertised function call could
    still name a built-in tool. Buffer and validate the response before releasing it.
    """
    def __init__(self, prompt: str, *, max_turns: int, max_tokens: int):
        self.prompt, self.max_turns, self.max_tokens = prompt, max_turns, max_tokens
        self.turns = 0
        self.allowed = {'mcp__permitforge__'+t for t in TOOLS}
        self.trace: list[dict] = []
        self.error: str | None = None
        self.retries: list[dict] = []

    def prepare(self, body: dict) -> dict:
        if self.turns >= self.max_turns:
            self.error = 'turn_limit'
            raise TimeoutError('turn limit reached')
        offered = []
        for tool in body.get('tools', []):
            if tool.get('type') == 'namespace' and tool.get('name') == 'mcp__permitforge':
                offered.extend({**t, 'name': 'mcp__permitforge__'+t['name']} for t in tool.get('tools', []))
            else:
                offered.append(tool)
        tools = [t for t in offered if t.get('type') == 'function' and t.get('name') in self.allowed]
        if {t['name'] for t in tools} != self.allowed:
            names = [t.get('name', t.get('type')) for t in body.get('tools', [])]
            kinds = [{k:v for k,v in i.items() if k in ('type', 'name')} | {'keys': list(i)} for i in body.get('input', [])]
            raise IsolationError(f'Codex did not discover exactly the four permitforge tools: {names}; input kinds={kinds}')
        if any(t.get('type') == 'namespace' and t.get('name') == 'mcp__permitforge' for t in body.get('tools', [])):
            tools = [{'type': 'namespace', 'name': 'mcp__permitforge', 'description': 'PermitForge public tools',
                      'tools': [{**t, 'name': t['name'].removeprefix('mcp__permitforge__')} for t in tools]}]
        # Never send Codex's implicit host context, AGENTS/skill instructions or filesystem paths.
        inputs = [{'role': 'user', 'content': [{'type': 'input_text', 'text': self.prompt}]}]
        for item in body.get('input', []):
            if item.get('type') in ('function_call', 'function_call_output', 'reasoning') or item.get('role') == 'assistant':
                if item.get('type') == 'function_call' and qualified(item) not in self.allowed:
                    raise IsolationError('non-permitforge tool in conversation')
                inputs.append(item)
        self.turns += 1
        if self.turns == self.max_turns:
            inputs.append({'role': 'user', 'content': [{'type': 'input_text', 'text': 'Last turn: submit your best feasible design now.'}]})
        result = {**body, 'tools': tools, 'instructions': INSTRUCTIONS,
                  'input': inputs, 'parallel_tool_calls': False, 'max_output_tokens': self.max_tokens}
        for key in ('previous_response_id', 'prompt_cache_key', 'temperature'):
            result.pop(key, None)
        return result

    def validate_event(self, event: dict) -> None:
        items = []
        if isinstance(event.get('item'), dict):
            items.append(event['item'])
        items.extend(event.get('response', {}).get('output', []))
        for item in items:
            kind = item.get('type')
            if kind == 'function_call':
                if qualified(item) not in self.allowed:
                    self.error = 'isolation'
                    raise IsolationError('provider returned a non-permitforge function call')
            elif kind not in ('message', 'reasoning', 'compaction'):
                self.error = 'isolation'
                raise IsolationError(f'provider returned unsupported output item: {kind}')


def proxy_app(boundary: CodexBoundary, *, upstream: str = UPSTREAM, session: Any = None,
              wall_clock: float = 7200, transport: Any = None):
    from fastapi import FastAPI
    from fastapi.responses import Response
    import httpx
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    deadline = time.monotonic()+wall_clock

    @app.post('/responses')
    async def responses(request: Request):
        if session is not None and session.submit_attempted:
            return Response('episode is over', status_code=409)
        try:
            body = boundary.prepare(await request.json())
        except (IsolationError, TimeoutError) as exc:
            return Response(str(exc), status_code=400)
        # No arbitrary destination or authorization material is exposed to the model/logs.
        headers = {k:v for k,v in request.headers.items() if k.lower() in (
            'authorization', 'chatgpt-account-id', 'openai-beta', 'originator', 'user-agent')}
        headers.update({'content-type': 'application/json', 'accept': 'text/event-stream'})
        for attempt in range(3):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                boundary.error = 'timeout'
                return Response('episode wall clock exhausted', status_code=408)
            try:
                timeout = httpx.Timeout(remaining, connect=min(10, remaining))
                async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                    response = await client.post(upstream, json=body, headers=headers)
                if response.status_code in (408, 429, 500, 502, 503, 504, 529) and attempt < 2:
                    boundary.retries.append({'turn': boundary.turns, 'attempt': attempt+1,
                                             'status': response.status_code, 'at': now()})
                    continue
                if response.status_code != 200:
                    boundary.error = 'infrastructure' if response.status_code >= 500 else 'provider'
                    return Response(response.content, status_code=response.status_code,
                                    media_type=response.headers.get('content-type', 'application/json'))
                events = []
                for line in response.text.splitlines():
                    if line.startswith('data:') and line[5:].strip() != '[DONE]':
                        event = json.loads(line[5:])
                        boundary.validate_event(event)
                        events.append(event)
                if not any(e.get('type') == 'response.completed' for e in events):
                    raise ValueError('incomplete provider response; not replayed')
                boundary.trace.append({'turn': boundary.turns, 'events': events})
                if session is not None and hasattr(session, 'checkpoint'):
                    session.checkpoint({'turns': boundary.turns, 'model_trace': boundary.trace,
                                        'retries': boundary.retries})
                return Response(response.content, media_type='text/event-stream')
            except IsolationError:
                return Response('isolation boundary rejected provider tool call', status_code=400)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                boundary.retries.append({'turn': boundary.turns, 'attempt': attempt+1, 'at': now(), 'error': type(exc).__name__})
                if attempt == 2:
                    boundary.error = 'infrastructure'
                    return Response('provider connection failed', status_code=502)
            except (httpx.ReadTimeout, ValueError):
                # A response may have been billed; never replay a partially sampled response.
                boundary.error = 'timeout'
                return Response('provider response incomplete', status_code=502)
    return app


def codex_command(model: str, endpoint: str, proxy: str, workdir: Path) -> list[str]:
    catalog_path = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'models_cache.json'
    catalog = json.loads(catalog_path.read_text())
    models = [m.copy() for m in catalog['models'] if m['slug'] == model]
    if not models:
        raise ValueError(f'No local Codex model metadata for {model}; refresh the CLI catalog first')
    for metadata in models:
        metadata.pop('tool_mode', None)
        metadata['use_responses_lite'] = False
        metadata['supports_search_tool'] = False
        metadata['experimental_supported_tools'] = []
    isolated_catalog = workdir/'model-catalog.json'
    isolated_catalog.write_text(json.dumps({'models': models}))
    config = {
        'model_catalog_json': json.dumps(str(isolated_catalog)),
        'approval_policy': '"never"', 'web_search': '"disabled"',
        'features.shell_tool': 'false', 'features.unified_exec': 'false',
        'features.apps': 'false', 'features.plugins': 'false', 'features.multi_agent': 'false',
        'features.browser_use': 'false', 'features.computer_use': 'false',
        'features.memories': 'false', 'features.skill_search': 'false',
        'features.skip_host_skill_discovery': 'true', 'features.workspace_dependencies': 'false',
        'features.code_mode': 'false', 'features.code_mode_only': 'false',
        'features.image_generation': 'false', 'features.view_image': 'false',
        'project_doc_max_bytes': '0', 'analytics.enabled': 'false', 'feedback.enabled': 'false',
        'model_provider': '"permit_eval"',
        'model_providers.permit_eval.name': '"PermitForge subscription boundary"',
        'model_providers.permit_eval.base_url': json.dumps(proxy),
        'model_providers.permit_eval.wire_api': '"responses"',
        'model_providers.permit_eval.requires_openai_auth': 'true',
        'model_providers.permit_eval.supports_websockets': 'false',
        'model_providers.permit_eval.request_max_retries': '0',
        'model_providers.permit_eval.stream_max_retries': '0',
        'mcp_servers.permitforge.url': json.dumps(endpoint),
        'mcp_servers.permitforge.enabled_tools': json.dumps(list(TOOLS)),
        'mcp_servers.permitforge.tool_timeout_sec': '1800',
        'mcp_servers.permitforge.required': 'true',
        'mcp_servers.permitforge.default_tools_approval_mode': '"approve"',
    }
    argv = ['codex', 'exec', '-m', model, '--json', '--ephemeral', '--ignore-user-config',
            '--ignore-rules', '--skip-git-repo-check', '--sandbox', 'read-only', '-C', str(workdir)]
    for key, value in config.items():
        argv += ['-c', key+'='+value]
    return argv+['-']


def parse_usage(events: list[dict]) -> dict:
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0, 'usd': None}
    for event in events:
        if event.get('type') == 'turn.completed':
            for key in ('input_tokens', 'output_tokens', 'cached_input_tokens'):
                usage[key] += event.get('usage', {}).get(key, 0)
    return usage


def recorded_usage(events: list[dict], trace: list[dict]) -> dict:
    if any(e.get('type') == 'turn.completed' for e in events):
        return {**parse_usage(events), 'source': 'codex-json'}
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0,
             'usd': None, 'source': 'provider-response-events', 'incomplete': True}
    found = False
    for turn in trace:
        for event in turn['events']:
            if event.get('type') == 'response.completed' and event.get('response', {}).get('usage'):
                u = event['response']['usage']
                found = True
                usage['input_tokens'] += u.get('input_tokens', 0)
                usage['output_tokens'] += u.get('output_tokens', 0)
                usage['cached_input_tokens'] += u.get('input_tokens_details', {}).get('cached_tokens', 0)
    if not found:
        usage.update(input_tokens=None, output_tokens=None, cached_input_tokens=None, source='unavailable')
    return usage


def run_codex(session: Any, model: str, *, max_turns: int = 40, max_tokens: int = 4096,
              wall_clock: float = 7200, prompt: str | None = None) -> dict:
    # This driver cannot fall back to API-key billing.
    login = subprocess.run(['codex', 'login', 'status'], capture_output=True, text=True, timeout=10)
    if login.returncode or 'ChatGPT' not in login.stdout+login.stderr:
        raise RuntimeError('Codex requires ChatGPT subscription login; API billing is not permitted')
    prompt = prompt or INSTRUCTIONS+'\nUse the permitforge tools. '+f'You have {max_turns} turns.'
    boundary = CodexBoundary(prompt, max_turns=max_turns, max_tokens=max_tokens)
    started = now()
    with tempfile.TemporaryDirectory(prefix='permit-eval-codex-') as work:
        with serve(gateway(session)) as endpoint, serve(proxy_app(boundary, session=session, wall_clock=wall_clock)) as proxy:
            argv = codex_command(model.removeprefix('codex:'), endpoint+'/mcp', proxy, Path(work))
            env = {k:v for k,v in os.environ.items() if not any(x in k.upper() for x in
                   ('API_KEY', 'ANTHROPIC', 'HF_TOKEN', 'PERMIT_EVAL_TOKEN', 'OPENAI_BASE_URL'))}
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
            try:
                stdout, stderr = process.communicate(prompt, timeout=wall_clock)
            except subprocess.TimeoutExpired:
                boundary.error = 'timeout'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    record = session.record()
    ended = ('submitted' if record.get('final') is not None else
             'turn_limit' if boundary.error == 'turn_limit' else
             'error' if boundary.error or process.returncode else 'no_submit')
    record.update(model=model, prompt_version=PROMPT_VERSION, started_at=started, finished_at=now(),
                  turns=boundary.turns, ended=ended, usage=recorded_usage(events, boundary.trace),
                  retries=boundary.retries, model_trace=boundary.trace, codex_events=events,
                  codex_exit_code=process.returncode, codex_stderr=stderr[-4000:],
                  error_type=boundary.error, submit_attempted=session.submit_attempted, retryable=False,
                  settings={'max_turns': max_turns, 'max_tokens': max_tokens, 'wall_clock': wall_clock,
                            'temperature': 'provider-default', 'evaluate_budget': 30,
                            'isolation': 'disabled host tools + filtered/validated subscription boundary'})
    return record
