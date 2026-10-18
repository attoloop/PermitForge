"""Isolated OAuth subscription driver: python -m eval.run_eval --models claudecode:MODEL ... ."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any

from permit_openenv.prompts import PROMPT_VERSION, instructions

from .claudecode_proxy import ClaudeBoundary, proxy_app
from .common import TOOLS, now
from .privacy import redact
from .settings import MAX_TOKENS, model_settings
from .transport import TERMINATING, gateway, serve


def claude_env(work: Path, proxy: str) -> dict[str, str]:
    original = Path(os.environ.get('CLAUDE_CONFIG_DIR', str(Path.home()/'.claude')))
    credentials = json.loads((original/'.credentials.json').read_text())
    oauth = credentials.get('claudeAiOauth', {})
    if not oauth.get('accessToken') or oauth.get('subscriptionType') not in ('pro', 'max', 'team', 'enterprise'):
        raise RuntimeError('Claude Code requires subscription OAuth credentials')
    state = work/'config'
    state.mkdir(mode=0o700)
    target = state/'.credentials.json'
    target.write_text(json.dumps({'claudeAiOauth': oauth}))
    target.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in (
        'ANTHROPIC', 'CLAUDE', 'CODEX', 'OPENAI', 'API_KEY', 'AUTH_TOKEN', 'HF_TOKEN', 'PERMIT_EVAL_TOKEN'))}
    env.update(CLAUDE_CONFIG_DIR=str(state), ANTHROPIC_BASE_URL=proxy,
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', CLAUDE_CODE_DISABLE_AUTO_MEMORY='1',
               DISABLE_AUTOUPDATER='1', DISABLE_TELEMETRY='1', DISABLE_ERROR_REPORTING='1',
               CLAUDE_CODE_MAX_RETRIES='0', CLAUDE_CODE_DISABLE_BUNDLED_SKILLS='1',
               CLAUDE_CODE_SKIP_PROMPT_HISTORY='1', CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(MAX_TOKENS))
    return env


def subscription_status(status: dict) -> bool:
    return (status.get('loggedIn') is True and status.get('authMethod') == 'claude.ai' and
            status.get('apiProvider', 'firstParty') == 'firstParty' and
            status.get('apiKeySource') in (None, '', 'none'))


def claude_command(model: str, prompt: str, mcp_config: Path, *, max_turns: int, effort: str) -> list[str]:
    return ['claude', '-p', '--model', model, '--effort', effort, '--system-prompt', prompt,
            '--tools', '', '--mcp-config', str(mcp_config), '--strict-mcp-config',
            '--allowedTools', ','.join('mcp__permitforge__'+t for t in TOOLS),
            '--output-format', 'stream-json', '--verbose', '--max-turns', str(max_turns),
            '--no-session-persistence', '--disable-slash-commands', '--setting-sources', '',
            '--settings', json.dumps({'disableAllHooks': True, 'autoMemoryEnabled': False,
                                     'enabledPlugins': {}})]


def usage_from(events: list[dict], trace: list[dict]) -> dict:
    reported = next((e['usage'] for e in reversed(events) if e.get('type') == 'result' and e.get('usage')), None)
    if reported is None:
        reports = [t['usage'] for t in trace]
        if not reports:
            reports = [e['message']['usage'] for e in events if e.get('type') == 'assistant' and
                       e.get('message', {}).get('usage')]
        if not reports:
            return {'input_tokens': None, 'output_tokens': None, 'usd': None, 'source': 'unavailable'}
        reported = {k: sum(r.get(k, 0) or 0 for r in reports) for k in (
            'input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')}
    return {**reported, 'usd': None, 'source': 'claude-stream-json' if any(
        e.get('type') == 'result' and e.get('usage') for e in events) else 'reported-partial'}


def run_claudecode(session: Any, model: str, *, max_turns: int = 40, max_tokens: int = MAX_TOKENS,
                   wall_clock: float = 1800, effort: str = 'xhigh') -> dict:
    sampling = model_settings(model, effort)
    budget = session.meta.get('evaluate_budget', 30)
    system = instructions(budget)
    prompt = f'Use the permitforge tools. You have {max_turns} turns; submit before the last turn ends.'
    capture = getattr(session, 'capture', None)
    boundary = ClaudeBoundary(system, prompt, max_turns=max_turns, max_tokens=max_tokens, effort=effort, capture=capture)
    started = now()
    events: list[dict] = []
    output, errors = [], []
    with tempfile.TemporaryDirectory(prefix='permit-eval-claude-') as temporary:
        work = Path(temporary)
        cwd = work/'empty'
        cwd.mkdir()
        with serve(gateway(session)) as endpoint, serve(proxy_app(boundary, session=session, wall_clock=wall_clock)) as proxy:
            env = claude_env(work, proxy)
            auth = subprocess.run(['claude', 'auth', 'status', '--json'], cwd=cwd, env=env,
                                  text=True, capture_output=True, timeout=10)
            try:
                status = json.loads(auth.stdout)
            except ValueError:
                status = {}
            if capture:
                capture.raw_json('auth-status.json', {k: status.get(k) for k in (
                    'loggedIn', 'authMethod', 'apiProvider', 'apiKeySource', 'subscriptionType')})
            if auth.returncode or not subscription_status(status):
                raise RuntimeError('Claude Code requires subscription login; API-key authentication refused')
            config = {'mcpServers': {'permitforge': {'type': 'http', 'url': endpoint+'/mcp'}}}
            config_path = work/'mcp.json'
            config_path.write_text(json.dumps(config))
            argv = claude_command(model.removeprefix('claudecode:'), system, config_path,
                                  max_turns=max_turns, effort=effort)
            if capture:
                capture.event('system', system)
                capture.event('user', prompt)
                capture.raw_json('command.json', {'argv': argv, 'cwd': str(cwd), 'stdin': prompt})
                capture.raw_json('config.json', config)
                capture.raw_text('claude-events.jsonl', '')
                capture.raw_text('claude-stderr.log', '')
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            def read(stream, chunks, name):
                for line in stream:
                    chunks.append(line)
                    if capture:
                        capture.raw_text(name, line, append=True)
                    if name != 'claude-events.jsonl':
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    events.append(event)
                    if event.get('type') == 'system' and event.get('subtype') == 'init':
                        if event.get('apiKeySource') not in (None, '', 'none'):
                            boundary.error = 'authentication'
                        if set(event.get('tools', [])) != boundary.allowed:
                            boundary.error = 'isolation'
                    if capture and event.get('type') == 'assistant':
                        message = event.get('message', {})
                        capture.event('assistant', message.get('content', []), usage=message.get('usage'))
                stream.close()
            readers = [threading.Thread(target=read, args=(process.stdout, output, 'claude-events.jsonl')),
                       threading.Thread(target=read, args=(process.stderr, errors, 'claude-stderr.log'))]
            for reader in readers:
                reader.start()
            try:
                process.stdin.write(prompt)
                process.stdin.close()
                deadline = time.monotonic()+wall_clock
                while process.poll() is None:
                    if session.submit_attempted and session.record().get('final') is not None:
                        break
                    if TERMINATING.is_set() or boundary.error in ('authentication', 'isolation'):
                        boundary.error = boundary.error or 'terminated'
                        break
                    if time.monotonic() >= deadline:
                        boundary.error = 'timeout'
                        break
                    try:
                        process.wait(timeout=.1)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                for reader in readers:
                    reader.join(timeout=5)
    if hasattr(session, 'finalize'):
        session.finalize()
    record = session.record()
    result = next((e for e in reversed(events) if e.get('type') == 'result'), {})
    turn_limit = result.get('subtype') == 'error_max_turns' or boundary.error == 'turn_limit'
    infrastructure = bool(boundary.error not in (None, 'timeout', 'turn_limit') or
                          (process.returncode or result.get('is_error')) and not turn_limit)
    ended = ('submitted' if record.get('final') is not None else 'turn_limit' if turn_limit else
             'error' if infrastructure or boundary.error else 'no_submit')
    record.update(model=model, prompt_version=PROMPT_VERSION, started_at=started,
                  finished_at=record.get('finished_at') or now(), ended=ended,
                  turns=boundary.turns or result.get('num_turns', 0), usage=usage_from(events, boundary.trace),
                  retries=boundary.retries, model_trace=boundary.trace, claude_events=redact(events),
                  claude_exit_code=process.returncode, claude_stderr=''.join(errors)[-4000:],
                  error_type='infrastructure' if infrastructure else boundary.error,
                  infrastructure_error=infrastructure and ended != 'submitted',
                  retryable=infrastructure and not session.submit_attempted, submit_attempted=session.submit_attempted,
                  traceback=boundary.traceback,
                  settings={**sampling, 'max_turns': max_turns, 'max_tokens': max_tokens, 'wall_clock': wall_clock,
                            'temperature': 'provider-default', 'evaluate_budget': budget,
                            'isolation': 'OAuth-only config copy + empty cwd + four-tool recording boundary'})
    return record
