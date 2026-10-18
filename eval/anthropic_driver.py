"""Messages tool loop; `python -m eval.run_eval --models anthropic:MODEL ...`."""
from __future__ import annotations

import json
from pathlib import Path
import time
import tomllib
import traceback
from typing import Any, Callable

from permit_openenv.prompts import PROMPT_VERSION, instructions, opening

from .common import TOOLS, now
from .settings import MAX_TOKENS, model_settings
from .budget import BudgetExhausted, request_bound


def prices() -> dict:
    return tomllib.loads(Path(__file__).with_name('prices.toml').read_text())['models']


def plain(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True)
    if hasattr(value, '__dict__'):
        return {k: plain(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def request_cost(model: str, usage: dict) -> float:
    table = prices()
    if model not in table:
        raise ValueError(f'No verified price for {model}')
    p = table[model]
    inputs = usage.get('input_tokens', 0)
    reads = usage.get('cache_read_input_tokens', 0) or 0
    writes = usage.get('cache_creation_input_tokens', 0) or 0
    creation = usage.get('cache_creation') or {}
    long_writes = creation.get('ephemeral_1h_input_tokens', 0) or 0
    short_writes = creation.get('ephemeral_5m_input_tokens', writes-long_writes) or 0
    if short_writes + long_writes != writes:
        raise ValueError('cache creation usage does not reconcile')
    multiplier = (p.get('long_context_multiplier', 1) if
                  inputs+reads+writes > p.get('long_context_threshold', float('inf')) else 1)
    return multiplier * (inputs*p['input'] + usage.get('output_tokens', 0)*p['output'] +
                         reads*p['cache_read'] + short_writes*p['cache_write_5m'] +
                         long_writes*p['cache_write_1h']) / 1_000_000


def transient(exc: Exception) -> bool:
    status = getattr(exc, 'status_code', None)
    return isinstance(exc, (ConnectionError, TimeoutError)) or status in (408, 429, 500, 502, 503, 504, 529) or type(exc).__name__ in {
        'APIConnectionError', 'APITimeoutError', 'ReadTimeout', 'ConnectTimeout', 'ConnectionClosedError'}


def run_anthropic(session: Any, model: str, *, api: Any = None, max_turns: int = 40,
                  max_tokens: int = MAX_TOKENS, wall_clock: float = 1800, effort: str = 'xhigh',
                  sleep: Callable[[float], None] = time.sleep, hard_budget: Any = None,
                  prompt_cache: bool = False) -> dict:
    model_id = model.removeprefix('anthropic:')
    request_cost(model_id, {})
    sampling = model_settings(model, effort)
    evaluate_budget = session.meta.get('evaluate_budget', 30)
    system = instructions(evaluate_budget)
    if api is None:
        import anthropic
        api = anthropic.Anthropic(max_retries=0, timeout=min(1800, wall_clock))
    started, deadline = now(), time.monotonic()+wall_clock
    tools = session.tools()
    if {t['name'] for t in tools} != set(TOOLS):
        raise ValueError('environment must expose exactly the four permitforge tools')
    messages = [{'role': 'user', 'content': opening(max_turns)}]
    capture = getattr(session, 'capture', None)
    if capture:
        capture.event('system', system)
        capture.event('user', messages[0]['content'])
    usage = {'input_tokens': 0, 'output_tokens': 0, 'usd': 0.0, 'requests': []}
    retries, model_trace = [], []
    ended, error, retryable, turn = 'no_submit', None, False, 0
    trace, infrastructure_error = None, False
    try:
        for turn in range(1, max_turns+1):
            if capture:
                capture.turn = turn
            if turn == max_turns:
                messages.append({'role': 'user', 'content': 'Last turn: submit your best feasible design now.'})
                if capture:
                    capture.event('user', messages[-1]['content'])
            for attempt in range(3):
                from .transport import TERMINATING
                if TERMINATING.is_set():
                    raise InterruptedError('harness terminated')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('episode wall clock exhausted')
                prefix = f'anthropic-{turn:04d}-{attempt+1}'
                request = dict(model=model_id, system=system, tools=tools,
                               messages=messages, max_tokens=max_tokens, timeout=min(1800, remaining),
                               output_config=sampling['output_config'], thinking=sampling['thinking'])
                if prompt_cache:
                    import copy
                    request = copy.deepcopy(request)
                    request['system'] = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]
                    request['tools'][-1]['cache_control'] = {'type': 'ephemeral'}
                    latest = request['messages'][-1]
                    if isinstance(latest['content'], str):
                        latest['content'] = [{'type': 'text', 'text': latest['content']}]
                    latest['content'][-1]['cache_control'] = {'type': 'ephemeral'}
                if capture:
                    capture.raw_json(prefix+'-request.json', request)
                budget_key = str(capture.root) + '/' + prefix if capture else model+'/'+started+'/'+prefix
                if hard_budget is not None:
                    hard_budget.reserve(budget_key, request_bound(model_id, max_tokens))
                try:
                    if capture and hasattr(api.messages, 'with_raw_response'):
                        wire = api.messages.with_raw_response.create(**request)
                        capture.raw_text(prefix+'-response.json', wire.http_response.text)
                        capture.raw_text(prefix+'-request.json', wire.http_response.request.content.decode())
                        response = wire.parse()
                    else:
                        response = api.messages.create(**request)
                        if capture:
                            capture.raw_json(prefix+'-response.json', plain(response))
                    break
                except Exception as exc:
                    if capture:
                        if getattr(exc, 'response', None) is not None:
                            capture.raw_text(prefix+'-response.txt', exc.response.text)
                        capture.failed(prefix, {'type': type(exc).__name__, 'message': str(exc)})
                    if not transient(exc) or attempt == 2:
                        raise
                    retries.append({'turn': turn, 'attempt': attempt+1, 'at': now(),
                                    'error': type(exc).__name__, 'reason': str(exc)[:800]})
                    sleep(min(2**attempt, max(0, deadline-time.monotonic())))
            u = plain(response.usage)
            usd = request_cost(model_id, u)
            if hard_budget is not None:
                hard_budget.settle(budget_key, usd)
            usage['requests'].append({**u, 'usd': usd})
            usage['input_tokens'] += u.get('input_tokens', 0)+(u.get('cache_read_input_tokens', 0) or 0)+(u.get('cache_creation_input_tokens', 0) or 0)
            usage['output_tokens'] += u.get('output_tokens', 0)
            usage['usd'] += usd
            blocks = [plain(b) for b in response.content]
            if capture:
                capture.event('assistant', blocks, usage=u)
            model_trace.append({'turn': turn, 'content': blocks, 'usage': u})
            if hasattr(session, 'checkpoint'):
                session.checkpoint({'usage': usage, 'turns': turn, 'retries': retries, 'model_trace': model_trace})
            messages.append({'role': 'assistant', 'content': blocks})
            calls = [b for b in blocks if b['type'] == 'tool_use']
            text = ''.join(b.get('text', '') for b in blocks if b['type'] == 'text')[:2000]
            if not calls:
                break
            results = []
            for call in calls:
                if time.monotonic() >= deadline:
                    raise TimeoutError('episode wall clock exhausted')
                if call['name'] not in TOOLS:
                    result = {'error': 'unknown tool'}
                else:
                    result = session.call(call['name'], call['input'], text)
                results.append({'type': 'tool_result', 'tool_use_id': call['id'], 'content': json.dumps(result)})
                if session.submit_attempted:
                    ended = 'submitted' if session.record().get('final') is not None else 'error'
                    break
            if session.submit_attempted:
                break
            messages.append({'role': 'user', 'content': results})
        else:
            ended = 'turn_limit'
    except Exception as exc:
        ended, error = 'error', {'type': type(exc).__name__, 'message': str(exc)[:800]}
        trace = traceback.format_exc()
        infrastructure_error = not (isinstance(exc, BudgetExhausted) or
                                    isinstance(exc, TimeoutError) and str(exc) == 'episode wall clock exhausted')
        if isinstance(exc, BudgetExhausted):
            ended = 'no_submit'
        retryable = transient(exc) and not session.submit_attempted
        if type(exc).__name__ in ('APITimeoutError', 'APIConnectionError', 'ReadTimeout'):
            usage.update(known_usd=usage['usd'], usd=None, billing_incomplete=True)
    if hasattr(session, 'finalize'):
        session.finalize()
    record = session.record()
    if record.get('final') is not None:
        ended, retryable = 'submitted', False
    record.update(model=model, prompt_version=PROMPT_VERSION, started_at=started, finished_at=record.get('finished_at') or now(),
                  turns=turn, ended=ended, usage=usage, retries=retries, model_trace=model_trace,
                  error=error, retryable=retryable, submit_attempted=session.submit_attempted,
                  error_type='budget' if error and error['type'] == 'BudgetExhausted' else
                             'infrastructure' if infrastructure_error else 'timeout' if error else None,
                  infrastructure_error=infrastructure_error, traceback=trace,
                  settings={**sampling, 'max_turns': max_turns, 'max_tokens': max_tokens, 'wall_clock': wall_clock,
                            'temperature': 'provider-default', 'evaluate_budget': evaluate_budget,
                            'prompt_cache': prompt_cache, 'prices': prices()[model_id]})
    return record
