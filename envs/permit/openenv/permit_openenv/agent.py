"""Anthropic tool-use loop; invoked only by an explicit model rollout."""
from __future__ import annotations

import json
import time
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction

from .episode import utc_now
from .prompts import PROMPT_VERSION
from .protocol import tool_result


def run_episode(client: Any, model: str, max_turns: int = 40, *, task_id: str | None = None,
                split: str = 'eval', index: int = 0, max_tokens: int = 4096) -> dict:
    provider, _, model_id = model.partition(':')
    if provider != 'anthropic' or not model_id:
        raise ValueError('model must be anthropic:<model>')
    if max_turns < 1:
        raise ValueError('max_turns must be positive')
    reset = client.reset(task_id=task_id) if task_id else client.reset(split=split, index=index)
    meta = reset.observation.metadata
    record = {
        'schema_version': 1, 'task_id': meta['task_id'], 'split': meta['split'],
        'station': meta['scenario']['station'], 'model': model, 'prompt_version': PROMPT_VERSION,
        'started_at': utc_now(), 'finished_at': None, 'scenario': meta['scenario'],
        'geometry': meta['geometry'], 'steps': [], 'evaluations_used': 0, 'ended': 'no_submit',
        'final': None, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'usd': None},
    }
    started = time.monotonic()
    tools = [{'name': t.name, 'description': t.description, 'input_schema': t.input_schema}
             for t in client.list_tools()]
    messages: list[dict] = [{'role': 'user', 'content': json.dumps(meta['scenario']) +
                            f'\nYou have {max_turns} turns. Submit before your final turn ends.'}]
    try:
        import anthropic
        api = anthropic.Anthropic(timeout=1800, max_retries=0)
        empty = 0
        for turn in range(max_turns):
            if turn == max_turns-1:
                messages.append({'role': 'user', 'content': 'Last turn: submit your best feasible design now.'})
            response = api.messages.create(model=model_id, system=meta['instructions'], tools=tools,
                                           messages=messages, max_tokens=max_tokens)
            usage = response.usage
            record['usage']['input_tokens'] += (usage.input_tokens +
                (getattr(usage, 'cache_read_input_tokens', 0) or 0) +
                (getattr(usage, 'cache_creation_input_tokens', 0) or 0))
            record['usage']['output_tokens'] += usage.output_tokens
            blocks = [b.model_dump(exclude_none=True) if hasattr(b, 'model_dump') else vars(b)
                      for b in response.content]
            messages.append({'role': 'assistant', 'content': blocks})
            assistant_text = ''.join(b.text for b in response.content if b.type == 'text')[:2000]
            calls = [b for b in response.content if b.type == 'tool_use']
            if not calls:
                empty += 1
                if empty >= 2:
                    break
                messages.append({'role': 'user', 'content': 'Use the tools; only submit_design submits an answer.'})
                continue
            empty = 0
            results = []
            for call in calls:
                if record['final'] is not None:
                    break
                step = client.step(CallToolAction(tool_name=call.name, arguments=call.input))
                result = tool_result(step.observation)
                record['steps'].append({'i': len(record['steps']), 'tool': call.name, 'args': call.input,
                                        'result': result, 't_s': round(time.monotonic()-started, 6),
                                        'assistant_text': assistant_text})
                if 'evaluations_left' in result:
                    record['evaluations_used'] = meta['evaluate_budget'] - result['evaluations_left']
                results.append({'type': 'tool_result', 'tool_use_id': call.id,
                                'content': json.dumps(result) + f'\nTurns left: {max_turns-turn-1}'})
                if call.name == 'submit_design' and 'reward' in result:
                    record['final'], record['ended'] = result, 'submitted'
            if record['final'] is not None:
                break
            messages.append({'role': 'user', 'content': results})
        else:
            record['ended'] = 'turn_limit'
    except Exception as exc:
        record['ended'] = 'error'
        record['error'] = f'{type(exc).__name__}: {str(exc)[:800]}'
    record['finished_at'] = utc_now()
    return record
