"""Recording Messages boundary for the isolated Claude Code subscription driver."""
from __future__ import annotations

import json
import time
import traceback
from typing import Any

from fastapi import Request

from .codex_driver import IsolationError
from .common import TOOLS, now
from .transport import TERMINATING


class ClaudeBoundary:
    def __init__(self, system: str, prompt: str, *, max_turns: int, max_tokens: int,
                 effort: str, capture: Any = None):
        self.system, self.prompt = system, prompt
        self.max_turns, self.max_tokens, self.effort = max_turns, max_tokens, effort
        self.capture = capture
        self.allowed = {'mcp__permitforge__'+t for t in TOOLS}
        self.turns = 0
        self.error: str | None = None
        self.traceback: str | None = None
        self.trace: list[dict] = []
        self.retries: list[dict] = []

    def prepare(self, body: dict) -> dict:
        if self.turns >= self.max_turns:
            self.error = 'turn_limit'
            raise TimeoutError('turn limit reached')
        if {t.get('name') for t in body.get('tools', [])} != self.allowed or len(body['tools']) != 4:
            raise IsolationError('Claude Code must expose exactly four PermitForge tools')
        if body.get('output_config', {}).get('effort') != self.effort:
            raise IsolationError('Claude Code effort differs from the pinned setting')
        messages = [{'role': 'user', 'content': self.prompt}]
        for item in body.get('messages', []):
            if item.get('role') == 'assistant':
                self.validate(item)
                messages.append(item)
            elif item.get('role') == 'user' and isinstance(item.get('content'), list):
                results = [b for b in item['content'] if b.get('type') == 'tool_result']
                if results:
                    messages.append({'role': 'user', 'content': results})
        self.turns += 1
        if self.capture:
            self.capture.turn = self.turns
        if self.turns == self.max_turns:
            notice = {'type': 'text', 'text': 'Last turn: submit your best feasible design now.'}
            if isinstance(messages[-1]['content'], str):
                messages[-1]['content'] = [{'type': 'text', 'text': messages[-1]['content']}]
            if messages[-1]['role'] == 'user':
                messages[-1]['content'].append(notice)
            else:
                messages.append({'role': 'user', 'content': [notice]})
            if self.capture:
                self.capture.event('user', notice)
        result = {k: v for k, v in body.items() if k in (
            'model', 'thinking', 'stream', 'tools', 'output_config')}
        result.update(system=[{'type': 'text', 'text': self.system}], messages=messages,
                      max_tokens=self.max_tokens)
        return result

    def validate(self, message: dict) -> None:
        for block in message.get('content', []):
            if block.get('type') == 'tool_use' and block.get('name') not in self.allowed:
                self.error = 'isolation'
                raise IsolationError('provider returned a non-PermitForge tool call')
            if block.get('type') not in ('text', 'thinking', 'redacted_thinking', 'tool_use'):
                self.error = 'isolation'
                raise IsolationError('unsupported provider output block')


def proxy_app(boundary: ClaudeBoundary, *, session: Any = None, wall_clock: float = 1800,
              upstream: str = 'https://api.anthropic.com', transport: Any = None):
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import Response
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    deadline = time.monotonic()+wall_clock

    @app.post('/v1/messages/count_tokens')
    async def count_tokens():
        # No provider token-count request or other side traffic can incur charges.
        return {'input_tokens': 0}

    @app.post('/v1/messages')
    async def messages(request: Request):
        if 'x-api-key' in request.headers or not request.headers.get('authorization', '').startswith('Bearer '):
            boundary.error = 'authentication'
            return Response('subscription OAuth required; API-key billing forbidden', status_code=403)
        if session is not None and session.submit_attempted:
            return Response('episode is over', status_code=409)
        try:
            body = boundary.prepare(await request.json())
        except (IsolationError, TimeoutError) as exc:
            boundary.error = boundary.error or 'isolation'
            boundary.traceback = traceback.format_exc()
            return Response(str(exc), status_code=400)
        headers = {k: v for k, v in request.headers.items() if k.lower() in (
            'authorization', 'anthropic-version', 'anthropic-beta', 'user-agent', 'x-app')}
        headers.update({'content-type': 'application/json', 'accept': 'text/event-stream'})
        for attempt in range(1, 4):
            prefix = f'claude-{boundary.turns:04d}-{attempt}'
            capture = boundary.capture
            if capture:
                capture.raw_json(prefix+'-request.json', body)
            remaining = deadline-time.monotonic()
            if remaining <= 0 or TERMINATING.is_set():
                boundary.error = 'timeout'
                return Response('episode wall clock exhausted', status_code=408)
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(remaining, connect=min(10, remaining)),
                                             transport=transport, trust_env=False) as client:
                    chunks = []
                    async with client.stream('POST', upstream+'/v1/messages', json=body, headers=headers) as response:
                        status = response.status_code
                        if capture:
                            capture.raw_json(prefix+'-status.json', {'status_code': status})
                        try:
                            async for chunk in response.aiter_bytes():
                                chunks.append(chunk)
                        finally:
                            raw = b''.join(chunks).decode('utf-8', errors='replace')
                            if capture:
                                capture.raw_text(prefix+'-response.sse', raw)
                if status in (408, 429, 500, 502, 503, 504, 529) and attempt < 3:
                    boundary.retries.append({'turn': boundary.turns, 'attempt': attempt, 'at': now(), 'status': status})
                    if capture:
                        capture.failed(prefix, {'status_code': status})
                    continue
                if status != 200:
                    boundary.error = 'provider'
                    return Response(raw, status_code=status)
                events = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith('data:')]
                if not any(e.get('type') == 'message_stop' for e in events):
                    raise ValueError('incomplete provider response; not replayed')
                message: dict = {'content': []}
                usage: dict = {}
                for event in events:
                    if event.get('type') == 'message_start':
                        usage.update(event.get('message', {}).get('usage', {}))
                    if event.get('type') == 'message_delta':
                        usage.update(event.get('usage', {}))
                    if event.get('type') == 'content_block_start':
                        message['content'].append(event['content_block'])
                boundary.validate(message)
                boundary.trace.append({'turn': boundary.turns, 'events': events, 'usage': usage})
                if session is not None and hasattr(session, 'checkpoint'):
                    session.checkpoint({'turns': boundary.turns, 'model_trace': boundary.trace,
                                        'retries': boundary.retries})
                return Response(raw, media_type='text/event-stream')
            except IsolationError:
                boundary.traceback = traceback.format_exc()
                return Response('isolation rejected provider tool', status_code=400)
            except (httpx.HTTPError, ValueError):
                boundary.error = 'infrastructure'
                boundary.traceback = traceback.format_exc()
                if capture:
                    capture.failed(prefix, boundary.traceback)
                return Response('provider response incomplete', status_code=502)
        raise AssertionError('unreachable')
    return app
