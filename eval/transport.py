"""Private recording sidecar: python -m eval.transport --port 8012 (PERMIT_EVAL_TOKEN required)."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any, Iterator

from permit_openenv.episode import STORE
from permit_openenv.protocol import tool_result
from permit_openenv.tasks import Task, TaskPack, load_pack

from .common import TOOLS, anchors_for


class EvalTask(Task):
    @property
    def anchors(self) -> dict:
        # The core Task property rejects SCHEMAS' explicit null-best / source:none shape.
        # Keep this compatibility adapter inside eval; no environment code is changed.
        return anchors_for(self)


def load_eval_pack(root: Path | None = None) -> TaskPack:
    pack = load_pack(root)
    return TaskPack([EvalTask(t.task_id, t.split, t.record, t.root) for t in pack.tasks
                     if t.record.get('eligible') is True])


def recording_app(token: str, *, pack: TaskPack | None = None, evaluator: Any = None, verifier: Any = None):
    from fastapi import Header, HTTPException
    from permit_openenv.server import create_server
    app = create_server(pack=pack, evaluator=evaluator, verifier=verifier)

    @app.get('/eval/records/{episode_id}', include_in_schema=False)
    def record(episode_id: str, authorization: str = Header(default='')):
        if not secrets.compare_digest(authorization, 'Bearer '+token):
            raise HTTPException(status_code=403, detail='recording credentials required')
        ep = STORE.get(episode_id)
        if ep is None:
            raise HTTPException(status_code=404, detail='episode not found')
        return ep.record()
    return app


@contextmanager
def serve(app: Any) -> Iterator[str]:
    import uvicorn
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=sock.getsockname()[1],
        log_level='error', ws_ping_timeout=1800, timeout_graceful_shutdown=5))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic()+30
    try:
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('local server failed to start')
            time.sleep(.02)
        yield f'http://127.0.0.1:{sock.getsockname()[1]}'
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


class Session:
    """One OpenEnv WebSocket session; the driver never exposes its reset/record endpoints."""
    def __init__(self, url: str, token: str, task_id: str, *, timeout: float = 7200, checkpoint: Any = None):
        self.url, self.token, self.task_id, self.timeout = url.rstrip('/'), token, task_id, timeout
        self.submit_attempted = False
        self.checkpoint = checkpoint or (lambda record: None)
        self._texts: list[str] = []
        self._lock = threading.RLock()

    def __enter__(self):
        from openenv.core.mcp_client import MCPToolClient
        self._context = MCPToolClient(self.url, message_timeout_s=self.timeout).sync()
        self.client = self._context.__enter__()
        try:
            reset = self.client.reset(task_id=self.task_id)
            self.meta = reset.observation.metadata
            self.checkpoint(self.record())  # Reject a missing recorder before model spend.
        except BaseException:
            self._context.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)

    def tools(self) -> list[dict]:
        specs = [{'name': t.name, 'description': t.description, 'input_schema': t.input_schema}
                 for t in self.client.list_tools()]
        if {t['name'] for t in specs} != set(TOOLS):
            raise ValueError('server does not expose exactly the four public tools')
        return specs

    def call(self, name: str, args: dict, text: str = '') -> dict:
        from openenv.core.env_server.mcp_types import CallToolAction
        if name not in TOOLS:
            return {'error': 'unknown tool'}
        with self._lock:
            if self.submit_attempted:
                return {'error': 'episode is over'}
            if name == 'submit_design':
                self.submit_attempted = True  # A lost response is never a reason to re-grade.
                self.checkpoint({**self.record(), 'submit_attempted': True, 'pending_submission': args})
            self._texts.append(text[:2000])
            result = self.client.step(CallToolAction(tool_name=name, arguments=args))
            self.checkpoint(self.record())
            return tool_result(result.observation)

    def record(self) -> dict:
        import httpx
        response = httpx.get(self.url+'/eval/records/'+self.meta['episode_id'],
                             headers={'Authorization': 'Bearer '+self.token}, timeout=30, trust_env=False)
        response.raise_for_status()
        record = response.json()
        for step, text in zip(record['steps'], self._texts):
            if text:
                step['assistant_text'] = text
        return record


def gateway(session: Session):
    """An episode-bound streamable HTTP MCP surface with no reset or private record route."""
    from fastmcp import FastMCP
    mcp = FastMCP('permitforge')
    for spec in session.tools():
        name = spec['name']
        if name == 'get_scenario':
            def get_scenario() -> str:
                return json.dumps(session.call('get_scenario', {}))
            mcp.tool(get_scenario, name=name, description=spec['description'])
        else:
            from fastmcp.tools import Tool
            from mcp.types import TextContent
            class Forward(Tool):
                async def run(self, arguments: dict) -> Any:
                    import asyncio
                    from fastmcp.tools.tool import ToolResult
                    result = await asyncio.to_thread(session.call, self.name, arguments)
                    return ToolResult(content=[TextContent(type='text', text=json.dumps(result))])
            mcp.add_tool(Forward(name=name, description=spec['description'], parameters=spec['input_schema']))
    return mcp.http_app(path='/mcp', stateless_http=True)


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8012)
    parser.add_argument('--tasks-root', type=Path)
    args = parser.parse_args()
    token = os.environ.get('PERMIT_EVAL_TOKEN')
    if not token:
        parser.error('set PERMIT_EVAL_TOKEN for the private record endpoint')
    uvicorn.run(recording_app(token, pack=load_eval_pack(args.tasks_root)), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
