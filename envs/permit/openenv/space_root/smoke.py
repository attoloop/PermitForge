"""HTTP/MCP smoke: uv run python space_root/smoke.py --url http://127.0.0.1:7860.

--start-local starts a temporary server on a free loopback port. Outputs measured
endpoint and tool timings; never calls a model API. Native submit can take minutes.
"""

import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient
from permit_openenv.protocol import tool_result

from bake_showcases import SHOWCASES


ROOT = Path(__file__).resolve().parents[4]


def smoke(url: str) -> dict:
    receipt = {'url': url, 'endpoints': {}, 'tasks': {}}
    for path in ('/healthz', '/web/'):
        started = time.monotonic()
        with urllib.request.urlopen(url + path, timeout=60) as response:
            body = response.read().decode()
            assert response.status == 200
            assert response.geturl() == url + path, response.geturl()
            if path == '/healthz':
                assert json.loads(body)['status'] == 'ok'
            else:
                assert '<html' in body.lower() and '<script' in body.lower()
        receipt['endpoints'][path] = {'status': 200, 'seconds': time.monotonic() - started}
    snippet = re.search(r'```python\n(.*?)\n```', (ROOT / 'README.md').read_text(), re.S).group(1)
    scope = {}
    exec(compile(snippet.replace('http://127.0.0.1:8012', url), 'README.md:client', 'exec'), scope)
    assert tool_result(scope['checked'].observation)['valid']
    receipt['readme_client'] = 'passed'
    with MCPToolClient(url, message_timeout_s=1800).sync() as env:
        for task_id in SHOWCASES:
            timings = {}
            started = time.monotonic()
            env.reset(task_id=task_id)
            timings['reset'] = time.monotonic() - started

            def call(name: str, arguments: dict) -> dict:
                started = time.monotonic()
                result = tool_result(env.step(CallToolAction(tool_name=name, arguments=arguments)).observation)
                timings[name] = time.monotonic() - started
                return result

            scenario = call('get_scenario', {})
            assert scenario['task_id'] == task_id and scenario['split'] == 'eval'
            evaluation = call('evaluate_design', {'design': scenario['draft_design']})
            assert evaluation['feasible'] and evaluation['evaluations_left'] == 29, evaluation
            assert evaluation['backend'] == 'fast evaluator', evaluation
            # Native submission only on the first showcase; compare every returned standard.
            submission = None
            if task_id == SHOWCASES[0]:
                print('submit_design: full native verification running', flush=True)
                submission = call('submit_design', {'design': scenario['draft_design']})
                assert submission['feasible'] and submission['reward'] == 0
                assert submission['prescore_usd_saved'] == 0
                assert submission['backend'] == 'full AERMOD', submission
                assert submission['standards'] == evaluation['standards']
            receipt['tasks'][task_id] = {'seconds': timings, 'evaluation': evaluation,
                                         'submission': submission}
            print(json.dumps({'task_id': task_id, 'seconds': timings}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--start-local', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/evidence/space-smoke.json')
    args = parser.parse_args()
    server = None
    with tempfile.TemporaryFile(mode='w+') as log:
        try:
            if args.start_local:
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                args.url = f'http://127.0.0.1:{port}'
                server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'permit_openenv.server:app',
                    '--host', '127.0.0.1', '--port', str(port), '--workers', '1',
                    '--ws-ping-timeout', '1800'], stdout=log, stderr=log)
                deadline = time.monotonic() + 180
                while True:
                    try:
                        urllib.request.urlopen(args.url + '/healthz', timeout=1).close()
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if server.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            raise RuntimeError(log.read()) from None
                        time.sleep(.2)
            receipt = smoke(args.url.rstrip('/'))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == '__main__':
    main()
