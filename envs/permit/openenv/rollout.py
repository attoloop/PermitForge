"""Run locally: uv run python rollout.py --smoke (fast evaluation, native submission).

Model use (requires an explicitly authorized API call):
    uv run --extra agents python rollout.py --model anthropic:MODEL --base-url http://127.0.0.1:8012
"""
import argparse
import json
from pathlib import Path

from openenv.core.env_server.mcp_types import CallToolAction

from permit_openenv.environment import PermitForgeEnvironment
from permit_openenv.protocol import tool_result


def smoke(task_id: str | None, split: str, index: int) -> dict:
    env = PermitForgeEnvironment()
    try:
        env.reset(task_id=task_id, split=split, index=index)
        scenario = tool_result(env.step(CallToolAction(tool_name='get_scenario', arguments={})))
        print('get_scenario:', json.dumps(scenario), flush=True)
        for name in ('check_design', 'evaluate_design', 'submit_design'):
            print(f'{name}: running (native verification may take minutes)', flush=True)
            result = tool_result(env.step(CallToolAction(tool_name=name, arguments={'design': scenario['draft_design']})))
            print(f'{name}:', json.dumps(result), flush=True)
            if name == 'check_design':
                assert result['valid'], result
            else:
                assert result['feasible'], result
        assert result['reward'] == 0.0 and result['prescore_usd_saved'] == 0, result
        return env.episode.record(model='smoke:draft')
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--task-id')
    parser.add_argument('--split', default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('evidence/smoke.json'))
    args = parser.parse_args()
    if args.smoke:
        record = smoke(args.task_id, args.split, args.index)
    else:
        if not args.base_url:
            parser.error('--model requires --base-url pointing to a running server')
        from openenv.core.mcp_client import MCPToolClient
        from permit_openenv.agent import run_episode
        with MCPToolClient(args.base_url, message_timeout_s=1800).sync() as client:
            record = run_episode(client, args.model, args.max_turns, task_id=args.task_id,
                                 split=args.split, index=args.index)
        for step in record['steps']:
            print(step['tool'], json.dumps(step['result']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'Episode record: {args.output}', flush=True)


if __name__ == '__main__':
    main()
