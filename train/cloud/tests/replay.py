"""Replay saved responses with installed TRL, without a model or native tools."""
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'train/cloud/data/smoke-result-grade/smoke/base-eval'


def replay_episodes(completion_limit: int = 100_000) -> list[dict]:
    from permit_openenv.tasks import load_pack
    from trl import GRPOTrainer
    from trl.chat_template_utils import add_response_schema, parse_response
    from train.cloud.config import CloudConfig
    from train.cloud.formatting import load_tokenizer
    from train.cloud.training import opening_ids
    from train.permit_env import TOOL_NAMES, opening_messages
    tok = add_response_schema(load_tokenizer(local_files_only=True))
    cfg = CloudConfig()
    opening = opening_ids(cfg, tok, load_pack(ROOT/'envs/permit/tasks'))
    results = []
    paths = sorted(BASE.rglob('episode.json'))
    assert len(paths) == 3, 'The retained base episodes are required for this regression'
    for path in paths:
        record = json.loads(path.read_text())
        rows = [json.loads(s) for s in path.with_name('transcript.jsonl').read_text().splitlines()]
        responses, feedback = [], []
        for row in rows:
            if 'response' in row:
                ids = tok.encode(row['response'], add_special_tokens=False)
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id)+1]
                responses.append(ids)
            else:
                feedback.append(row)
        replies = iter(responses[1:] + [[tok.eos_token_id]])
        tools = iter(feedback)
        executed, prefixes = [], []

        def dispatch(name, **kwargs):
            row = next(tools)
            assert (name, kwargs) == (row['tool'], row['args'])
            executed.append(name)
            return json.dumps(row['result'], separators=(',', ':'))

        def generate(inputs, *args, **kwargs):
            prefixes.append(len(inputs[0]))
            return [next(replies)], None

        loop = SimpleNamespace(processing_class=tok, _tokenizer=tok, _is_vlm=False,
            chat_template=None, chat_template_kwargs={'enable_thinking': False},
            max_tool_calling_iterations=cfg.train_turns, max_completion_length=completion_limit,
            use_vllm=False, tools=True,
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144)),
            _sync_tool_dicts=[{name: partial(dispatch, name) for name in TOOL_NAMES}],
            _async_tool_dicts=[{}], _generate_single_turn=generate)
        loop._get_tool_suffix_ids = partial(GRPOTrainer._get_tool_suffix_ids, loop)
        outcome = GRPOTrainer._tool_call_loop(loop, [opening_messages(cfg.train_turns)], [opening],
            [responses[0]], [[parse_response(tok, responses[0], prefix=opening)]], None, None, {})
        results.append(dict(task_id=record['task_id'], opening_tokens=len(opening),
            completion_tokens=len(outcome[2][0]), total_tokens=len(opening)+len(outcome[2][0]),
            before_submit_total_tokens=prefixes[-2] if executed[-1:] == ['submit_design'] else None,
            submit_feedback_retained=any(m.get('name') == 'submit_design' for m in outcome[1][0]),
            submitted='submit_design' in executed, reply_tokens=max(map(len, responses)),
            complete_seconds=(datetime.fromisoformat(record['finished_at']) -
                              datetime.fromisoformat(record['started_at'])).total_seconds()))
    return results
