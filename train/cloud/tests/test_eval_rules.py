"""Cloud GRPO trains under the evaluation's rules and opening (operator, 2026-10-18)."""
from pathlib import Path

import pytest

from train.cloud.config import CloudConfig
from train.cloud.training import training_options

ROOT = Path(__file__).resolve().parents[3]


def test_training_turns_and_context_follow_the_evaluation():
    config = CloudConfig()
    assert config.train_turns == config.eval_turns == 40
    options = training_options(config, 'grpo', Path('unused'), smoke=True)
    # TRL allows at most this many executed tool rounds, the same maximum as the evaluation's 40 turns.
    assert options['max_tool_calling_iterations'] == 40
    assert config.max_length == config.max_prompt_length + options['max_completion_length']
    with pytest.raises(ValueError, match='evaluation'):
        CloudConfig(train_turns=8)


def test_training_opening_equals_the_shared_evaluation_opening():
    from permit_openenv.prompts import INSTRUCTIONS, opening
    from permit_openenv.tasks import load_pack
    from train.cloud.training import grpo_rows
    from train.permit_env import PermitTrainingEnv, opening_messages
    config = CloudConfig()
    pack = load_pack(ROOT/'envs/permit/tasks')
    row = grpo_rows(config, [pack.get('074897d0f159-s47')])[0]
    observation = PermitTrainingEnv(pack=pack, opening_turns=config.train_turns).reset(**row)
    # TRL appends reset()'s observation to the last prompt message.
    prompt = row['prompt'][:-1] + [{**row['prompt'][-1], 'content': row['prompt'][-1]['content']+observation}]
    assert prompt == opening_messages(config.eval_turns)
    assert prompt == [{'role': 'system', 'content': INSTRUCTIONS}, {'role': 'user', 'content': opening(40)}]
    # The cloud evaluation builds its messages with the same function.
    assert 'messages = opening_messages(config.eval_turns)' in (ROOT/'train/cloud/evaluation.py').read_text()


def test_local_trainer_keeps_its_compact_opening():
    from permit_openenv.tasks import load_pack
    from train.permit_env import PermitTrainingEnv, scenario_text
    pack = load_pack(ROOT/'envs/permit/tasks')
    env = PermitTrainingEnv(pack=pack)
    assert env.reset(task_id='074897d0f159-s47') == scenario_text(pack.get('074897d0f159-s47'))


def test_training_tool_names_and_descriptions_equal_the_evaluation_server():
    import asyncio
    pytest.importorskip('fastmcp', reason='the evaluation MCP server is part of the cloud lock, not train/.venv')
    from transformers.utils import get_json_schema
    from permit_openenv.environment import PermitForgeEnvironment
    from permit_openenv.tasks import load_pack
    from train.permit_env import PermitTrainingEnv, TOOL_NAMES
    pack = load_pack(ROOT/'envs/permit/tasks')
    specs = asyncio.run(PermitForgeEnvironment(pack=pack)._require_mcp_server().list_tools())
    served = {tool.name: tool.description for tool in specs}
    env = PermitTrainingEnv(pack=pack)
    trained = {name: get_json_schema(getattr(env, name))['function']['description'] for name in TOOL_NAMES}
    assert trained == served


def test_rendered_opening_fits_the_prompt_limit():
    # Rendered as TRL does (bound environment methods as tools); measured 983 tokens.
    from permit_openenv.tasks import load_pack
    from train.cloud.formatting import load_tokenizer
    from train.cloud.training import opening_ids
    config = CloudConfig()
    ids = opening_ids(config, load_tokenizer(local_files_only=True), load_pack(ROOT/'envs/permit/tasks'))
    assert len(ids) <= config.max_prompt_length


def test_plan_caps_fit_the_measured_evaluation_timings():
    import math
    import statistics
    from datetime import datetime
    import json
    from train.cloud.config import EVAL_EPISODE_SECONDS, GRPO_STEP_CAP_SECONDS, GRPO_STEP_SECONDS, stage_plan
    plan = {row['stage']: row for row in stage_plan(CloudConfig(recipe='grpo', grpo_steps=45))}
    episodes = [json.loads(p.read_text()) for p in
                (ROOT/'train/cloud/data/smoke-result-grade/smoke/base-eval').rglob('episode.json')]
    assert len(episodes) == 3
    seconds = [(datetime.fromisoformat(r['finished_at']) -
                datetime.fromisoformat(r['started_at'])).total_seconds() for r in episodes]
    assert EVAL_EPISODE_SECONDS == math.ceil(statistics.mean(seconds))
    for stage in ('base-eval', 'eval'):
        assert plan[stage]['estimated_seconds'] == 30*EVAL_EPISODE_SECONDS
        assert plan[stage]['wall_seconds'] >= 30*max(seconds)
    assert GRPO_STEP_CAP_SECONDS >= 2*GRPO_STEP_SECONDS
    assert plan['grpo']['wall_seconds'] >= 45*GRPO_STEP_CAP_SECONDS
