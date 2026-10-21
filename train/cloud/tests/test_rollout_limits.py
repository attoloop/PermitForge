"""WP2.16: full recorded trajectories and stock TRL termination diagnostics."""
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from train.cloud.config import CloudConfig
from train.cloud.tests.replay import replay_episodes
from train.cloud.tests.checkout import requires_task_data  # noqa: E402


def test_complete_base_episodes_fit_with_headroom_through_submission():
    config = CloudConfig()
    measured = replay_episodes()
    bounded = replay_episodes(config.max_completion_length)
    for row, actual in zip(measured, bounded, strict=True):
        assert row['submitted'] and row['submit_feedback_retained']
        assert config.max_completion_length >= 1.2*row['completion_tokens']
        assert config.max_length >= config.max_prompt_length + row['completion_tokens']
        assert actual == row, 'The configured TRL loop discarded a saved response or tool result'
        assert row['reply_tokens'] <= config.eval_max_tokens


@requires_task_data
@pytest.mark.parametrize('mode', ['stopped_tools', 'budget_feedback', 'budget_reply', 'turn_limit',
                                 'submitted', 'submitted_budget', 'multiple_calls', 'unknown_tool'])
def test_installed_loop_reports_termination_and_keeps_budget_cut_targets(mode):
    from trl import GRPOTrainer
    from trl.trainer import grpo_trainer as module
    from train.cloud.training import rollout_diagnostics
    from train.permit_env import PermitTrainingEnv
    from permit_openenv.tasks import load_pack
    from train.cloud.tests.replay import ROOT
    records = []
    env = PermitTrainingEnv(pack=load_pack(ROOT/'envs/permit/tasks'), trace_sink=records.append)
    env.reset(task_id='074897d0f159-s47')
    submitted = mode.startswith('submitted')
    name = 'submit_design' if submitted else ('unknown' if mode == 'unknown_tool' else 'get_scenario')
    call = {'type':'function', 'function':{'name':name, 'arguments':{'design':{}} if submitted else {}}}
    initial = [3]*7+[2] if mode in ('budget_feedback','submitted_budget') else [3,2]
    calls = [] if mode == 'stopped_tools' else [call]*(2 if mode == 'multiple_calls' else 1)
    budget = 10 if 'budget' in mode else 10000
    loop = SimpleNamespace(max_tool_calling_iterations=1 if mode == 'turn_limit' else 40,
        max_completion_length=budget, use_vllm=False, tools=True, _is_vlm=False,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144)),
        _tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0),
        _sync_tool_dicts=[{'get_scenario':env.get_scenario, 'submit_design':env.submit_design}],
        _async_tool_dicts=[{}], _get_tool_suffix_ids=lambda messages:[4]*4,
        _generate_single_turn=lambda *a, **kw: ([[9]*10 if mode == 'budget_reply' else [9,2]],None))
    parsed = {'role':'assistant','content':''}
    if mode == 'turn_limit':
        parsed['tool_calls'] = [call]
    with patch.object(module,'parse_response',return_value=parsed):
        outcome = GRPOTrainer._tool_call_loop(loop, [[{'role':'user','content':'x'}]], [[1]],
            [initial], [[{'role':'assistant','tool_calls':calls}]], None, None, {})
    hook = partial(rollout_diagnostics, tokenizer=loop._tokenizer,
        config=SimpleNamespace(max_completion_length=budget, train_turns=loop.max_tool_calling_iterations))
    assert hook(completions=outcome[1],completion_ids=outcome[2],environments=[env]) == [None]
    assert env.get_reward() == 0.0
    expected = ('submitted' if submitted else 'budget' if 'budget' in mode else
                'turn_limit' if mode == 'turn_limit' else 'stopped_tools')
    assert records[0]['ended'] == expected
    assert records[0]['completion_tokens'] == len(outcome[2][0])
    assert records[0]['budget_cut'] == ('budget' in mode)
    assert records[0]['submitted'] == submitted
    assert sum(outcome[0][0]) > 0
    env.reset(task_id='074897d0f159-s47')
    assert 'completion_tokens' not in env._record(), 'Pooled environments must clear diagnostics'


def test_budget_cuts_are_counted_per_optimizer_step_and_reported(tmp_path):
    import json
    from train.cloud.training import rollout_metrics
    from train.cloud.smoke import smoke_summary, write_report
    traces = [dict(optimizer_step=0,ended='budget',budget_cut=True,completion_tokens=8),
              dict(optimizer_step=0,ended='submitted',budget_cut=False,completion_tokens=20),
              dict(optimizer_step=1,ended='stopped_tools',budget_cut=False,completion_tokens=3)]
    metrics = rollout_metrics(traces)
    assert metrics['budget_cuts_per_step'] == [
        {'optimizer_step':1,'rollouts':2,'budget_cuts':1},
        {'optimizer_step':2,'rollouts':1,'budget_cuts':0}]
    assert metrics['ending_counts'] == {'budget':1,'submitted':1,'stopped_tools':1}
    summary = smoke_summary([],metrics,{},elapsed_seconds=1,hourly_usd=5)
    (tmp_path/'metrics.json').write_text(json.dumps(summary))
    write_report(tmp_path)
    text = (tmp_path/'REPORT.md').read_text()
    assert 'Budget cuts per optimizer step' in text
    assert '| 1 | 2 | 1 |' in text and '| 2 | 1 | 0 |' in text
