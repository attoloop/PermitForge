import inspect
import json
import math

import pytest

from permit_core.reward import reward
from permit_openenv.prompts import INSTRUCTIONS
from train.permit_env import PermitTrainingEnv, training_instructions


def test_class_contract(env):
    methods = {name for name, value in inspect.getmembers(PermitTrainingEnv, inspect.isfunction)
               if not name.startswith('_')}
    assert methods == {'reset', 'get_reward', 'get_scenario', 'check_design', 'evaluate_design', 'submit_design'}
    for name in methods - {'reset', 'get_reward'}:
        assert inspect.getdoc(getattr(env, name))
        assert inspect.signature(getattr(env, name)).return_annotation
    opening = env.reset(split='train', index=0)
    assert 'synthetic-train' in opening
    assert 'anchors' not in opening and 'calibration' not in opening
    assert training_instructions() == INSTRUCTIONS
    assert '8 tool turns' in training_instructions(8)
    assert '30 evaluations' not in training_instructions(8)


def test_real_fast_reward_and_terminal_state(env, design):
    env.reset(task_id='synthetic-train')
    assert env.get_reward() == 0
    checked = json.loads(env.check_design(design))
    evaluated = json.loads(env.evaluate_design(design))
    submitted = json.loads(env.submit_design(design))
    assert checked['cost'] == evaluated['cost'] == submitted['cost'] == 6
    assert evaluated['standards'] == submitted['standards']
    assert submitted['prescore_usd_saved'] == 16
    assert env.get_reward() == reward(16, 22) == 16/22
    assert 'anchors' not in submitted
    assert submitted['pct_saved'] == pytest.approx(100*16/22)
    for name in ('get_scenario', 'check_design', 'evaluate_design', 'submit_design'):
        args = () if name == 'get_scenario' else (design,)
        assert 'error' in json.loads(getattr(env, name)(*args))
    env.reset(index=0)
    assert env.get_reward() == 0


def test_invalid_is_free_and_budget_exhaustion(env, design):
    env.reset(index=0)
    assert not json.loads(env.evaluate_design({}))['valid']
    assert json.loads(env.get_scenario())['evaluations_left'] == 30
    for _ in range(30):
        env.evaluate_design(design)
    assert json.loads(env.evaluate_design(design))['evaluations_left'] == 0
    assert json.loads(env.submit_design(design))['reward'] == 16/22


@pytest.mark.parametrize('choice', [{}, {'S1': {'control': 'none', 'height_m': 10}}])
def test_invalid_and_infeasible_score_zero(env, choice):
    env.reset(index=0)
    result = json.loads(env.submit_design(choice))
    assert not result['feasible'] and result['reward'] == 0
    assert env.get_reward() == 0


def test_grading_failure_is_nan_and_sanitized(task_pack, design):
    def broken(task):
        raise OSError('/private/cache/path')
    env = PermitTrainingEnv(pack=task_pack[0], fields_loader=broken)
    env.reset(index=0)
    result = env.submit_design(design)
    assert '/private' not in result
    assert math.isnan(env.get_reward())


def test_train_split_only(env):
    with pytest.raises(ValueError, match='train'):
        env.reset(split='eval', index=0)
    with pytest.raises(ValueError, match='train'):
        env.reset(task_id='synthetic-eval')


@pytest.mark.parametrize('smoke', [False, True])
def test_training_needs_no_anchors(task_pack, design, smoke):
    pack, fields = task_pack
    task = pack.at('train', 0)
    assert 'anchors' not in task.record
    env = PermitTrainingEnv(pack=pack, fields_loader=lambda _: fields, smoke=smoke)
    env.reset(index=0)
    env.submit_design(design)
    assert env.get_reward() == 16/22
    assert 'anchors' not in env._record() and 'provisional' not in env._record()
