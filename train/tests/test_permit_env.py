import inspect
import json
import math

import pytest

from permit_core.reward import reward
from permit_openenv.prompts import INSTRUCTIONS
from train.permit_env import PermitTrainingEnv, resolve_anchors, training_instructions


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
    assert env.get_reward() == reward(16, submitted['anchors']) == .4
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
    assert json.loads(env.submit_design(design))['reward'] == .4


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


def test_provisional_only_for_smoke(task_pack, tmp_path):
    task = task_pack[0].at('train', 0)
    task.record.pop('anchors')
    diagnostics = {'id': task.task_id, 'split': 'train', 'baselines': {
        'good': {'feasible': True, 'usd_saved': 12},
        'failed': {'feasible': False, 'usd_saved': 1000}},
        'search': {'feasible': True, 'usd_saved': 16}}
    (tmp_path / f'{task.task_id}.json').write_text(json.dumps(diagnostics))
    with pytest.raises(ValueError, match='anchor'):
        resolve_anchors(task, smoke=False, diagnostics_dir=tmp_path)
    anchors = resolve_anchors(task, smoke=True, diagnostics_dir=tmp_path)
    assert anchors['best'] == 16 and anchors['provisional'] is True
    assert anchors['source'] == 'provisional_baseline_search_smoke'
    diagnostics['id'] = 'another-task'
    (tmp_path / f'{task.task_id}.json').write_text(json.dumps(diagnostics))
    with pytest.raises(ValueError, match='identity'):
        resolve_anchors(task, smoke=True, diagnostics_dir=tmp_path)


def test_search_anchor_rejected_outside_smoke(task_pack):
    task = task_pack[0].at('train', 0)
    task.record['anchors']['source'] = 'search'
    with pytest.raises(ValueError, match='calibration'):
        resolve_anchors(task)


def test_reward_increases_beyond_anchor(env):
    env.reset(index=0)
    result = json.loads(env.submit_design({'S1': {'control': 'none', 'height_m': 20}}))
    assert result['reward'] > 0
    assert reward(32, {'worst': 0, 'best': 16}) > .4


def test_unlabelled_anchor_is_not_calibration(task_pack):
    task = task_pack[0].at('train', 0)
    task.record['anchors'].pop('source')
    with pytest.raises(ValueError, match='calibration'):
        resolve_anchors(task)
