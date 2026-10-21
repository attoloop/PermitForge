"""public policies: uv run pytest -q tests/test_baselines.py."""

from copy import deepcopy
import json

import pytest

from permit_core.baselines import POLICIES, scenario_view, v1_coordinate_descent
from permit_core.evaluate import evaluate
from permit_core.reward import reward
from wp14_fixtures import synthetic
from conftest import requires_native_binaries, requires_task_data


@pytest.mark.parametrize('name', ['draft', 'all_none_at_gep', 'uniform_moderate_at_gep',
                                  'cheapest_first_greedy', 'random_30', 'v1_coordinate_descent'])
def test_policies_public_contract(name):
    task, pack = synthetic()
    view = scenario_view(task)
    original = deepcopy(view)
    calls = []
    def tool(design):
        calls.append(deepcopy(design))
        result = evaluate(task, design, pack)
        assert result['valid'], result
        return {**result, 'evaluations_left': 30-len(calls)}
    result = POLICIES[name](view, tool)
    assert view == original
    assert result['evaluations_used'] == len(calls)
    assert len(calls) <= (25 if name == 'v1_coordinate_descent' else 30)
    assert result['usd_saved'] >= 0 if result['feasible'] else result['usd_saved'] == -1
    if result['feasible']:
        assert evaluate(task, result['design'], pack)['feasible']
        assert result['usd_saved'] == view['draft_cost'] - result['cost']
    json.dumps(result, allow_nan=False)
    expected = {'draft': 28, 'all_none_at_gep': 4, 'uniform_moderate_at_gep': 16,
                'cheapest_first_greedy': 8, 'v1_coordinate_descent': 12}
    if name in expected:
        assert result['cost'] == expected[name]
    if name == 'all_none_at_gep':
        assert not result['feasible']


def test_seeded_random_budget_and_no_feasible_sentinel():
    task, pack = synthetic()
    view = scenario_view(task)
    def run(seed):
        calls = []
        def tool(design):
            calls.append(deepcopy(design))
            return evaluate(task, design, pack)
        return POLICIES['random_30'](view, tool, seed=seed), calls
    first, calls = run(42)
    assert run(42) == (first, calls)
    assert run(43)[1] != calls
    assert len(calls) == 30
    failed = POLICIES['random_30'](view, lambda d: {'valid': True, 'feasible': False, 'cost': 1})
    assert failed == {'design': None, 'feasible': False, 'cost': None,
                      'usd_saved': -1, 'evaluations_used': 30}
    view['evaluations_left'] = 3
    assert POLICIES['random_30'](view, lambda d: evaluate(task, d, pack))['evaluations_used'] == 3


def test_v1_initial_designs_and_move_order():
    task, pack = synthetic()
    view = scenario_view(task)
    calls = []
    def tool(design):
        calls.append(deepcopy(design))
        return evaluate(task, design, pack)
    result = v1_coordinate_descent(view, tool)
    assert calls[0] == view['draft_design']
    for i, (level, height) in enumerate((('none', 20), ('none', 10), ('moderate', 20),
                                        ('moderate', 10), ('maximum', 20), ('maximum', 10)), 1):
        assert calls[i] == {s['id']: {'control': s['id']+'_'+level, 'height_m': height}
                            for s in view['sources']}
    # Uniform moderate at minimum height is the cheapest feasible initial point;
    # its height cannot be lowered, so S1's control is tried before S2's.
    assert calls[7]['S1'] == {'control': 'S1_none', 'height_m': 10}
    assert calls[7]['S2'] == {'control': 'S2_moderate', 'height_m': 10}
    assert calls[-1] == result['design']


def test_v1_final_disagreement_is_not_a_result():
    task, pack = synthetic()
    view = scenario_view(task)
    counts = {}
    def tool(design):
        key = json.dumps(design, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        result = evaluate(task, design, pack)
        if counts[key] > 1 and design != view['draft_design']:
            result['feasible'] = False
        return result
    with pytest.raises(ValueError, match='fresh evaluation'):
        v1_coordinate_descent(view, tool)


def test_gate_failures_and_saved_fraction_monotonicity():
    scores = [reward(x, 100) for x in (-1, 0, 1, 5, 10, 20, 40, 100)]
    assert scores == sorted(scores)
    assert scores[:2] == [0, 0]


def test_none_alias_when_public_menu_omits_explicit_none_rows():
    task, pack = synthetic()
    view = scenario_view(task)
    view['controls'] = [c for c in view['controls'] if not c['id'].endswith('_none')]
    result = POLICIES['all_none_at_gep'](view, lambda d: evaluate(task, d, pack))
    assert all(row['control'] == 'none' for row in result['design'].values())
    assert result['cost'] == 4 and result['usd_saved'] == -1


@requires_task_data
@requires_native_binaries
@pytest.mark.native
def test_real_draft_with_verify_evaluator(tmp_path, evidence_dir):
    from permit_core.model import load_v2_instance
    from permit_core.baselines import verify_evaluation
    from conftest import REPO
    task = load_v2_instance(REPO / 'envs/permit/tasks', '074897d0f159-s47')
    result = POLICIES['draft'](scenario_view(task), lambda d: verify_evaluation(task, d, tmp_path))
    assert result['feasible'] and result['usd_saved'] == 0
    assert result['evaluations_used'] == 1
    (evidence_dir / 'wp14-native-smoke.json').write_text(json.dumps(result, indent=2)+'\n')
