"""WP2.0b public disclosure, bounded inputs, and terminal lifecycle regressions."""
import copy
import json

import pytest

from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
from test_environment import call


def test_disclosure_boundary_reset_tools_state_and_errors(env):
    task = env.pack.get(env.state.task_id)
    original = task.record
    task.record = dict(original, anchors={'worst': 0, 'best': 1234567, 'saturation': 4938268, 'source': 'panel'},
                       best_known_search_cost=987654, baselines={'private': 'hidden'})
    try:
        reset = env.reset(task_id=task.task_id)
        scenario, _ = call(env, 'get_scenario')
        assert reset.metadata['scenario'] == scenario
        assert 'geometry' not in reset.metadata
        surfaces = [reset.model_dump(), scenario, env.state.model_dump()]
        for name in ('check_design', 'evaluate_design'):
            surfaces.append(call(env, name, design=scenario['draft_design'])[0])
            surfaces.append(call(env, name, design={})[0])
        class Broken:
            def evaluate(self, *args):
                raise RuntimeError('/private/native/path anchor=1234567 baselines hidden')
        env.evaluator = Broken()
        surfaces.append(call(env, 'evaluate_design', design=scenario['draft_design'])[0])
        surfaces.append(env.state.model_dump())
        for surface in surfaces:
            text = json.dumps(surface)
            for forbidden in ('geometry', 'receptors', 'receptor_values', 'receptor_margins', 'buildings',
                              'wind_rose', 'anchors', 'best_known_search_cost', 'baselines', '/private/', '1234567'):
                assert forbidden not in text
        env.verifier = Broken()
        final, _ = call(env, 'submit_design', design=scenario['draft_design'])
        assert 'anchors' not in final
        assert 'best_known_search_cost' not in final
        assert '/private/' not in json.dumps([final, env.state.model_dump()])
        assert env.episode.record()['geometry']['receptors']
        assert '/private/native/path' in json.dumps(env.episode.record()['diagnostics'])
        for name in ('get_scenario', 'check_design', 'evaluate_design', 'submit_design'):
            assert 'episode is over' in call(env, name, **({} if name == 'get_scenario' else {'design': {}}))[0]['error']
    finally:
        task.record = original


@pytest.mark.parametrize('case', ['sources', 'key', 'bytes'])
def test_design_size_limits_are_free_and_not_retained(env, case):
    from permit_openenv.protocol import MAX_DESIGN_BYTES, MAX_KEY_LENGTH, MAX_SOURCES
    design = copy.deepcopy(env.episode.task.view()['draft_design'])
    if case == 'sources':
        design = {f'S{i}': {'control': 'none', 'height_m': 65} for i in range(MAX_SOURCES + 1)}
    elif case == 'key':
        design['k' * (MAX_KEY_LENGTH + 1)] = design.pop('S1')
    else:
        design['S1']['control'] = 'x' * (MAX_DESIGN_BYTES + 1)
    result, _ = call(env, 'evaluate_design', design=design)
    assert result['errors'] == ['E_SIZE: design exceeds input limits']
    assert env.state.evaluations_used == 0
    assert len(json.dumps(env.episode.steps[-1]['args'])) < 100


def test_retained_history_and_diagnostics_are_bounded(env):
    from permit_openenv.episode import MAX_STEPS, MAX_DIAGNOSTICS
    for _ in range(MAX_STEPS + 5):
        env._dispatch('check_design', {'design': {}})
    assert len(env.episode.steps) == MAX_STEPS
    assert env.episode.record()['steps_dropped'] == 5
    assert len(env.episode.record()['diagnostics']) <= MAX_DIAGNOSTICS


def test_close_finalizes_once_without_submission(env):
    ep = env.episode
    assert ep.record()['finished_at'] is None
    env.close()
    stamp = ep.finished_at
    assert stamp and ep.done and ep.ended == 'no_submit'
    assert ep.record()['reward'] == 0 and ep.final is None
    env.close()
    assert ep.finished_at == stamp and env.state.done and env.state.reward == 0


def test_prompt_actionable_semantics():
    assert PROMPT_VERSION == 2
    for phrase in ('µg/m³', 'USD/year', 'limit − background − value', 'smallest margin',
                   'draft is feasible', 'exhausted', 'invalid submission'):
        assert phrase in INSTRUCTIONS


def test_transport_request_limit_precedes_json_parsing(pack, monkeypatch):
    from fastapi.testclient import TestClient
    from permit_openenv.server import create_server
    monkeypatch.setenv('ENABLE_WEB_INTERFACE', 'false')
    with TestClient(create_server(pack=pack)) as client:
        response = client.post('/step', content=b'x' * (65536 + 1))
    assert response.status_code == 413
    assert response.json() == {'error': 'request exceeds input limits'}


def test_unknown_task_error_is_stable(env):
    with pytest.raises(ValueError, match='^unknown task selection$'):
        env.reset(task_id='/private/nonexistent/task')


def test_framework_tool_error_is_public_and_bounded(env):
    result, observation = call(env, 'check_design', wrong='/private/tool/argument')
    assert '/private/' not in json.dumps(observation.model_dump())
    assert len(json.dumps(result)) < 200
