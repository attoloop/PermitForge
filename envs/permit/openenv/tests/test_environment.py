import copy
import json

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from permit_core.reward import reward
from permit_openenv.protocol import tool_result
from permit_openenv.tasks import TaskPack
from permit_openenv.environment import PermitForgeEnvironment
from conftest import StubEvaluator


def call(env, tool, **args):
    obs = env.step(CallToolAction(tool_name=tool, arguments=args))
    return tool_result(obs), obs


def test_tool_schemas_and_scenario(env):
    obs = env.step(ListToolsAction())
    schemas = {t.name: t.input_schema for t in obs.tools}
    assert set(schemas) == {'get_scenario', 'check_design', 'evaluate_design', 'submit_design'}
    assert not schemas['get_scenario'].get('required')
    for name in ('check_design', 'evaluate_design', 'submit_design'):
        schema = schemas[name]
        assert schema['required'] == ['design']
        design = schema['properties']['design']
        assert design['type'] == 'object'
        row = design['additionalProperties']
        if '$ref' in row:
            row = schema['$defs'][row['$ref'].split('/')[-1]]
        assert set(row['required']) == {'control', 'height_m'}
        assert row['properties']['height_m']['type'] == 'number'
    result, obs = call(env, 'get_scenario')
    assert set(result) >= {'task_id', 'split', 'station', 'years', 'sources', 'controls',
                           'standards', 'stack_usd_per_m', 'draft_design', 'draft_cost', 'evaluations_left'}
    assert set(result['sources'][0]) >= {'id', 'type', 'x_m', 'y_m', 'uncontrolled_g_s',
                                       'height_min_m', 'gep_m', 'height_menu_m'}
    assert set(result['standards'][0]) >= {'id', 'pollutant', 'form', 'limit', 'background'}
    assert obs.reward == 0
    assert env.get_metadata().name == 'PermitForge v2'


@pytest.mark.parametrize('bad', [None, [], {}, {'S1': {}}, 'not a design'])
def test_invalid_is_free(env, bad):
    result, obs = call(env, 'evaluate_design', design=bad)
    assert result['valid'] is False
    assert result['errors']
    assert result['evaluations_left'] == 30
    assert env.state.evaluations_used == 0
    assert not env.evaluator.calls
    assert obs.reward == 0


@pytest.mark.parametrize('field,value', [('height_m', True), ('height_m', '82.27'),
                                        ('height_m', float('inf')), ('height_m', 40.123),
                                        ('control', 'S2_maximum'), ('extra', 1)])
def test_hostile_rows_are_free(env, field, value):
    design = call(env, 'get_scenario')[0]['draft_design']
    design['S1'][field] = value
    result, _ = call(env, 'evaluate_design', design=design)
    assert result['valid'] is False
    assert env.state.evaluations_used == 0


def test_budget_exhaustion_and_submit_once(env):
    design = call(env, 'get_scenario')[0]['draft_design']
    checked, obs = call(env, 'check_design', design=design)
    assert checked['valid'] and checked['cost'] > 0 and obs.reward == 0
    for left in range(29, -1, -1):
        result, obs = call(env, 'evaluate_design', design=design)
        assert result['evaluations_left'] == left
        assert obs.reward == 0 and not obs.done
    result, _ = call(env, 'evaluate_design', design=design)
    assert 'error' in result and result['evaluations_left'] == 0
    assert len(env.evaluator.calls) == 30
    final, obs = call(env, 'submit_design', design=design)
    assert obs.done and obs.reward == 0.0
    assert final['prescore_usd_saved'] == 0 and final['reward'] == 0.0
    assert 'anchors' not in final
    again, obs = call(env, 'submit_design', design=design)
    assert 'error' in again and obs.reward == 0
    assert len(env.verifier.calls) == 1
    assert env.state.done


def test_reward_and_rubric_match_core(env):
    task = copy.copy(env.pack.get(env.state.task_id))
    task.record = dict(task.record, anchors={'worst': 0, 'best': 1e6, 'saturation': 4e6, 'source': 'search'})
    instance = PermitForgeEnvironment(TaskPack([task]), evaluator=StubEvaluator(), verifier=StubEvaluator())
    try:
        instance.reset(task_id=task.task_id)
        design = call(instance, 'get_scenario')[0]['draft_design']
        design['S1']['control'] = 'S1_moderate'
        final, obs = call(instance, 'submit_design', design=design)
        assert final['reward'] == reward(final['prescore_usd_saved'], task.draft_cost)
        assert obs.reward == final['reward'] > 0
        assert instance.rubric.feasible.last_score == 1
        assert instance.rubric.cost_quality.last_score == final['reward']
        assert len(instance.verifier.calls) == 1 and not instance.evaluator.calls
    finally:
        instance.close()


def test_invalid_submit_ends_and_no_submit_scores_zero(env):
    final, obs = call(env, 'submit_design', design={})
    assert obs.done and obs.reward == 0
    assert final['valid'] is False and final['prescore_usd_saved'] == -1
    assert not env.verifier.calls
    env.reset(task_id='003bd328a392-s47')
    assert env.episode.record()['final'] is None
    assert env.episode.record()['ended'] == 'no_submit'


def test_episode_matches_contract_and_reset(env):
    scenario = call(env, 'get_scenario')[0]
    call(env, 'evaluate_design', design=scenario['draft_design'])
    call(env, 'submit_design', design=scenario['draft_design'])
    record = env.episode.record(model='test:model')
    assert set(record) >= {'schema_version', 'task_id', 'split', 'station', 'model', 'prompt_version',
                           'started_at', 'finished_at', 'scenario', 'geometry', 'steps',
                           'evaluations_used', 'ended', 'final', 'usage'}
    assert record['schema_version'] == 1 and record['ended'] == 'submitted'
    assert record['evaluations_used'] == 1
    assert record['usage'] == {'input_tokens': 0, 'output_tokens': 0, 'usd': None}
    for i, step in enumerate(record['steps']):
        assert set(step) >= {'i', 'tool', 'args', 'result', 't_s'}
        assert step['i'] == i
    geometry = record['geometry']
    assert set(geometry) == {'task_id', 'station', 'buildings', 'receptors', 'stacks', 'wind_rose'}
    assert len(geometry['wind_rose']['counts']) == 16
    assert all(len(row) == 6 for row in geometry['wind_rose']['counts'])
    assert geometry['wind_rose']['speed_bins_m_s'] == [0.5, 2, 4, 6, 8, 10]
    assert geometry['buildings'][0]['corners']
    json.dumps(record, allow_nan=False)
    env.reset(split='train', index=0)
    assert env.state.evaluations_used == 0 and not env.state.done
    assert env.episode.record()['steps'] == []
    assert env.num_tasks('train') == len(env.list_tasks('train'))
    assert env.get_task('train', 0)['task_id'] == env.state.task_id
