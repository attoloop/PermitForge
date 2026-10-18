"""WP2.11: local stubs only; never spend console credit or start model episodes."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from eval.common import ROOT, records
from eval.tests.test_contract import Session, api_response, episode


@pytest.mark.parametrize('budget', [1, 30, 100, 200])
def test_prompt_template_preserves_default_and_changes_budget(budget):
    from permit_openenv.prompts import INSTRUCTIONS, instructions
    assert hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() == '75fa2efb3393976c7c85209945f0ffbd97a18449cd75f2f00fdf54d2a8ce0f13'
    assert f'you have {budget} evaluations' in instructions(budget)
    assert instructions(30) == INSTRUCTIONS


@pytest.mark.parametrize('budget', [0, 201, True, 1.5])
def test_invalid_evaluation_budget(budget):
    from permit_openenv.prompts import instructions
    with pytest.raises(ValueError):
        instructions(budget)


def test_budget_flows_through_server_and_rescore():
    from eval.transport import Session as RealSession, load_eval_pack, recording_app, serve
    from eval.rescore import rescore_record
    from permit_openenv.tasks import TaskPack
    from eval.tests.test_transport import Physics
    task = load_eval_pack().tasks[0]
    with serve(recording_app('test-only', pack=TaskPack([task]), evaluator=Physics(),
                             verifier=Physics(), evaluate_budget=100)) as url:
        with RealSession(url, 'test-only', task.task_id) as session:
            assert session.meta['evaluate_budget'] == 100
            assert 'you have 100 evaluations' in session.meta['instructions']
            design = session.call('get_scenario', {})['draft_design']
            assert session.call('evaluate_design', {'design': {}})['evaluations_left'] == 100
            for count in range(1, 101):
                assert session.call('evaluate_design', {'design': design})['evaluations_left'] == 100-count
            assert 'error' in session.call('evaluate_design', {'design': design})
            assert session.record()['evaluations_used'] == 100
            session.call('submit_design', {'design': design})
            record = session.record()
            assert record['evaluate_budget'] == 100
    rescored = rescore_record(record, task, verify_fn=lambda *a: {'valid': True, 'feasible': True,
                                                               'cost': float(task.draft_cost)})
    assert rescored['final']['evaluations_left'] == 0


def test_cache_breakpoints_recorded_and_default_unchanged(tmp_path):
    from eval.anthropic_driver import run_anthropic, request_cost
    from eval.capture import EpisodeCapture
    for enabled in (False, True):
        calls = []
        def create(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return api_response('get_scenario' if len(calls) == 1 else 'submit_design')
        session = Session()
        session.meta['evaluate_budget'] = 100
        session.capture = EpisodeCapture(tmp_path/str(enabled))
        out = run_anthropic(session, 'anthropic:claude-fable-5-1', effort='medium', prompt_cache=enabled,
                           api=SimpleNamespace(messages=SimpleNamespace(create=create)))
        assert out['settings']['prompt_cache'] is enabled and out['settings']['evaluate_budget'] == 100
        for body in calls:
            assert len(body['tools']) == 4
            if enabled:
                assert body['system'][-1]['cache_control'] == {'type': 'ephemeral'}
                assert body['tools'][-1]['cache_control'] == {'type': 'ephemeral'}
                assert body['messages'][-1]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
                assert sum('cache_control' in b for m in body['messages'] if isinstance(m['content'], list)
                           for b in m['content']) == 1
            else:
                assert isinstance(body['system'], str) and 'cache_control' not in json.dumps(body)
        bodies = list((session.capture.root/'raw').glob('*-request.json'))
        assert len(bodies) == 2
    assert request_cost('claude-fable-5-1', {'input_tokens': 100, 'output_tokens': 10,
        'cache_read_input_tokens': 100, 'cache_creation_input_tokens': 100}) == pytest.approx(.002775)


def test_run_manifest_hash_resume_budget_and_cache(tmp_path):
    from eval.run_eval import provenance, run_panel
    tasks = [dict(task_id='a', station='X', split='eval')]
    base = {'evaluate_budget': 30, 'prompt_cache': False}
    hashes = {provenance(['codex:gpt-6-astra'], base | changes, 1, 0)['protocol_sha256']
              for changes in ({}, {'evaluate_budget': 100}, {'prompt_cache': True})}
    assert len(hashes) == 3
    run_panel(tmp_path, tasks, ['codex:gpt-6-astra'], lambda m, t: episode(m),
              settings=base | {'evaluate_budget': 100})
    manifest = json.loads((tmp_path/'run.json').read_text())
    assert manifest['settings']['evaluate_budget'] == manifest['sampling']['evaluate_budget'] == 100
    with pytest.raises(ValueError, match='resume'):
        run_panel(tmp_path, tasks, ['codex:gpt-6-astra'], lambda *a: pytest.fail('rerun'), settings=base)


def test_episode_launch_cap_two_and_remaining_tasks(tmp_path):
    from eval.run_eval import run_panel
    tasks = [dict(task_id=str(i), station='X', split='eval') for i in range(4)]
    called = []
    def driver(model, task, hard_budget):
        assert hard_budget is None  # No request-level cutoff for an admitted episode.
        called.append(task['task_id'])
        return episode(model, task['task_id']) | {'usage': {'usd': 2.0}}
    run_panel(tmp_path, tasks, ['anthropic:claude-fable-5-1'], driver, budget_usd=4, prior=2,
              settings={'evaluate_budget': 100, 'experiment': 'extended-budget'})
    assert called == ['0', '1']
    assert all(r.get('error_type') != 'budget' for _, r in records([tmp_path]))
    remaining = json.loads((tmp_path/'remaining-tasks.json').read_text())
    assert [t['task_id'] for t in remaining['tasks']] == ['2', '3']
    assert remaining['model'] == 'claudecode:claude-fable-5-1'


@pytest.mark.parametrize('leg', ['credit', 'subscription'])
def test_extended_dry_run_schedules_without_providers(tmp_path, monkeypatch, capsys, leg):
    from eval import run_eval
    monkeypatch.setattr(run_eval.preflight, 'drivers', lambda *a: pytest.fail('provider preflight'))
    args = ['run_eval', '--run', 'extended-budget' if leg == 'credit' else 'extended-budget-sub',
            '--models', *(['codex:gpt-6-astra', 'anthropic:claude-fable-5-1'] if leg == 'credit'
                         else ['claudecode:claude-fable-5-1']), '--subsets', str(ROOT/'results/subsets.json'),
            '--split', 'both', '--effort', 'medium', '--evaluate-budget', '100', '--max-turns', '150',
            '--wall-clock', '7200', '--repeats', '1', '--dry-run', '--output-root', str(tmp_path)]
    if leg == 'credit':
        args += ['--prompt-cache', '--budget-usd', '100']
    else:
        remaining = tmp_path/'remaining.json'
        remaining.write_text(json.dumps({'experiment': 'extended-budget', 'model': 'claudecode:claude-fable-5-1',
                                        'tasks': [json.loads((ROOT/'results/subsets.json').read_text())['eval'][0]]}))
        args += ['--remaining-tasks', str(remaining)]
    monkeypatch.setattr('sys.argv', args)
    run_eval.main()
    out = json.loads(capsys.readouterr().out)
    assert out['settings']['evaluate_budget'] == 100 and out['settings']['max_turns'] == 150
    assert out['settings']['wall_clock'] == 7200 and out['settings']['effort'] == 'medium'
    assert len(out['schedule']) == (40 if leg == 'credit' else 1)
    splits = [r['split'] for r in out['schedule']]
    assert splits == sorted(splits, key=lambda x: x != 'eval')
    assert not list(tmp_path.glob('*/run.json'))


def test_whole_episode_reservations_learn_measured_cost_and_survive_resume(tmp_path):
    from eval.run_eval import run_panel
    tasks = [dict(task_id=str(i), station='X', split='eval') for i in range(5)]
    called = []
    def driver(model, task):
        called.append(task['task_id'])
        return episode(model, task['task_id']) | {'usage': {'usd': 3.0}}
    settings = {'experiment': 'extended-budget', 'evaluate_budget': 100}
    run_panel(tmp_path, tasks, ['anthropic:claude-fable-5-1'], driver, budget_usd=6, prior=5, settings=settings)
    assert called == ['0', '1']  # First measured cost replaces the conservative prior.
    run_panel(tmp_path, tasks, ['anthropic:claude-fable-5-1'], driver, budget_usd=6, prior=5, settings=settings)
    assert called == ['0', '1']
    assert len(json.loads((tmp_path/'remaining-tasks.json').read_text())['tasks']) == 3


def test_subscription_requires_completed_credit_leg(tmp_path):
    from eval.extended import consume_remaining
    tasks = [dict(task_id='a', station='X', split='eval')]
    path = tmp_path/'remaining.json'
    path.write_text(json.dumps({'experiment': 'extended-budget', 'model': 'claudecode:claude-fable-5-1', 'tasks': tasks}))
    with pytest.raises(ValueError, match='source_run'):
        consume_remaining(path, tasks)
    assert consume_remaining(path, tasks, preview=True) == tasks
    (tmp_path/'run.json').write_text(json.dumps({'status': 'running'}))
    value = json.loads(path.read_text()) | {'source_run': str(tmp_path)}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='finish'):
        consume_remaining(path, tasks)


def test_main_card_refuses_extended_experiment(tmp_path):
    from eval.card import build_card
    (tmp_path/'run.json').write_text(json.dumps({'status': 'complete', 'settings': {
        'experiment': 'extended-budget', 'evaluate_budget': 100}}))
    with pytest.raises(ValueError, match='extended|30-evaluation'):
        build_card(tmp_path, ROOT/'results/subsets.json')
