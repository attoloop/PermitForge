"""WP2.1 requirements; all providers and native work are fake unless stated."""
import copy
import json
from types import SimpleNamespace

import pytest

from eval.common import model_slug, write_json
from eval.anthropic_driver import run_anthropic, prices, request_cost
from eval.run_eval import Budget, run_panel
from eval.report import summarize, cluster_ci, paired
from eval.best_observed import best_observed
from eval.common import records
from eval.rescore import rescore_record


class Session:
    def __init__(self, task='t1'):
        self.meta = {'task_id': task, 'split': 'train', 'scenario': {'station': 'X', 'draft_cost': 100},
                     'geometry': {'receptors': [[0, 0]]}}
        self.steps = []
        self.final = None
        self.submit_attempted = False
    def tools(self):
        return [{'name': n, 'description': n, 'input_schema': {'type': 'object'}} for n in
                ('get_scenario', 'check_design', 'evaluate_design', 'submit_design')]
    def call(self, name, args, text=''):
        result = {'valid': True, 'feasible': True, 'evaluations_left': 29}
        if name == 'submit_design':
            self.submit_attempted = True
            result.update(reward=0, prescore_usd_saved=20, pct_saved=20, cost=80)
            self.final = result
        self.steps.append({'i': len(self.steps), 'tool': name, 'args': args, 'result': result,
                           'receptor_values': {'x': [1]}, 'receptor_margins': {'x': [2]},
                           't_s': 0, 'assistant_text': text[:2000]})
        return result
    def record(self):
        return {'schema_version': 1, 'task_id': self.meta['task_id'], 'split': 'train', 'station': 'X',
                'scenario': self.meta['scenario'], 'geometry': self.meta['geometry'],
                'steps': copy.deepcopy(self.steps), 'final': self.final, 'evaluations_used': 1}


def api_response(name='submit_design', **usage):
    return SimpleNamespace(content=[SimpleNamespace(type='text', text='x'*2100),
        SimpleNamespace(type='tool_use', name=name, input={'design': {}}, id='call')],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, **usage))


def test_anthropic_records_contract_and_no_temperature():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return api_response()
    rec = run_anthropic(Session(), 'anthropic:claude-sonnet-5-5',
                        api=SimpleNamespace(messages=SimpleNamespace(create=create)))
    from permit_openenv.prompts import INSTRUCTIONS
    assert calls[0]['system'] == INSTRUCTIONS and 'temperature' not in calls[0]
    assert len(calls[0]['tools']) == 4
    assert rec['ended'] == 'submitted' and rec['turns'] == 1
    assert rec['usage']['usd'] == pytest.approx(.0004)
    assert rec['steps'][0]['receptor_values'] == {'x': [1]}
    assert len(rec['steps'][0]['assistant_text']) == 2000


def test_price_tiers_cache_and_sources():
    table = prices()
    assert len(table) == 3
    for value in table.values():
        assert value['source'].startswith('https://') and value['verified_at'] == '2026-10-17'
    assert request_cost('claude-haiku-5-5', {'input_tokens': 100000, 'output_tokens': 100}) == pytest.approx(.01005)
    assert request_cost('claude-haiku-5-5', {'input_tokens': 100000, 'cache_read_input_tokens': 1,
                                         'output_tokens': 100}) == pytest.approx(.05025005)
    assert request_cost('claude-opus-5-5', {'input_tokens': 0, 'cache_creation_input_tokens': 100,
        'cache_creation': {'ephemeral_5m_input_tokens': 60, 'ephemeral_1h_input_tokens': 40},
        'cache_read_input_tokens': 100, 'output_tokens': 0}) == pytest.approx(.00064)
    with pytest.raises(ValueError):
        request_cost('unknown', {})


def test_retry_transient_only_and_never_after_submit():
    count = 0
    def create(**kw):
        nonlocal count
        count += 1
        if count < 3:
            raise ConnectionError('test infrastructure')
        return api_response()
    rec = run_anthropic(Session(), 'anthropic:claude-sonnet-5-5',
                        api=SimpleNamespace(messages=SimpleNamespace(create=create)), sleep=lambda _: None)
    assert count == 3 and len(rec['retries']) == 2 and rec['final']
    session = Session()
    def failed_submit(*args):
        session.submit_attempted = True
        raise ConnectionError('lost grade response')
    session.call = failed_submit
    rec = run_anthropic(session, 'anthropic:claude-sonnet-5-5',
                        api=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: api_response())))
    assert rec['ended'] == 'error' and rec['submit_attempted'] and not rec['retryable']


def test_budget_reservations_and_resume(tmp_path):
    budget = Budget(10, concurrency=2, prior=3)
    assert budget.reserve('a', True)
    assert budget.reserve('b', True)
    budget.finish('a', 5)
    assert not budget.reserve('c', True)
    budget.finish('b', 1)
    assert not budget.reserve('c', True)
    assert budget.reserve('free', False)
    calls = []
    def driver(model, task):
        calls.append((model, task['task_id']))
        return episode(model, task['task_id'], station=task['station'])
    tasks = [{'task_id': 'a', 'split': 'eval', 'station': 'X'},
             {'task_id': 'b', 'split': 'eval', 'station': 'Y'}]
    run_panel(tmp_path, tasks, ['codex:m', 'codex:n'], driver, concurrency=2)
    assert [t for _, t in calls] == ['a', 'a', 'b', 'b']
    calls.clear()
    run_panel(tmp_path, tasks, ['codex:m', 'codex:n'], driver, concurrency=2)
    assert calls == []
    assert len(json.loads((tmp_path/'index.json').read_text())['episodes']) == 4


def episode(model='codex:m', task='a', saved=20, station='X', feasible=True):
    return {'schema_version': 1, 'task_id': task, 'split': 'eval', 'station': station, 'model': model,
            'prompt_version': 1, 'scenario': {'draft_cost': 100}, 'geometry': {},
            'started_at': 'start', 'finished_at': 'end', 'steps': [{'tool': 'submit_design', 'args': {'design': {}}}],
            'ended': 'submitted', 'turns': 2, 'evaluations_used': 1,
            'final': {'valid': True, 'feasible': feasible, 'prescore_usd_saved': saved if feasible else -1,
                      'pct_saved': saved if feasible else 0, 'reward': saved/100 if feasible else 0},
            'usage': {'input_tokens': 1, 'output_tokens': 2, 'usd': None}}


def test_cluster_report_paired_and_ignores_legacy_anchors():
    records = [episode(task='a'), episode(task='b', saved=0, station='Y', feasible=False)]
    records[1]['final']['reward'] = 0
    row = summarize(records)
    assert row['mean_usd_saved'] == 9.5  # raw infeasible sentinel is retained
    assert row['feasible'] == .5 and row['mean'] == .1
    assert row['ci95'] == [0, .2] and row['station_clusters'] == 2
    assert row['failure_categories']['infeasible'] == 1
    records[0]['final']['anchors'] = {'source': 'none', 'best': None}
    assert summarize(records)['mean'] == .1
    assert cluster_ci([1, 3], ['X', 'X']) is None
    assert paired([episode()], [episode(model='codex:n', saved=10)])['mean_usd_saved_delta'] == 10


def test_best_observed_never_changes_task_files(tmp_path):
    run = tmp_path/'run'
    write_json(run/'m'/'a.json', episode())
    write_json(run/'n'/'a.json', episode(model='codex:n', saved=40))
    tasks = tmp_path/'tasks.jsonl'
    tasks.write_text(json.dumps({'task_id': 'a', 'search': {'usd_saved': 999}})+'\n')
    before = tasks.read_bytes()
    result = best_observed([r for _,r in records([run])])
    assert result['a']['saved_fraction'] == .4
    assert tasks.read_bytes() == before


def test_rescore_uses_saved_submission_and_preserves_trace():
    rec = episode()
    calls = []
    def verify(task, design):
        calls.append(design)
        return {'valid': True, 'feasible': True, 'cost': 90, 'standards': [],
                'receptor_values': {'x': [1]}, 'receptor_margins': {'x': [2]}}
    task = SimpleNamespace(draft_cost=100, record={})
    out = rescore_record(rec, task, verify_fn=verify)
    assert calls == [{}] and out['final']['reward'] == .1 and out['final']['prescore_usd_saved'] == 10
    assert out['steps'][0]['result'] == out['final']
    assert out['steps'][0]['receptor_values'] == {'x': [1]}
    assert out['rescore_history'][0]['final'] == rec['final']
    rec['steps'] = []
    rec['final'] = None
    rescore_record(rec, task, verify_fn=verify)
    assert len(calls) == 1


def test_slug_cannot_escape_or_collide():
    assert '/' not in model_slug('codex:../../a')
    assert model_slug('codex:a/b') != model_slug('codex:a__b')


def test_no_submit_turn_limit_invalid_and_bad_request_are_not_retried():
    calls = []
    def invalid(**kw):
        calls.append(kw)
        raise ValueError('bad request')
    r = run_anthropic(Session(), 'anthropic:claude-sonnet-5-5', api=SimpleNamespace(messages=SimpleNamespace(create=invalid)))
    assert len(calls) == 1 and r['ended'] == 'error' and not r['retryable']
    no_calls = SimpleNamespace(content=[SimpleNamespace(type='text', text='done')],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    r = run_anthropic(Session(), 'anthropic:claude-sonnet-5-5', api=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw:no_calls)))
    assert r['ended'] == 'no_submit' and r['final'] is None
    r = run_anthropic(Session(), 'anthropic:claude-sonnet-5-5', max_turns=2,
         api=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw:api_response('check_design'))))
    assert r['ended'] == 'turn_limit' and r['turns'] == 2


def test_resume_never_regrades_failures_or_force(tmp_path):
    def driver(model, task):
        r = episode(model, task['task_id'], feasible=False)
        r['ended'], r['final'] = 'no_submit', None
        return r
    tasks = [{'task_id': 'a', 'split': 'eval', 'station': 'X'}]
    run_panel(tmp_path, tasks, ['codex:m'], driver)
    def fail(*args):
        raise AssertionError('must not retry terminal failure')
    run_panel(tmp_path, tasks, ['codex:m'], fail)
    with pytest.raises(ValueError, match='force'):
        run_panel(tmp_path, tasks, ['codex:m'], fail, force=True)


def test_best_observed_excludes_baselines_and_infeasible():
    result = best_observed([episode(model='baseline:draft', saved=999), episode(saved=0),
                            episode(model='codex:n', saved=999, feasible=False)])
    assert result['a']['saved_fraction'] == 0


def test_rescore_infrastructure_failure_keeps_original():
    r = episode()
    original = copy.deepcopy(r)
    task = SimpleNamespace(draft_cost=100, record={})
    with pytest.raises(RuntimeError):
        rescore_record(r, task, verify_fn=lambda *a:{'valid':True,'feasible':False,'errors':['E_AERMOD_FATAL']})
    assert r == original


def test_budget_rejects_nonfinite_and_resume_charges_unknown_spend(tmp_path):
    for value in (float('nan'), float('inf'), -1):
        with pytest.raises(ValueError):
            Budget(value, concurrency=1)
    calls = []
    def unknown(model, task):
        calls.append(task['task_id'])
        r = episode(model, task['task_id'])
        r['usage']['usd'] = None
        r['usage']['billing_incomplete'] = True
        return r
    tasks = [{'task_id': t, 'split':'eval', 'station':'X'} for t in ['a','b','c']]
    run_panel(tmp_path, tasks, ['anthropic:m'], unknown, budget_usd=5, prior=3)
    assert calls == ['a']
    run_panel(tmp_path, tasks, ['anthropic:m'], unknown, budget_usd=5, prior=3)
    assert calls == ['a']


def test_interrupted_attempt_is_checkpointed_and_not_restarted(tmp_path):
    tasks = [{'task_id':'a','split':'eval','station':'X'}]
    def interrupted(model, task, checkpoint):
        r = episode(model, task['task_id'])
        checkpoint(r)
        raise RuntimeError('process interrupted after grade')
    run_panel(tmp_path, tasks, ['codex:m'], interrupted)
    r = json.loads(next((tmp_path/model_slug('codex:m')).glob('*/r1/episode.json')).read_text())
    assert r['final']['feasible']
    def no_retry(*a, **kw):
        raise AssertionError('must not re-grade interrupted submitted attempt')
    run_panel(tmp_path, tasks, ['codex:m'], no_retry)
