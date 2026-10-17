"""WP2.8 requirements; no external model requests."""
import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from eval.tests.test_contract import Session, api_response, episode


def test_highest_effort_and_thinking_round_trip(tmp_path):
    from eval.anthropic_driver import run_anthropic
    from eval.capture import EpisodeCapture
    for model in ('claude-opus-5-5', 'claude-sonnet-5-5'):
        calls = []
        blocks = [dict(type='thinking', thinking='returned summary', signature='signed-block'),
                  dict(type='redacted_thinking', data='opaque'),
                  dict(type='tool_use', id='c', name='get_scenario', input={})]
        def create(**kw):
            calls.append(copy.deepcopy(kw))
            return (SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=100, output_tokens=20))
                    if len(calls) == 1 else api_response())
        session = Session()
        session.capture = EpisodeCapture(tmp_path/model)
        record = run_anthropic(session, 'anthropic:'+model,
                               api=SimpleNamespace(messages=SimpleNamespace(create=create)))
        assert calls[0]['output_config'] == {'effort': 'max'}
        assert calls[0]['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
        assert calls[0]['max_tokens'] == 32768
        assert calls[1]['messages'][1] == {'role': 'assistant', 'content': blocks}
        assert record['settings']['effort'] == 'max'
        assert 'returned summary' in (tmp_path/model/'transcript.jsonl').read_text()


def test_codex_effort_config_and_boundary(tmp_path):
    from eval.codex_driver import codex_command, CodexBoundary
    from eval.tests.test_codex import request
    argv = codex_command('gpt-6.1-sol', 'http://localhost:1', 'http://localhost:2', tmp_path)
    assert 'model_reasoning_effort="xhigh"' in argv
    body = request() | {'reasoning': {'effort': 'xhigh'}}
    result = CodexBoundary('test', max_turns=40, max_tokens=32768).prepare(body)
    assert result['reasoning']['effort'] == 'xhigh'


def test_shortlist_and_selection_deterministic_station_spread():
    from eval.select_tasks import shortlist, choose
    rows = [dict(task_id=f'{split}-{station}-{i}', split=split, station=str(station),
                 eligible=True, n_sources=i+2, near_binding=i, uncontrolled_exceedance=i,
                 hardness=i/10, variance_proxy=(6-i)/100, archetype='test')
            for split, n in [('eval', 5), ('train', 10)] for station in range(n) for i in range(6)]
    selected = shortlist(rows)
    assert Counter(r['split'] for r in selected) == {'eval': 30, 'train': 30}
    assert selected == shortlist(list(reversed(rows)))
    for split, stations in [('eval', 5), ('train', 10)]:
        result = choose([r for r in selected if r['split'] == split])
        assert len(result) == 10 and len({r['station'] for r in result}) == stations
        assert Counter(r['selection_group'] for r in result) == {'hard': 8, 'easy': 2}
        assert result == choose(list(reversed([r for r in selected if r['split'] == split])))


def test_probe_metrics_failure_zero_and_best_of_all():
    from eval.select_tasks import metrics
    result = metrics(100, {'draft': {'usd_saved': 0, 'feasible': True},
                          'cheapest_first_greedy': {'usd_saved': 20, 'feasible': True},
                          'v1_coordinate_descent': {'usd_saved': 30, 'feasible': True},
                          'search': {'usd_saved': 40, 'feasible': True}},
                     [{'usd_saved': v, 'feasible': v >= 0} for v in [-1, 0, 10, 20, 30, 40, 50, 60]])
    assert result['best_saved'] == 60
    assert result['hardness'] == pytest.approx(2/3)
    assert result['random_saved_fractions'][0] == 0
    assert result['variance_proxy'] > 0


def test_round_robin_order_projection_and_resume(tmp_path):
    from eval.run_eval import run_panel
    models = ['codex:gpt-6.1-sol', 'anthropic:claude-sonnet-5-5',
              'anthropic:claude-opus-5-5', 'codex:gpt-5.6-sol']
    tasks = [dict(task_id=t, split=s, station=t) for t, s in [('train', 'train'), ('eval', 'eval')]]
    calls = []
    def driver(model, task):
        calls.append((model, task['split'], task['repeat']))
        rec = episode(model, task['task_id'], station=task['station'])
        rec['usage']['usd'] = .5 if model.startswith('anthropic:') else None
        return rec
    kwargs = dict(order='round-robin', repeats=4, codex_concurrency=1,
                  anthropic_concurrency=1, budget_usd=190)
    run_panel(tmp_path, tasks, models, driver, **kwargs)
    assert [r for _, _, r in calls] == sorted(r for _, _, r in calls)
    for repeat in range(1, 5):
        round_calls = [(m, s) for m, s, r in calls if r == repeat]
        assert [s for _, s in round_calls] == ['eval']*4 + ['train']*4
        assert [(m, s) for m, s in round_calls if m.startswith('anthropic:')] == [
            ('anthropic:claude-opus-5-5', 'eval'), ('anthropic:claude-sonnet-5-5', 'eval'),
            ('anthropic:claude-sonnet-5-5', 'train'), ('anthropic:claude-opus-5-5', 'train')]
    projection = json.loads((tmp_path/'projection.json').read_text())
    assert projection['after_round'] == 1
    assert projection['models']['anthropic:claude-opus-5-5']['remaining_episodes'] == 6
    assert projection['models']['anthropic:claude-opus-5-5']['projected_remaining_usd'] == 3
    calls.clear()
    run_panel(tmp_path, tasks, models, driver, **kwargs)
    assert calls == []


def test_hard_request_budget_persists_unknown_and_concurrent_reservations(tmp_path):
    from eval.budget import HardBudget, BudgetExhausted
    budget = HardBudget(tmp_path/'budget.json', 10)
    budget.reserve('a', 6)
    with pytest.raises(BudgetExhausted):
        budget.reserve('b', 6)
    budget.settle('a', 2)
    budget.reserve('b', 6)
    recovered = HardBudget(tmp_path/'budget.json', 10)
    assert recovered.committed == 8
    with pytest.raises(BudgetExhausted):
        recovered.reserve('c', 3)


def test_board_effort_label():
    from eval.report import build_report, markdown
    rec = episode()
    rec['settings'] = {'effort': 'xhigh'}
    assert 'codex:m@xhigh' in markdown(build_report([rec]))


def test_probe_worker_limit():
    from eval.select_tasks import validate_workers
    validate_workers(4, 3)
    with pytest.raises(ValueError):
        validate_workers(4, 4)


@pytest.mark.parametrize('jobs', [4, 12])
def test_warm_capacity_and_retention(tmp_path, monkeypatch, jobs):
    from eval.warm_fields import warm
    leases = []
    class Lease:
        computed = True
        def __init__(self, task):
            self.path = tmp_path/task/'pack.json'
            self.path.parent.mkdir()
            self.path.write_text(json.dumps({'identity': task, 'files': {'a.npy': [100, 0]}}))
            self.closed = False
            leases.append(self)
        def release(self):
            self.closed = True
    def ensure(task, *, jobs):
        assert not any(lease.closed for lease in leases)
        return Lease(task)
    monkeypatch.setattr('eval.warm_fields.load_v2_instance', lambda root, tid: tid)
    monkeypatch.setattr('eval.warm_fields.ensure_pack', ensure)
    result = warm(tmp_path, ['a', 'b'], field_jobs=jobs)
    assert len(result['tasks']) == 2 and all(lease.closed for lease in leases)


def test_dry_run_has_no_driver_preflight_or_run_artifacts(tmp_path, monkeypatch, capsys):
    from eval import run_eval
    monkeypatch.setattr(run_eval.preflight, 'drivers', lambda *a: pytest.fail('dry-run sampled/preflighted a provider'))
    import sys
    monkeypatch.setattr(sys, 'argv', ['run_eval', '--run', 'dry', '--models', 'codex:gpt-6.1-sol',
                        '--split', 'train', '--limit', '1', '--repeats', '4', '--order', 'round-robin',
                        '--dry-run', '--output-root', str(tmp_path)])
    run_eval.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result['schedule']) == 4 and result['settings']['max_tokens'] == 32768
    assert not list(tmp_path.iterdir())


def test_providers_can_progress_independently(tmp_path):
    import threading
    from eval.run_eval import run_panel
    ready = threading.Event()
    def driver(model, task):
        if model.startswith('anthropic:'):
            assert ready.wait(2), 'Codex was blocked behind the paid queue'
        else:
            ready.set()
        return episode(model, task['task_id'])
    tasks = [dict(task_id=t, split='eval', station='X') for t in ('a', 'b')]
    run_panel(tmp_path, tasks, ['anthropic:claude-opus-5-5', 'codex:gpt-6.1-sol'], driver,
              repeats=1, order='round-robin', codex_concurrency=1, anthropic_concurrency=1, budget_usd=190)
    assert json.loads((tmp_path/'run.json').read_text())['status'] == 'complete'


def test_paid_driver_cannot_dispatch_over_hard_cap(tmp_path):
    from eval.anthropic_driver import run_anthropic
    from eval.budget import HardBudget, request_bound
    api = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: pytest.fail('paid dispatch')))
    cap = request_bound('claude-opus-5-5', 32768)-.01
    result = run_anthropic(Session(), 'anthropic:claude-opus-5-5', api=api,
                           hard_budget=HardBudget(tmp_path/'budget.json', cap))
    assert result['ended'] == 'no_submit' and result['error_type'] == 'budget'


def test_missing_warm_pack_fails_without_compute(tmp_path, monkeypatch):
    from eval.warm_fields import pin_warm
    from permit_core import fields
    monkeypatch.setattr('eval.warm_fields.load_v2_instance', lambda root, tid: SimpleNamespace(id=tid))
    monkeypatch.setattr(fields, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(fields, 'pack_identity', lambda task: 'identity')
    monkeypatch.setattr(fields, 'ensure_pack', lambda *a, **kw: pytest.fail('episode computed fields'))
    with pytest.raises(ValueError, match='missing/stale'):
        with pin_warm(tmp_path, ['missing']):
            pytest.fail('missing pack accepted')


def test_budget_denies_paid_launches_but_finishes_subscription(tmp_path):
    from eval.run_eval import run_panel
    called = []
    def driver(model, task):
        called.append(model)
        return episode(model, task['task_id'])
    run_panel(tmp_path, [dict(task_id='a', split='eval', station='X')],
              ['anthropic:claude-opus-5-5', 'codex:gpt-6.1-sol'], driver,
              order='round-robin', repeats=4, budget_usd=1, prior=.1)
    assert called == ['codex:gpt-6.1-sol']*4
    assert json.loads((tmp_path/'run.json').read_text())['status'] == 'budget_stopped'


def test_report_can_select_split_from_combined_panel(tmp_path, monkeypatch):
    from eval import report
    from eval.common import write_json
    run = tmp_path/'run'
    for split in ('eval', 'train'):
        write_json(run/'model'/split/'r1/episode.json', episode(task=split) | {'split': split})
    output = tmp_path/'eval.md'
    monkeypatch.setattr('sys.argv', ['report', str(run), '--split', 'eval', '--output', str(output)])
    report.main()
    assert json.loads(output.with_suffix('.json').read_text())['models'][0]['tasks'] == 1


def test_cancelled_probe_queue_does_not_start_more_native_work(tmp_path, monkeypatch):
    from eval import select_tasks
    (tmp_path/'.stop').touch()
    monkeypatch.setattr(select_tasks, 'probe', lambda *a, **kw: pytest.fail('cancelled queue computed fields'))
    with pytest.raises(InterruptedError):
        select_tasks.probe_cached((tmp_path, {'task_id': 'task'}, tmp_path, 3, 'fingerprint', False))
