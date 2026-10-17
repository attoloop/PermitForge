"""WP2.7 acceptance tests; synthetic episodes only, no model/native jobs."""
import copy
import json
from pathlib import Path

import pytest

from eval.common import model_slug, records, write_json
from eval.report import build_report, cluster_ci
from eval.run_eval import run_panel
from permit_core.reward import reward


def episode(model='codex:test', task='a', repeat=1, saved=20, station='X'):
    return {'schema_version': 1, 'model': model, 'task_id': task, 'repeat': repeat,
            'split': 'eval', 'station': station, 'scenario': {'draft_cost': 100},
            'steps': [{'tool': 'submit_design', 'args': {'design': {}}}],
            'ended': 'submitted', 'final': {'valid': True, 'feasible': True,
            'prescore_usd_saved': saved, 'reward': saved/100}, 'usage': {'usd': .1}}


def test_reward_is_saved_fraction_without_anchors():
    assert reward(20, 100) == .2
    assert reward(-1, 100) == reward(0, 100) == 0
    assert reward(100, 100) == 1
    for bad in (None, True, float('nan'), float('inf')):
        assert reward(bad, 100) == 0
    for bad in (0, -1, None, True, float('nan')):
        with pytest.raises(ValueError, match='draft'):
            reward(20, bad)


def test_repeat_resume_and_layout(tmp_path):
    tasks = [{'task_id': 'a', 'station': 'X', 'split': 'eval'}]
    calls = []
    def driver(model, task):
        calls.append(task['repeat'])
        return episode(model=model, repeat=task['repeat'])
    run_panel(tmp_path, tasks, ['codex:test'], driver, repeats=4, concurrency=2)
    assert sorted(calls) == [1, 2, 3, 4]
    root = tmp_path/model_slug('codex:test')/'a'
    assert all((root/f'r{k}'/'episode.json').exists() for k in range(1, 5))
    assert len(records([tmp_path])) == 4
    run_panel(tmp_path, tasks, ['codex:test'], driver, repeats=4, concurrency=2)
    assert len(calls) == 4
    assert {r['repeat'] for r in json.loads((tmp_path/'index.json').read_text())['episodes']} == {1, 2, 3, 4}


def test_import_legacy_repeat_one_and_protocol_guard(tmp_path):
    tasks = [{'task_id': 'a', 'station': 'X', 'split': 'eval'}]
    source, target = tmp_path/'pilot', tmp_path/'repeated'
    run_panel(source, tasks, ['codex:test'], lambda m,t: episode(model=m), repeats=1)
    # Emulate the pilot's canonical pre-repeat layout.
    root = source/model_slug('codex:test')/'a'
    value = json.loads((root/'r1'/'episode.json').read_text())
    value.pop('repeat')
    value['reward'] = .99
    import shutil
    shutil.rmtree(root/'r1')
    (root/'r1.json').unlink(missing_ok=True)
    write_json(root/'episode.json', value)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls = []
    def driver(m, t):
        calls.append(t['repeat'])
        return episode(model=m, repeat=t['repeat'])
    run_panel(target, tasks, ['codex:test'], driver, repeats=4, import_runs=[source])
    assert calls == [2, 3, 4]
    imported = json.loads((target/model_slug('codex:test')/'a/r1/episode.json').read_text())
    assert imported['reward'] == imported['final']['reward'] == .2
    assert all(p.read_bytes() == b for p,b in before.items())
    manifest = json.loads((source/'run.json').read_text())
    manifest['protocol_sha256'] = 'different'
    write_json(source/'run.json', manifest)
    with pytest.raises(ValueError, match='protocol'):
        run_panel(tmp_path/'bad', tasks, ['codex:test'], driver, repeats=4, import_runs=[source])


def test_task_weighted_variance_and_secondary_metric(tmp_path):
    items = [episode(repeat=1, saved=0), episode(repeat=2, saved=20),
             episode(task='b', saved=80, station='Y')]
    report = build_report(items)
    row = report['models'][0]
    assert row['mean'] == pytest.approx(.45)  # equal task weight, not equal episode weight
    assert row['ci95'] == cluster_ci([.1, .8], ['X', 'Y'])
    a, b = row['per_task']
    assert a['task_id'] == 'a' and a['mean'] == .1 and a['std'] == pytest.approx(2**.5/10)
    assert (a['min'], a['max'], a['mean_pairwise_spread']) == (0, .2, .2)
    assert b['std'] is None and b['mean_pairwise_spread'] is None
    assert row['grpo_signal'] == 1 and row['repeated_tasks'] == 1
    assert row['fraction_of_best_observed'] == .75
    from eval.report import write_variance
    write_variance(tmp_path/'variance', report)
    assert len(json.loads((tmp_path/'variance.json').read_text())) == 2
    assert 'mean_pairwise_spread' in (tmp_path/'variance.csv').read_text()
    with pytest.raises(ValueError, match='duplicate'):
        build_report(items + [copy.deepcopy(items[0])])


def test_best_observed_is_read_only_and_zero_denominator():
    from eval.best_observed import best_observed
    data = [episode(), episode(repeat=2, saved=40), episode(task='b', saved=0),
            episode(model='baseline:test', saved=99)]
    original = copy.deepcopy(data)
    best = best_observed(data)
    assert best['a']['saved_fraction'] == .4
    assert best['b']['saved_fraction'] == 0
    assert data == original
    row = build_report([episode(saved=0)])['models'][0]
    assert row['fraction_of_best_observed'] is None


def test_subset_selection_is_deterministic_and_covers_pilot():
    from eval.subsets import PILOT_TASKS, select_subsets
    root = Path(__file__).resolve().parents[2]/'envs/permit/tasks'
    rows = [json.loads(s) for p in sorted(root.glob('permit-v2-*/tasks.jsonl')) for s in p.read_text().splitlines()]
    selected = select_subsets(rows, seed=27)
    assert selected == select_subsets(list(reversed(rows)), seed=27)
    assert len(selected['eval']) == 20 and len(selected['train']) == 10
    assert set(PILOT_TASKS) <= {r['task_id'] for r in selected['eval']}
    from collections import Counter
    assert set(Counter(r['station'] for r in selected['eval']).values()) == {4}
    assert len({r['station'] for r in selected['train']}) == 10
    for split in ('train', 'eval'):
        assert {r['n_sources'] for r in selected[split]} == set(range(2, 7))
        assert len({r['archetype'] for r in selected[split]}) == 5
    assert all(not any('anchor' in k for k in r) for r in rows)


def test_repeat_infrastructure_retry_is_local(tmp_path):
    tasks = [{'task_id': 'a', 'station': 'X', 'split': 'eval'}]
    calls = []
    def driver(m, t):
        calls.append(t['repeat'])
        if t['repeat'] == 2 and calls.count(2) == 1:
            raise ConnectionError('synthetic transport failure')
        return episode(model=m, repeat=t['repeat'])
    run_panel(tmp_path, tasks, ['codex:test'], driver, repeats=3)
    assert len(records([tmp_path])) == 2
    run_panel(tmp_path, tasks, ['codex:test'], driver, repeats=3)
    assert calls == [1, 2, 3, 2]
    assert len(records([tmp_path])) == 3


def test_report_merge_requires_matching_protocol(tmp_path):
    from eval.common import report_records
    for name, protocol in [('a', 'one'), ('b', 'two')]:
        write_json(tmp_path/name/'run.json', {'protocol_sha256': protocol})
        write_json(tmp_path/name/'m/a/episode.json', episode(repeat=1 if name == 'a' else 2))
    with pytest.raises(ValueError, match='protocol'):
        report_records([tmp_path/'a', tmp_path/'b'])


def test_invalid_or_unsubmitted_legacy_final_is_not_scored():
    from eval.common import saved_fraction
    for key, value in [('valid', False), ('feasible', False)]:
        item = episode()
        item['final'][key] = value
        assert saved_fraction(item) == 0
    item = episode()
    item['ended'] = 'no_submit'
    assert saved_fraction(item) == 0


def test_rescore_updates_top_level_reward_and_zero_no_submit():
    from types import SimpleNamespace
    from eval.rescore import rescore_record
    item = episode()
    item['reward'] = .99
    task = SimpleNamespace(draft_cost=100, record={})
    result = rescore_record(item, task, verify_fn=lambda *a: {'valid': True, 'feasible': True, 'cost': 80})
    assert result['reward'] == result['final']['reward'] == .2
    item.update(ended='no_submit', final=None, steps=[])
    assert rescore_record(item, task)['reward'] == 0
