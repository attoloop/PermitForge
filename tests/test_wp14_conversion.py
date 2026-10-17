"""Cache, reporting and admission: uv run --project envs/permit/core pytest -q tests/test_wp14_conversion.py."""

from copy import deepcopy
import json

import pytest

from convert import admit, run_baselines
from convert.reverify_drafts import input_sha256
from convert.v1_to_v2 import read_rows, write_json, write_rows


def record(ident='fixture', split='train'):
    return {'id': ident, 'split': split, 'station': 'SYNTHETIC', 'sha256': {}, 'met_sha256': {},
            'eligible': True, 'archetype': 'synthetic', 'n_sources': 2}


def results(row=None, *, saved=20, greedy=18, random=18):
    row = row or record()
    def policy(amount):
        return {'design': {'S1': {'control': 'none', 'height_m': 10}},
                'feasible': True, 'cost': 100-amount, 'usd_saved': amount,
                'evaluations_used': 1, 'verified': True}
    return {'schema_version': 1, 'id': row['id'], 'split': row['split'], 'complete': True,
            'task_sha256': input_sha256(row), 'draft_cost': 100,
            'baselines': {name: policy(0 if name == 'draft' else greedy if name == 'cheapest_first_greedy'
                                      else random if name == 'random_30' else 0)
                          for name in run_baselines.POLICIES},
            'search': {**policy(saved), 'nodes': 2, 'closed': False, 'seconds': .1}}


def test_admission_uses_only_native_arm2_draft():
    draft = {'feasible': True, 'no2_tier': 'ARM2', 'cost': '100'}
    for diagnostic in (None, results(saved=0), results(greedy=99), results(random=99)):
        assert admit.decision(draft, diagnostic) == (True, [])
    assert admit.decision(None, None)[0] is None
    assert admit.decision({**draft, 'feasible': False}, results())[0] is False
    with pytest.raises(ValueError, match='ARM2'):
        admit.decision({**draft, 'no2_tier': 'PVMRM'}, None)


def test_resume_and_changed_inputs_rejected(tmp_path, monkeypatch):
    row = record()
    pack = tmp_path / 'fixture.npz'
    pack.write_bytes(b'synthetic pack fingerprint')
    calls = []
    def compute(tasks, row, field_path, cache, **kwargs):
        calls.append(row['id'])
        return results(row)
    monkeypatch.setattr(run_baselines, 'compute_task', compute)
    first = run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')
    assert run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache') == first
    assert calls == ['fixture']
    pack.write_bytes(b'changed field pack')
    with pytest.raises(ValueError, match='stale'):
        run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')


def test_interrupted_task_resumes_completed_policies(tmp_path, monkeypatch):
    row = record()
    pack = tmp_path / 'fixture.npz'
    pack.write_bytes(b'synthetic')
    def interrupted(tasks, row, field_path, cache, *, progress, checkpoint, **kwargs):
        progress['baselines']['draft'] = results(row)['baselines']['draft']
        checkpoint(progress)
        raise KeyboardInterrupt
    monkeypatch.setattr(run_baselines, 'compute_task', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')
    def resumed(tasks, row, field_path, cache, *, progress, **kwargs):
        assert set(progress['baselines']) == {'draft'}
        return results(row)
    monkeypatch.setattr(run_baselines, 'compute_task', resumed)
    result = run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')
    assert result['complete']
    (tmp_path / 'cache/fixture.json').write_text('{')
    with pytest.raises(ValueError, match='cache'):
        run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')


def test_compute_task_uses_public_evaluate_and_arm2_verifies_submissions(tmp_path, monkeypatch):
    from types import SimpleNamespace
    view = {'sources': [{'id': 'S1', 'height_min_m': 10, 'height_menu_m': [10, 20], 'gep_m': 20}],
            'controls': [{'id': 'S1_none', 'applies_to': ['S1'], 'annualized_usd': 0},
                         {'id': 'S1_moderate', 'applies_to': ['S1'], 'annualized_usd': 5},
                         {'id': 'S1_maximum', 'applies_to': ['S1'], 'annualized_usd': 10}],
            'stack_usd_per_m': 1, 'draft_design': {'S1': {'control': 'S1_maximum', 'height_m': 20}},
            'draft_cost': 20, 'evaluations_left': 30}
    task = SimpleNamespace(id='fixture')
    pack = object()
    def evaluation(task_arg, design, pack_arg):
        assert task_arg is task and pack_arg is pack
        return {'valid': True, 'feasible': True, 'cost': float(run_baselines.outcome(view, design, None, 0)['cost'])}
    calls = []
    def verify(task_arg, design, work):
        calls.append(json.dumps(design, sort_keys=True))
        return evaluation(task_arg, design, pack)
    monkeypatch.setattr(run_baselines, 'load_v2_instance', lambda *a: task)
    monkeypatch.setattr(run_baselines, 'load_pack', lambda *a: pack)
    monkeypatch.setattr(run_baselines, 'scenario_view', lambda *a, **kw: view)
    monkeypatch.setattr(run_baselines, 'evaluate', evaluation)
    monkeypatch.setattr(run_baselines, 'verify_evaluation', verify)
    monkeypatch.setattr(run_baselines, 'search', lambda *a, **kw: results()['search'])
    checkpoints = []
    progress = {'baselines': {}, 'complete': False}
    result = run_baselines.compute_task(tmp_path, record(), tmp_path / 'fields', tmp_path / 'cache',
                                        seed=1, max_nodes=1, seconds=1, progress=progress,
                                        checkpoint=lambda p: checkpoints.append(deepcopy(p)))
    assert result['complete'] and len(checkpoints) == len(run_baselines.POLICIES)
    assert len(calls) == len(set(calls))
    assert all(p['verified'] for p in result['baselines'].values())


def test_resume_refuses_changed_task_files(tmp_path, monkeypatch):
    row = record()
    pack = tmp_path / 'fixture.npz'
    pack.write_bytes(b'synthetic')
    monkeypatch.setattr(run_baselines, 'compute_task', lambda *a, **kw: results(row))
    run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')
    (tmp_path / 'data/fixture').mkdir(parents=True)
    (tmp_path / 'data/fixture/task.json').write_text('changed')
    with pytest.raises(ValueError, match='sha256'):
        run_baselines.run_one(tmp_path, row, pack, tmp_path / 'cache')


def test_report_both_splits_gate_failure_and_saved_fraction(tmp_path):
    rows = [record('a', 'train'), record('b', 'eval')]
    cache = tmp_path / 'cache'
    for row in rows:
        data = results(row)
        data['baselines']['all_none_at_gep'].update(feasible=False, usd_saved=-1, verified=True)
        write_json(cache / f"{row['id']}.json", data)
    rows[1]['anchors'] = {'worst': 0, 'best': 40, 'saturation': 160, 'source': 'panel'}
    output = tmp_path / 'baselines.md'
    run_baselines.report(rows, cache, output)
    text = output.read_text()
    assert 'train' in text and 'eval' in text
    assert 'mean USD saved' in text and '% saved' in text and 'feasible rate' in text
    assert 'best-known search' in text and 'sentinel' in text
    assert '0.18' in text  # Reward is independent of the stale anchor fixture.
    assert '| cheapest_first_greedy | 18.0000 | 18.0000 | 1.0000 | 0.1800 |' in text
    assert 'optimum' not in text.lower()


def test_admission_writes_only_records_logs_and_manifests_idempotently(tmp_path):
    rows = [record('a', 'train'), record('b', 'eval'), record('c', 'eval')]
    for split in ('train', 'eval'):
        write_rows(tmp_path / f'permit-v2-{split}/tasks.jsonl', [r for r in rows if r['split'] == split])
        write_rows(tmp_path / f'permit-v2-{split}/build_log.jsonl', [])
    write_rows(tmp_path / 'build_log.jsonl', [])
    for row in rows[:2]:
        write_json(tmp_path / 'drafts' / f"{row['id']}.json", {
            'id': row['id'], 'input_sha256': input_sha256(row), 'no2_tier': 'ARM2', 'feasible': True,
            'cost': '100', 'design_values': {}, 'margins': {}, 'native_seconds': {},
            'errors': [], 'binary_sha256': {}, 'failed_standards': []})
        write_json(tmp_path / 'cache' / f"{row['id']}.json",
                   results(row, greedy=19 if row['id'] == 'b' else 18))
    summary = admit.apply(tmp_path, tmp_path / 'drafts', tmp_path / 'cache')
    assert summary['admitted'] == 2 and summary['dropped'] == 0 and summary['pending'] == 1
    updated = {r['id']: r for r in read_rows(tmp_path)}
    assert updated['a']['eligible'] is True and updated['a']['admitted'] is True
    assert updated['b']['eligible'] is True and updated['b']['admitted'] is True
    assert updated['c']['admitted'] is None
    assert 'anchors' not in updated['a']
    assert 'anchors' not in updated['b']
    # The >10% decision point concerns failed ARM2 drafts, not naive-policy admission.
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['decision_required'] is False
    assert manifest['admission_counts'] == summary
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert admit.apply(tmp_path, tmp_path / 'drafts', tmp_path / 'cache') == summary
    assert all(p.read_bytes() == value for p, value in before.items())
    bad = deepcopy(results(rows[0]))
    bad['task_sha256'] = 'wrong'
    write_json(tmp_path / 'cache/a.json', bad)
    assert admit.apply(tmp_path, tmp_path / 'drafts', tmp_path / 'cache') == summary


def test_pending_baselines_preserve_arm2_draft_eligibility(tmp_path):
    row = record('a')
    for split in ('train', 'eval'):
        write_rows(tmp_path / f'permit-v2-{split}/tasks.jsonl', [row] if split == 'train' else [])
    write_json(tmp_path / 'drafts/a.json', {
        'id': 'a', 'input_sha256': input_sha256(row), 'no2_tier': 'ARM2', 'feasible': True,
        'cost': '100', 'design_values': {}, 'margins': {}, 'native_seconds': {},
        'errors': [], 'binary_sha256': {}, 'failed_standards': []})
    admit.apply(tmp_path, tmp_path / 'drafts', tmp_path / 'cache')
    updated = read_rows(tmp_path)[0]
    assert updated['admitted'] is True and updated['eligible'] is True
    # A pending task remains selectable for the later baseline run.
    assert run_baselines.run_tasks(tmp_path, tmp_path / 'cache')['selected'] == 1


def test_jobs_and_missing_fields(tmp_path):
    for jobs in (0, 13):
        with pytest.raises(ValueError, match='jobs'):
            run_baselines.run_tasks(tmp_path, tmp_path / 'cache', jobs=jobs)
    for split in ('train', 'eval'):
        write_rows(tmp_path / f'permit-v2-{split}/tasks.jsonl', [record(split, split)])
    summary = run_baselines.run_tasks(tmp_path, tmp_path / 'cache', jobs=1)
    assert summary['completed'] == 0 and summary['missing_fields'] == ['eval', 'train']
    with pytest.raises(ValueError, match='unknown'):
        run_baselines.run_tasks(tmp_path, tmp_path / 'cache', jobs=1, only=['unknown'])


def test_record_writer_drops_legacy_reward_metadata(tmp_path):
    from convert.v1_to_v2 import write_manifests, read_rows
    row = record('a', 'train')
    row.update(anchors={'worst': 0, 'best': 40}, anchors_frozen_at='old',
               anchor_source='panel', anchor_runs=['old'])
    write_manifests(tmp_path, [row])
    assert not any('anchor' in k for k in read_rows(tmp_path)[0])
