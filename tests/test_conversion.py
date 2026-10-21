"""Conversion requirements: uv run pytest -q tests/test_conversion.py."""

from copy import deepcopy
from decimal import Decimal, localcontext, ROUND_UP
import os
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from convert.v1_to_v2 import convert, height_menu, read_rows, check_integrity
from convert.reverify_drafts import finalize, reverify_one, run_drafts
from permit_core.model import load_v2_instance

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'convert/v1-src'
requires_archive = pytest.mark.skipif(not ARCHIVE.exists(), reason='convert/v1-src (the v1 extraction) is not in a public checkout')
IDS = ['003bd328a392-s47', '01addc2a86f1-s47', '0c0f1dfdc277-s47']


@pytest.mark.parametrize(('low', 'high', 'expected'), [
    ('10.00', '17.00', ['10', '11', '12', '13', '14', '15', '16', '17']),
    ('64.99', '65.00', ['64.99', '65']),
    ('64.50', '65.00', ['64.5', '65']),
    ('10.00', '13.50', ['10', '10.5', '11', '11.5', '12', '12.5', '13', '13.5']),
    ('10.25', '13.75', ['10.25', '11', '12', '13', '13.75']),
    ('65.00', '65.00', ['65']),
])
def test_height_menu(low, high, expected):
    with localcontext() as context:
        context.rounding = ROUND_UP
        assert height_menu(low, high) == [Decimal(x) for x in expected]


@pytest.mark.parametrize(('low', 'high'), [('66', '65'), ('NaN', '65'), ('10', 'Infinity')])
def test_bad_bounds(low, high):
    with pytest.raises(ValueError):
        height_menu(low, high)


@pytest.fixture
def converted(tmp_path):
    if not ARCHIVE.exists():
        pytest.skip('convert/v1-src (the v1 extraction) is not in a public checkout')
    src, out = tmp_path / 'archive', tmp_path / 'tasks'
    src.mkdir()
    archive_manifest = json.loads((ARCHIVE / 'EXTRACT_MANIFEST.json').read_text())
    manifest = json.loads((ARCHIVE / 'family/manifest.json').read_text())
    manifest['instances'] = [deepcopy(r) for r in manifest['instances'] if r['id'] in IDS]
    # Two splits and a mixed-source refinery: exercise the published metadata.
    for index, row in enumerate(manifest['instances']):
        row['coverage_tags'] = ['split:eval' if index == 2 else 'split:train']
        station = archive_manifest['instances'][row['id']]['met_station']
        source = ARCHIVE / row['path'] / 'task'
        shutil.copytree(source, src / row['path'] / 'task')
        met = src / 'met' / station
        met.mkdir(parents=True, exist_ok=True)
        for suffix in ['sfc', 'pfl']:
            (met / f'{station}.{suffix}').write_text(station + suffix)
    (src / 'family').mkdir()
    (src / 'family/manifest.json').write_text(json.dumps(manifest))
    (src / 'EXTRACT_MANIFEST.json').write_text(json.dumps(archive_manifest))
    convert(src, out)
    return src, out


def test_record_round_trip(converted):
    src, out = converted
    rows = read_rows(out)
    assert len(rows) == 3
    for row in rows:
        original = json.loads((src / 'instances' / row['id'] / 'task/task.json').read_text(),
                              parse_float=Decimal)
        scenario = load_v2_instance(out, row['id'])
        task = scenario.task
        for key in ['scenario', 'costs', 'background', 'locked_keyword_table',
                    'locked_native_template', 'bpip_stack_records', 'guards']:
            assert task[key] == original[key]
        assert task['design_value_requirements'] == original['scenario']['naaqs_thresholds']
        assert task['gep_method'] == original['scenario']['gep_method']
        assert task['legacy'] == {'instance_id': row['id'],
                                 'v1_commit': '00393129a2afbcff3824eca0b27d094b1cbb4844',
                                 'v1_task_name': 'permit-forge'}
        assert task['archetype'] == original['scenario']['facility']['archetype']
        assert task['no2_tier'] == 'ARM2'
        assert task['ozone']['usage'] == 'unused'
        assert task['evaluate_budget'] == 30
        assert scenario.met_path.is_dir()
        assert len(scenario.receptors) == 184
        for source in scenario.sources:
            assert source.gep_height_m in scenario.height_menu[source.id]
            assert scenario.draft_design[source.id]['control'] == source.id + '_maximum'
            assert scenario.draft_design[source.id]['stack_height_m'] == source.gep_height_m
        assert row['eligible'] is None
    legacy = [json.loads(s) for s in (out / 'permit-v1-legacy/tasks.jsonl').read_text().splitlines()]
    assert {r['id'] for r in legacy} == {r['id'] for r in rows}
    assert all(r['no2_tier'] == 'as_published' and r['height_menu'] is None
               and r['height_mode'] == 'continuous' for r in legacy)
    summary = json.loads((out / 'manifest.json').read_text())
    assert summary['counts']['total'] == 3
    assert summary['counts']['split'] == {'train': 2, 'eval': 1}
    assert sum(summary['counts']['station'].values()) == 3
    assert sum(summary['counts']['archetype'].values()) == 3
    assert sum(summary['counts']['n_sources'].values()) == 3
    assert all((out / f'permit-v2-{s}/manifest.json').is_file() for s in ['train', 'eval'])


def test_integrity_checks_every_file_and_met(converted):
    _, out = converted
    assert check_integrity(out) == []
    row = read_rows(out)[0]
    root = out / 'data' / row['id']
    assert set(row['sha256']) == {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    path = root / 'draft/bpip.inp'
    old = path.read_bytes()
    path.write_bytes(old + b'changed')
    assert any('draft/bpip.inp' in e for e in check_integrity(out))
    path.write_bytes(old)
    extra = root / 'unexpected.txt'
    extra.write_text('extra')
    assert any('unexpected.txt' in e for e in check_integrity(out))
    extra.unlink()
    met = next((out / 'met').rglob('*.sfc'))
    met.write_text('changed')
    assert any('.sfc' in e for e in check_integrity(out))


def test_repeat_conversion_does_not_reset_admission(converted, tmp_path):
    src, out = converted
    rows_before = read_rows(out)
    with pytest.raises(FileExistsError):
        convert(src, out)
    assert read_rows(out) == rows_before


def fake_verify(scenario, design, workdir, *, no2_tier):
    assert no2_tier == 'ARM2'
    assert Path(workdir).is_dir()
    assert design == scenario.draft_design
    return SimpleNamespace(feasible=True, design_values={}, errors=[], native_seconds={'NO2': 1.25},
                           gep={}, cost=Decimal('10'), binary_sha256={})


def test_resume_uses_cache_without_verify(converted, tmp_path):
    _, out = converted
    row = read_rows(out)[0]
    cache = tmp_path / 'cache'
    first = reverify_one(out, row, cache, verify_fn=fake_verify)
    def forbidden(*args, **kwargs):
        pytest.fail('cached task was re-verified')
    assert reverify_one(out, row, cache, verify_fn=forbidden) == first
    assert first['native_seconds'] == {'NO2': 1.25}
    assert json.loads((cache / f"{row['id']}.json").read_text()) == first
    cached = cache / f"{row['id']}.json"
    cached.write_text('{')
    with pytest.raises(ValueError, match='cache'):
        reverify_one(out, row, cache, verify_fn=forbidden)


def test_changed_input_rejects_cache(converted, tmp_path):
    _, out = converted
    row = read_rows(out)[0]
    cache = tmp_path / 'cache'
    reverify_one(out, row, cache, verify_fn=fake_verify)
    changed = deepcopy(row)
    changed['met_sha256']['KMSP.sfc'] = 'bad'
    with pytest.raises(ValueError):
        reverify_one(out, changed, cache, verify_fn=fake_verify)


def test_finalize_partial_failure_idempotent(converted, tmp_path):
    _, out = converted
    cache = tmp_path / 'cache'
    rows = read_rows(out)
    reverify_one(out, rows[0], cache, verify_fn=fake_verify)
    def infeasible(*args, **kwargs):
        result = fake_verify(*args, **kwargs)
        result.feasible = False
        result.errors = [{'code': 'E_NAAQS', 'detail': 'NO2_1h failed'}]
        result.design_values = {'NO2_1h': SimpleNamespace(value=Decimal('200'), background=Decimal('1'),
                                                        limit=Decimal('188'), margin=Decimal('-13'))}
        return result
    reverify_one(out, rows[1], cache, verify_fn=infeasible)
    summary = finalize(out, cache)
    updated = {r['id']: r for r in read_rows(out)}
    assert updated[rows[0]['id']]['eligible'] is True
    assert updated[rows[1]['id']]['eligible'] is False
    assert 'NO2_1h' in updated[rows[1]['id']]['eligibility_reason']
    assert updated[rows[2]['id']]['eligible'] is None
    assert summary['counts']['eligibility'] == {'eligible': 1, 'ineligible': 1, 'pending': 1}
    assert summary['failing_standards'] == {'NO2_1h': 1}
    assert summary['eligibility_by_split'] == {
        'train': {'eligible': 1, 'ineligible': 1, 'pending': 0},
        'eval': {'eligible': 0, 'ineligible': 0, 'pending': 1},
    }
    assert summary['decision_required'] is True
    before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert finalize(out, cache) == summary
    assert all(p.read_bytes() == data for p, data in before.items())
    assert check_integrity(out) == []
    legacy = (out / 'permit-v1-legacy/tasks.jsonl').read_text()
    assert all(json.loads(s)['eligible'] is None for s in legacy.splitlines())


def test_finalize_rejects_cache_for_other_input(converted, tmp_path):
    _, out = converted
    cache = tmp_path / 'cache'
    row = read_rows(out)[0]
    reverify_one(out, row, cache, verify_fn=fake_verify)
    path = cache / f"{row['id']}.json"
    result = json.loads(path.read_text())
    result['input_sha256'] = 'bad'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        finalize(out, cache)


def test_job_limit_and_unknown_ids(converted, tmp_path):
    _, out = converted
    for n in [0, (os.cpu_count() or 1) + 1]:
        with pytest.raises(ValueError):
            run_drafts(out, tmp_path / 'cache', jobs=n)
    with pytest.raises(ValueError):
        run_drafts(out, tmp_path / 'cache', jobs=1, only=['unknown'])


@requires_archive
def test_converted_archive_integrity():
    out = REPO / 'envs/permit/tasks'
    rows = read_rows(out)
    original = json.loads((ARCHIVE / 'family/manifest.json').read_text())['instances']
    assert {r['id'] for r in rows} == {r['id'] for r in original}
    assert len(rows) == 360
    assert sum(r['split'] == 'train' for r in rows) == 230
    assert sum(r['split'] == 'eval' for r in rows) == 130
    assert check_integrity(out) == []


def test_finalize_rejects_changed_files(converted, tmp_path):
    _, out = converted
    cache = tmp_path / 'cache'
    row = read_rows(out)[0]
    reverify_one(out, row, cache, verify_fn=fake_verify)
    (out / 'data' / row['id'] / 'receptors.inp').write_text('changed')
    with pytest.raises(ValueError, match='receptors.inp'):
        finalize(out, cache)


def test_run_drafts_resumes_without_worker_submission(converted, tmp_path, monkeypatch):
    from convert import reverify_drafts
    _, out = converted
    cache = tmp_path / 'cache'
    row = read_rows(out)[0]
    reverify_one(out, row, cache, verify_fn=fake_verify)
    def forbidden(*args, **kwargs):
        pytest.fail('cached task submitted to worker')
    monkeypatch.setattr(reverify_drafts.ProcessPoolExecutor, 'submit', forbidden)
    summary = run_drafts(out, cache, jobs=3, only=[row['id']])
    assert summary == {'selected': 1, 'cached': 1, 'completed': 1, 'decision_required': False}
