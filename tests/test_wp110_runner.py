"""Task-major ensure/release, including failed jobs and explicit eviction."""
from types import SimpleNamespace
import json

import pytest

from convert import run_baselines as runner


@pytest.mark.parametrize('fails', [False, True])
def test_task_major_lease_lasts_through_policies_and_search(tmp_path, monkeypatch, fails):
    row = {'id': 'fixture', 'split': 'train', 'station': 'TEST', 'sha256': {}, 'met_sha256': {}}
    events = []
    manifest = tmp_path/'pack.json'
    manifest.write_text('{}')
    class Pack:
        path = manifest
        def release(self, *, evict=False):
            events.append(('release', evict))
    def ensure(task, *, jobs):
        events.append(('ensure', jobs))
        return Pack()
    def compute(*args, **kwargs):
        assert isinstance(kwargs['pack'], Pack)
        events.append('policies+search')
        if fails:
            raise RuntimeError('interrupted')
        return {'complete': True}
    monkeypatch.setattr(runner, 'ensure_pack', ensure)
    monkeypatch.setattr(runner, 'load_v2_instance', lambda *a: SimpleNamespace(id='fixture'))
    monkeypatch.setattr(runner, 'fingerprint', lambda *a: 'key')
    monkeypatch.setattr(runner, 'compute_task', compute)
    if fails:
        with pytest.raises(RuntimeError):
            runner.run_one(tmp_path, row, None, tmp_path/'results', field_jobs=2, evict_after=True)
    else:
        runner.run_one(tmp_path, row, None, tmp_path/'results', field_jobs=2, evict_after=True)
    assert events == [('ensure', 2), 'policies+search', ('release', True)]


def test_only_matching_native_draft_can_seed_verification_cache(tmp_path, monkeypatch):
    from convert.reverify_drafts import input_sha256
    row = {'id': 'fixture', 'station': 'TEST', 'sha256': {}, 'met_sha256': {}}
    binaries = tmp_path/'bin'
    binaries.mkdir()
    for name in ('aermod', 'bpipprm'):
        (binaries/name).write_text(name)
    monkeypatch.setattr(runner, 'models_dir', lambda: binaries)
    data = {'id': 'fixture', 'input_sha256': input_sha256(row), 'no2_tier': 'ARM2',
            'feasible': True, 'cost': '20', 'design_values': {}, 'margins': {},
            'native_seconds': {}, 'errors': [], 'failed_standards': [],
            'binary_sha256': {name: runner.sha256(binaries/name) for name in ('aermod', 'bpipprm')}}
    path = tmp_path/'fixture.json'
    path.write_text(json.dumps(data))
    assert runner.cached_draft(row, tmp_path)['cost'] == 20
    (binaries/'aermod').write_text('different binary')
    assert runner.cached_draft(row, tmp_path) is None
    data['input_sha256'] = 'wrong'
    path.write_text(json.dumps(data))
    assert runner.cached_draft(row, tmp_path) is None


def test_integrity_is_checked_before_on_demand_native_work(tmp_path, monkeypatch):
    row = {'id': 'fixture', 'station': 'TEST', 'sha256': {'task.json': 'expected'}, 'met_sha256': {}}
    calls = []
    monkeypatch.setattr(runner, 'load_v2_instance', lambda *a: calls.append('load'))
    monkeypatch.setattr(runner, 'ensure_pack', lambda *a, **kw: calls.append('ensure'))
    with pytest.raises(ValueError, match='sha256'):
        runner.run_one(tmp_path, row, None, tmp_path/'results')
    assert calls == []
