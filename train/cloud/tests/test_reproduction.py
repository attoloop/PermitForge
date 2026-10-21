"""The retained reproducer validates the lock and reviewed files, independent of git HEAD."""
from pathlib import Path
import runpy
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPRODUCER = ROOT/'internal/reports/WP2.15-evidence/reproduce.py'
requires_reproducer = pytest.mark.skipif(not REPRODUCER.exists(), reason='the stack reproducer is a local review tool, not part of the checkout')


def test_bundle_excludes_the_readme_that_records_its_digest():
    from train.cloud.bundle import bundle_files
    files = bundle_files(ROOT)
    assert ROOT/'train/cloud/README.md' not in files
    assert ROOT/'train/cloud/DESIGN.md' in files


@requires_reproducer
def test_reproducer_requires_the_locked_torch_distribution():
    reproduce = runpy.run_path(str(ROOT/'internal/reports/WP2.15-evidence/reproduce.py'))
    check = reproduce['check_versions']
    lock = tomllib.loads((ROOT/'train/cloud/uv.lock').read_text())
    check({'torch':'2.12.1+cu129', 'triton':'3.7.1'}, lock)
    with pytest.raises(AssertionError, match='torch'):
        check({'torch':'2.12.1', 'triton':'3.7.1'}, lock)


@requires_reproducer
def test_reproducer_uses_reviewed_hashes_and_rejects_changed_files(tmp_path):
    import hashlib
    reproduce = runpy.run_path(str(ROOT/'internal/reports/WP2.15-evidence/reproduce.py'))
    check = reproduce['check_identities']
    path = tmp_path/'reviewed.py'
    path.write_text('reviewed code')
    expected = {'reviewed.py':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert check(tmp_path,expected) == expected
    path.write_text('changed code')
    with pytest.raises(AssertionError, match='reviewed.py'):
        check(tmp_path,expected)
