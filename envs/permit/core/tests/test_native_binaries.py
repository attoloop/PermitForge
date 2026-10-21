"""The native binaries that grade submissions are the ones the lift evidence was recorded with."""
import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / 'results/evidence/faithful-lift'


def test_native_binaries_match_the_lift_evidence():
    binaries = Path(os.environ.get('PERMIT_MODELS_DIR') or ROOT / 'envs/permit/core/native_build/bin')
    if not (binaries / 'aermod').exists() or not (binaries / 'bpipprm').exists():
        pytest.skip('native binaries are not built; run sh envs/permit/core/native_build/build.sh')
    expected = {}
    for path in sorted(EVIDENCE.glob('*-s47.json')):
        for name, row in json.loads(path.read_text())['binary_sha256'].items():
            expected.setdefault(name, set()).add(row['ours'])
    assert set(expected) == {'aermod', 'bpipprm'}
    for name, hashes in expected.items():
        assert len(hashes) == 1, f'the evidence records more than one {name} hash'
        digest = hashlib.sha256((binaries / name).read_bytes()).hexdigest()
        assert digest == next(iter(hashes)), f'{name} differs from the binary the evidence was recorded with'
