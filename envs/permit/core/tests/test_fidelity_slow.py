"""Re-run the first ten recorded gate comparisons, including native ARM2 verification."""

import json
from pathlib import Path

import pytest
from conftest import requires_native_binaries, requires_task_data, requires_v1

ROOT = Path(__file__).resolve().parents[4]


@requires_v1
@requires_task_data
@requires_native_binaries
@pytest.mark.slow
@pytest.mark.native
def test_fidelity_slow(tmp_path):
    path = ROOT / 'results/evidence/fidelity.json'
    if not path.exists():
        pytest.skip('run convert/fidelity.py to create the fidelity evidence first')
    evidence = json.loads(path.read_text())
    cases = evidence['comparisons'][:10]
    assert len(cases) == 10
    from permit_core.evaluate import evaluate
    from permit_core.fields import cached_pack, certified_prune, precompute
    from permit_core.model import load_v1_instance
    from permit_core.verify import verify
    task = None
    for case in cases:
        if task is None or task.id != case['task_id']:
            task = load_v1_instance(case['task_id'])
            # The field cache is size-capped and evicts; recompute whatever was evicted.
            precompute([task.id], jobs=4)
            pack, _ = certified_prune(task, cached_pack(task))
        fast = evaluate(task, case['design'], pack)
        ledger = {sid: {'control': row['control'], 'stack_height_m': row['height_m']}
                  for sid, row in case['design'].items()}
        native = verify(task, ledger, tmp_path, no2_tier='ARM2')
        assert native.design_values, native.errors
        assert fast['feasible'] == native.feasible
        for standard in fast['standards']:
            expected = native.design_values[standard['id']].value
            from decimal import Decimal
            assert Decimal(str(standard['value'])) == expected, (task.id, case['index'], standard['id'])
