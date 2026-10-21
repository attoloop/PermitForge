import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.native
def test_rollout_smoke_with_verify(tmp_path):
    package = Path(__file__).resolve().parents[1]
    output = tmp_path / 'smoke.json'
    environment = dict(os.environ)
    environment.pop('PERMIT_EVALUATOR', None)
    result = subprocess.run([sys.executable, 'rollout.py', '--smoke', '--task-id', '003bd328a392-s47',
                             '--output', str(output)], cwd=package,
                            env=environment,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert [s['tool'] for s in record['steps']] == [
        'get_scenario', 'check_design', 'evaluate_design', 'submit_design']
    assert record['final']['feasible'] and record['final']['reward'] == 0.0
    assert record['final']['prescore_usd_saved'] == 0
    assert record['evaluations_used'] == 1
    evidence = tmp_path / 'native-smoke'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'smoke.json').write_text(output.read_text())
    (evidence / 'smoke-trajectory.txt').write_text(result.stdout)
