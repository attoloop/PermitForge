"""Current reward documentation must match reproducible saved-fraction reports."""
from pathlib import Path

from convert.run_baselines import report
from convert.v1_to_v2 import read_rows

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_report_is_current(tmp_path):
    output = tmp_path/'baselines.md'
    report(read_rows(ROOT/'envs/permit/tasks'), ROOT/'convert/cache/baselines', output)
    assert output.read_bytes() == (ROOT/'results/baselines.md').read_bytes()


def test_reward_provenance_and_audit_have_no_retired_calibration():
    assert not (ROOT/'envs/permit/core/tests/fixtures/calibration.py.txt').exists()
    for name in ('SOURCES.md', 'results/reward-hacks.md', 'results/evidence/README.md'):
        text = (ROOT/name).read_text().lower()
        assert 'saved fraction' in text or 'saved-fraction' in text, name
        for stale in ('model calibration', 'model-derived anchors', 'calibrated', 'club-mix',
                      'calibration.py', 'calibration pending', 'gating and calibration'):
            assert stale not in text, (name, stale)
