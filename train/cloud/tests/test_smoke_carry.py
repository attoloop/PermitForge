"""A GRPO-only smoke carries a passed base evaluation from an earlier run, with its provenance recorded."""
import json
from pathlib import Path

import pytest

from train.cloud import smoke
from train.cloud.run import write_json

URI = 'hf://buckets/Attoloop/permitforge-qwen38-runs/smoke-evalrules-c1/smoke/base-eval'


def test_carried_base_evaluation_is_restored_with_provenance(tmp_path, monkeypatch):
    calls = []
    def fake_restore(local: Path, uri: str, seconds: float) -> None:
        calls.append((local, uri))
        local.mkdir(parents=True)
        (local/'index.json').write_text('{}')
    monkeypatch.setattr(smoke, 'restore', fake_restore)
    smoke.carry_base_eval(URI, tmp_path/'smoke')
    assert calls == [(tmp_path/'smoke/base-eval', URI)]
    assert json.loads((tmp_path/'smoke/base-eval/PROVENANCE.json').read_text()) == {'carried_from': URI}
    for bad in ['s3://elsewhere/base-eval', 'hf://datasets/Attoloop/x/base-eval', URI+'/../grpo']:
        with pytest.raises(ValueError):
            smoke.carry_base_eval(bad, tmp_path/'other')
    assert not calls[1:]


def test_report_names_the_carried_base_evaluation(tmp_path):
    output = tmp_path/'smoke'
    (output/'base-eval').mkdir(parents=True)
    write_json(output/'base-eval/PROVENANCE.json', {'carried_from': URI})
    write_json(output/'metrics.json', smoke.smoke_summary([], {}, {}, elapsed_seconds=1.0, hourly_usd=5.0))
    smoke.write_report(output)
    assert f'Base evaluation carried from {URI}' in (output/'REPORT.md').read_text()


def test_smoke_readiness_requires_a_fitting_worst_case_step():
    rows = [dict(task_id=str(i), ended='submitted', reward=.1, history=[
        {'tool':'check_design', 'result':{'valid':True}}]) for i in range(3)]
    training = dict(status='passed', weights_changed=True, adapter_reload_equal=True,
        optimizer_steps=2, seconds_per_step=100, groups=[{'reward_std':.1}])
    kernels = {'status':'passed'}
    assert smoke.smoke_summary(rows, training, {}, elapsed_seconds=1, hourly_usd=5, kernels=kernels)['ready_for_grpo'] is False
    training['worst_case'] = {'fits': False, 'peak_reserved_bytes': 140_000_000_000}
    assert smoke.smoke_summary(rows, training, {}, elapsed_seconds=1, hourly_usd=5, kernels=kernels)['ready_for_grpo'] is False
    training['worst_case'] = {'fits': True, 'peak_reserved_bytes': 100_000_000_000, 'completion_tokens': 40960}
    assert smoke.smoke_summary(rows, training, {}, elapsed_seconds=1, hourly_usd=5, kernels=kernels)['ready_for_grpo'] is True
