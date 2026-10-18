import json
from pathlib import Path

import pytest
import torch


def test_probe_matrix_has_real_and_regression_geometry_without_relaxed_tolerances():
    from train.cloud.kernel_cases import CASES, TOLERANCES
    assert TOLERANCES == dict(rtol=.03, atol=.002, relative_l2=.02)
    real = [c for c in CASES if c.name.startswith('model')]
    assert {c.seed for c in real} >= {42, 212}
    assert {c.length for c in real} >= {63, 64, 65, 129}
    assert all((c.key_heads, c.value_heads, c.key_dim, c.value_dim) == (16, 48, 128, 128) for c in real)
    assert any(c.left_padding > 0 and c.batch == 2 for c in real)
    issue = [c for c in CASES if c.name.startswith('issue640')]
    assert any((c.batch,c.length,c.key_heads,c.key_dim,c.dtype,c.scale,c.normalize) ==
               (1,1024,4,100,'float16',1.,False) for c in issue)


def test_inner_dispatch_must_execute_tilelang_not_merely_import_it():
    from train.cloud.kernel_cases import require_inner_backend
    with pytest.raises(RuntimeError, match='TileLang'):
        require_inner_backend({'fla.ops.common.chunk_o.chunk_bwd_dqkwg': 5})
    require_inner_backend({'fla.ops.common.backends.tilelang.chunk_bwd.chunk_bwd_dqkwg_tilelang': 2})


def test_selection_uses_only_complete_passing_measurements_and_baseline_is_mandatory():
    from train.cloud.qualification import select_configuration
    baseline = dict(name='baseline', status='passed', generation_seconds=8., training_seconds=4.,
                    settings=dict(attention='sdpa', persistent=False, microbatch=1, liger=False))
    fast = dict(baseline, name='cached', generation_seconds=2.)
    bad = dict(fast, name='bad', status='failed', generation_seconds=.01)
    assert select_configuration([baseline,fast,bad])['name'] == 'cached'
    with pytest.raises(RuntimeError, match='baseline'):
        select_configuration([dict(baseline,status='failed'),fast])
    with pytest.raises(RuntimeError):
        select_configuration([dict(baseline,training_seconds=float('nan'))])


def test_smoke_caps_allow_qualification_but_never_exceed_three_hours_or_fifteen_dollars():
    from train.cloud.config import CloudConfig, stage_plan
    from train.cloud.run import main
    cfg = CloudConfig()
    stage = stage_plan(cfg)[0]
    assert 6600 < stage['wall_seconds'] <= 10200
    assert stage['usd_cap'] <= 15
    assert stage['estimated_seconds'] > 1515
    with pytest.raises(ValueError, match='Smoke caps'):
        stage_plan(cfg, wall_caps={'smoke':10801})
    assert callable(main)


def test_optional_packages_are_individually_pinned():
    import tomllib
    project = tomllib.loads((Path(__file__).parents[1]/'pyproject.toml').read_text())['project']
    assert project['optional-dependencies']['attention'] == ['kernels==0.16.2']
    assert project['optional-dependencies']['loss'] == ['liger-kernel==0.8.2']


def test_gpu_gate_failure_is_recorded_and_does_not_select_candidate(tmp_path):
    from train.cloud.qualification import gate
    def fail():
        raise RuntimeError('injected finite gradient mismatch')
    report = {}
    assert gate('fa3', fail, report, tmp_path/'qualification.json') is None
    saved = json.loads((tmp_path/'qualification.json').read_text())
    assert saved['fa3']['status'] == 'failed'
    assert 'gradient mismatch' in saved['fa3']['error']


def test_numerical_guard_rejects_finite_corruption_and_records_failure():
    from train.cloud.kernel_cases import compare
    evidence = {}
    with pytest.raises(RuntimeError, match='numerical agreement'):
        compare('gradient',torch.ones(8),torch.ones(8)*2,evidence)
    assert evidence['gradient']['passed'] is False


def test_fixed_trajectories_use_actual_train_tasks_and_tool_truncation_masks():
    from types import SimpleNamespace
    from permit_openenv.tasks import load_pack
    from train.cloud.config import CloudConfig
    from train.cloud.fixed_trajectories import fixtures, loss_inputs
    from train.cloud.formatting import load_tokenizer
    from train.cloud.selection import selection_manifest
    root = Path(__file__).resolve().parents[3]
    pack = load_pack(root/'envs/permit/tasks')
    selection = selection_manifest(pack,root/'results/subsets.json')
    tasks = [pack.get(t) for t in selection['smoke_train']]
    trainer = SimpleNamespace(processing_class=load_tokenizer(local_files_only=True),
                              accelerator=SimpleNamespace(device='cpu'))
    fixed = fixtures(trainer,tasks,CloudConfig())
    assert fixed == fixtures(trainer,tasks,CloudConfig())
    assert len(fixed['rows']) == 4
    assert {r['task_id'] for r in fixed['rows']} == set(selection['smoke_train'])
    rows = loss_inputs(fixed,trainer)
    assert rows[-1]['completion_mask'].sum() == 0
    assert (rows[0]['tool_mask'] == 0).any() and rows[0]['tool_mask'].sum() > 0
    for row in fixed['rows']:
        assert row['turns'][1][:len(row['turns'][0])] == row['turns'][0]
        assert all(seq[-1] == trainer.processing_class.eos_token_id for seq in row['forced'])


def test_behavior_only_smoke_cannot_qualify_an_untested_stack():
    from train.cloud.smoke import smoke_summary
    rows = [dict(ended='submitted',reward=.1,steps=[dict(result={'valid':True})]) for _ in range(3)]
    training = dict(status='passed',optimizer_steps=2,weights_changed=True,adapter_reload_equal=True,
                    groups=[dict(reward_std=.1)])
    assert not smoke_summary(rows,training,{},elapsed_seconds=1,hourly_usd=5)['ready_for_grpo']


def test_inner_trace_observes_execution_and_restores_profiler():
    import sys
    from types import FunctionType
    from train.cloud.kernel_cases import inner_calls,require_inner_backend
    def chunk_bwd_dqkwg_tilelang():
        return 1
    fn = FunctionType(chunk_bwd_dqkwg_tilelang.__code__,
        {'__name__':'fla.ops.common.backends.tilelang.chunk_bwd'})
    before = sys.getprofile()
    with inner_calls() as calls:
        fn()
    require_inner_backend(calls)
    assert sys.getprofile() is before


def test_matrix_reference_uses_nonzero_state_padding_and_grouped_heads():
    import inspect
    from types import SimpleNamespace
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from train.cloud.kernel_cases import Case,qualify_matrix
    from train.cloud.kernels import NAMES
    functions = SimpleNamespace(**{n:inspect.unwrap(getattr(qwen,n)) for n in NAMES})
    with torch.backends.mkldnn.flags(enabled=False):
        result = qualify_matrix(functions,fast=False,device='cpu',cases=[
            Case('cpu_reference',42,65,batch=2,key_heads=2,value_heads=4,key_dim=8,value_dim=8,left_padding=3),
            Case('cpu_scale',212,65,key_heads=2,value_heads=2,key_dim=8,value_dim=8,normalize=False,scale=1.)])
    assert result['status'] == 'passed'
    assert all('dh0' in row['comparisons'] for row in result['cases'])


def test_failed_fast_gdn_is_reported_then_reference_must_pass(monkeypatch,tmp_path):
    import inspect
    from train.cloud import kernels
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name in kernels.NAMES:
        original = inspect.unwrap(getattr(qwen,name))
        def factory(original,module):
            def implementation(*a,**kw):
                raise AssertionError('Probe is stubbed, never execute GPU')
            implementation.__module__ = module
            def dispatch(*a,**kw):
                return implementation(*a,**kw)
            dispatch.__wrapped__ = original
            return dispatch
        module = 'fla.test' if name.startswith('torch_') else 'causal_conv1d.test'
        monkeypatch.setattr(qwen,name,factory(original,module))
    calls = []
    def qualify(functions,**kwargs):
        calls.append(kwargs['fast'])
        if kwargs['fast']:
            raise RuntimeError('finite gradient corruption')
        return dict(status='passed')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(kernels,'qualify',qualify)
    result = kernels.probe('auto',output=tmp_path/'kernels.json')
    assert calls == [True,False] and result['status'] == 'fallback'
    assert 'gradient corruption' in result['failed_candidate']['error']
    assert json.loads((tmp_path/'kernels.json').read_text()) == result


def test_later_run_rejects_different_kernel_fallback():
    from train.cloud.qualification import check_kernel_selection
    passed = dict(status='fast',dispatch={'chunk':'fla'},numerical_agreement={'status':'passed'})
    check_kernel_selection(passed,passed)
    with pytest.raises(ValueError,match='differs'):
        check_kernel_selection(dict(passed,status='fallback'),passed)


def test_bootstrap_enforces_total_provider_cost_before_unpack(monkeypatch):
    import sys
    from train.cloud import hf_job
    monkeypatch.setattr(sys,'argv',['hf_job.py','--bundle','unused','--bundle-sha256','unused',
        '--job-seconds','10800','--','--recipe','smoke','--hourly-usd','6'])
    with pytest.raises(ValueError,match='three hours'):
        hf_job.main()
