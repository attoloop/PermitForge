"""A small numerical check must catch finite wrong gradients, without changing dispatch."""
import inspect
import json

import pytest
import torch

from train.cloud import kernels


def reference():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    return inspect.unwrap(qwen.torch_chunk_gated_delta_rule)


def test_small_sanity_checks_outputs_state_and_every_input_gradient():
    result = kernels.numerical_check(reference(), device='cpu')
    assert result['status'] == 'passed'
    assert set(result['comparisons']) == {'out','state','dq','dk','dv','dg','dbeta','dh0'}
    assert result['shape'] == [1,65,2,128]


def test_sanity_accepts_a_single_bf16_rounding_step():
    # The first H200 run differed from the fp32 reference by 2**-7 on one dk element (relative L2 0.0043).
    def rounded(*args, **kwargs):
        def nudge(grad):
            grad = grad.contiguous().clone()
            grad.view(-1)[grad.abs().argmin()] += 2**-7
            return grad
        args[1].register_hook(nudge)
        return reference()(*args, **kwargs)
    assert kernels.numerical_check(rounded, device='cpu')['status'] == 'passed'


def test_tolerances_are_fla_test_chunk_error_ratios():
    # flash-linear-attention v0.5.2 tests/ops/test_gdn.py::test_chunk, assert_close error ratios.
    assert kernels.TOLERANCES == dict(out=.005, state=.005, dq=.008, dk=.008, dv=.008,
                                      dg=.02, dbeta=.02, dh0=.008)


def gate_noise(ratio):
    def noisy(*args, **kwargs):
        def add(grad):
            noise = torch.randn(grad.shape, generator=torch.Generator().manual_seed(7)).to(grad)
            return grad + noise*(ratio*grad.norm()/noise.norm())
        args[3].register_hook(add)
        return reference()(*args, **kwargs)
    return noisy


def test_gate_gradient_within_fla_ratio_passes():
    # The CUDA 12.9 H200 run measured dg relative L2 0.0161 (Attoloop/6ad46fb65ccb47ba0f7c2337).
    result = kernels.numerical_check(gate_noise(.016), device='cpu')
    assert result['comparisons']['dg']['error_ratio'] < kernels.TOLERANCES['dg']


def test_gate_gradient_beyond_fla_ratio_fails_and_reports_every_tensor():
    with pytest.raises(RuntimeError, match='numerical') as error:
        kernels.numerical_check(gate_noise(.03), device='cpu')
    assert "['dg']" in str(error.value)
    assert all(name in str(error.value) for name in kernels.TOLERANCES)


@pytest.mark.parametrize('fault', ['output','state','gradient','gate_gradient','nonfinite'])
def test_sanity_rejects_silent_gdn_corruption(fault):
    def corrupt(*args, **kwargs):
        out, state = reference()(*args, **kwargs)
        if fault == 'output':
            out = out + .25
        elif fault == 'state':
            state = state + .25
        elif fault == 'gradient':
            out = out.detach() + 2*(out-out.detach())
        elif fault == 'gate_gradient':
            args[3].register_hook(lambda grad: grad*2)
        else:
            out = out*float('nan')
        return out, state
    with pytest.raises(RuntimeError, match='numerical'):
        kernels.numerical_check(corrupt, device='cpu')


def test_probe_failure_records_error_and_does_not_replace_kernel(monkeypatch, tmp_path):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    before = qwen.torch_chunk_gated_delta_rule
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    calls = []
    def fail(fn, **kwargs):
        calls.append(fn)
        raise RuntimeError('injected bad gradient')
    monkeypatch.setattr(kernels,'numerical_check',fail)
    path = tmp_path/'kernels.json'
    with pytest.raises(RuntimeError, match='bad gradient'):
        kernels.probe(output=path)
    assert calls == [before] and qwen.torch_chunk_gated_delta_rule is before
    saved = json.loads(path.read_text())
    assert saved['status'] == 'failed' and 'bad gradient' in saved['error']


def test_cpu_probe_never_qualifies_gpu_kernels(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert kernels.probe()['status'] == 'gpu_unavailable'
