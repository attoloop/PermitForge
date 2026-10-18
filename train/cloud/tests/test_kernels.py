import inspect
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
import torch

from train.cloud import kernels

ROOT = Path(__file__).resolve().parents[3]
NAMES = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
         'causal_conv1d_fn', 'causal_conv1d_update')


@pytest.fixture(autouse=True)
def cpu_convolution_backward():
    # This host's oneDNN lacks bf16 backward; Torch's native CPU path supports it.
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def references():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    return SimpleNamespace(**{name: inspect.unwrap(getattr(qwen, name)) for name in NAMES})


def test_hopper_workaround_and_memory_monitor_are_locked():
    project = tomllib.loads((ROOT/'train/cloud/pyproject.toml').read_text())['project']
    extra = project['optional-dependencies']['fast-kernels']
    assert 'tilelang[nvcc]==0.1.15' in extra
    assert 'apache-tvm-ffi==0.1.12' in extra
    assert 'nvidia-cuda-nvcc==13.2.86' in extra
    assert 'nvidia-cuda-cccl==13.2.86' in extra
    assert 'nvidia-ml-py==13.580.82' in project['dependencies']


def test_numerical_qualification_accepts_reference_and_covers_all_gradients():
    result = kernels.qualify(references(), device='cpu')
    assert result['status'] == 'passed'
    for length in (64, 65):
        for name in ('out', 'state', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0'):
            assert f'chunk_{length}/{name}' in result['comparisons']
    for name in ('out', 'state'):
        assert 'decode/'+name in result['comparisons']
    for name in ('out', 'dx', 'dweight', 'dbias', 'update', 'cache'):
        assert 'conv/'+name in result['comparisons']
    assert all(row['max_abs_error'] == 0 for row in result['comparisons'].values())


@pytest.mark.parametrize('fault', ['output', 'state', 'gradient', 'gate_gradient', 'nonfinite'])
def test_numerical_qualification_rejects_silent_gdn_corruption(fault):
    ref = references()
    def corrupt(*args, **kwargs):
        out, state = ref.torch_chunk_gated_delta_rule(*args, **kwargs)
        if fault == 'output':
            out = out + .25
        elif fault == 'state':
            state = state + .25
        elif fault == 'gradient':
            out = out.detach() + 2 * (out - out.detach())
        elif fault == 'gate_gradient':
            args[3].register_hook(lambda grad: grad * 2)
        else:
            out = out * float('nan')
        return out, state
    candidate = SimpleNamespace(**vars(ref))
    candidate.torch_chunk_gated_delta_rule = corrupt
    with pytest.raises(RuntimeError, match='numerical agreement'):
        kernels.qualify(candidate, reference=ref, device='cpu')


@pytest.mark.parametrize('operation', ['torch_recurrent_gated_delta_rule', 'causal_conv1d_fn',
                                       'causal_conv1d_update'])
def test_numerical_qualification_rejects_decode_and_convolution_corruption(operation):
    ref = references()
    def corrupt(*args, **kwargs):
        result = getattr(ref, operation)(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0] + .25, result[1]
        return result + .25
    candidate = SimpleNamespace(**vars(ref))
    setattr(candidate, operation, corrupt)
    with pytest.raises(RuntimeError, match='numerical agreement'):
        kernels.qualify(candidate, reference=ref, device='cpu')


def test_off_forces_reference_even_when_fast_packages_are_importable(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    def forbidden(*args, **kwargs):
        raise AssertionError('fast kernel must not be used with off')
    for name in NAMES:
        wrapped = getattr(qwen, name)
        def factory(original):
            def dispatch(*args, **kwargs):
                return forbidden(*args, **kwargs)
            dispatch.__wrapped__ = inspect.unwrap(original)
            return dispatch
        monkeypatch.setattr(qwen, name, factory(wrapped))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(kernels, 'qualify', lambda functions, **kw: {'status': 'passed'})
    result = kernels.probe('off')
    assert result['status'] == 'fallback'
    assert all(value.startswith(qwen.__name__+'.') for value in result['dispatch'].values())
    assert qwen.torch_chunk_gated_delta_rule is inspect.unwrap(qwen.torch_chunk_gated_delta_rule)


def test_cpu_probe_never_qualifies_gpu_kernels(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    result = kernels.probe()
    assert result['status'] == 'gpu_unavailable'
    assert result.get('numerical_agreement') is None


def test_required_policy_rejects_reference_dispatch(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name, implementation in vars(references()).items():
        monkeypatch.setattr(qwen, name, implementation)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(kernels, 'qualify', lambda *a, **kw: {'status': 'passed'})
    with pytest.raises(RuntimeError, match='Required fast kernels'):
        kernels.probe('required')


@pytest.mark.parametrize('fault', ['gradient', 'cache'])
def test_numerical_qualification_rejects_convolution_backward_and_cache_corruption(fault):
    ref = references()
    candidate = SimpleNamespace(**vars(ref))
    if fault == 'gradient':
        def corrupt(*args, **kwargs):
            out = ref.causal_conv1d_fn(*args, **kwargs)
            return out.detach() + 2*(out-out.detach())
        candidate.causal_conv1d_fn = corrupt
    else:
        def corrupt(x, state, *args, **kwargs):
            out = ref.causal_conv1d_update(x, state, *args, **kwargs)
            state.add_(1)
            return out
        candidate.causal_conv1d_update = corrupt
    with pytest.raises(RuntimeError, match='numerical agreement'):
        kernels.qualify(candidate, reference=ref, device='cpu')

