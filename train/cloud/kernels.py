"""Qualify pinned Qwen kernels before model loading: python -m train.cloud.kernels."""
import argparse
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace

NAMES = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
         'causal_conv1d_fn', 'causal_conv1d_update')


def qualify(functions: ModuleType | SimpleNamespace, *, reference: SimpleNamespace | None = None,
            device: str = 'cuda', matrix: bool = False, fast: bool = False,
            evidence: dict | None = None) -> dict:
    """Compare identical bf16 inputs/cotangents with unwrapped, non-FLA Torch code."""
    if matrix:
        from .kernel_cases import qualify_matrix
        return qualify_matrix(functions,fast=fast,device=device,evidence=evidence)
    import torch
    if reference is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        reference = SimpleNamespace(**{name: inspect.unwrap(getattr(qwen, name)) for name in NAMES})
    generator = torch.Generator(device=device).manual_seed(212)
    comparisons = {}
    # bf16 kernels round intermediate products; bound both individual and aggregate error.
    tolerances = dict(rtol=.03, atol=.002, relative_l2=.02)

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    def compare(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
        if actual is None or expected is None or actual.shape != expected.shape:
            raise RuntimeError('Kernel numerical agreement failed: '+name+' missing tensor/shape')
        a, b = actual.detach().float(), expected.detach().float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('Kernel numerical agreement failed: '+name+' nonfinite tensor')
        error = a-b
        relative_l2 = (error.norm()/b.norm().clamp_min(1e-12)).item()
        comparisons[name] = dict(max_abs_error=error.abs().max().item(), relative_l2_error=relative_l2)
        if (not torch.allclose(a, b, rtol=tolerances['rtol'], atol=tolerances['atol'])
                or relative_l2 > tolerances['relative_l2']):
            raise RuntimeError(f'Kernel numerical agreement failed: {name}: {comparisons[name]}')

    def run_chunk(fn: Callable, tensors: list[torch.Tensor],
                  cotangents: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inputs = [t.detach().clone().requires_grad_() for t in tensors]
        out, state = fn(*inputs[:5], initial_state=inputs[5], output_final_state=True,
                        use_qk_l2norm_in_kernel=True)
        grads = torch.autograd.grad((out, state), inputs, cotangents)
        return (out, state, *grads)

    started = time.monotonic()
    for length in (64, 65):
        tensors = [randn(1, length, 2, 32) for _ in range(3)]
        tensors += [-torch.rand(1, length, 2, device=device, generator=generator),
                    torch.rand(1, length, 2, device=device, dtype=torch.bfloat16, generator=generator),
                    .1*randn(1, 2, 32, 32, dtype=torch.float32)]
        cotangents = (randn(1, length, 2, 32), randn(1, 2, 32, 32, dtype=torch.float32))
        actual = run_chunk(functions.torch_chunk_gated_delta_rule, tensors, cotangents)
        expected = run_chunk(reference.torch_chunk_gated_delta_rule, tensors, cotangents)
        for name, a, b in zip(('out', 'state', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0'), actual, expected):
            compare(f'chunk_{length}/'+name, a, b)

    with torch.no_grad():
        # Decode from the independently computed prefill states, including cache agreement.
        args = [t[:, :1].detach().clone() for t in tensors[:5]]
        actual_decode = functions.torch_recurrent_gated_delta_rule(*args, initial_state=actual[1].clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected_decode = reference.torch_recurrent_gated_delta_rule(*args, initial_state=expected[1].clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        for name, a, b in zip(('out', 'state'), actual_decode, expected_decode):
            compare('decode/'+name, a, b)

    # Channel-first contiguous input meets the native convolution stride requirements.
    conv_inputs = [randn(1, 64, 65), randn(64, 4), randn(64)]
    dout = randn(1, 64, 65)
    conv_results = []
    for implementation in (functions, reference):
        x, weight, bias = [t.detach().clone().requires_grad_() for t in conv_inputs]
        out = implementation.causal_conv1d_fn(x, weight, bias=bias, activation='silu')
        grads = torch.autograd.grad(out, (x, weight, bias), dout)
        with torch.no_grad():
            cache = x[:, :, -4:].detach().clone()
            update = implementation.causal_conv1d_update(x[:, :, :1].detach().clone(), cache,
                weight.detach(), bias=bias.detach(), activation='silu')
        conv_results.append((out, *grads, update, cache))
    for name, a, b in zip(('out', 'dx', 'dweight', 'dbias', 'update', 'cache'), *conv_results):
        compare('conv/'+name, a, b)
    if device == 'cuda':
        torch.cuda.synchronize()
    return dict(status='passed', seed=212, dtype='bfloat16', tolerances=tolerances,
                comparisons=comparisons, seconds=time.monotonic()-started)


def probe(policy: str = 'auto', *, output: Path | None = None) -> dict:
    import torch
    if policy not in ('auto', 'required', 'off'):
        raise ValueError('Unknown kernel policy')
    versions = {}
    for package in ('torch', 'triton', 'transformers', 'flash-linear-attention', 'fla-core',
                    'causal-conv1d', 'tilelang', 'nvidia-cuda-nvcc', 'nvidia-cuda-cccl', 'nvidia-ml-py'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    evidence = dict(policy=policy, versions=versions, status='gpu_unavailable', dispatch={})
    if not torch.cuda.is_available():
        return evidence
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if policy == 'off':
        for name in NAMES:
            setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    for name in NAMES:
        wrapped = getattr(qwen, name)
        implementation = inspect.getclosurevars(wrapped).nonlocals.get('implementation', wrapped)
        evidence['dispatch'][name] = implementation.__module__+'.'+implementation.__name__
    full = (all(evidence['dispatch'][n].startswith('fla.') for n in NAMES[:2]) and
            all(evidence['dispatch'][n].startswith('causal_conv1d.') for n in NAMES[2:]))
    optimized = any(value.startswith(('fla.','causal_conv1d.')) for value in evidence['dispatch'].values())
    fast_chunk = evidence['dispatch'][NAMES[0]].startswith('fla.')
    evidence['status'] = 'fast' if full else 'fallback'
    if policy == 'required' and not full:
        evidence['status'] = 'failed'
        if output:
            output.write_text(json.dumps(evidence,indent=2)+'\n')
        raise RuntimeError('Required fast kernels unavailable: '+str(evidence['dispatch']))
    candidate = evidence['numerical_agreement'] = {}
    try:
        evidence['numerical_agreement'] = qualify(qwen,matrix=True,fast=fast_chunk,evidence=candidate)
    except Exception as exc:
        candidate.update(status='failed',error=type(exc).__name__+': '+str(exc))
        evidence['status'] = 'failed'
        if output:
            output.write_text(json.dumps(evidence,indent=2)+'\n')
        if policy == 'required' or not optimized:
            raise
        evidence['failed_candidate'] = evidence.pop('numerical_agreement')
        for name in NAMES:
            setattr(qwen,name,inspect.unwrap(getattr(qwen,name)))
        baseline = evidence['numerical_agreement'] = {}
        evidence['fallback_dispatch'] = {name:getattr(qwen,name).__module__+'.'+name for name in NAMES}
        try:
            evidence['numerical_agreement'] = qualify(qwen,matrix=True,fast=False,evidence=baseline)
            evidence['status'] = 'fallback'
        finally:
            if output:
                output.write_text(json.dumps(evidence,indent=2)+'\n')
    evidence['note'] = 'Real geometry, issue #640, seeds, padding, states and gradients; frozen tolerances.'
    if output:
        output.write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--policy', choices=('auto', 'required', 'off'), default='auto')
    args = p.parse_args()
    result = probe(args.policy,output=args.output)
    payload = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
