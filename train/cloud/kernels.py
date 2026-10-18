"""One linear-attention sanity check: python -m train.cloud.kernels --output kernels.json."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

from .run import write_json

if TYPE_CHECKING:
    from torch import Tensor


def recurrence(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
               initial_state: Tensor) -> tuple[Tensor, Tensor]:
    """Independent fp32 recurrence on the same quantized inputs as the selected kernel."""
    import torch
    q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
    q = q * (q.square().sum(-1, keepdim=True)+1e-6).rsqrt() * q.shape[-1]**-.5
    k = k * (k.square().sum(-1, keepdim=True)+1e-6).rsqrt()
    state = initial_state.float()
    output = []
    for i in range(q.shape[1]):
        state = state*g[:, i].exp()[..., None, None]
        delta = (v[:, i]-(state*k[:, i, ..., None]).sum(-2))*beta[:, i, ..., None]
        state = state+k[:, i, ..., None]*delta[..., None, :]
        output.append((state*q[:, i, ..., None]).sum(-2))
    return torch.stack(output, dim=1), state


# flash-linear-attention v0.5.2 tests/ops/test_gdn.py::test_chunk compares this kernel with a naive
# recurrence using these assert_close error ratios; the check adopts the library's own standard.
TOLERANCES = dict(out=.005, state=.005, dq=.008, dk=.008, dv=.008, dg=.02, dbeta=.02, dh0=.008)


def numerical_check(function: Callable, *, device: str = 'cuda') -> dict:
    """One bf16 chunk-boundary case; check output, state and all input gradients."""
    import torch
    started = time.monotonic()
    rng = torch.Generator(device=device).manual_seed(212)
    shape = (1, 65, 2, 128)
    def randn(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=rng)
    inputs = [randn(shape) for _ in range(3)] + [
        -torch.rand(shape[:3], device=device, generator=rng),
        torch.rand(shape[:3], device=device, dtype=torch.bfloat16, generator=rng),
        .1*randn((1, 2, 128, 128), torch.float32)]
    cotangents = (randn(shape), randn((1, 2, 128, 128), torch.float32))
    results = []
    for reference in (False, True):
        leaves = [x.detach().clone().requires_grad_() for x in inputs]
        if reference:
            out, state = recurrence(*leaves)
        else:
            out, state = function(*leaves[:5], initial_state=leaves[5],
                                 output_final_state=True, use_qk_l2norm_in_kernel=True)
        gradients = torch.autograd.grad((out, state), leaves, cotangents)
        results.append((out, state, *gradients))
    comparisons, failed = {}, []
    for name, actual, expected in zip(TOLERANCES, *results):
        a, b = actual.detach().float(), expected.detach().float()
        # FLA's error ratio: RMS of the difference over RMS of the reference.
        ratio = ((a-b).square().mean().sqrt()/(b.square().mean().sqrt()+1e-8)).item()
        comparisons[name] = dict(error_ratio=ratio, tolerance=TOLERANCES[name],
                                 max_abs_error=(a-b).abs().max().item())
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not ratio <= TOLERANCES[name]:
            failed.append(name)
    if failed:
        raise RuntimeError(f'Kernel numerical agreement failed: {failed}: {comparisons}')
    return dict(status='passed', shape=list(shape), seed=212, dtype='bfloat16', tolerances=TOLERANCES,
                comparisons=comparisons, seconds=time.monotonic()-started)


def probe(*, output: Path | None = None) -> dict:
    """Observe Transformers' selected implementation; never patch it or retry a failure."""
    import torch
    versions = {}
    for package in ('torch','triton','transformers','trl','peft','flash-linear-attention','fla-core'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    evidence = dict(status='gpu_unavailable', versions=versions)
    try:
        if torch.cuda.is_available():
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            function = qwen.torch_chunk_gated_delta_rule
            implementation = inspect.getclosurevars(function).nonlocals.get('implementation', function)
            evidence['implementation'] = implementation.__module__+'.'+implementation.__name__
            evidence['numerical_check'] = numerical_check(function)
            evidence['status'] = 'passed'
    except Exception as exc:
        evidence.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if output:
            write_json(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = probe(output=args.output)
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
