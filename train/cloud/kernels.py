"""Inspect local pinned Qwen Gated DeltaNet dispatch: python -m train.cloud.kernels."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import time


def probe(policy: str = 'auto') -> dict:
    import torch
    versions = {}
    for package in ('torch','triton','flash-linear-attention','fla-core','causal-conv1d'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    evidence = dict(policy=policy, versions=versions, status='gpu_unavailable', dispatch={})
    if not torch.cuda.is_available():
        return evidence
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
             'causal_conv1d_fn', 'causal_conv1d_update')
    for name in names:
        wrapped = getattr(qwen,name)
        implementation = inspect.getclosurevars(wrapped).nonlocals.get('implementation', wrapped)
        evidence['dispatch'][name] = implementation.__module__+'.'+implementation.__name__
    full = (all(evidence['dispatch'][n].startswith('fla.') for n in names[:2]) and
            all(evidence['dispatch'][n].startswith('causal_conv1d.') for n in names[2:]))
    evidence['status'] = 'fast' if full else 'fallback'
    # Exercise the installed training kernel, its gradient and recurrent decode, before loading 27B.
    q,k,v = [torch.randn(1,64,2,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    g = -torch.rand(1,64,2,device='cuda')
    beta = torch.rand(1,64,2,device='cuda',dtype=torch.bfloat16)
    started = time.monotonic()
    out,state = qwen.torch_chunk_gated_delta_rule(q,k,v,g,beta,output_final_state=True,
                                                use_qk_l2norm_in_kernel=True)
    out.float().square().mean().backward()
    decoded,_ = qwen.torch_recurrent_gated_delta_rule(q[:,:1].detach(),k[:,:1].detach(),
        v[:,:1].detach(),g[:,:1],beta[:,:1],initial_state=state.detach(),output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    x = torch.randn(1,64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    weight = torch.randn(64,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    conv = qwen.causal_conv1d_fn(x,weight,activation='silu')
    conv.float().square().mean().backward()
    with torch.no_grad():
        update = qwen.causal_conv1d_update(x[:,:,:1].detach(),x[:,:,-4:].detach().clone(),
                                          weight.detach(),activation='silu')
    torch.cuda.synchronize()
    if any(not torch.isfinite(t).all() for t in (out,decoded,conv,update)) or any(
            t.grad is None or not torch.isfinite(t.grad).all() for t in (q,k,v,x,weight)):
        raise RuntimeError('Gated DeltaNet kernel qualification produced nonfinite outputs/gradients')
    evidence.update(forward_backward_decode_seconds=time.monotonic()-started,
        note='Tiny dispatch/finite-gradient check includes JIT; full smoke establishes memory and step cost.')
    return evidence


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    result = probe()
    payload = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
    print(payload,end='')


if __name__ == '__main__':
    main()
