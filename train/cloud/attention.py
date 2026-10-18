"""Qualify only the pinned FA3 full-attention artifact; leave GDN dispatch alone."""
import base64
import hashlib
from pathlib import Path
import time

from .kernel_cases import compare

FA3_REPO = 'kernels-community/flash-attn3'
FA3_REVISION = '3c1451f803c146c54222305a8c350f7f46bb5135'
FA3 = FA3_REPO+'@'+FA3_REVISION
FA3_BINARY = '_flash_attn3_cuda_373a45c.abi3.so'
FA3_BINARY_SHA256 = base64.b64decode('iqm7SeF0hyvk4suMJgjy1m3im1E/cE+42Rdx+c0531Y=').hex()


def qualify_attention(evidence: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.hub_kernels import load_and_register_attn_kernel
    start = time.monotonic()
    kernel = load_and_register_attn_kernel(FA3)
    binary = Path(kernel.__file__).parent/FA3_BINARY
    with binary.open('rb') as handle:
        digest = hashlib.file_digest(handle,'sha256').hexdigest()
    if digest != FA3_BINARY_SHA256:
        raise RuntimeError('Pinned FA3 binary digest mismatch')
    evidence.update(artifact=FA3,binary_sha256=digest,implementation=kernel.flash_attn_varlen_func.__module__,
                    geometry=dict(query_heads=24,kv_heads=4,head_dim=256),comparisons={})
    for seed in (42,212):
        for name,qlens,klens in (('prefill',(64,64),(64,64)),('padding',(63,65),(63,65)),
                                ('decode',(1,1),(65,129))):
            rng = torch.Generator(device='cuda').manual_seed(seed)
            inputs = [torch.randn(shape,device='cuda',dtype=torch.bfloat16,generator=rng)
                      for shape in ((sum(qlens),24,256),(sum(klens),4,256),(sum(klens),4,256))]
            cotangent = torch.randn(inputs[0].shape,device='cuda',dtype=torch.bfloat16,generator=rng)
            q,k,v = [x.detach().clone().requires_grad_() for x in inputs]
            cuq = torch.tensor([0,qlens[0],sum(qlens)],dtype=torch.int32,device='cuda')
            cuk = torch.tensor([0,klens[0],sum(klens)],dtype=torch.int32,device='cuda')
            out = kernel.flash_attn_varlen_func(q,k,v,cuq,cuk,max(qlens),max(klens),causal=True)
            if isinstance(out,tuple):
                out = out[0]
            grads = torch.autograd.grad(out,(q,k,v),cotangent)
            qr,kr,vr = [x.float().detach().requires_grad_() for x in inputs]
            parts = []
            qi,ki = 0,0
            with sdpa_kernel(SDPBackend.MATH):
                for nq,nk in zip(qlens,klens,strict=True):
                    mask = torch.arange(nk,device='cuda')[None,:] <= (
                        torch.arange(nq,device='cuda')[:,None]+nk-nq)
                    parts.append(F.scaled_dot_product_attention(qr[qi:qi+nq].transpose(0,1),
                        kr[ki:ki+nk].transpose(0,1),vr[ki:ki+nk].transpose(0,1),
                        attn_mask=mask,enable_gqa=True).transpose(0,1))
                    qi,ki = qi+nq,ki+nk
            ref = torch.cat(parts)
            ref_grads = torch.autograd.grad(ref,(qr,kr,vr),cotangent.float())
            for part,a,b in zip(('out','dq','dk','dv'),(out,*grads),(ref,*ref_grads),strict=True):
                compare(f'{seed}/{name}/{part}',a,b,evidence['comparisons'])
    torch.cuda.synchronize()
    evidence['seconds'] = time.monotonic()-start
    return evidence


def set_attention(model, name: str) -> dict:
    model.set_attn_implementation(name)
    text = model.config.get_text_config()
    if text._attn_implementation != name:
        raise RuntimeError('Requested attention backend was not applied')
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    function = ALL_ATTENTION_FUNCTIONS[name]
    return dict(requested=name,wrapper=function.__module__+'.'+function.__name__)
