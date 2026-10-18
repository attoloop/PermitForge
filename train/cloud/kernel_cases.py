"""Fixed H200 geometry and numerical guards; run via python -m train.cloud.kernels."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import inspect
import sys
import time

TOLERANCES = dict(rtol=.03, atol=.002, relative_l2=.02)


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    length: int
    batch: int = 1
    key_heads: int = 16
    value_heads: int = 48
    key_dim: int = 128
    value_dim: int = 128
    left_padding: int = 0
    dtype: str = 'bfloat16'
    scale: float | None = None
    normalize: bool = True


CASES = tuple(Case(f'model_{seed}_{length}', seed, length,
                  batch=2 if length == 129 else 1, left_padding=7 if length == 129 else 0)
              for seed in (42,212) for length in (63,64,65,129)) + tuple(
    Case(f'issue640_{seed}',seed,1024,key_heads=4,value_heads=4,key_dim=100,value_dim=100,
         dtype='float16',scale=1.,normalize=False) for seed in (42,212))


def check_model_geometry(config) -> None:
    expected = dict(num_hidden_layers=64,num_attention_heads=24,num_key_value_heads=4,head_dim=256,
        linear_num_key_heads=16,linear_num_value_heads=48,linear_key_head_dim=128,
        linear_value_head_dim=128,linear_conv_kernel_dim=4)
    if any(getattr(config,k,None) != value for k,value in expected.items()):
        raise ValueError('Loaded model geometry differs from the frozen H200 probe')
    if list(config.layer_types).count('full_attention') != 16:
        raise ValueError('Loaded hybrid layer layout differs from the frozen H200 probe')


def compare(name, actual, expected, evidence: dict, *, tolerances: dict = TOLERANCES) -> None:
    import torch
    if actual is None or expected is None or actual.shape != expected.shape:
        evidence[name] = dict(passed=False, error='missing tensor/shape')
        raise RuntimeError('Kernel numerical agreement failed: '+name+' missing tensor/shape')
    a,b = actual.detach().float(),expected.detach().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        evidence[name] = dict(passed=False,error='nonfinite tensor')
        raise RuntimeError('Kernel numerical agreement failed: '+name+' nonfinite tensor')
    error = a-b
    relative = (error.norm()/b.norm().clamp_min(1e-12)).item()
    passed = bool(torch.allclose(a,b,rtol=tolerances['rtol'],atol=tolerances['atol']) and
                  relative <= tolerances['relative_l2'])
    evidence[name] = dict(passed=passed,max_abs_error=error.abs().max().item(),relative_l2_error=relative)
    if not passed:
        raise RuntimeError(f'Kernel numerical agreement failed: {name}: {evidence[name]}')


@contextmanager
def inner_calls():
    """Observe executed Python entry points, without changing backend dispatch."""
    counts = Counter()
    previous = sys.getprofile()
    def trace(frame, event, arg):
        if event == 'call':
            module = frame.f_globals.get('__name__','')
            name = frame.f_code.co_name
            if module.startswith('fla.ops.') and name.startswith('chunk_bwd_dqkwg'):
                counts[module+'.'+name] += 1
        if previous is not None:
            previous(frame,event,arg)
    sys.setprofile(trace)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def require_inner_backend(calls: dict) -> None:
    key = 'fla.ops.common.backends.tilelang.chunk_bwd.chunk_bwd_dqkwg_tilelang'
    if not calls.get(key,0):
        raise RuntimeError('TileLang inner backward did not execute: '+str(calls))


def recurrence(q,k,v,g,beta,initial_state,*,normalize=True,scale=None):
    """Independent fp32 token recurrence on the exact quantized test inputs."""
    import torch
    q,k,v,g,beta = [x.float() for x in (q,k,v,g,beta)]
    if normalize:
        q = q * (q.square().sum(-1,keepdim=True)+1e-6).rsqrt()
        k = k * (k.square().sum(-1,keepdim=True)+1e-6).rsqrt()
    q = q * (scale if scale is not None else q.shape[-1]**-.5)
    state = initial_state.float()
    out = []
    for i in range(q.shape[1]):
        state = state*g[:,i].exp()[...,None,None]
        delta = (v[:,i]-(state*k[:,i,...,None]).sum(-2))*beta[:,i,...,None]
        state = state+k[:,i,...,None]*delta[...,None,:]
        out.append((state*q[:,i,...,None]).sum(-2))
    return torch.stack(out,dim=1),state


def qualify_matrix(functions, *, fast: bool, device: str = 'cuda', cases=CASES,
                   evidence: dict | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    evidence = evidence if evidence is not None else {}
    evidence.update(tolerances=dict(TOLERANCES),cases=[],status='running')
    started = time.monotonic()
    for case in cases:
        rng = torch.Generator(device=device).manual_seed(case.seed)
        dtype = getattr(torch,case.dtype)
        def rand(shape, *, normal=True, dtype=dtype):
            fn = torch.randn if normal else torch.rand
            return fn(shape,device=device,dtype=dtype,generator=rng)
        b,t,h,hv,k,v = case.batch,case.length,case.key_heads,case.value_heads,case.key_dim,case.value_dim
        inputs = [rand((b,t,h,k),normal=case.normalize) for _ in range(2)]
        inputs += [rand((b,t,hv,v),normal=case.normalize),
                   -torch.rand((b,t,hv),device=device,generator=rng),
                   rand((b,t,hv),normal=False),.1*rand((b,hv,k,v),dtype=torch.float32)]
        if not case.normalize:
            inputs[3] = F.logsigmoid(rand((b,t,hv),normal=False,dtype=torch.float32))/.1
            inputs[4] = inputs[4].sigmoid()
        if case.left_padding:
            # A padded prefix is a no-op, including for the deliberately nonzero state.
            for tensor in inputs[:5]:
                tensor[0,:case.left_padding] = 0
        cotangents = (rand((b,t,hv,v)),rand((b,hv,k,v),dtype=torch.float32))
        row = dict(case=asdict(case),comparisons={},inner_backend={})
        evidence['cases'].append(row)
        def run(reference):
            leaves = [x.detach().clone().requires_grad_() for x in inputs]
            q,k = leaves[:2]
            if not case.normalize:
                q,k = [F.normalize(x,p=2,dim=-1) for x in (q,k)]
            q,k = [x.repeat_interleave(hv//h,dim=2) for x in (q,k)]
            if reference:
                out,state = recurrence(q,k,*leaves[2:],normalize=case.normalize,scale=case.scale)
            else:
                fn = functions.torch_chunk_gated_delta_rule
                kwargs = dict(initial_state=leaves[-1],output_final_state=True,
                              use_qk_l2norm_in_kernel=case.normalize)
                if case.scale is not None:
                    impl = inspect.getclosurevars(fn).nonlocals.get('implementation',fn)
                    if impl.__module__.startswith('transformers.'):
                        q = q.float()*(case.scale*case.key_dim**.5)
                    else:
                        kwargs['scale'] = case.scale
                out,state = fn(q,k,*leaves[2:5],**kwargs)
            grads = torch.autograd.grad((out,state),leaves,cotangents)
            return (out.detach(),state.detach(),*[x.detach() for x in grads])
        with inner_calls() as calls:
            actual = run(False)
        row['inner_backend'] = dict(calls)
        if fast:
            require_inner_backend(calls)
        expected = run(True)
        for name,a,e in zip(('out','state','dq','dk','dv','dg','dbeta','dh0'),actual,expected,strict=True):
            compare(name,a,e,row['comparisons'])
        with torch.no_grad():
            q,k = [x[:,-1:].repeat_interleave(hv//h,dim=2) for x in inputs[:2]]
            args = [q,k,*[x[:,-1:] for x in inputs[2:5]]]
            a = functions.torch_recurrent_gated_delta_rule(*args,initial_state=actual[1].clone(),
                output_final_state=True,use_qk_l2norm_in_kernel=True)
            e = recurrence(*args,expected[1].clone())
            for name,av,ev in zip(('decode_out','decode_state'),a,e,strict=True):
                compare(name,av,ev,row['comparisons'])
        del actual,expected
    # Real Q/K/V convolution width = 2*16*128 + 48*128, kernel size 4.
    conv = evidence['convolution'] = {}
    for seed in (42,212):
        torch.manual_seed(seed)
        data = [torch.randn(*shape,device=device,dtype=torch.bfloat16) for shape in
                ((2,10240,65),(10240,4),(10240,))]
        dout = torch.randn_like(data[0])
        results = []
        for impl in (functions,qwen):
            fn = impl.causal_conv1d_fn if impl is functions else inspect.unwrap(impl.causal_conv1d_fn)
            update = impl.causal_conv1d_update if impl is functions else inspect.unwrap(impl.causal_conv1d_update)
            x,w,bias = [x.detach().clone().requires_grad_() for x in data]
            out = fn(x,w,bias=bias,activation='silu')
            grads = torch.autograd.grad(out,(x,w,bias),dout)
            with torch.no_grad():
                state = x[:,:,-4:].clone()
                decoded = update(x[:,:,:1].clone(),state,w,bias=bias,activation='silu')
            results.append((out,*grads,decoded,state))
        for name,a,e in zip(('out','dx','dw','db','decode','state'),*results,strict=True):
            compare(f'{seed}/{name}',a,e,conv)
    if device == 'cuda':
        torch.cuda.synchronize()
    evidence.update(status='passed',seconds=time.monotonic()-started)
    return evidence
