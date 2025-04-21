#PRWKV cxa073 Implementation
#2025 OpenMOSE

from .config import LAYER_CONFIG
from .linears import make_linear_att,make_linear_ffn,make_linear_head,make_emb,make_linear_ffn_experts,QuantLinear

import functools
import os, math, gc, importlib
import torch
import time
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.cpp_extension import load
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from einops import rearrange

allow_ops_in_compiled_graph()

# about FLA
#from fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7

# for infctx
from .infctx_module import *

#global
HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
HEAD = int(os.environ["RWKV_HEAD"])
BATCH_SIZE = int(os.environ["RWKV_MIRCO_BSZ"])
KernelMode = int(os.environ["FLA_MODE"])

ModelGeneration = os.environ["RWKV_MY_TESTING"] #x060, x070, xa070
RunningDevice = os.environ["RWKV_MY_ARCHITECTURE"] # cuda, rocm
TrainingType = os.environ["RWKV_TRAIN_TYPE"]

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop


if 'xa07' in ModelGeneration:
    print('PRWKV v7 Mode')
    if 'flatriton' in RunningDevice or 'infctx' in TrainingType or ('state' in TrainingType and KernelMode == 1): #infctx or FLA Mode
        print('cx070 Flash-Linear-Attention Kernel Mode')
        @torch.jit.ignore
        def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            w = -w.float().exp().to(r)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
            return o, state
        def RUN_RWKV7_INFCTX(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            w = -w.float().exp().to(r)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
            return o, state
        def RUN_RWKV7_RECURRENT(r, k, v, w, a, b, s, HEAD_SIZE=64): # for sampling
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            w = -w.float().exp().to(r)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            o, state = fused_recurrent_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
            return o, state
        @torch.jit.ignore
        def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64): #compatible with cuda implement
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            w = -w.float().exp().to(r)
            o, _ = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False, head_first=False)
            return o
    else:
        if 'cuda' in RunningDevice:
            print('cxa070 Wind CUDA Kernel Mode')
            CHUNK_LEN = 16
            flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

            class WindBackstepping(torch.autograd.Function):
                @staticmethod
                def forward(ctx, w,q,k,v,z,b,fp32mode=False):
                    B,T,H,C = w.shape 
                    #print(f'T = {T} CHUNK_LEN = {CHUNK_LEN}')
                    assert T%CHUNK_LEN == 0
                    if fp32mode:
                        w=w.to(dtype=torch.bfloat16)
                        q=q.to(dtype=torch.bfloat16)
                        k=k.to(dtype=torch.bfloat16)
                        v=v.to(dtype=torch.bfloat16)
                        z=z.to(dtype=torch.bfloat16)
                        b=b.to(dtype=torch.bfloat16)
                        fpmode = torch.tensor(32)
                    else:
                        fpmode = torch.tensor(16)

                    assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
                    assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)

                    torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
                    ctx.save_for_backward(w,q,k,v,z,b,s,sa,fpmode)
                    if fpmode == torch.tensor(32):
                        return y.float()
                    return y
                @staticmethod
                def backward(ctx, dy):
                    
                    assert all(i.is_contiguous() for i in [dy])
                    w,q,k,v,z,b,s,sa,fpmode = ctx.saved_tensors
                    if fpmode == torch.tensor(16):
                        assert all(i.dtype==torch.bfloat16 for i in [dy])
                    else:
                        dy = [i.to(torch.bfloat16) if i.is_contiguous() else i.contiguous().to(torch.bfloat16) for i in dy]
                        if isinstance(dy, list):
                            dy = torch.cat(dy, dim=0)

                    dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
                    torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
                    if fpmode == torch.tensor(32):
                        return dw.float(),dq.float(),dk.float(),dv.float(),dz.float(),db.float()
                    return dw,dq,dk,dv,dz,db

            def RUN_CUDA_RWKV7g(q,w,k,v,a,b,HEAD_SIZE=64):
                B,T,HC = q.shape
                q,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [q,w,k,v,a,b]]
                return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
            
            def RUN_RWKV7_RECURRENT(r, k, v, w, a, b, s, HEAD_SIZE=64): # for sampling
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                w=-torch.exp(w)
                #w = -w.float().exp().to(r)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                o, state = fused_recurrent_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
                #o = rearrange(o, 'b h l d -> b l (h d)')
                return o, state
            
        else:
            if 'bigheadtriton' in RunningDevice:
                from  .tritonbighead import RUN_CUDA_RWKV7g
            elif 'backstepping_longhead' in RunningDevice:
                import os, torch as th
                from torch.utils.cpp_extension import load

                print('backstepping longhead mode with CUDA or HIP')

                CHUNK_LEN = 16

                class RWKV7_longhead(th.autograd.Function):
                    @staticmethod
                    def forward(ctx, q,w,k,v,a,b,s0):
                        B,T,H,C = w.shape
                        assert T%CHUNK_LEN == 0
                        if not th.compiler.is_compiling():
                            assert hasattr(th.ops.wind_backstepping_longhead, 'forward'), 'Requires a load kernel from load_backstepping_longhead(head_size)'
                            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
                            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
                            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
                            assert list(s0.shape) == [B,H,C,C]
                        B,T,H,C = w.shape
                        y = th.empty_like(v)
                        sT = th.empty_like(s0)
                        if any(i.requires_grad for i in [w,q,k,v,a,b,s0]):
                            s = th.empty(B,H,T//CHUNK_LEN,C,C, dtype=th.float32,device=w.device)
                            sa = th.empty(B,T,H,C, dtype=th.float32,device=w.device)
                        else:
                            s = sa = None
                        th.ops.wind_backstepping_longhead.forward(w,q,k,v,a,b, s0,y,s,sa,sT)
                        ctx.save_for_backward(w,q,k,v,a,b,s,sa)
                        return y, sT
                    @staticmethod
                    def backward(ctx, dy, dsT):
                        w,q,k,v,a,b,s,sa = ctx.saved_tensors
                        B,T,H,C = w.shape
                        if not th.compiler.is_compiling():
                            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
                            assert all(i.is_contiguous() for i in [dy,dsT])

                        dv,ds0 = [th.empty_like(x) for x in [v,dsT]]
                        dw,dq,dk,da,db = [th.zeros(B,T,H,C, device=w.device) for i in range(5)]
                        th.ops.wind_backstepping_longhead.backward(w,q,k,v,a,b, dy,s,sa,dsT, dw,dq,dk,dv,da,db,ds0)
                        return dq,dw,dk,dv,da,db,ds0

                def attn_backstepping_longhead(r,w,k,v,a,b, s0 = None):
                    B,T,H,C = w.shape
                    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
                    return RWKV7_longhead.apply(r,w,k,v,a,b, s0)

                def load_backstepping_longhead(head_size, batchsz_times_heads_estimate = 8*64):
                    if hasattr(th.ops.wind_backstepping_longhead, 'forward'): return
                    device_props = th.cuda.get_device_properties(th.cuda.current_device())
                    if 'AMD' in device_props.name:
                        value_chunk_size = 16
                        CUDA_FLAGS = [f'-D_C_={head_size}', f'-D_K_={value_chunk_size}', f'-D_CHUNK_LEN_={CHUNK_LEN}', '-O3', '-ffast-math', '-DAMD', '--offload-arch=gfx942']
                    else:
                        value_chunk_size = 32
                        if th.cuda.get_device_properties(th.cuda.current_device()).multi_processor_count >= batchsz_times_heads_estimate * head_size / 32:
                            value_chunk_size = 32
                        CUDA_FLAGS = ['-res-usage', f'-D_C_={head_size} -D_K_={value_chunk_size}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
                    path = os.path.dirname(__file__) + '/../cuda/'
                    load(name="wind_backstepping_longhead", sources=[os.path.join(path,'backstepping_longhead.cu'), os.path.join(path,'backstepping_longhead.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
                    assert hasattr(th.ops.wind_backstepping_longhead, 'forward')

                load_backstepping_longhead(HEAD_SIZE,BATCH_SIZE*HEAD)

                def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE, mask=None,  dot_prec = 'fp32'):
                    #mask and dot_prec is dummy
                    B,T,HC = w.shape
                    C = HEAD_SIZE
                    H = HC//C
                    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
                    return attn_backstepping_longhead(r,w,k,v,a,b,s0)[0].view(B,T,HC)
            else:
                print('cxa 070 Wind Triton Kernel Mode')
        
                import torch as th
                import triton
                import triton.language as tl

                @triton.jit
                def IND4(a,b,c,d,nb,nc,nd):
                    return ((a*nb+b)*nc+c)*nd+d
                @triton.jit
                def IND5(a,b,c,d,e,nb,nc,nd,ne):
                    return (((a*nb+b)*nc+c)*nd+d)*ne+e

                @triton.jit
                def _prod(a,b): return a*b

                # inv(I-A) where A is a strictly lower triangular nxn matrix
                @triton.jit
                def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
                    i = tl.arange(0,n)
                    prod = (i[None,:]==i[:,None]).to(tl.float32)
                    for j in range(n-1):
                        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
                    return prod.trans()

                @triton.jit
                def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
                    bi = tl.program_id(1)
                    hi = tl.program_id(0)

                    i = tl.arange(0,C)[None,:]
                    state = tl.load(s0_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)
                    for t0 in range(T//dT):
                        t = t0*dT+tl.arange(0,dT)[:,None]
                        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

                        w = (-sw.exp()).exp()
                        fw = tl.reduce(w, 0, _prod, keep_dims=True)
                        incl_pref = tl.cumprod(w,axis=0)
                        non_incl_pref = incl_pref / w
                        inv_incl_pref = 1 / incl_pref

                        wq = sq * incl_pref
                        wa = sa * non_incl_pref
                        kwi = sk * inv_incl_pref
                        bwi = sb * inv_incl_pref

                        mask1 = (t > t.trans())
                        ab = tl_dot(prec, wa, bwi.trans()) * mask1
                        ak = tl_dot(prec, wa, kwi.trans()) * mask1

                        ab_inv = tri_minv(ab, dT, prec)

                        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
                        u = tl_dot(prec, ab_inv, ab_u)
                        mask2 = (t >= t.trans())
                        qk = tl_dot(prec, wq, kwi.trans()) * mask2
                        qb = tl_dot(prec, wq, bwi.trans()) * mask2
                        yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + tl_dot(prec, wq, state.trans())
                        tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

                        tl.store(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C), state.to(tl.float32))
                        state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
                    tl.store(sT_+IND4(bi,hi,i.trans(),i, H,C,C), state.to(tl.bfloat16))

                @triton.jit
                def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_, dw_,dq_,dk_,dv_,da_,db_,ds0_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
                    bi = tl.program_id(1)
                    hi = tl.program_id(0)

                    i = tl.arange(0,C)[None,:]
                    dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)

                    for t0 in range(T//dT-1,-1,-1):
                        t = t0*dT+tl.arange(0,dT)[:,None]

                        state = tl.load(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C)).to(tl.float32)

                        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                        sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

                        dw_fac = -sw.exp()
                        w = dw_fac.exp()
                        fw = tl.reduce(w, 0, _prod, keep_dims=True)
                        incl_pref = tl.cumprod(w,axis=0)
                        non_incl_pref = incl_pref / w
                        inv_incl_pref = 1 / incl_pref

                        wq = sq * incl_pref
                        wa = sa * non_incl_pref
                        kwi = sk * inv_incl_pref
                        bwi = sb * inv_incl_pref

                        mask1 = (t > t.trans())
                        ab = tl_dot(prec, wa, bwi.trans()) * mask1
                        ak = tl_dot(prec, wa, kwi.trans()) * mask1

                        ab_inv = tri_minv(ab, dT, prec)

                        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
                        u = tl_dot(prec, ab_inv, ab_u)
                        mask2 = (t >= t.trans())
                        qk = tl_dot(prec, wq, kwi.trans()) * mask2
                        qb = tl_dot(prec, wq, bwi.trans()) * mask2

                        du = tl_dot(prec, qb.trans(), sdy) + tl_dot(prec, bwi*fw, dstate.trans())
                        dab_u = tl_dot(prec, ab_inv.trans(), du)

                        dv = tl_dot(prec, qk.trans(), sdy) + tl_dot(prec, kwi*fw, dstate.trans()) + tl_dot(prec, ak.trans(), dab_u)
                        tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

                        dab = tl_dot(prec, tl_dot(prec, ab_inv.trans(), du), u.trans()) * mask1
                        dak = tl_dot(prec, dab_u, sv.trans()) * mask1
                        dab_u_state = tl_dot(prec, dab_u, state)
                        da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
                        tl.store(da_+IND4(bi,t,hi,i, T,H,C), da.to(tl.bfloat16))

                        dqb = tl_dot(prec, sdy, u.trans()) * mask2
                        dqk = tl_dot(prec, sdy, sv.trans()) * mask2
                        dy_state = tl_dot(prec, sdy, state)
                        dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
                        tl.store(dq_+IND4(bi,t,hi,i, T,H,C), dq.to(tl.bfloat16))

                        fw_u_dstate = fw * tl_dot(prec, u, dstate)
                        db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
                        tl.store(db_+IND4(bi,t,hi,i, T,H,C), db.to(tl.bfloat16))

                        fw_v_dstate = fw * tl_dot(prec, sv, dstate)
                        dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
                        tl.store(dk_+IND4(bi,t,hi,i, T,H,C), dk.to(tl.bfloat16))

                        dw0 = fw * tl.sum(state*dstate, axis=0,keep_dims=True)
                        for k in range(t0*dT,t0*dT+dT):
                            lmask = (t<k).trans()
                            A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
                            A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
                            A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
                            A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
                            dw = tl.sum(A, axis=0,keep_dims=True) + dw0

                            wk = tl.load(w_+IND4(bi,k,hi,i, T,H,C)).to(tl.float32)
                            dw *= -wk.exp()
                            tl.store(dw_+IND4(bi,k,hi,i, T,H,C), dw.to(tl.bfloat16))

                        dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
                    tl.store(ds0_+IND4(bi,hi,i.trans(),i, H,C,C), dstate.to(tl.bfloat16))


                class TritonRWKV7(th.autograd.Function):
                    @staticmethod
                    def forward(ctx, w,q,k,v,z,b,s0, dot_prec):
                        K = 16
                        B,T,H,C = w.shape
                        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
                        y = th.empty_like(v)
                        sT = th.empty_like(s0)
                        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
                        fw_attn_triton[(H,B)](w,q,k,v,z,b, s0,y,s,sT, B,T,H,C,K, dot_prec)
                        ctx.dot_prec = dot_prec
                        ctx.save_for_backward(w,q,k,v,z,b,s)
                        return y, sT
                    @staticmethod
                    def backward(ctx, dy, dsT):
                        K = 16
                        w,q,k,v,z,b,s = ctx.saved_tensors
                        B,T,H,C = w.shape
                        dw,dq,dk,dv,dz,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,z,b,dsT]]
                        bw_attn_triton[(H,B)](w,q,k,v,z,b, dy,s,dsT, dw,dq,dk,dv,dz,db,ds0, B,T,H,C,K, ctx.dot_prec)
                        return dw,dq,dk,dv,dz,db,ds0,None

                @triton.jit
                def tl_dot(prec:tl.constexpr, a, b):
                    if prec == 'fp32':
                        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
                    elif prec == 'tf32':
                        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
                    elif prec == 'bf16':
                        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
                    else:
                        tl.static_assert(False)

                def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64, dot_prec = 'fp32'):
                    B,T,HC = w.shape
                    C = HEAD_SIZE
                    H = HC//C
                    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
                    return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)
                def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64, dot_prec = 'fp32'):
                    B,T,HC = w.shape
                    C = HEAD_SIZE
                    H = HC//C
                    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                    s0 = s
                    return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC), None
            
    
        
    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            """
            Qwen2RMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
        

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads along the head dimension (GQA).
        Input:  (B, T, H_kv, D)
        Output: (B, T, H_kv * n_rep, D)
        """
        B, T, H_kv, D = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        # Expand head dim
        hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
        hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
        return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()


    
    class PRWKV_Tmix_cxa075(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.my_testing = args.my_testing

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
            self.kv_n_head = args.gqa_kv_heads
            assert args.dim_att % self.n_head == 0
            H = self.n_head
            N = self.head_size
            C = args.n_embd

            with torch.no_grad():
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, C)
                for i in range(C):
                    ddd[0, 0, i] = i / C

                def ortho_init(x, scale):
                    with torch.no_grad():
                        shape = x.shape
                        if len(shape) == 2:
                            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                            nn.init.orthogonal_(x, gain=gain * scale)
                        elif len(shape) == 3:
                            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                            for i in range(shape[0]):
                                nn.init.orthogonal_(x[i], gain=gain * scale)
                        else:
                            assert False
                        return x

                D_DECAY_LORA = 64
                D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
                self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
                decay_speed = torch.ones(C)
                for n in range(C):
                    decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

                D_AAA_LORA = 64
                D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
                self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
                self.a0 = nn.Parameter(torch.zeros(1,1,C))

                D_MV_LORA = 32
                D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

                D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
                # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

                self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
                self.k_a = nn.Parameter(torch.ones(1,1,C))
                self.r_k = nn.Parameter(torch.zeros(H,N))

                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

                Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

                self.receptance = make_linear_att(C, C, bias=True,n_layer=self.layer_id,pname='att.receptance')
                #GQAStyle
                self.key = make_linear_att(C, self.head_size*self.kv_n_head, bias=True,n_layer=self.layer_id,pname='att.key')
                self.value = make_linear_att(C, self.head_size*self.kv_n_head, bias=True,n_layer=self.layer_id,pname='att.value')
                self.output = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.output')

          

        #@torch.compile
        def forward(self, x, v_first,passthrough = False):
            B, T, C = x.size()
            H = self.n_head
        
            xr = xw = xk = xv = xa = xg = x            

            r = self.receptance(xr,passthrough)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk,passthrough)
            v = self.value(xv,passthrough)

            g_delta = torch.sigmoid(xg @ self.g1) @ self.g2
            g = 1.0 + g_delta

            k = k.view(B, T, self.kv_n_head, self.head_size)
            v = v.view(B, T, self.kv_n_head, self.head_size)

            # repeat k/v heads if n_kv_heads < n_heads
            #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
            k = repeat_kv(k, self.n_head // self.kv_n_head)#reshape(B,T,-1) #(B,T,C)
            v = repeat_kv(v, self.n_head // self.kv_n_head)#reshape(B,T,-1) #(B,T,C)

            k = k.view(B, T, -1)
            v = v.view(B, T, -1)

            #so now all B,T,C tensors

            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)

            k = k * (1 + (a-1) * self.k_a)

            x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,HEAD_SIZE=self.head_size).view(B, T, C)


            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x*g,passthrough)
            return x, v_first
        
        #@torch.no_grad() # for sampling ppo
        #@torch.compile
        def forward_rnn(self, x, v_first, last_state: TimeMixState,passthrough=False):
            
            B, T, C = x.size()
            H = self.n_head
            xx = self.time_shift(x) - x

            shift_state = last_state.shift_state
            wkv_state = last_state.wkv_state.clone().contiguous() 

            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

            xr = xw = xk = xv = xa = xg = x   

            shift_state = x[:, -1]

            r = self.receptance(xr,passthrough)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk,passthrough)
            v = self.value(xv,passthrough)

            g_delta = torch.sigmoid(xg @ self.g1) @ self.g2
            g = 1.0 + g_delta

            k = k.view(B, T, self.kv_n_head, self.head_size)
            v = v.view(B, T, self.kv_n_head, self.head_size)

            # repeat k/v heads if n_kv_heads < n_heads
            #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
            k = repeat_kv(k, self.n_head // self.kv_n_head)#reshape(B,T,-1) #(B,T,C)
            v = repeat_kv(v, self.n_head // self.kv_n_head)#reshape(B,T,-1) #(B,T,C)

            k = k.view(B, T, -1)
            v = v.view(B, T, -1)

            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
            k = k * (1 + (a-1) * self.k_a)

            x, wkv_state = RUN_RWKV7_RECURRENT(r,k,v,w,-kk, kk*a,s=wkv_state).view(B, T, C)

            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x*g,passthrough)

            return x, v_first,TimeMixState(shift_state,wkv_state)
        
    
    

    
        
    class Qwen2MLP(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.hidden_size = args.n_embd
            self.intermediate_size = args.dim_ffn

            self.gate = make_linear_ffn(self.hidden_size, self.intermediate_size, bias=False,n_layer=self.layer_id,pname='ffn.gate')
            self.up = make_linear_ffn(self.hidden_size, self.intermediate_size, bias=False,n_layer=self.layer_id,pname='ffn.up')
            self.down = make_linear_ffn(self.intermediate_size, self.hidden_size, bias=False,n_layer=self.layer_id,pname='ffn.down')

        def forward(self, x,passthrough=False):
            down_proj = self.down(F.silu(self.gate(x,passthrough)) * self.up(x,passthrough),passthrough)
            return down_proj
        
        def forward_rnn(self, x,last_state: ChannelMixState,passthrough=False):
            down_proj = self.down(F.silu(self.gate(x,passthrough)) * self.up(x,passthrough),passthrough)
            return down_proj, last_state
        
    class Phi35MLP(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.hidden_size = args.n_embd
            self.intermediate_size = args.dim_ffn

            self.gate_up = make_linear_ffn(self.hidden_size, self.intermediate_size*2, bias=False,n_layer=self.layer_id,pname='ffn.gate')
            self.down = make_linear_ffn(self.intermediate_size, self.hidden_size, bias=False,n_layer=self.layer_id,pname='ffn.down')

        def forward(self, x,passthrough=False):
            up_states = self.gate_up(x,passthrough)

            gate, up_states = up_states.chunk(2, dim=-1)
            up_states = up_states * F.silu(gate)

            return self.down(up_states,passthrough)
        
        def forward_rnn(self, x,last_state: ChannelMixState,passthrough=False):
            up_states = self.gate_up(x,passthrough)

            gate, up_states = up_states.chunk(2, dim=-1)
            up_states = up_states * F.silu(gate)

            return self.down(up_states,passthrough), last_state
        
    class Phi35MLP_infctx(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.hidden_size = args.n_embd
            self.intermediate_size = args.dim_ffn

            self.gate_up = make_linear_ffn(self.hidden_size, self.intermediate_size*2, bias=False,n_layer=self.layer_id,pname='ffn.gate')
            #self.up = make_linear_ffn(self.hidden_size, self.intermediate_size, bias=False,n_layer=self.layer_id,pname='ffn.up')
            self.down = make_linear_ffn(self.intermediate_size, self.hidden_size, bias=False,n_layer=self.layer_id,pname='ffn.down')
        #@torch.compile()
        def forward(self, x,last_state: ChannelMixState,passthrough=False):
            up_states = self.gate_up(x,passthrough)

            gate, up_states = up_states.chunk(2, dim=-1)
            up_states = up_states * F.silu(gate)

            return self.down(up_states,passthrough), last_state
        
 

        
        
    class Qwen2MLP_infctx(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.hidden_size = args.n_embd
            self.intermediate_size = args.dim_ffn

            self.gate = make_linear_ffn(self.hidden_size, self.intermediate_size, bias=False,n_layer=self.layer_id,pname='ffn.gate')
            self.up = make_linear_ffn(self.hidden_size, self.intermediate_size, bias=False,n_layer=self.layer_id,pname='ffn.up')
            self.down = make_linear_ffn(self.intermediate_size, self.hidden_size, bias=False,n_layer=self.layer_id,pname='ffn.down')

        def forward(self, x ,last_state: ChannelMixState,passthrough=False):
            down_proj = self.down(F.silu(self.gate(x,passthrough)) * self.up(x,passthrough),passthrough)
            return down_proj, last_state #dummy
        



