#RWKV v7 Wind(CUDA,Triton),Flash-Linear-Attention(Triton)

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
from fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7

# for infctx
from .infctx_module import *

#global
HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
KernelMode = int(os.environ["FLA_MODE"])

ModelGeneration = os.environ["RWKV_MY_TESTING"] #x060, x070
RunningDevice = os.environ["RWKV_MY_ARCHITECTURE"] # cuda, rocm
TrainingType = os.environ["RWKV_TRAIN_TYPE"]

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop


if 'x070' in ModelGeneration:
    print('RWKV v7 Mode')
    if 'flatriton' in RunningDevice or 'infctx' in TrainingType or ('state' in TrainingType and KernelMode == 1): #infctx or FLA Mode
        print('x070 Flash-Linear-Attention Kernel Mode')
        @torch.jit.ignore
        def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            w = -w.float().exp().to(r)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            #s=s.unsqueeze(0).repeat(B, 1, 1, 1)
            #s = s.permute(0,1,3,2).contiguous()
            o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
            #state = state.permute(0,1,3,2).contiguous()
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
            w=-torch.exp(w)
            #w = -w.float().exp().to(r)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            o, state = fused_recurrent_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
            #o = rearrange(o, 'b h l d -> b l (h d)')
            return o, state
        # def RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, a, b, num_heads:int) -> torch.Tensor:
        #     B,T,_ = r.shape
        #     r,log_neglog_w,k,v,a,b = [i.view(B,T,num_heads,-1) for i in [r,log_neglog_w,k,v,a,b]]
        #     log_w = -log_neglog_w.float().exp().to(r)
        #     output, state = fused_recurrent_rwkv7(r=r, log_w=log_w, k=k, v=v, a=a, b=b, output_final_state=False)
        #     return output, state
        @torch.jit.ignore
        def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64): #compatible with cuda implement
            print('FLA chunk_wkv7')
            B,T,HC = w.shape
            C = HEAD_SIZE
            H = HC//C
            #w=-torch.exp(w)
            r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
            w = -w.float().exp().to(r)
            o, _ = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False, head_first=False)
            return o
    else:
        if 'cuda' in RunningDevice:
            print('x070 Wind CUDA Kernel Mode')
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
            print('x070 Wind Triton Kernel Mode')
    
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
            def RUN_RWKV7_STATE(r, w, k, v, a, b, s=None, HEAD_SIZE=64, dot_prec = 'fp32'):
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                if s is None:
                    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
                else:
                    s0 = s

                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                x, state = TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)
                return x.view(B,T,HC), state


    
    class RWKV_Tmix_x070(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.my_testing = args.my_testing

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
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

                self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
                self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
                self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
                self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

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
                if C == 2560 or C == 2048:
                    D_DECAY_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_DECAY_LORA = D_DECAY_LORA * 2
                # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
                self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
                decay_speed = torch.ones(C)
                for n in range(C):
                    decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

                D_AAA_LORA = 64
                if C == 2560 or C == 2048:
                    D_AAA_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    print('openmose mytesting mode')
                    D_AAA_LORA = D_AAA_LORA * 2
                # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
                self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
                self.a0 = nn.Parameter(torch.zeros(1,1,C))

                D_MV_LORA = 32
                if C == 2560 or C == 2048:
                    D_MV_LORA = 64
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_MV_LORA = D_MV_LORA * 4
                # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

                D_GATE_LORA = 128
                if C == 2560:
                    D_GATE_LORA = 320
                elif C == 2048:
                    D_GATE_LORA = 256
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_GATE_LORA = D_GATE_LORA * 4
                # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
                # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

                self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
                self.k_a = nn.Parameter(torch.ones(1,1,C))
                self.r_k = nn.Parameter(torch.zeros(H,N))

                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

                Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

               
                self.receptance = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.receptance')
                self.key = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.key')
                self.value = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.value')
                self.output = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.output')


                self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!



        def forward(self, x, v_first,passthrough = False):
            B, T, C = x.size()
            H = self.n_head
            xx = self.time_shift(x) - x

            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g

            

            r = self.receptance(xr,passthrough)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk,passthrough)
            v = self.value(xv,passthrough)
            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
            g = torch.sigmoid(xg @ self.g1) @ self.g2

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
            k = k * (1 + (a-1) * self.k_a)

            x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,HEAD_SIZE=self.head_size)

            x = self.ln_x(x.view(B * T, C)).view(B, T, C)

            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x * g,passthrough)
            return x, v_first
        
        #@torch.no_grad() # for sampling ppo
        @torch.compile
        def forward_rnn(self, x, v_first, last_state: TimeMixState,passthrough=False):
            
            B, T, C = x.size()
            H = self.n_head
            #xx = self.time_shift(x) - x

            shift_state = last_state.shift_state
            wkv_state = last_state.wkv_state.clone().contiguous() 

            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g

            shift_state = x[:, -1]

            r = self.receptance(xr,passthrough)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk,passthrough)
            v = self.value(xv,passthrough)
            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
            g = torch.sigmoid(xg @ self.g1) @ self.g2

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
            k = k * (1 + (a-1) * self.k_a)

            x, wkv_state = RUN_RWKV7_RECURRENT(r,k,v,w,-kk, kk*a,s=wkv_state)

            #x = x.view(B,T,-1).to(dtype=r.dtype)
            #x = x.permute(0, 2, 1).contiguous()  # (B,H*N,T)

            x = self.ln_x(x.view(B * T, C)).view(B, T, C)

            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x * g,passthrough)

            return x, v_first,TimeMixState(shift_state,wkv_state)
        
    
    class RWKV_Tmix_x070_state(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.my_testing = args.my_testing

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
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

                self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
                self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
                self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
                self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

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
                if C == 2560 or C == 2048:
                    D_DECAY_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_DECAY_LORA = D_DECAY_LORA * 2
                # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
                self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
                decay_speed = torch.ones(C)
                for n in range(C):
                    decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

                D_AAA_LORA = 64
                if C == 2560 or C == 2048:
                    D_AAA_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    print('openmose mytesting mode')
                    D_AAA_LORA = D_AAA_LORA * 2
                # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
                self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
                self.a0 = nn.Parameter(torch.zeros(1,1,C))

                D_MV_LORA = 32
                if C == 2560 or C == 2048:
                    D_MV_LORA = 64
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_MV_LORA = D_MV_LORA * 4
                # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

                D_GATE_LORA = 128
                if C == 2560:
                    D_GATE_LORA = 320
                elif C == 2048:
                    D_GATE_LORA = 256
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_GATE_LORA = D_GATE_LORA * 4
                # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
                # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

                self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
                self.k_a = nn.Parameter(torch.ones(1,1,C))
                self.r_k = nn.Parameter(torch.zeros(H,N))

                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

                #for State-tuning
                if self.args.direct_state_tuning:
                    self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))
                if self.args.post_wkv_tuning:
                    #self.time_state_offset = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))
                    self.time_offset = nn.Parameter(torch.zeros(args.n_embd))



                Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

                
                self.receptance = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.receptance')
                self.key = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.key')
                self.value = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.value')
                self.output = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.output')


                self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!



        def forward(self, x, v_first,passthrough):
            B, T, C = x.size()
            H = self.n_head
            xx = self.time_shift(x) - x

            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g

            r = self.receptance(xr,passthrough)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk,passthrough)
            v = self.value(xv,passthrough)
            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual


            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
            g = torch.sigmoid(xg @ self.g1) @ self.g2

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
            k = k * (1 + (a-1) * self.k_a)

            if self.args.direct_state_tuning:
                input_state = self.time_state.unsqueeze(0).repeat(B, 1, 1, 1)
                #print(input_state.shape)
                x, out_state = RUN_RWKV7_STATE(r,w,k,v,-kk, kk*a,s=input_state,HEAD_SIZE=self.head_size)
            else:
                x, out_state = RUN_RWKV7_STATE(r,w,k,v,-kk, kk*a,s=None,HEAD_SIZE=self.head_size)

            if self.args.post_wkv_tuning:
                x = x * (1 + self.time_offset)

            #print(x)

            x = self.ln_x(x.view(B * T, C)).view(B, T, C)

            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x * g,passthrough)

            return x, v_first, out_state
        

    class RWKV_Tmix_x070_infctx(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.my_testing = args.my_testing

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
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

                self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
                self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
                self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
                self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

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
                if C == 2560 or C == 2048:
                    D_DECAY_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_DECAY_LORA = D_DECAY_LORA * 2
                # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
                self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
                decay_speed = torch.ones(C)
                for n in range(C):
                    decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

                D_AAA_LORA = 64
                if C == 2560 or C == 2048:
                    D_AAA_LORA = 96
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    print('openmose mytesting mode')
                    D_AAA_LORA = D_AAA_LORA * 2
                # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
                self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
                self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
                self.a0 = nn.Parameter(torch.zeros(1,1,C))

                D_MV_LORA = 32
                if C == 2560 or C == 2048:
                    D_MV_LORA = 64
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_MV_LORA = D_MV_LORA * 4
                # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

                D_GATE_LORA = 128
                if C == 2560:
                    D_GATE_LORA = 320
                elif C == 2048:
                    D_GATE_LORA = 256
                elif C == 4096 and 'x070Upgraded' in args.my_testing:
                    D_GATE_LORA = D_GATE_LORA * 4
                # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
                # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

                self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
                self.k_a = nn.Parameter(torch.ones(1,1,C))
                self.r_k = nn.Parameter(torch.zeros(H,N))

                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

                #for State-tuning
                #self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

                Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

                
                self.receptance = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.receptance')
                self.key = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.key')
                self.value = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.value')
                self.output = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.output')


                self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!



        def forward(self, x, v_first, last_state: TimeMixState):
            B, T, C = x.size()
            H = self.n_head
            #xx = self.time_shift(x) - x
            
            shift_state = last_state.shift_state
            wkv_state = last_state.wkv_state.clone().contiguous() 

            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x


            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g

            #print(f'x shape = {x.shape}')

            shift_state = x[:,-1,:]

            r = self.receptance(xr)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
            k = self.key(xk)
            v = self.value(xv)
            if self.layer_id == 0:
                v_first = v # store the v of the first layer
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
            g = torch.sigmoid(xg @ self.g1) @ self.g2

            kk = k * self.k_k
            kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
            k = k * (1 + (a-1) * self.k_a)

            x , wkv_state = RUN_RWKV7_INFCTX(r,k,v,w,-kk, kk*a,wkv_state)

            x = self.ln_x(x.view(B * T, C)).view(B, T, C)

            x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
            x = self.output(x * g)
            
            return x, v_first,TimeMixState(shift_state,wkv_state)

    
        
    class RWKV_CMix_x070(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

     
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id,pname='ffn.key')
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.value')

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
            # self.value.weight.data.zero_()

        def forward(self, x,passthrough=False):
            xx = self.time_shift(x) - x
            #xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
            
            
            k = x + xx * self.x_k
            k = torch.relu(self.key(k,passthrough)) ** 2

            return self.value(k,passthrough)#,ChannelMixState(x[:, -1])
        
        @torch.no_grad() # for sampling ppo
        def forward_rnn(self, x,last_state: ChannelMixState,passthrough = False):
            #xx = self.time_shift(x) - x
            xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
            
            
            k = x + xx * self.x_k
            k = torch.relu(self.key(k,passthrough)) ** 2

            return self.value(k,passthrough),ChannelMixState(x[:, -1])
        
    class RWKV_CMix_x070_infctx(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']


            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id,pname='ffn.key')
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.value')

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
            # self.value.weight.data.zero_()

        def forward(self, x,last_state: ChannelMixState):
            #xx = self.time_shift(x) - x
            xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
            
            
            k = x + xx * self.x_k
            k = torch.relu(self.key(k)) ** 2

            return self.value(k),ChannelMixState(x[:, -1])
        
    #---------------------------------------------------------
    # Router層の定義：hiddenベクトルから各Expertへのスコアを出す
    #---------------------------------------------------------
    class Router(nn.Module):
        def __init__(self, input_dim, num_experts):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_experts,bias=False)
            nn.init.kaiming_uniform_(getattr(self,'linear').weight, a=math.sqrt(5))
            #nn.init.zeros_(self.linear.bias)
            #nn.init.normal_(self.linear.weight, mean=0, std=0.01)

        def forward(self, x):
            # x: (batch*seq, input_dim)
            # 出力: (batch*seq, num_experts)
            return self.linear(x)
        
    class RWKV_CMix_x070_LoRAExperts(nn.Module):
        def __init__(self, args, layer_id, Shared_key,Shared_value,ExpertNo=0):
            # 最初にsuper().__init__()を呼び出す
            super().__init__()
            self.layer_id = layer_id
            self.key = make_linear_ffn_experts(args.n_embd, args.dim_ffn, shared_weight = Shared_key,bias=False,n_layer=self.layer_id,pname='ffn.key',ExpertNo = ExpertNo)
            self.value = make_linear_ffn_experts(args.dim_ffn, args.n_embd, shared_weight = Shared_value, bias=False,n_layer=self.layer_id,pname='ffn.value',ExpertNo=ExpertNo)
        def forward(self, hidden):
            k = torch.relu(self.key(hidden)) ** 2
            return self.value(k)
      

    class RWKV_CMix_x070_MoLE_(nn.Module):
        def __init__(self, args, layer_id, num_experts=4):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            self.ActiveExperts = args.moe_active  
            self.num_experts = num_experts

            #for save in checkpoint
            

            self.shared_expert = True

            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 からほぼ0へ
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
                self.moe_info = nn.Parameter(torch.tensor([float(num_experts), float(args.moe_active)]))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            # baseweight is frozen permanently
            self.key = QuantLinear(args.n_embd, args.dim_ffn, bias=False, n_layer=self.layer_id)
            self.value = QuantLinear(args.dim_ffn, args.n_embd, bias=False, n_layer=self.layer_id)


            for i in range(self.num_experts):
                setattr(self, f"expert_{i}", RWKV_CMix_x070_LoRAExperts(self.args, self.layer_id, self.key, self.value))


            if self.shared_expert:
                self.router = Router(args.n_embd, self.num_experts - 1)
            else:
                self.router = Router(args.n_embd, self.num_experts)

            nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        def forward(self, hidden, input_ids): # input_ids for hashrouter. but currently not use.


            # time-shift 処理
            shifted = self.time_shift(hidden)
            if len(shifted.size()) == 2:
                shifted = shifted.unsqueeze(1)
            delta_hidden_to_shifted = shifted - hidden
            hidden_with_tokenshift = hidden + delta_hidden_to_shifted * self.x_k
            # [B, S, n_embd] => [B*S, n_embd]
            flat_hidden = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))
            B = flat_hidden.size(0)  # B*S のトータルトークン数

            flat_value = torch.zeros_like(flat_hidden)  

            out_0 = getattr(self, f"expert_{0}")(flat_hidden)  
            flat_value += out_0  # ゲーティングなしで全量加算

            router_scores = self.router(flat_hidden)  
            AdaptiveActiveExperts = self.ActiveExperts - 1
            router_scores_others = router_scores
            topk_values, topk_experts = torch.topk(router_scores_others, k=AdaptiveActiveExperts, dim=-1)
            gating = F.softmax(topk_values, dim=-1)  # 同じshape [B, (AdaptiveActiveExperts)]
            gating_full = torch.zeros_like(router_scores_others)  # [B, num_experts-1]
            gating_full.scatter_(dim=1, index=topk_experts, src=gating)
            usage = gating_full.mean(dim=0)  # -> (num_experts-1,)
            load_balance_loss = ((usage - (1.0 / (self.num_experts - 1))) ** 2).sum()

            topk_experts_flat = topk_experts.reshape(-1)  # 各トークンが割り当てられたexpert番号(=0〜(num_experts-2))
            gating_flat = gating.reshape(-1)              # そのトークンのゲーティング値

            source_indices = torch.arange(B, device=flat_hidden.device).unsqueeze(1).expand(B, AdaptiveActiveExperts).reshape(-1)
            real_expert_id = topk_experts_flat  # [0..(num_experts-2)]

 

            for e in range(1, self.num_experts):
                mask_e = (real_expert_id == e)
                if not mask_e.any():
                    continue
                indices_e = mask_e.nonzero(as_tuple=True)[0]
                input_e = flat_hidden[source_indices[indices_e]]
                out_e = getattr(self, f"expert_{e}")(input_e)
                out_e = out_e * gating_flat[indices_e].unsqueeze(-1)
                flat_value.index_add_(0, source_indices[indices_e], out_e)

            # (B*S, n_embd) => [B_, S_, n_embd] に戻す
            kv = flat_value.view(hidden.size(0), hidden.size(1), hidden.size(2))
            return kv, load_balance_loss
        
    class RWKV_CMix_x070_MoLE(nn.Module):
        def __init__(self, args, layer_id, num_experts=4):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            # 利用する active expert の数
            self.ActiveExperts = args.moe_active  
            self.num_experts = num_experts
            self.shared_expert = True  # expert_0 は全トークンに対して必ず呼ばれる

            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 からほぼ0へ
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                # time-shift 用の重み
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
                # MoE 情報（チェックポイント保存用）
                self.moe_info = nn.Parameter(torch.tensor([float(num_experts), float(args.moe_active)]))

            # layer の処理モード（必要に応じて）
            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            # 元の Dense FFN の重みを使って各 expert を初期化する
            self.key = QuantLinear(args.n_embd, args.dim_ffn, bias=False, n_layer=self.layer_id)
            self.value = QuantLinear(args.dim_ffn, args.n_embd, bias=False, n_layer=self.layer_id)
            for i in range(self.num_experts):
                setattr(self, f"expert_{i}", RWKV_CMix_x070_LoRAExperts(self.args, self.layer_id, self.key, self.value))

            # shared_expert==True の場合、router は expert_1 以降のスコアのみを出す
            if self.shared_expert:
                self.router = Router(args.n_embd, self.num_experts - 1)
            else:
                self.router = Router(args.n_embd, self.num_experts)

        def forward(self, hidden, input_ids):  # input_ids は現状使用していません
            # 1. time-shift 処理
            shifted = self.time_shift(hidden)
            if len(shifted.size()) == 2:
                shifted = shifted.unsqueeze(1)
            delta_hidden_to_shifted = shifted - hidden
            hidden_with_tokenshift = hidden + delta_hidden_to_shifted * self.x_k

            # [B, S, n_embd] -> [B*S, n_embd]
            flat_hidden = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))
            B = flat_hidden.size(0)

            # 2. 各 expert へのルーティングスコアを生成
            if self.shared_expert:
                # router は expert_1～expert_(num_experts-1) のスコアを出すので、expert_0 のスコアは 0 とする
                router_scores_shared = self.router(flat_hidden)  # shape: [B, num_experts-1]
                expert0_score = torch.zeros(B, 1, device=flat_hidden.device, dtype=router_scores_shared.dtype)
                router_scores_all = torch.cat([expert0_score, router_scores_shared], dim=-1)  # [B, num_experts]
            else:
                router_scores_all = self.router(flat_hidden)  # [B, num_experts]

            # 3. トークンごとに active expert を top-k 選択し、正規化する
            if self.ActiveExperts < self.num_experts:
                topk_values, topk_indices = torch.topk(router_scores_all, k=self.ActiveExperts, dim=-1)
                gating_active = F.softmax(topk_values, dim=-1)  # shape: [B, ActiveExperts]
                gating_full = torch.zeros_like(router_scores_all)
                gating_full.scatter_(1, topk_indices, gating_active)
            else:
                gating_full = F.softmax(router_scores_all, dim=-1)
                topk_indices = torch.arange(self.num_experts, device=flat_hidden.device).unsqueeze(0).expand(B, self.num_experts)
                gating_active = gating_full

            # load balance loss（各 expert の利用が均等になるように）
            usage = gating_full.mean(dim=0)  # [num_experts]
            load_balance_loss = ((usage - (1.0 / self.num_experts)) ** 2).sum()

            # 4. 各 expert の出力を、対応する gating 重みで加重平均する
            flat_value = torch.zeros_like(flat_hidden)
            for e in range(self.num_experts):
                # 各トークンに対し、top-k 選択された expert が e である箇所を抽出
                mask = (topk_indices == e)  # shape: [B, ActiveExperts]（ActiveExperts == num_experts の場合は [B, num_experts]）
                if mask.sum() == 0:
                    continue
                token_indices = torch.nonzero(mask, as_tuple=False)  # 各行：[トークン番号, topk内の位置]
                tokens = token_indices[:, 0]  # flat_hidden 内のトークンインデックス
                gating_weights = gating_active[tokens, token_indices[:, 1]]  # 該当する gating 重み

                # 該当トークンに対して expert_e の出力を計算
                input_e = flat_hidden[tokens]
                out_e = getattr(self, f"expert_{e}")(input_e)
                out_e = out_e * gating_weights.unsqueeze(-1)
                # 型を flat_value と合わせる（BFloat16 ならここで変換）
                out_e = out_e.to(flat_value.dtype)
                flat_value.index_add_(0, tokens, out_e)

            # [B*S, n_embd] -> [B, S, n_embd] に変形して出力
            kv = flat_value.view(hidden.size(0), hidden.size(1), hidden.size(2))
            return kv, load_balance_loss



##########################################################################################################




def parallel_ema(write: torch.Tensor, decay: float) -> torch.Tensor:
    B, T, C = write.shape
    device = write.device
    power = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1)
    decay_factors = decay ** power
    weighted = write * decay_factors
    ema = torch.cumsum(weighted.flip(dims=[1]), dim=1).flip(dims=[1])
    return ema
from typing import Tuple, Optional

def parallel_ema_stateful(
    write: torch.Tensor,           # (B, T, C)
    decay: float,
    prev_ema: Optional[torch.Tensor] = None  # (B, C)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    数値的に安定で、prev_ema = 0 でも通常のEMAと一致する設計
    """
    B, T, C = write.shape
    device = write.device
    dtype = write.dtype

    # 初期状態：prev_ema or 0
    if prev_ema is None:
        ema0 = torch.zeros(B, 1, C, dtype=dtype, device=device)
    else:
        ema0 = prev_ema.unsqueeze(1)  # (B, 1, C)

    # 系列 t=0～T-1 用に積み重ねる
    decay_tensor = torch.full((1, T, 1), decay, dtype=dtype, device=device)
    decay_factors = torch.cumprod(decay_tensor, dim=1)  # (1, T, 1)

    weighted = write * torch.flip(decay_factors, dims=[1])  # weight decay^{T-t-1}
    weighted = torch.cat([ema0 * decay_factors[:, :1, :], weighted], dim=1)  # 初期値込み

    # 加算して逆に戻す
    ema = torch.cumsum(weighted, dim=1)[:, 1:, :]  # (B, T, C)

    return ema, ema[:, -1, :].detach()
import time
import torch
from torch.autograd import Function
import triton
import triton.language as tl
from typing import Optional, Tuple
@triton.jit
def _ema_kernel(
    B, T, C, decay,
    write_ptr, ema_ptr, state_ptr,
    stride_bw, stride_tw, stride_cw,
    stride_be, stride_te, stride_ce,
    stride_bs, stride_cs,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    # load initial state
    s = tl.load(state_ptr + b * stride_bs + c * stride_cs)
    # compute EMA over time
    for t in range(T):
        w = tl.load(write_ptr + b * stride_bw + t * stride_tw + c * stride_cw)
        s = decay * s + w
        tl.store(ema_ptr + b * stride_be + t * stride_te + c * stride_ce, s)
    # store final state
    tl.store(state_ptr + b * stride_bs + c * stride_cs, s)

# ---------------- Triton backward kernel ----------------
@triton.jit
def _ema_bwd_kernel(
    B, T, C, decay,
    grad_ema_ptr, grad_state_ptr, grad_write_ptr,
    stride_bw, stride_tw, stride_cw,
    stride_ge_b, stride_ge_t, stride_ge_c,
    stride_gs_b, stride_gs_c,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    # load gradient of final state
    s_grad = tl.load(grad_state_ptr + b * stride_gs_b + c * stride_gs_c)
    # reverse time loop
    for t in range(T - 1, -1, -1):
        g = tl.load(grad_ema_ptr + b * stride_ge_b + t * stride_ge_t + c * stride_ge_c)
        # at last step, add grad_state without decay scaling
        s_grad = g + (decay * s_grad if t != T - 1 else s_grad)
        tl.store(grad_write_ptr + b * stride_bw + t * stride_tw + c * stride_cw, s_grad)

class EMAFunction(Function):
    @staticmethod
    def forward(ctx, write: torch.Tensor, decay: float, prev_ema: Optional[torch.Tensor]):
        B, T, C = write.shape
        ema = torch.empty_like(write)
        state = prev_ema.clone() if prev_ema is not None else torch.zeros(B, C, device=write.device, dtype=write.dtype)
        grid = (B * C,)
        _ema_kernel[grid](
            B, T, C, decay,
            write, ema, state,
            write.stride(0), write.stride(1), write.stride(2),
            ema.stride(0), ema.stride(1), ema.stride(2),
            state.stride(0), state.stride(1),
        )
        ctx.decay = decay
        ctx.save_for_backward(write)
        return ema, state

    @staticmethod
    def backward(ctx, grad_ema: torch.Tensor, grad_state: torch.Tensor):
        write, = ctx.saved_tensors
        decay = ctx.decay
        B, T, C = write.shape
        grad_write = torch.zeros_like(write)
        grid = (B * C,)
        _ema_bwd_kernel[grid](
            B, T, C, decay,
            grad_ema, grad_state, grad_write,
            write.stride(0), write.stride(1), write.stride(2),
            grad_ema.stride(0), grad_ema.stride(1), grad_ema.stride(2),
            grad_state.stride(0), grad_state.stride(1),
        )
        return grad_write, None, None

# wrapper
def parallel_ema_stateful_triton_autograd(write: torch.Tensor, decay: float, prev_ema: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if prev_ema is not None:
        prev_ema = prev_ema.to(dtype=torch.float32)
    return EMAFunction.apply(write.float(), decay, prev_ema)
class RWKV_Tmix_x070m(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

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
                return x


        D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
        D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
        D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
        D_GATE_LORA = 128#max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion

        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        self.w0 = nn.Parameter(torch.ones(1, 1, C))

        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))

        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.ones(1, 1, C))

        self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
        self.k_a = nn.Parameter(torch.ones(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(self.n_head, self.head_size))



        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = make_linear_att(C, C, bias=False, n_layer=layer_id, pname='att.receptance')
        self.key = make_linear_att(C, C, bias=False, n_layer=layer_id, pname='att.key')
        self.value = make_linear_att(C, C, bias=False, n_layer=layer_id, pname='att.value')
        self.output = make_linear_att(C, C, bias=False, n_layer=layer_id, pname='att.output')

        self.ln_x = nn.GroupNorm(self.n_head, C, eps=64e-5)

        D_NM = 64
        self.nm_A = nn.Parameter(torch.zeros(C, D_NM))
        self.nm_B = nn.Parameter(ortho_init(torch.zeros(D_NM, C), 0.1))
        self.nm_alpha = nn.Parameter(torch.ones(1, 1, C) * 0.1)
        self.nm_decay = 0.98

        #for memory tuning
        #self.nm_init_mem = nn.Parameter(torch.zeros(1, C))

    def forward(self, x, v_first, passthrough=False):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        #Neural Memory
        delta = (x @ self.nm_A) @ self.nm_B
        gate = torch.sigmoid(delta)
        write = gate * delta
        #init_mem = self.nm_init_mem.expand(B, -1)  # (B, C)
        init_mem = None

        #for MemoryTuning + Gate
        # この init_mem を parallel_ema_stateful に入れる
        mem_stack, next_mem = parallel_ema_stateful_triton_autograd(write, decay=self.nm_decay, prev_ema=init_mem)

        #print(self.nm_alpha)
        

        #Gate only
        #mem_stack = parallel_ema(write, self.nm_decay).to(dtype=torch.bfloat16)

        #xr = x + (xx + mem_stack.to(dtype=torch.bfloat16)) * self.x_r * self.nm_alpha
        xr = x + xx * self.x_r + mem_stack.to(dtype=torch.bfloat16) * self.nm_alpha
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr, passthrough)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk, passthrough)
        v = self.value(xv, passthrough)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a, HEAD_SIZE=self.head_size)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g, passthrough)

        return x, v_first

    def forward_rnn(self, x, v_first, shift_state, nm_mem, wkv_state, passthrough=False):
        B, T, C = x.size()
        H = self.n_head

        xx = torch.cat([shift_state.unsqueeze(1), x[:, :-1]], dim=1) - x
        delta = (x @ self.nm_A) @ self.nm_B
        gate = torch.sigmoid(delta)
        write = gate * delta
        nm_mem = self.nm_decay * nm_mem + write

        xr = x + (xx + nm_mem) * self.x_r * self.nm_alpha
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        shift_state = x[:, -1]

        r = self.receptance(xr, passthrough)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk, passthrough)
        v = self.value(xv, passthrough)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x, wkv_state = RUN_RWKV7_RECURRENT(r, k, v, w, -kk, kk * a, s=wkv_state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g, passthrough)

        return x, v_first, shift_state, nm_mem, wkv_state



