import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear


import torch
import torch.nn as nn
from typing import Optional,List
import types, gc, os, time, re
from torch.nn import functional as F
import numpy as np
import os, sys
import time
import bitsandbytes as bnb
import functools
from einops import rearrange


#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script

class RWKV_7(nn.Module):
    # x070 Multi batch Implementation
    # modified from RWKV-LM v7 demo_fast code @ BlinkDL
    # Ongoing cuda custom kernel.(if we can avoid atomicadd(its difficult solve on Rocm lol.))

    @MyStatic
    def x070_TimeMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        B, _ ,_= x.shape  # B, T, H*N
        #xx = x_prev - x
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x 

        #print(f'xx shape = {xx.shape}')
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        #print(f'k shape = {k.shape} k_k shape = {k_k.shape} (k * k_k) shape = {(k * k_k).shape}')

        kk = torch.nn.functional.normalize((k * k_k).view(B,H,N), dim=-1, p=2.0).view(B,1,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

        vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
        #print(f'a shape = {a.shape} kk = {kk.shape}')
        ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
        #state = state * w.view(B,H,1,N) + state @ ab.float() + vk.float()
        state = state * w.view(B,H,1,N) + state @ ab + vk
        xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1))

        xx = torch.nn.functional.group_norm(xx.view(B,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,H*N)    
        xx = (xx + ((r * k * r_k).view(B,H,N).sum(dim=-1, keepdim=True) * v.view(B,H,N)).view(B,H*N)).view(B,1,H*N)
        #print(f'TimeMix Before Return XX shape ={ xx.shape}')
        return (xx * g) @ O_, x, state, v_first
    
    
    @MyStatic
    def x070_TimeMix_one_hybrid(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        #dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

       

        #r = xr @ R_
        r = hybrid_matmul(xr,R_)
        w = torch.tanh(xw @ w1) @ w2
        k = hybrid_matmul(xk,K_)

        #v = xv @ V_
        v = hybrid_matmul(xv,V_)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        ######## cuda-free method 
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        #w = torch.exp(-0.606531 * torch.sigmoid((w0 + w)))

        state = state.transpose(-1, -2).contiguous().float()

        t=0
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
        ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
        state = state * w_.view(B,H,1,N) + state @ ab.float() + vk.float()
        #state = state * w_.view(B,H,1,N) + state @ ab + vk
        xx[:,t] = (state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)

        state = state.transpose(-1, -2).contiguous()

        #xx = xx.permute(0, 2, 1)  # (B,H*N,T)
        
        xx=xx.view(B,-1)
        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)

        # 元の形状 (B,T,H*N) に戻す
        #xx = xx.permute(0, 2, 1)
        xx=xx.view(B,1,-1)
        

        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        return hybrid_matmul((xx * g) , O_), x[:,-1], state, v_first
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
        

    
    @MyStatic
    def x070_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):
        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        #r = xr @ R_
        r = hybrid_matmul(xr,R_)

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5

        #k = xk @ K_
        k = hybrid_matmul(xk,K_)

        #v = xv @ V_
        v = hybrid_matmul(xv,V_)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,g,aa,bb,xx,v_first
    
    def x070_TimeMix_fla_Step2(r, w, k, v, aa, bb,state,FullyFusedMode = True):

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)

        w=-torch.exp(w)
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        if T>128 and FullyFusedMode == False:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        return xx, state
    @MyStatic
    def x070_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,g,O_,x,xx,state,v_first,ln_w,ln_b):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
        return hybrid_matmul((xx * g) , O_), x[:,-1], state.float(), v_first
        #return hybrid_matmul((xx * g) , O_), x[:,-1], state, v_first
        #hybrid_matmul


    #@MyStatic
    def x070_TimeMix_fla_combined(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        w=-torch.exp(w)

        aa=-kk
        bb=kk*a

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        #state = state.permute(0,1,3,2).contiguous()
        if T>128:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        #state = state.permute(0,1,3,2).contiguous()


        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return (xx * g) @ O_, x[:,-1], state.float(), v_first
    

    def x070_TimeMix_seq_fla(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        #w = torch.tanh(xw @ w1) @ w2
        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        w=-torch.exp(w)

        aa=-kk
        bb=kk*a

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]

        state = state.permute(0,1,3,2).contiguous()
        #xx, state = chunk_rwkv7(r_.float(), w_.float(), k_.float(), v_.float(), aa_.float(), bb_.float(), scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        state = state.permute(0,1,3,2).contiguous()


        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return (xx * g) @ O_, x[:,-1], state.float(), v_first


    
    @MyStatic
    def x070_TimeMix_seq(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        #w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        ######## cuda-free method 
        #w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w)))

        state = state.permute(0,1,3,2).contiguous()
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
            vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
            ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
            #state = state * w_.view(B,H,1,N) + state @ ab.float() + vk.float()
            state = state * w_.view(B,H,1,N) + state @ ab + vk
            xx[:,t] = (state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)

        state = state.permute(0,1,3,2).contiguous()

        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        return (xx * g) @ O_, x[:,-1], state, v_first
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
    
    @MyStatic
    def x070_ChannelMix_one(x, x_prev, x_k, K_, V_):
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x 
        k = x + xx * x_k
        k = torch.relu(k @ K_) ** 2
        return k @ V_, x

    @MyStatic
    def x070_ChannelMix_seq(x, x_prev, x_k, K_, V_):
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x  # (B,T,H*N)
        k = x + xx * x_k
        #k = torch.relu(k @ K_) ** 2
        k = torch.relu(hybrid_matmul(k , K_)) ** 2

        #hybrid_matmul
        #return k @ V_, x[:,-1,:]
        return hybrid_matmul(k , V_), x[:,-1,:]