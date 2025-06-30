from ..config import LAYER_CONFIG
from ..linears import make_linear_att,make_linear_ffn,make_linear_head,make_emb

import functools
import os, math, gc, importlib
import torch
import time

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from torch.utils.cpp_extension import load
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from einops import rearrange

allow_ops_in_compiled_graph()

# about FLA
from fla.ops.rwkv6 import chunk_rwkv6

# for infctx
from ..infctx_module import *

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

if 'x060' in ModelGeneration:
    print('RWKV v6 Mode')
    if 'infctx' in TrainingType or KernelMode == 1: #infctx or FLA Mode
        print('x060 Flash-Linear-Attention Kernel Mode')

        @torch.jit.ignore
        def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
            r = rearrange(r, 'b l (h d) -> b h l d', h = H)
            k = rearrange(k, 'b l (h d) -> b h l d', h = H)
            v = rearrange(v, 'b l (h d) -> b h l d', h = H)
            w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
            o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
            x = rearrange(o, 'b h l d -> b l (h d)')
            return x, state
        @torch.jit.ignore
        def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
            r = rearrange(r, 'b l (h d) -> b h l d', h = H)
            k = rearrange(k, 'b l (h d) -> b h l d', h = H)
            v = rearrange(v, 'b l (h d) -> b h l d', h = H)
            w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
            o, _ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=None, output_final_state=False)
            x = rearrange(o, 'b h l d -> b l (h d)')
            return x
        
    else:
        if 'rocm' in RunningDevice:
            wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                        verbose=True, extra_cuda_cflags=["-fopenmp -ffast-math -munsafe-fp-atomics --gpu-max-threads-per-block=120", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                                verbose=True, extra_cuda_cflags=["-fopenmp -ffast-math -munsafe-fp-atomics --gpu-max-threads-per-block=120",  f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
        else:
            wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])  
            wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
        class WKV_6(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu)
        @torch.jit.ignore
        def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
            
            return WKV_6.apply(B, T, C, H, r, k, v, w, u)        

        class WKV_6STATE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert s.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    assert s.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u, s)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u, s = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    gs = torch.sum(gs, 0).view(H, C//H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu, gs)

        def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
            return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
        

    ##########################################################################################################

    class RWKV_Tmix_x060(MyModule):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
            assert args.dim_att % self.n_head == 0

            with torch.no_grad():
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                if args.n_embd == 4096:
                    TIME_MIX_EXTRA_DIM = 64 
                self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                decay_speed = torch.ones(args.dim_att)
                for n in range(args.dim_att):
                    decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

                TIME_DECAY_EXTRA_DIM = 64
                if args.n_embd == 4096:
                    TIME_DECAY_EXTRA_DIM = 128
                self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-1e-4, 1e-4))

                tmp = torch.zeros(args.dim_att)
                for n in range(args.dim_att):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.receptance')
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.key')
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.value')
            self.output = make_linear_att(args.dim_att, args.n_embd, bias=False,n_layer=self.layer_id,pname='att.output')
            self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.gate')




            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

        #@MyFunction
        
        def jit_func(self, x):
            B, T, C = x.size()

            xx = self.time_shift(x) - x

            xxx = x + xx * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            xw = x + xx * (self.time_maa_w + mw)
            xk = x + xx * (self.time_maa_k + mk)
            xv = x + xx * (self.time_maa_v + mv)
            xr = x + xx * (self.time_maa_r + mr)
            xg = x + xx * (self.time_maa_g + mg)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)
            g = F.silu(self.gate(xg))

            ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
            w = self.time_decay + ww

            return r, k, v, g, w

        @MyFunction
        def jit_func_2(self, x, g):
            B, T, C = x.size()
            x = x.view(B * T, C)
            
            x = self.ln_x(x).view(B, T, C)
            x = self.output(x * g)
            return x

        def forward(self, x):
            B, T, C = x.size()
            H = self.n_head

            r, k, v, g, w = self.jit_func(x)
            x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

            return self.jit_func_2(x, g)

########################################################################################################


    class RWKV_CMix_x060(MyModule):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id,pname='ffn.key')
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.receptance')
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.value')

        @MyFunction
        def forward(self, x):
            xx = self.time_shift(x) - x
            xk = x + xx * self.time_maa_k
            xr = x + xx * self.time_maa_r

            k = self.key(xk)
            k = torch.relu(k) ** 2
            kv = self.value(k)
            return torch.sigmoid(self.receptance(xr)) * kv
        

    class RWKV_Tmix_x060_state(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
            assert args.dim_att % self.n_head == 0

            with torch.no_grad():
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
                if args.n_embd==4096:
                    D_MIX_LORA = D_MIX_LORA*2
                self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

                # fancy time_decay
                decay_speed = torch.ones(args.dim_att)
                for n in range(args.dim_att):
                    decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

                D_DECAY_LORA = 64
                if args.n_embd==4096:
                    D_DECAY_LORA = D_DECAY_LORA*2
                self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
                self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

                tmp = torch.zeros(args.dim_att)
                for n in range(args.dim_att):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
                self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.receptance')
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.key')
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.value')
            self.output = make_linear_att(args.dim_att, args.n_embd, bias=False,n_layer=self.layer_id,pname='att.output')
            self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.gate')



            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

        def jit_func(self, x):
            B, T, C = x.size()

            xx = self.time_shift(x) - x

            xxx = x + xx * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            xw = x + xx * (self.time_maa_w + mw)
            xk = x + xx * (self.time_maa_k + mk)
            xv = x + xx * (self.time_maa_v + mv)
            xr = x + xx * (self.time_maa_r + mr)
            xg = x + xx * (self.time_maa_g + mg)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)
            g = F.silu(self.gate(xg))

            ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
            w = self.time_decay + ww

            return r, k, v, g, w

        def jit_func_2(self, x, g):
            B, T, C = x.size()
            x = x.view(B * T, C)
            
            x = self.ln_x(x).view(B, T, C)
            x = self.output(x * g)
            return x

        def forward(self, x):
            B, T, C = x.size()
            H = self.n_head

            r, k, v, g, w = self.jit_func(x)
            x = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=self.time_state)

            return self.jit_func_2(x, g)
        


    class RWKV_Tmix_x060_infctx(MyModule):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.head_size = args.head_size_a
            self.n_head = args.dim_att // self.head_size
            assert args.dim_att % self.n_head == 0

            with torch.no_grad():
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
                if args.n_embd==4096:
                    D_MIX_LORA = D_MIX_LORA*2
                self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

                # fancy time_decay
                decay_speed = torch.ones(args.dim_att)
                for n in range(args.dim_att):
                    decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

                D_DECAY_LORA = 64
                if args.n_embd==4096:
                    D_DECAY_LORA = D_DECAY_LORA*2
                self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
                self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

                tmp = torch.zeros(args.dim_att)
                for n in range(args.dim_att):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.receptance')
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.key')
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.value')
            self.output = make_linear_att(args.dim_att, args.n_embd, bias=False,n_layer=self.layer_id,pname='att.output')
            self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id,pname='att.gate')
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))
        #@torch.jit.script_method
        
        #@MyFunction
        @torch.compile
        def jit_func(self, x, shift_state):
            B, T, C = x.size()
            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

            xxx = x + xx * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            xw = x + xx * (self.time_maa_w + mw)
            xk = x + xx * (self.time_maa_k + mk)
            xv = x + xx * (self.time_maa_v + mv)
            xr = x + xx * (self.time_maa_r + mr)
            xg = x + xx * (self.time_maa_g + mg)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)
            g = F.silu(self.gate(xg))

            ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
            w = self.time_decay + ww

            return r, k, v, g, w, x[:, -1]

        #@torch.compile
        def jit_func_2(self, x, g, timemixstate:TimeMixState):
            B, T, C = x.size()
            x = x.view(B * T, C)
            
            x = self.ln_x(x).view(B, T, C)
            x = self.output(x * g)
            return x, timemixstate
        #@torch.compile
        def forward(self, x, last_state: TimeMixState):
            B, T, C = x.size()
            H = self.n_head
            shift_state = last_state.shift_state
            r, k, v, g, w, lx = self.jit_func(x, shift_state)
            ######
            wkv_state = last_state.wkv_state.clone().contiguous() #Mose modified
            x, wkv_state = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=wkv_state)
            #wkv_state = last_state.wkv_state
            return self.jit_func_2(x, g, TimeMixState(lx, wkv_state))

    class RWKV_CMix_x060_infctx(MyModule):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, args.n_embd)
                for i in range(args.n_embd):
                    ddd[0, 0, i] = i / args.n_embd
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id,pname='ffn.key')
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.receptance')
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.value')

        #@torch.compile
        def forward(self, x, last_state: ChannelMixState):
            xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
            xk = x + xx * self.time_maa_k
            xr = x + xx * self.time_maa_r

            k = self.key(xk)
            k = torch.relu(k) ** 2
            kv = self.value(k)
            return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])
        
        