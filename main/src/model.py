########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from .adopt import ADOPT
import functools
import os, math, gc, importlib
import torch
import time
from torch.nn.utils.rnn import pad_sequence
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .infctx_module import *

from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
#from lion_pytorch import Lion
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from einops import rearrange
allow_ops_in_compiled_graph()

from .config import LAYER_CONFIG

from .linears import make_linear_att,make_linear_ffn,make_linear_head,make_emb

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
    
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.001 , epsilon=1e-8):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.epsilon = epsilon
    # def forward(self, pred, target,paththrough=False):
    #     n_classes = pred.size(-1)
    #     one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    #     smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
    #     log_prob = F.log_softmax(pred, dim=-1)
    #     return torch.sum(-smooth_one_hot * log_prob, dim=-1)


    def forward(self, pred, target): #Approch1
        n_classes = pred.size(-1)
        softplus_pred = F.softplus(pred)
        prob = softplus_pred / (softplus_pred.sum(dim=-1, keepdim=True) + self.epsilon)
        log_prob = torch.log(prob + self.epsilon)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.sum(dim=-1) / n_classes
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        return loss


    def forward2(self, pred, target): #Approch2
        n_classes = pred.size(-1)
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.sum(dim=-1) / n_classes
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss#.mean()
    

 


########################################################################################################
# CUDA Kernel
########################################################################################################

    
from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
FLAMODE = 1
if 'x060' in os.environ["RWKV_MY_TESTING"]:
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    if FLAMODE:
                        @torch.jit.ignore
                        def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                            r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                            k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                            v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                            w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                            o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
                            x = rearrange(o, 'b h l d -> b l (h d)')
                            return x, state

if 'x060' in os.environ["RWKV_MY_TESTING"] and os.environ["RWKV_TRAIN_TYPE"] != 'infctx':
    if 'rocm' in os.environ["RWKV_MY_ARCHITECTURE"]:
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


if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    if 'rocm' in os.environ["RWKV_MY_ARCHITECTURE"]:
        flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-fopenmp -ffast-math -munsafe-fp-atomics --gpu-max-threads-per-block=120"]
        load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    else:
        flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
        load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            #print(f'T = {T} CHUNK_LEN = {CHUNK_LEN}')
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            #print(f's shape = {s.shape}')
            
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            #print(f'sa shape = {sa.shape}')
            #)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################

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
            if C == 2560:
                D_DECAY_LORA = 96
            # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            if C == 2560:
                D_AAA_LORA = 96
            # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            if C == 2560:
                D_MV_LORA = 64
            # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            D_GATE_LORA = 128
            if C == 2560:
                D_GATE_LORA = 320
            # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']

            LinearMode = 0

            if Processing_Mode == 'full':
                LinearMode = 0

            elif Processing_Mode == 'freeze':
                Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
                if Quant_Mode == 'none':
                    LinearMode = 0
                else:
                    LinearMode = 1
            else:
                LinearMode = 1

            if LinearMode == 0:
                self.receptance = nn.Linear(C, C, bias=False)
                self.key = nn.Linear(C, C, bias=False)
                self.value = nn.Linear(C, C, bias=False)
                self.output = nn.Linear(C, C, bias=False)
            else:
                self.receptance = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.receptance')
                self.key = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.key')
                self.value = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.value')
                self.output = make_linear_att(C, C, bias=False,n_layer=self.layer_id,pname='att.output')


            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

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

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1

        if LinearMode == 0:
            self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
            self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)
        else:
            self.key = make_linear_ffn(args.n_embd, args.n_embd * 4, bias=False,n_layer=self.layer_id,pname='ffn.key')
            self.value = make_linear_ffn(args.n_embd * 4, args.n_embd, bias=False,n_layer=self.layer_id,pname='ffn.value')

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)




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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1
        
        if LinearMode == 0:
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        else:
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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1

        if LinearMode == 0:
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        else:
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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1

        if LinearMode == 0:
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)

        else:
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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1
        
        if LinearMode == 0:
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        else:
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

        LinearMode = 0

        if Processing_Mode == 'full':
            LinearMode = 0

        elif Processing_Mode == 'freeze':
            Quant_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['quant']
            if Quant_Mode == 'none':
                LinearMode = 0
            else:
                LinearMode = 1
        else:
            LinearMode = 1

        if LinearMode == 0:
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        else:
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
########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

########################################################################################################

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    class Block(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.ln1 = nn.LayerNorm(args.n_embd)
            self.ln2 = nn.LayerNorm(args.n_embd)

            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(args.n_embd)

            self.att = RWKV_Tmix_x070(args, layer_id)  
            self.ffn = RWKV_CMix_x070(args, layer_id)


        def forward(self, x, v_first):
            if self.layer_id == 0:
                x = self.ln0(x)

            x_attn, v_first = self.att(self.ln1(x), v_first)
            x = x + x_attn

            x = x + self.ffn(self.ln2(x))
            return x, v_first
if 'x060' in os.environ["RWKV_MY_TESTING"]:
    class Block(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.ln1 = nn.LayerNorm(args.n_embd)
            self.ln2 = nn.LayerNorm(args.n_embd)

            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(args.n_embd)
                # if args.my_pos_emb > 0:
                #     self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                #     self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

            #if self.layer_id == 0 and self.args.pre_ffn > 0:
            #    self.ffnPre = RWKV_ChannelMix(args, 0)
            #else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.att = RWKV_Tmix_x060_infctx(args, layer_id)
                elif os.environ["RWKV_TRAIN_TYPE"] == 'state':
                    self.att = RWKV_Tmix_x060_state(args, layer_id)
                else:
                    self.att = RWKV_Tmix_x060(args, layer_id)

            if 'g' in os.environ["RWKV_MY_TESTING"]:
                self.ffn = MishGLU(args, layer_id)
            else:
                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                        self.ffn = RWKV_CMix_x060_infctx(args, layer_id)
                    else:
                        self.ffn = RWKV_CMix_x060(args, layer_id)
    

            if args.dropout > 0:
                self.drop0 = nn.Dropout(p = args.dropout)
                self.drop1 = nn.Dropout(p = args.dropout)

        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def forward(self, x, last_state: BlockState, x_emb=None):
                args = self.args
                B, T, C = x.size()
                if self.layer_id == 0:
                    x = self.ln0(x)
                    #if args.my_pos_emb > 0:
                    #    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                    #    x = x + pos_emb

                if self.args.dropout == 0:
                    if self.layer_id == 0 and args.pre_ffn > 0:
                        x = x + self.ffnPre(self.ln1(x))
                    else:
                        att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
                        x = x + att_out
                    ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
                    x = x + ffn_out
                else:
                    if self.layer_id == 0 and args.pre_ffn > 0:
                        x = self.drop0(x + self.ffnPre(self.ln1(x)))
                    else:
                        x = self.drop0(x + self.att(self.ln1(x)))
                    x = self.drop1(x + self.ffn(self.ln2(x)))
                # if self.layer_id == 0 and args.pre_ffn > 0:
                #     x = x + self.ffnPre(self.ln1(x))
                #     x = x if self.args.dropout == 0 else self.drop0(x)
                # else:
                #     att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
                #     x = x + att_out
                #     x = x if self.args.dropout == 0 else self.drop0(x)
                # ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
                # x = x + ffn_out
                # x = x if self.args.dropout == 0 else self.drop1(x)

    
                return x, BlockState(att_state, fnn_state)
        else:
            def forward(self, x, x_emb=None):
                args = self.args
                B, T, C = x.size()
                if self.layer_id == 0:
                    x = self.ln0(x)

                if self.args.dropout == 0:
                    if self.layer_id == 0 and args.pre_ffn > 0:
                        x = x + self.ffnPre(self.ln1(x))
                    else:
                        x = x + self.att(self.ln1(x))
                    x = x + self.ffn(self.ln2(x))
                else:
                    if self.layer_id == 0 and args.pre_ffn > 0:
                        x = self.drop0(x + self.ffnPre(self.ln1(x)))
                    else:
                        x = self.drop0(x + self.att(self.ln1(x)))
                    x = self.drop1(x + self.ffn(self.ln2(x)))

    
                return x
        
 

if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, token_amount):
            ctx.save_for_backward(y)
            ctx.token_amount = token_amount
            return loss

        @staticmethod
        def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            if ctx.token_amount == 0:
                return (grad_output, None, None)
            factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
                # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
                gy.scatter_(-1, ids, maxx * factor * grad_output)
            else:
                gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy, None)
    class MemoryEfficientL2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, attention_mask):
            # 必要な情報のみを保存
            ctx.save_for_backward(y, attention_mask)
            ctx.token_amount = attention_mask.sum().item()
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y, attention_mask = ctx.saved_tensors
            
            with torch.no_grad():
                factor = 1e-4 / ctx.token_amount
                mask = attention_mask.unsqueeze(-1)
                
                gy = torch.zeros_like(y)
                masked_y = y.masked_fill(~mask.bool(), float('-inf'))
                maxx, ids = torch.max(masked_y, -1, keepdim=True)
                
                if os.environ.get("WN_FIX_L2WRAP"):
                    gy.scatter_(-1, ids, maxx * factor * grad_output)
                else:
                    gy.scatter_(-1, ids, maxx * factor)
                
                gy.mul_(mask)  
                
            return (grad_output, gy, None)
    class L2Wrap_infctx(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, factor, currentMask):
            # Currently (8th July 2023), save_for_backward, causes an issue with
            # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
            # 
            # Due to L2Wrap being a major hotspot, we should monitor this for future support.
            # so that once its resolved, we can include the L2Wrap step in the torch.compile path
            #
            # See also:
            # - checkpointed_step
            ctx.save_for_backward(y, factor, currentMask)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y, factor, currentMask = ctx.saved_tensors

            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)

            # We ensure the mask is reshaped accordingly, and apply it against gy
            gy = gy * currentMask.reshape(gy.shape[0],gy.shape[1],1) # currentMask[:, None][None, :]
            return (grad_output, gy, None, None)
else:
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            ctx.save_for_backward(y)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy)
        
        
NowCurrentlyGPUNo = 0


class RWKV(pl.LightningModule):
    def __init__(self, args,load_dict=None,realtime_quant=False):
        super().__init__()

        device = self.device
        print(f"Device in __init__: {device}")

        #GPUNo = int(os.environ["RWKV_GLOBAL_NO"])
        #os.environ["RWKV_GLOBAL_NO"] = str(GPUNo + 1)
        data =''
        with open('internaltemp.dat', 'r') as f:
            data = f.read()

        print(f'filedata : {data}')
        GPUNo = int(data)
        time.sleep(0.1)

        with open('internaltemp.dat', 'w') as f:
            f.write(str((GPUNo+1)))
            time.sleep(0.5)

        print(f'Target GPUNo: {GPUNo}')

        target_gpu = f'cuda:{GPUNo}'
        self.target_gpu = target_gpu
        
        self.args = args
        self.inputtalkmax = 0
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = make_emb(args.vocab_size, args.n_embd)

        

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = make_linear_head(args.n_embd, args.vocab_size, bias=False)
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        print('Make Blocks')

        self.blocks = nn.ModuleList()  # 空のModuleListを初期化

        if load_dict is not None:
            for i in range(args.n_layer):
                print('make block')
                block = Block(args, i)
                block.requires_grad_(False)
                print('load block weights')
                self.load_block_weights(block, load_dict, i)
                self.blocks.append(block)
                print('start quantize')
                #print(LAYER_CONFIG)

                if realtime_quant:
                    for name, m in self.blocks.named_modules():
                        if hasattr(m, "quant") and callable(getattr(m, "quant")) and f'{str(i)}.' in name:
                                m.quant(args.quant_mode,target_gpu)
                                print(f'{name} Quant')
            self.load_element_weights(self,'emb',load_dict)
            self.load_element_weights(self,'ln_out',load_dict)
            self.load_element_weights(self,'head',load_dict)
            if realtime_quant:
                for name, m in self.named_modules():
                    print(f'pname = {name}')
                    if hasattr(m, "quant") and callable(getattr(m, "quant")) and f'head' in name:
                            m.quant(args.quant_mode,target_gpu)
                            print(f'{name} Quant')
        else:
            self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        global NowCurrentlyGPUNo
        self.CurrentCudaNo = NowCurrentlyGPUNo

        print('finish blocks')
    def setup(self, stage):

        if self.device.type == 'cuda':
            print(f"Model initialized on GPU: {self.device.index}")
        else:
            print("Model initialized on CPU")

    def load_block_weights(self, block, load_dict, layer_id):
        block_prefix = f'blocks.{layer_id}.'
        block_state_dict = {}
        keys_to_delete = []

        for key, value in load_dict.items():
            #print(key)
            if key.startswith(block_prefix):
                new_key = key[len(block_prefix):]  # block_prefixを除去
                block_state_dict[new_key] = value


        block.load_state_dict(block_state_dict, strict=False)

    def load_element_weights(self,element,element_name, load_dict):
        block_prefix = element_name
        block_state_dict = {}
        keys_to_delete = []

        for key, value in load_dict.items():
            if key.startswith(block_prefix):
                new_key = key#key[len(block_prefix):]  # block_prefixを除去
                block_state_dict[new_key] = value

        element.load_state_dict(block_state_dict, strict=False)

    def configure_optimizers(self):
        args = self.args

        if args.lr_advanced:
            print('LR Advanced Mode. will get info from LayerProfile')
            default_lr_init = args.lr_init
            default_lr_final = args.lr_final


            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = []

 

            for n, p in self.named_parameters():
                print(f'LR Check {n}')
                if ('emb' in n  or 'ln_in' in n):# and LAYER_CONFIG['emb']['mode'] == 'full':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['emb']['lr_init']), 
                                            'lr_final':float(LAYER_CONFIG['emb']['lr_final']) , 
                                            'weight_decay':float(LAYER_CONFIG['emb']['weight_decay']), 
                                            'pname':'emb'})
                        print(optim_groups)
                    #exit()
                elif ('head' in n or 'ln_out' in n) and LAYER_CONFIG['head']['mode']:# != 'freeze':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['head']['lr_init']),
                                            'lr_final':float(LAYER_CONFIG['head']['lr_final']),
                                            'weight_decay':float(LAYER_CONFIG['head']['weight_decay']) ,
                                            'pname':'head'})
                else:
                    print('Layer Check')
                    Found = False
                    for i in range(args.n_layer):
                        blockname = f'blocks.{i}.'
                        if blockname in n:
                            print(n)
                        if blockname in n and 'time_state' in n and args.state:
                            print(f"State-tuning {n} Set lr_init {float(LAYER_CONFIG[f'{str(i)}']['lr_init_state'])} lr_final {float(LAYER_CONFIG[f'{str(i)}']['lr_final_state'])}")
                            if p.requires_grad:
                                optim_groups.append({"params":[param_dict[n]], "weight_decay": 0.0,
                                                    'lr_init':float(LAYER_CONFIG[f'{str(i)}']['lr_init_state']), 
                                                    'lr_final':float(LAYER_CONFIG[f'{str(i)}']['lr_final_state']),  
                                                    'pname':n
                                                    })
                            Found = True
                            break
                        elif blockname in n and LAYER_CONFIG[f'{str(i)}']['mode'] != 'freeze':
                            #if n in  LAYER_CONFIG[f'{str(i)}']['RejectParts'] and len(LAYER_CONFIG[f'{str(i)}']['RejectParts']) > 0:
                            if any(word in n for word in LAYER_CONFIG[f'{str(i)}']['RejectParts']) and LAYER_CONFIG[f'{str(i)}']['RejectParts'][0] != '':
                                print(f'Rejected {n}')
                                Found = True
                                break

                            lr_x = 1.0
                            if 'time_decay' in n: # for x060
                                lr_x = 2.0

                            print(f"WeightParameter {n} Set lr_init {float(LAYER_CONFIG[f'{str(i)}']['lr_init'])} lr_final {float(LAYER_CONFIG[f'{str(i)}']['lr_final'])}")
                            if p.requires_grad:
                                optim_groups.append({"params":[param_dict[n]], 
                                                    'lr_init':float(LAYER_CONFIG[f'{str(i)}']['lr_init'])*lr_x, 
                                                    'lr_final':float(LAYER_CONFIG[f'{str(i)}']['lr_final'])*lr_x,  
                                                    'weight_decay':float(LAYER_CONFIG[f'{str(i)}']['weight_decay']),
                                                    'pname':n
                                                    })
                            Found = True
                            break
                    if Found==False:
                        print( f'{n} is not found optimizer strategy')
                        #exit()

            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups,  betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)


        else:
            lr_decay = set()
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                    if args.limited_lora == False:
                        lr_1x.add(n)
                elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                        if args.limited_lora == False:
                            lr_1x.add(n)
                elif (("time_decay" in n) or ("time_faaaa" in n)) and (args.layerwise_lr > 0):
                        if args.limited_lora == False:
                            lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                        if args.limited_lora == False:
                            lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    if args.limited_lora == False:
                        lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

            lr_decay = sorted(list(lr_decay))
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))

            param_dict = {n: p for n, p in self.named_parameters()}

            
            if args.layerwise_lr > 0:
                    optim_groups = [
                        #{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                        #{"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                    ]
                    print(optim_groups)
                    #exit()
            else:
                optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
                #print(f'optim_groups  {optim_groups}')
                #raise optim_groups

            if args.weight_decay > 0:
                optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
                return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
                if args.optim == 'adopt':
                    print('Adopt Mode')
                    return ADOPT(optim_groups,lr=self.args.lr_init)
                return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    if os.environ["RWKV_TRAIN_TYPE"] == 'infctx': # BPTT Mode 

        def forward(self, idx,  last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
            args = self.args
            B, T = idx.size()
            assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."
            C = args.n_embd
            H =  args.dim_att // args.head_size_a
            assert C==H*args.head_size_a
            
            x = self.emb(idx)
            x_emb = x
            new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                            x.device, x.dtype)
            if args.dropout > 0:
                x = self.drop0(x)

            for i, (block, block_state) in enumerate(zip(self.blocks,
                BlockStateList(last_shift_states, last_wkv_states))):
                # x = x.to(block.device)
                if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                    x, new_block_state = torch_checkpoint(block, x, block_state, x_emb, use_reentrant=False)
                    #x, new_block_state = block(x, block_state, x_emb)

                else:
                    #x, new_block_state = torch_checkpoint(block, x, block_state, x_emb, use_reentrant=False)
                    x, new_block_state = deepspeed.checkpointing.checkpoint(block, x,block_state, x_emb)
                    #x, new_block_state = block(x, block_state, x_emb)
                    #x, new_block_state = torch_checkpoint(block, x, block_state, x_emb, use_reentrant=False)
                    #x, new_block_state = deepspeed.checkpointing.checkpoint
                new_states[i] = new_block_state#.clone().detach()

            x = self.ln_out(x)

            if args.head_qk > 0:
                q = self.head_q(x)[:, :T, :]
                k = self.head_k(x)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
                c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

                if "32" in os.environ["RWKV_FLOAT_MODE"]:
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size)
                elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
                elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

                x = self.head(x) + c
            else:
                x = self.head(x)

            return x, new_states.shift_states, new_states.wkv_states
        
        def distillation_loss(self, student_logits, teacher_probs, temperature):
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        def kl_divergence_loss(self, student_logits, teacher_probs, temperature):
            student_probs = F.softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div((student_probs + 1e-8).log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)
    
        def training_step(self, batch): #batch_idx
            args = self.args
            T_train = args.chunk_ctx  #chunk size

            if args.distillation:
                #temperature = args.temperature
                #alpha = args.alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                top_k_values = batch['top_k_values']
                top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']


                #target = input_ids[:,1:]
                #input_ids = input_ids[:,:-1]
                

                B, T = input_ids.shape
                total_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                kl_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                smooth_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                token_amount = 0
                C = args.n_embd
                H =  args.dim_att // args.head_size_a
                assert C==H*args.head_size_a
                states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
                    self.emb.weight.dtype)

                def checkpointed_step2(chunk_input_ids,chunk_target_ids, chunk_top_k_values, chunk_top_k_indices, 
                                    chunk_attention_mask, prev_loss, prev_smooth_loss, prev_kl_loss, last_shift_states,last_wkv_states, prev_token_amount):
                    # Forward pass
                    targets = chunk_target_ids.contiguous().view(-1)
                    mask = chunk_attention_mask.contiguous().view(-1)
                    sum_mask = torch.sum(mask).item()
                    if sum_mask == 0:
                        status = 'skip'
                        return prev_loss,prev_smooth_loss,prev_kl_loss, last_shift_states, last_wkv_states, prev_token_amount, status
                    
                    student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)

                    # Label Smoothing Loss
                    label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                    student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                    smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                    # Top-k teacher logits KL-divergence loss
                    teacher_probs = chunk_top_k_values#[:, :-1]
                    teacher_indices = chunk_top_k_indices#[:, :-1]
                    student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)
                    kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_probs, args.temperature)
     
                    current_token_amount = chunk_input_ids.shape[1]#sum_mask

                    # Combine losses
                    #print(f'summask = {sum_mask} maskshape = {mask.shape[0]}')
                    if sum_mask == mask.shape[0]:
                        loss = args.alpha * smooth_loss.mean() + (1 - args.alpha) * kl_loss.mean()
                        smooth_loss = smooth_loss.mean()
                        kl_loss = kl_loss.mean()
                        loss = L2Wrap.apply(loss, student_logits, current_token_amount)
                    else:
                        smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                        loss = smooth_loss
                        kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                        loss = args.alpha * smooth_loss + (1 - args.alpha) * kl_loss
                        loss = L2Wrap.apply(loss, student_logits, current_token_amount)

                    
                    new_token_amount = prev_token_amount + current_token_amount
                    if new_token_amount > 0:
                        new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                        new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
                        new_kl_loss = prev_kl_loss * (prev_token_amount / new_token_amount) + kl_loss * (current_token_amount / new_token_amount)
                    else:
                        new_loss = prev_loss
                        new_smooth_loss = smooth_loss
                        new_kl_loss = kl_loss

                    status = 'proceed'
                    return new_loss, new_smooth_loss, new_kl_loss, new_shift_states, new_wkv_states, new_token_amount, status
                
                proceedtokens = 0
                
                for i in range(math.ceil(T / T_train)):
                    chunk_start = i * T_train
                    chunk_end = (i + 1) * T_train#min((i + 1) * T_train, T)
                    #print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')

                    # chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                    # chunk_target = target[:, chunk_start:chunk_end]
                    # chunk_top_k_values = top_k_values[:, chunk_start:chunk_end]
                    # chunk_top_k_indices = top_k_indices[:, chunk_start:chunk_end]
                    # chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]

                    # print(f'{chunk_input_ids.shape} {chunk_target.shape} {chunk_top_k_values.shape} {chunk_top_k_indices.shape} {chunk_attention_mask.shape}')

                    
                    total_loss, smooth_loss,kl_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(
                        checkpointed_step2,
                        input_ids[:, chunk_start:chunk_end],
                        target[:, chunk_start:chunk_end],
                        top_k_values[:, chunk_start:chunk_end],
                        top_k_indices[:, chunk_start:chunk_end],
                        attention_mask[:, chunk_start:chunk_end],
                        total_loss,
                        smooth_loss,
                        kl_loss,
                        states.shift_states,
                        states.wkv_states,
                        token_amount,
                        use_reentrant=False
                    )
                    #states = BlockStateList(new_shift_states, new_wkv_states)
                    states.shift_states = new_shift_states
                    states.wkv_states = new_wkv_states

                    if status == 'skip':
                        break

                    if status == 'proceed':
                        proceedtokens = proceedtokens + (chunk_end-chunk_start)

                
                self.trainer.smooth_loss = float(smooth_loss)
                self.trainer.kl_loss = float(kl_loss)
                self.trainer.realproceedtokens =float(proceedtokens)

                return total_loss.float()#, states
            
            if args.sft and 0:
                #temperature = args.temperature
                #alpha = args.alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                attention_mask = batch['attention_mask']


                #target = input_ids[:,1:]
                #attention_mask = attention_mask[:,:-1]
                #input_ids = input_ids[:,:-1]

                B, T = input_ids.shape
                total_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                #kl_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                smooth_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                token_amount = 0
                C = args.n_embd
                H =  args.dim_att // args.head_size_a
                assert C==H*args.head_size_a
                states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
                    self.emb.weight.dtype)



                def checkpointed_step2(chunk_input_ids,chunk_target_ids,#, chunk_top_k_values, chunk_top_k_indices, 
                                    chunk_attention_mask,
                                      prev_loss,
                                        prev_smooth_loss,
                                          #prev_kl_loss,
                                            last_shift_states,last_wkv_states, prev_token_amount):
                    # Forward pass
                    targets = chunk_target_ids.contiguous().view(-1)
                    mask = chunk_attention_mask.contiguous().view(-1)
                    #print(f'mask = {mask}')
                    sum_mask = torch.sum(mask).item()
                    
                    if sum_mask == 0:
                        status = 'skip'
                        return prev_loss,prev_smooth_loss,last_shift_states, last_wkv_states, prev_token_amount, status
                    
                    student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)
                    print(f'logit sum0={torch.sum(student_logits)}')
                    # Label Smoothing Loss
                    label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                    student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                    #print(f'logit sum1={torch.sum(student_logits_shifted)}')
                    #smooth_loss = label_smoothing_loss(student_logits_shifted, targets)
                    #student_logits_shifted = student_logits_shifted[:, :targets.size(1)]
                    #print(f'student_logits_shifted = {student_logits_shifted.shape} targets = {targets.shape}')
                    # if smoothing == 0:
                    #     #smooth_loss = label_smoothing_loss(student_logits_shifted, targets, True) #Through
                    #     smooth_loss = F.cross_entropy(student_logits_shifted.view(-1, student_logits_shifted.size(-1)), targets.reshape(-1))
                    # else:
                    smooth_loss = label_smoothing_loss(student_logits_shifted, targets)
                    #smooth_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.reshape(-1), reduction='none')
                    #del student_logits_shifted
                    del targets
                  

                    #student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                    #smooth_loss = smooth_cross_entropy(student_logits_shifted,targets,smoothing)
     
                    current_token_amount = chunk_input_ids.shape[1]#sum_mask

                    print(f'current token amount = {current_token_amount}')

                    # Combine losses
                    if sum_mask == mask.shape[0]:
                        print('nomask')
                        loss = smooth_loss.mean() + 1e-8 #args.alpha * smooth_loss.mean() + (1 - args.alpha) * kl_loss.mean()
                        smooth_loss = smooth_loss.mean()
                        #kl_loss = kl_loss.mean()
                        loss = L2Wrap.apply(loss, student_logits, current_token_amount)
                    else:
                        print('masked')
                        print(f'smooth_loss = {smooth_loss.shape} mask = {mask.shape}')
                        smooth_loss = torch.sum(smooth_loss * mask + 1e-8) / sum_mask
                        loss = smooth_loss
                        #kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                        #loss = smooth_loss#args.alpha * smooth_loss + (1 - args.alpha) * kl_loss
                        print(f'logits after:{student_logits[0][int(sum_mask):int(sum_mask)+32]}')
                        #loss = L2Wrap.apply(loss, student_logits[:,0:int(sum_mask)], int(sum_mask))

                        loss = L2Wrap.apply(loss, student_logits,current_token_amount)
                    
                    new_token_amount = prev_token_amount + current_token_amount
                    if new_token_amount > 0:
                        print(f'loss ={float(loss)}')
                        new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                        
                        #print(f'new_loss ={float(new_loss)}')
                        new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
                        #new_kl_loss = prev_kl_loss * (prev_token_amount / new_token_amount) + kl_loss * (current_token_amount / new_token_amount)
                    else:
                        new_loss = prev_loss
                        new_smooth_loss = smooth_loss
                        #new_kl_loss = kl_loss

                    status = 'proceed'
                    return new_loss, new_smooth_loss,new_shift_states, new_wkv_states, new_token_amount, status
                
                proceedtokens = 0
                
                for i in range(math.ceil(T / T_train)):
                    chunk_start = i * T_train
                    chunk_end = (i + 1) * T_train#min((i + 1) * T_train, T)
                    #print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')
                    total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(
                    #total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = deepspeed.checkpointing.checkpoint(
                        checkpointed_step2,
                        input_ids[:, chunk_start:chunk_end],
                        target[:, chunk_start:chunk_end],
                        attention_mask[:, chunk_start:chunk_end],
                        total_loss,
                        smooth_loss,
                        states.shift_states,
                        states.wkv_states,
                        token_amount,
                        use_reentrant=False
                    )
                    #if status == 'skip':
                    #    break
                    states = BlockStateList(new_shift_states, new_wkv_states)
                    #states.shift_states = new_shift_states
                    #states.wkv_states = new_wkv_states

                    if status == 'proceed':
                        proceedtokens = proceedtokens + (chunk_end-chunk_start)

                
                self.trainer.smooth_loss = float(smooth_loss)
                #self.trainer.kl_loss = float(kl_loss)
                self.trainer.realproceedtokens =float(proceedtokens)

                #print(f'total_loss = {float(total_loss)}')

                return total_loss#, states
            


            if args.sft:

                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                attention_mask = batch['attention_mask'].float()
                max_len = int(attention_mask.sum(dim=1).max().item())

                B, T = input_ids.shape
                total_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                smooth_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
                token_amount = 0
                C = args.n_embd
                H =  args.dim_att // args.head_size_a
                assert C==H*args.head_size_a
                states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
                    self.emb.weight.dtype)
                
                # def robust_masked_mean(values, mask, clip_min=-100, clip_max=100):
                #     EPSILON = 1e-8
                #     assert torch.is_tensor(values), "values must be a tensor"
                #     assert torch.is_tensor(mask), "mask must be a tensor"
                #     assert values.shape == mask.shape, f"Shape mismatch: values {values.shape} vs mask {mask.shape}"
                #     mask = mask.float()
                #     if not ((mask == 0) | (mask == 1)).all():
                #         print("Warning: mask contains values other than 0 and 1")
                #         mask = (mask > 0.5).float()
                #     clipped_values = torch.clamp(values, clip_min, clip_max)
                #     scale = torch.max(torch.abs(clipped_values))
                #     if scale > EPSILON:
                #         scaled_values = clipped_values / scale
                #     else:
                #         scaled_values = clipped_values
                #     sum_mask = torch.sum(mask)
                #     if sum_mask <= EPSILON:
                #         return torch.zeros_like(values.mean())
                #     masked_sum = torch.sum(scaled_values * mask)
                #     mean = masked_sum / (sum_mask + EPSILON)
                #     if scale > EPSILON:
                #         mean = mean * scale
                #     if torch.isnan(mean) or torch.isinf(mean):
                #         print("Warning: Result is NaN or Inf")
                #         return torch.zeros_like(mean)
                    
                #     return mean



                def checkpointed_step2(chunk_input_ids,chunk_target_ids,
                                    chunk_attention_mask,
                                      prev_loss,
                                        prev_smooth_loss,
                                            last_shift_states,last_wkv_states, prev_token_amount):
                    # Forward pass
                    targets = chunk_target_ids.contiguous().view(-1)
                    mask = chunk_attention_mask.contiguous().view(-1)
                    batchsize = chunk_attention_mask.shape[0]
                    #print(f'mask = {mask}')
                    sum_mask = torch.sum(mask).item()

                    avg_mask_sum = torch.sum(mask) / batchsize
                    L2Wrap_factor = 1e-4 / avg_mask_sum


                    #print(sum_mask)
                    
                    if sum_mask == 0:
                        status = 'skip'
                        #print('skip')
                        return prev_loss,prev_smooth_loss,last_shift_states, last_wkv_states, prev_token_amount, status
                    
                    student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)
                    #print(f'logit sum0={torch.sum(student_logits)}')
                    # Label Smoothing Loss
                    label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                    student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1)).float()

                    smooth_loss = label_smoothing_loss(student_logits_shifted.float(), targets)
                    #smooth_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.reshape(-1), reduction='none')


                    current_token_amount = chunk_input_ids.shape[1]#sum_mask

                    #print(f'current token amount = {current_token_amount}')


                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    #smooth_loss = robust_masked_mean(smooth_loss, mask)


                    loss = smooth_loss
                    loss = L2Wrap.apply(loss, student_logits, current_token_amount)

                    #MemoryEfficientL2Wrap
                    #if loss <= 0.0:
                    #    loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
                    #loss = L2Wrap_infctx.apply(loss, student_logits,L2Wrap_factor, mask)

                    #print(f'checkpoint loss = {loss}')
                    
                    new_token_amount = prev_token_amount + current_token_amount
                    if new_token_amount > 0:
                        new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                        new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
                    else:
                        new_loss = prev_loss
                        new_smooth_loss = smooth_loss


                     

                    status = 'proceed'
                    #print(f'new_loss loss = {new_loss}')
                    return new_loss, new_smooth_loss,new_shift_states, new_wkv_states, new_token_amount, status
                
                proceedtokens = 0

                batchmax_tokens = max_len

                # remainder = max_len % 1024
                # if remainder == 0:
                #     max_len = max_len
                # else:
                #     padding = 1024 - remainder
                #     max_len = max_len + padding
                
                # if max_len < 1024:
                #     max_len = 1024


                #print(f'T = {T}')

                T_train = args.chunk_ctx#min(args.chunk_ctx,max_len)
                #print(f'T_train = {T_train}')

                #print(f'math.ceil(T / T_train) = {math.ceil(T / T_train)}')

                chunk_start = 0
                while True:
                    chunk_end = chunk_start + T_train
                    if chunk_end > T:
                        chunk_end = T

                    #print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')
                    total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(

                        checkpointed_step2,
                        input_ids[:, chunk_start:chunk_end],
                        target[:, chunk_start:chunk_end],
                        attention_mask[:, chunk_start:chunk_end],
                        total_loss,
                        smooth_loss,
                        states.shift_states.clone().detach(),
                        states.wkv_states.clone().detach(),
                        token_amount,
                        use_reentrant=False
                    )
                    if status == 'skip':
                        break
                    states = BlockStateList(new_shift_states, new_wkv_states)


                    if status == 'proceed':
                        proceedtokens = proceedtokens + (chunk_end-chunk_start)

                    chunk_start = chunk_end
                    if chunk_start >= batchmax_tokens:
                        break

                self.trainer.smooth_loss = float(smooth_loss)
                self.trainer.realproceedtokens =float(proceedtokens)

                #print(f'total_loss dtype = {total_loss.dtype}')

                #print(f'totalloss = {total_loss}')

                
                # for i in range(math.ceil(T / T_train)):
                #     print('loop start-------------------------------------------------------------------------------')
                #     print(f'i = {i}')
                #     print(f'T_train = {T_train}')
                #     print(f'math.ceil(T / T_train) = {math.ceil(T / T_train)}')
                #     chunk_start = i * T_train
                #     chunk_end = (i + 1) * T_train#min((i + 1) * T_train, T)
                #     print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')
                #     total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(
                #     #total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = deepspeed.checkpointing.checkpoint(
                #         checkpointed_step2,
                #         input_ids[:, chunk_start:chunk_end],
                #         target[:, chunk_start:chunk_end],
                #         attention_mask[:, chunk_start:chunk_end],
                #         total_loss,
                #         smooth_loss,
                #         states.shift_states,
                #         states.wkv_states,
                #         token_amount,
                #         use_reentrant=False
                #     )
                #     if status == 'skip':
                #         break
                #     states = BlockStateList(new_shift_states, new_wkv_states)
                #     #states.shift_states = new_shift_states
                #     #states.wkv_states = new_wkv_states

                #     if status == 'proceed':
                #         proceedtokens = proceedtokens + (chunk_end-chunk_start)

                
                # self.trainer.smooth_loss = float(smooth_loss)
                # #self.trainer.kl_loss = float(kl_loss)
                # self.trainer.realproceedtokens =float(proceedtokens)

                # #print(f'total_loss = {float(total_loss)}')

                return total_loss#, states


           

            idx, targets = batch
            B, T = idx.shape
            C = args.n_embd
            H =  args.dim_att // args.head_size_a
            assert C==H*args.head_size_a
            states = BlockStateList.create(args.n_layer, B, C, H, idx.device,
                self.emb.weight.dtype)

            def checkpointed_step(idx, targets, prev_loss, last_shift_states,
                                last_wkv_states, prev_token_amount):
                logits, new_shift_states, new_wkv_states = self(idx, last_shift_states, last_wkv_states)
                current_token_amount = (targets!=-100).sum() #这样是不是更合适？
                current_token_amount = idx.shape[1]
                if current_token_amount == 0:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),reduction='sum')
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                    
                    loss = L2Wrap.apply(loss, logits, current_token_amount)
                new_token_amount = prev_token_amount+current_token_amount
                if new_token_amount>0:
                    new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                        current_token_amount / new_token_amount)
                else:
                    new_loss = prev_loss

                return new_loss, new_shift_states, new_wkv_states, new_token_amount
            
            total_loss = torch.tensor(0.,dtype=self.emb.weight.dtype).requires_grad_()
            token_amount = 0
            i = 0
            print(f'idx {idx.shape} targets {targets.shape}')
            for i in range(math.ceil(T / T_train)):
                print(f'start = {i * T_train} end = {(i + 1) * T_train} diff = {(i + 1) * T_train - i * T_train}' )
                total_loss,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                    checkpointed_step,
                    idx[:, i * T_train:(i + 1) * T_train],
                    targets[:, i * T_train:(i + 1) * T_train],
                    total_loss,
                    states.shift_states,
                    states.wkv_states,
                    token_amount,
                    use_reentrant=False
                )
                states = BlockStateList(new_shift_states, new_wkv_states)

            
            print()
            return total_loss
        

    
    else: #Normal Trianing Mode = have limit context size 

        def forward(self, idx):
            args = self.args
            B, T = idx.size()
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

            x = self.emb(idx)
            x_emb = x

            if args.dropout > 0:
                x = self.drop0(x)
            if 'x070' in os.environ["RWKV_MY_TESTING"]:
                    v_first = torch.empty_like(x)
                    for block in self.blocks:
                        if args.grad_cp == 1:
                            layer_mode = LAYER_CONFIG[f'{str(block.layer_id)}']['mode']
                            if layer_mode == 'full' or layer_mode == 'freeze':
                                #x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
                                x, v_first = torch_checkpoint(block, x, v_first,use_reentrant=False)
                            else:
                                x, v_first = torch_checkpoint(block, x, v_first ,use_reentrant=False)
                                #x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first )
                        else:
                            x, v_first = block(x, v_first)
            else:
                for block in self.blocks:
                    if args.grad_cp == 1:
                        layer_mode = LAYER_CONFIG[f'{str(block.layer_id)}']['mode']
                        if layer_mode == 'full' or layer_mode == 'freeze':
                            #x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                            x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                        else:
                            x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                            #x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                        #x = deepspeed.checkpointing.checkpoint(block, x)
                    else:
                        x = block(x)

            x = self.ln_out(x)

            if args.head_qk > 0:
                q = self.head_q(x)[:, :T, :]
                k = self.head_k(x)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
                c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

                if "32" in os.environ["RWKV_FLOAT_MODE"]:
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size)
                elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
                elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                    c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

                x = self.head(x) + c
            else:
                x = self.head(x)

            return x
    
        def compute_logps_simple_mask(self, chosen_inputs, logits, attention_mask=None):
            #chosen_inputs = chosen_inputs[:, :-1]
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=2)
            
            # Gather log probabilities
            gathered_log_probs = torch.gather(log_probs, dim=2, index=chosen_inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1]
            else:
                attention_mask = torch.ones_like(gathered_log_probs)
            
            # Apply mask to log probabilities
            masked_log_probs = gathered_log_probs * attention_mask
            
            # Compute sequence log probabilities
            sequence_logps = masked_log_probs.sum(dim=1)
            
            # Compute effective sequence lengths (sum of attention mask)
            effective_lengths = attention_mask.sum(dim=1)
            
            # Normalize log probabilities by effective sequence length
            normalized_sequence_logps = sequence_logps / effective_lengths
            
            return sequence_logps

        def distillation_loss(self, student_logits, teacher_probs, temperature):
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        def kl_divergence_loss(self, student_logits, teacher_probs, temperature):
            student_probs = F.softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div(student_probs.log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)


        def training_step(self, batch, batch_idx):
            
            args = self.args

            if args.distillation:
                temperature = args.temperature
                alpha = args.alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                top_k_values = batch['top_k_values']
                top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']

                #max_len = int(input_ids.shape[1])#int(attention_mask.sum(dim=1).max().item())

                max_len = int(attention_mask.sum(dim=1).max().item())
                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    top_k_values = top_k_values[:, :max_len]
                    top_k_indices = top_k_indices[:, :max_len, :]
                    attention_mask = attention_mask[:, :max_len]

                # Forward: input_ids[:, :-1]を使用
                student_logits = self(input_ids)

                # 評価: input_ids[:, 1:]を使用
                #targets = input_ids[:, 1:].contiguous().view(-1) #

                targets = target.contiguous().view(-1)
                #del input_ids

                # マスクの調整
                #mask = attention_mask[:, 1:].contiguous().view(-1) #.contiguous()
                mask = attention_mask.contiguous().view(-1)
                #del attention_mask
                sum_mask = torch.sum(mask).item()

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)

                # Label Smoothing Loss
                label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                # Top-k teacher logitsを使用したKL-divergence loss
                teacher_probs = top_k_values#[:, :-1]
                teacher_indices = top_k_indices#[:, :-1]

                #print(f'teacher_probs shape = {teacher_probs.shape}')

                
                # 学生モデルのlogitsからTop-k値を取得
                student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)

                #print(f'student_top_k_logits shape = {student_top_k_logits.shape}')
                
                kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_probs, temperature)

                #del student_top_k_logits

                # Lossの計算
                if sum_mask == mask.shape[0]:
                    loss = alpha * smooth_loss.mean() + (1 - alpha) * kl_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                    loss = alpha * smooth_loss + (1 - alpha) * kl_loss

                self.trainer.smooth_loss = float(smooth_loss.mean())
                self.trainer.kl_loss = float(kl_loss.mean())
                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)
            

            if args.sft:
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                attention_mask = batch['attention_mask']

                #max_len = int(input_ids.shape[1])#int(attention_mask.sum(dim=1).max().item())

                max_len = int(attention_mask.sum(dim=1).max().item())
                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]

                # Forward: input_ids[:, :-1]を使用
                student_logits = self(input_ids)

                # 評価: input_ids[:, 1:]を使用
                #targets = input_ids[:, 1:].contiguous().view(-1) #

                targets = target.contiguous().view(-1)
                #del input_ids

                # マスクの調整
                #mask = attention_mask[:, 1:].contiguous().view(-1) #.contiguous()
                mask = attention_mask.contiguous().view(-1)
                #del attention_mask
                sum_mask = torch.sum(mask).item()

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)

                # Label Smoothing Loss
                label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)


                # Lossの計算
                if sum_mask == mask.shape[0]:
                    loss = smooth_loss.mean()# + (1 - alpha) * kl_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    #kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                    loss = smooth_loss# + (1 - alpha) * kl_loss

                self.trainer.smooth_loss = float(smooth_loss.mean())
                #self.trainer.kl_loss = float(kl_loss.mean())
                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)





            


            if args.dpo:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0


                for s in range(bsz):
                    chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob = batch_orpo[s]

                    # マスクの作成
                    chosen_mask = (chosen_output != 0).float()  # パディングは0と仮定
                    reject_mask = (reject_output != 0).float()

                    
                    # 両方のテンソルの長さを取得
                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)


                    # 最大長を計算
                    max_len = max(len1, len2)

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    if 'x070' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

                    # 長さが異なる場合、短いテンソルをパディング
                    if len1 < max_len:
                        # len1がmax_lenになるようにパディングを追加 (右側にパディング)
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        # len2がmax_lenになるようにパディングを追加 (右側にパディング)
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))
    

                    SFT_idx = []
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0) # make batch with Chosen and Reject  

                    RT = self(SFT_idx)


                    #print(RT)
                    outputs_pos = RT[0].unsqueeze(0)
                    outputs_neg = RT[1].unsqueeze(0)

                    #del RT
                    del SFT_idx

                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()
                    
                    def cross_entropy(pred, target):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        #loss = loss * mask.view(-1)
                        return loss#loss.sum() / mask.sum()

                    l2_pos_loss = masked_cross_entropy(outputs_pos,chosen_output,chosen_mask)

                    loss_chosen = cross_entropy(outputs_pos,chosen_output)
                    loss_reject = cross_entropy(outputs_neg,reject_output)


                    chosen_prob = -torch.sum(loss_chosen[len1-length_chosen:len1])
                    reject_prob = -torch.sum(loss_reject[len2-length_reject:len2])


                    #reject_prob = -torch.sum(loss_reject[-length_reject:])
                    pref_ratio = args.dpo_beta * (chosen_prob - reject_prob - chosen_ref_prob + reject_ref_prob)
                    pref_matches += (pref_ratio > 0)
                    #pref_ratio = - F.logsigmoid(pref_ratio)
                    pref_ratio = F.softplus(-pref_ratio)
                    

                    final_loss = (l2_pos_loss*args.dpo_alpha) + pref_ratio
                    final_loss = L2Wrap.apply(final_loss, outputs_pos)

                    loss2 = loss2 + final_loss # FinalLoss
                    loss1 = loss1 + l2_pos_loss # SFT Loss
                    lossorpoonly = lossorpoonly + pref_ratio #Pref-ratio
                    

                
                loss2 = loss2 / bsz
                loss1 = loss1 / bsz
                lossorpoonly = lossorpoonly / bsz


                self.trainer.loss_2_dpo = float(lossorpoonly)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                return loss2
            

            elif args.orpo and args.orpo_mode == 0:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                
                #SFT_targets = []
                for s in range(bsz):
                    chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob = batch_orpo[s]

                    # マスクの作成
                    chosen_mask = (chosen_output != 0).float()  # パディングは0と仮定
                    reject_mask = (reject_output != 0).float()

                    
                    # 両方のテンソルの長さを取得
                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)

                    #print(f'len1 = {len1}')
                    #print(f'len2 = {len2}')

                    # 最大長を計算
                    max_len = max(len1, len2)

                    if 'x070' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    # 長さが異なる場合、短いテンソルをパディング
                    if len1 < max_len:
                        # len1がmax_lenになるようにパディングを追加 (右側にパディング)
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        # len2がmax_lenになるようにパディングを追加 (右側にパディング)
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))
    

                    SFT_idx = []
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0) # make batch with Chosen and Reject  

                    RT = self(SFT_idx)


                    #print(RT)
                    outputs_pos = RT[0].unsqueeze(0)
                    outputs_neg = RT[1].unsqueeze(0)

                    #del RT
                    del SFT_idx

                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()

                    l2_pos_loss = masked_cross_entropy(outputs_pos,chosen_output,chosen_mask)


                    pos_prob = self.compute_logps_simple_mask(chosen_output.unsqueeze(0),outputs_pos,chosen_mask.unsqueeze(0))
                    neg_prob = self.compute_logps_simple_mask(reject_output.unsqueeze(0),outputs_neg,reject_mask.unsqueeze(0))


                    orpo_ratio = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
                    
                    pref_matches += (orpo_ratio > 0)

                    orpo_ratio = F.logsigmoid(orpo_ratio)

                    orpo_ratio = -orpo_ratio*args.orpo_alpha

                    orpo_loss = torch.mean(((l2_pos_loss*(1.0-args.orpo_alpha))+orpo_ratio)) #maybe no need torch.mean
                    loss1 = loss1 + l2_pos_loss

                    orpo_loss = L2Wrap.apply(orpo_loss, RT) #im not sure is this correct? outputs_pos or RT ? 

                    loss2 = loss2 + orpo_loss
                    lossorpoonly = lossorpoonly + orpo_ratio

                
                loss2 = loss2 / bsz
                loss1 = loss1 / bsz
                lossorpoonly = lossorpoonly / bsz


                self.trainer.loss_2_orpo = float(lossorpoonly)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                return loss2
            

            elif args.orpo and args.orpo_mode == 1:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                
                #SFT_targets = []
                for s in range(bsz):
                    chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob = batch_orpo[s]

                    # マスクの作成
                    chosen_mask = (chosen_output != 0).float()  # パディングは0と仮定
                    reject_mask = (reject_output != 0).float()

                    
                    # 両方のテンソルの長さを取得
                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)

                    #print(f'len1 = {len1}')
                    #print(f'len2 = {len2}')

                    # 最大長を計算
                    max_len = max(len1, len2)

                    if 'x070' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    # 長さが異なる場合、短いテンソルをパディング
                    if len1 < max_len:
                        # len1がmax_lenになるようにパディングを追加 (右側にパディング)
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        # len2がmax_lenになるようにパディングを追加 (右側にパディング)
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))
    

                    SFT_idx = []
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0) # make batch with Chosen and Reject  

                    RT = self(SFT_idx)


                    #print(RT)
                    outputs_pos = RT[0].unsqueeze(0)
                    outputs_neg = RT[1].unsqueeze(0)

                    #del RT
                    del SFT_idx

                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()
                    
                  
                    def cross_entropy(pred, target):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        #loss = loss * mask.view(-1)
                        return loss#loss.sum() / mask.sum()

                    l2_pos_loss = masked_cross_entropy(outputs_pos,chosen_output,chosen_mask)

                    loss_chosen = cross_entropy(outputs_pos,chosen_output)
                    loss_reject = cross_entropy(outputs_neg,reject_output)

                    pos_prob = -torch.sum(loss_chosen[len1-length_chosen:len1])
                    neg_prob = -torch.sum(loss_reject[len2-length_reject:len2])


                    #pos_prob = self.compute_logps_simple_mask(chosen_output.unsqueeze(0),outputs_pos,chosen_mask.unsqueeze(0))
                    #neg_prob = self.compute_logps_simple_mask(reject_output.unsqueeze(0),outputs_neg,reject_mask.unsqueeze(0))


                    orpo_ratio = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
                    
                    pref_matches += (orpo_ratio > 0)

                    orpo_ratio = F.logsigmoid(orpo_ratio)

                    orpo_ratio = -orpo_ratio*args.orpo_alpha

                    orpo_loss = torch.mean(((l2_pos_loss*(1.0-args.orpo_alpha))+orpo_ratio)) #maybe no need torch.mean
                    loss1 = loss1 + l2_pos_loss

                    orpo_loss = L2Wrap.apply(orpo_loss, RT) #im not sure is this correct? outputs_pos or RT ? 

                    loss2 = loss2 + orpo_loss
                    lossorpoonly = lossorpoonly + orpo_ratio

                
                loss2 = loss2 / bsz
                loss1 = loss1 / bsz
                lossorpoonly = lossorpoonly / bsz


                self.trainer.loss_2_orpo = float(lossorpoonly)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                return loss2


            



            if args.my_qa_mask != 1:
                idx, targets = batch
                logits = self(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # if '0' in os.environ["RWKV_MY_TESTING"]:
                #     print('logits', logits)
                #     torch.set_printoptions(threshold=10000)
                #     print('idx', idx)
                #     exit(0)
            else:
                idx, targets, mask = batch
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()
                # if sum_mask == 0:
                #     return torch.tensor([0.0], requires_grad=True)

                logits = self(idx)
                if sum_mask == mask.shape[0]:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                    # print('rank', self.global_rank, 'loss', loss.item())
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                    # loss_raw = loss
                    loss = torch.sum(loss * mask) / sum_mask

            return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
