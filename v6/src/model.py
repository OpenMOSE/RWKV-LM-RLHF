########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

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
from torch._lowrank import svd_lowrank
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import bitsandbytes as bnb

from .infctx_module import *
from einops import rearrange
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6

from .config import LAYER_CONFIG

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
#if os.environ["RWKV_JIT_ON"] == "1":
#    #MyModule = torch.jit.ScriptModule
#    MyFunction = torch.jit.script_method



def inspect_pytorch_variable(var, name="Variable"):
    print(f"Inspecting {name}:")
    
    # 型の確認
    print(f"Type: {type(var)}")
    
    # テンソルの場合
    if isinstance(var, torch.Tensor):
        print(f"Shape: {var.shape}")
        print(f"Dtype: {var.dtype}")
        print(f"Device: {var.device}")
        print(f"Requires grad: {var.requires_grad}")
        
        # GPUにある場合、どのGPUかを表示
        if var.is_cuda:
            print(f"GPU index: {var.get_device()}")
        
        print(f"Data:\n{var}")
    
    # リストやタプルの場合
    elif isinstance(var, (list, tuple)):
        print(f"Length: {len(var)}")
        print("Contents:")
        for i, item in enumerate(var):
            inspect_pytorch_variable(item, f"{name}[{i}]")
    
    # 辞書の場合
    elif isinstance(var, dict):
        print(f"Keys: {list(var.keys())}")
        print("Contents:")
        for key, value in var.items():
            inspect_pytorch_variable(value, f"{name}['{key}']")
    
    # その他の型の場合
    else:
        print(f"Value: {var}")
    
    print()  # 空行を追加


def rwkv_quantize(quant_type, weight):

    global NowCurrentlyGPUNo
    #print(f'cuda:{NowCurrentlyGPUNo}')
    if quant_type=='4bit':
        qweight, qstate= bnb.functional.quantize_4bit((weight.data))
    elif quant_type=='nf4':
        qweight, qstate= bnb.functional.quantize_nf4((weight.data))
    elif quant_type=='fp4':
        qweight, qstate= bnb.functional.quantize_fp4((weight.data))
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize((weight.data))
    return qweight, qstate


def rwkv_dequantize(quant_type, weight, qstate):
    device = weight.device
    #print(f'weight device = {device}')
    #inspect_pytorch_variable(qstate)
    #if qstate is None:
    #    print("qstate is None")

    #if qstate.device != weight.device:
    #qstate = qstate.clone().detach().to(device)
    #print(qstate)

    #attributes = dir(qstate)

    #print(attributes)
    #qstate=qstate.to(device)
    #i#nspect_pytorch_variable(qstate)
    if quant_type=='4bit':
        deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate)
    elif quant_type=='nf4':
        #print(qstate)
        deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate)
    elif quant_type=='fp4':
        deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate)
    elif quant_type=='int8':
        deweight= bnb.functional.dequantize(weight.data,state=qstate)
    return deweight.to(torch.bfloat16)


class HeadLoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r = int(LAYER_CONFIG[f'head']['rank'])
        alpha = int(LAYER_CONFIG[f'head']['alpha'])

        dropout = float(LAYER_CONFIG[f'head']['dropout'])

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.r = r
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.pissa = False
        #self.is_quant = False

    def pissa_load(self, init_A, init_B):
        self.pissa = True
        self.weight.data = self.weight.data - init_B @ init_A


    def pissa_init(self, svd_niter):

        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A
    #def quant(self, quant_type):
    #    self.is_quant = True
    #    self.quant_type = quant_type
    #    self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))

    def forward(self, x):
        if self.pissa:
            return (
                F.linear(x, self.weight) + 
                F.linear(F.linear(x, self.lora_A), self.lora_B))
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 

        
LORA_CONFIG = {
    "r": 0,
    "alpha": 0,
    "dropout": 0,
    "parts": {"att", "ln", "time", "ffn"},
    "quant": False,
}


    
class LoraEmbedding(nn.Module): #Not working well. please help
    def __init__(self, num_embeddings, embedding_dim, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.weight = nn.Embedding(num_embeddings, embedding_dim)
        self.lora_A = nn.Parameter(torch.zeros(r, num_embeddings))
        self.lora_B = nn.Parameter(torch.zeros(embedding_dim, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # Initialize LoRA parameters
        nn.init.normal_(self.lora_A, std=1 / r)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Regular embedding
        embedded = self.weight(x)
        
        # LoRA path
        lora_embedded = self.dropout(F.embedding(x, self.lora_A.T))  # [batch_size, seq_len, r]
        lora_embedded = lora_embedded @ self.lora_B.T  # [batch_size, seq_len, embedding_dim]
        
        return embedded + self.scaling * lora_embedded


class LoraLinear(nn.Module): # from RWKV-PEFT @JL-er Thanks :) Chaos Modified

    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        #r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
        #    "alpha"], LORA_CONFIG["dropout"]
        r = int(LAYER_CONFIG[f'{str(n_layer)}']['rank'])
        alpha = int(LAYER_CONFIG[f'{str(n_layer)}']['alpha'])
        d = LAYER_CONFIG[f'{str(n_layer)}']
        #print(f'trying intialize layer:{n_layer} {d}')
        dropout = float(LAYER_CONFIG[f'{str(n_layer)}']['dropout'])

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.r = r
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.pissa = False
        self.is_quant = False

    def pissa_load(self, init_A, init_B):
        self.pissa = True
        self.weight.data = self.weight.data - init_B @ init_A


    def pissa_init(self, svd_niter):

        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A
    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to(target_gpu))

    def forward(self, x):

        if self.is_quant:
            if self.pissa:
                return (
                    F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate)) + 
                    F.linear(F.linear(x, self.lora_A), self.lora_B))
            return (
                F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate)) + self.scaling *
                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 

        if self.pissa:
            return (
                F.linear(x, self.weight) + 
                F.linear(F.linear(x, self.lora_A), self.lora_B))
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))  
    

class QuantLinear(nn.Module): # from RWKV-PEFT @JL-er Thanks :)
    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False

    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to(target_gpu))
    def forward(self, x):

        if self.is_quant:
            return F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate))
        else:
            return F.linear(x, self.weight)
        

@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    layer_id = kwargs.get('n_layer')
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze':
        return QuantLinear(*args, **kwargs)
    elif "att" in LAYER_CONFIG[f'{str(layer_id)}']["parts"]:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs) # will Error


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    layer_id = kwargs.get('n_layer')
    #print(f'ffn layerid = {layer_id}')
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze':
        return QuantLinear(*args, **kwargs)
    elif "ffn" in LAYER_CONFIG[f'{str(layer_id)}']["parts"]:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs) # will Error
    

@functools.wraps(HeadLoraLinear)
def make_linear_head(*args, **kwargs):
    #layer_id = kwargs.get('n_layer')
    #print(f'ffn layerid = {layer_id}')
    if LAYER_CONFIG[f'head']['mode'] == 'full' or LAYER_CONFIG[f'head']['mode'] == 'freeze':
        return nn.Linear(*args, **kwargs)
    else:
        return HeadLoraLinear(*args, **kwargs)
    

@functools.wraps(LoraEmbedding)
def make_emb(*args, **kwargs):
    #layer_id = kwargs.get('n_layer')
    #print(f'ffn layerid = {layer_id}')
    if LAYER_CONFIG[f'emb']['mode'] == 'full' or LAYER_CONFIG[f'emb']['mode'] == 'freeze':
        return nn.Embedding(*args, **kwargs)
    else:
        return LoraEmbedding(*args, **kwargs)


# class LabelSmoothingLoss(torch.nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.smoothing = smoothing

#     def forward(self, pred, target):
#         n_classes = pred.size(-1)
#         one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
#         smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
#         log_prob = F.log_softmax(pred, dim=-1)
#         return torch.sum(-smooth_one_hot * log_prob, dim=-1)
    
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        return torch.sum(-smooth_one_hot * log_prob, dim=-1)



########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MY_TESTING"]:
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                        r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                        k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                        v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                        w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                        o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
                        x = rearrange(o, 'b h l d -> b l (h d)')
                        return x, state

if 'x060' in os.environ["RWKV_MY_TESTING"]:
    if 'rocm' in os.environ["RWKV_MY_ARCHITECTURE"]:
        wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-fopenmp -ffast-math -munsafe-fp-atomics --gpu-max-threads-per-block=120 -enable-vectorize-compares", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
    else:
        wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
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

    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################

 


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

        #print(f'model.py timemix x060 LAYER_CONFIG = {LAYER_CONFIG}')

        Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']
        #print(f'Processing Mode = {Processing_Mode}')

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
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.output = make_linear_att(args.dim_att, args.n_embd, bias=False,n_layer=self.layer_id)
            self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)




        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
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

# class RWKV_ChannelMix(MyModule):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

#         with torch.no_grad():  # fancy init of time_mix
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#             self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
#             self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
#         self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
#         self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
#         self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x)
#         xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
#         xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
#         k = self.key(xk)
#         k = torch.relu(k) ** 2
#         kv = self.value(k)
#         return torch.sigmoid(self.receptance(xr)) * kv

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
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id)
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False,n_layer=self.layer_id)
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    


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
            #self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False)
        # self.key = make_linear_att(args.n_embd, args.dim_att, bias=False)
        # self.value = make_linear_att(args.n_embd, args.dim_att, bias=False)
        # self.output = make_linear_att(args.dim_att, args.n_embd, bias=False)
        # self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False)
        Processing_Mode = LAYER_CONFIG[f'{str(self.layer_id)}']['mode']
        #print(f'Processing Mode = {Processing_Mode}')

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
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
            self.output = make_linear_att(args.dim_att, args.n_embd, bias=False,n_layer=self.layer_id)
            self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False,n_layer=self.layer_id)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
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

    @MyFunction
    def jit_func_2(self, x, g, timemixstate:TimeMixState):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x, timemixstate

    def forward(self, x, last_state: TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        shift_state = last_state.shift_state
        r, k, v, g, w, lx = self.jit_func(x, shift_state)
        ######
        wkv_state = last_state.wkv_state.clone().contiguous()
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

        # self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False)
        # self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False)
        # self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False)
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
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False,n_layer=self.layer_id)
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False,n_layer=self.layer_id)
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False,n_layer=self.layer_id)

    @MyFunction
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


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.att = RWKV_Tmix_x060_infctx(args, layer_id)
                else:
                    self.att = RWKV_Tmix_x060(args, layer_id)
            # else:
            #     self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.ffn = RWKV_CMix_x060_infctx(args, layer_id)
                else:
                    self.ffn = RWKV_CMix_x060(args, layer_id)
            # else:
            #     self.ffn = RWKV_ChannelMix(args, layer_id)
        
        # if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
        #     self.tiny_ln = nn.LayerNorm(args.n_embd)
        #     self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
        #     self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
        def forward(self, x, last_state: BlockState, x_emb=None):
            args = self.args
            B, T, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
                if args.my_pos_emb > 0:
                    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                    x = x + pos_emb

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

            if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
                xx = self.tiny_ln(x)
                q = self.tiny_q(xx)[:, :T, :]
                k = self.tiny_k(xx)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
                c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
                x = x + c @ self.tiny_v(x_emb)
            return x, BlockState(att_state, fnn_state)
    else:
        def forward(self, x, x_emb=None):
            args = self.args
            B, T, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
                if args.my_pos_emb > 0:
                    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                    x = x + pos_emb

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

            if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
                xx = self.tiny_ln(x)
                q = self.tiny_q(xx)[:, :T, :]
                k = self.tiny_k(xx)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
                c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
                x = x + c @ self.tiny_v(x_emb)
            return x
        
    # def forward(self, x, x_emb=None):
    #     args = self.args
    #     B, T, C = x.size()
    #     if self.layer_id == 0:
    #         x = self.ln0(x)
    #         if args.my_pos_emb > 0:
    #             pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
    #             x = x + pos_emb

    #     if self.args.dropout == 0:
    #         if self.layer_id == 0 and args.pre_ffn > 0:
    #             x = x + self.ffnPre(self.ln1(x))
    #         else:
    #             x = x + self.att(self.ln1(x))
    #         x = x + self.ffn(self.ln2(x))
    #     else:
    #         if self.layer_id == 0 and args.pre_ffn > 0:
    #             x = self.drop0(x + self.ffnPre(self.ln1(x)))
    #         else:
    #             x = self.drop0(x + self.att(self.ln1(x)))
    #         x = self.drop1(x + self.ffn(self.ln2(x)))

    #     # if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
    #     #     xx = self.tiny_ln(x)
    #     #     q = self.tiny_q(xx)[:, :T, :]
    #     #     k = self.tiny_k(xx)[:, :T, :]
    #     #     c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
    #     #     c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
    #     #     x = x + c @ self.tiny_v(x_emb)
    #     return x


# class L2Wrap(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, loss, y):
#         ctx.save_for_backward(y)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         y = ctx.saved_tensors[0]
#         # to encourage the logits to be close to 0
#         factor = 1e-4 / (y.shape[0] * y.shape[1])
#         maxx, ids = torch.max(y, -1, keepdim=True)
#         gy = torch.zeros_like(y)
#         gy.scatter_(-1, ids, maxx * factor)
#         return (grad_output, gy)

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


        #global_rank = self.global_rank
        #print(f"Local Rank: {global_rank}")
        print(f'Target GPUNo: {GPUNo}')

        target_gpu = f'cuda:{GPUNo}'
        
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

        #make_emb

        self.emb = make_emb(args.vocab_size, args.n_embd)

        

        self.ln_out = nn.LayerNorm(args.n_embd)
        #make_linear_head
        #self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.head = make_linear_head(args.n_embd, args.vocab_size, bias=False)

        #if args.head_qk > 0:
        #    self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
        #    self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
        #    self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


        print('Make Blocks')

        #self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
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
        else:
            self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        global NowCurrentlyGPUNo
        self.CurrentCudaNo = NowCurrentlyGPUNo
        #NowCurrentlyGPUNo = NowCurrentlyGPUNo + 1

        print('finish blocks')
    def setup(self, stage):
        # この方法でGPU番号を取得できます
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
                #keys_to_delete.append(key)
                #print(f'{key} loaded')

        block.load_state_dict(block_state_dict, strict=False)

        # 使用済みの重みをload_dictから削除
        #for key in keys_to_delete:
        #    del load_dict[key]

    def load_element_weights(self,element,element_name, load_dict):
        block_prefix = element_name
        block_state_dict = {}
        keys_to_delete = []

        for key, value in load_dict.items():
            #print(f'{key} {block_prefix}')
            if key.startswith(block_prefix):
                new_key = key#key[len(block_prefix):]  # block_prefixを除去
                block_state_dict[new_key] = value
                #keys_to_delete.append(key)
                #print(f'{key} loaded')

        element.load_state_dict(block_state_dict, strict=False)

        # 使用済みの重みをload_dictから削除
        #3for key in keys_to_delete:
        #    del load_dict[key]

        

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                if args.limited_lora == False:
                    lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    if args.limited_lora == False:
                        lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    if args.limited_lora == False:
                        lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
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
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

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
                if args.grad_cp == 1 and i > 0:  # and i < len(self.blocks)-1
                    x, new_block_state = torch_checkpoint(block, x, block_state, use_reentrant=False)
                else:
                    x, new_block_state = block(x, block_state)
                new_states[i] = new_block_state

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
        
        # def kl_divergence_loss(self, student_logits, teacher_probs, temperature):
        #     student_probs = F.softmax(student_logits / temperature, dim=-1)
        #     teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
        #     return F.kl_div(student_probs.log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)

        def kl_divergence_loss(self, student_logits, teacher_probs, temperature):
            student_probs = F.softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div((student_probs + 1e-8).log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)
    
        def training_step(self, batch): #batch_idx
            args = self.args
            T_train = args.chunk_ctx  #chunk size

            if args.distillation:
                temperature = args.temperature
                alpha = args.alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                top_k_values = batch['top_k_values']
                top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']

                # max_len = int(attention_mask.sum(dim=1).max().item())

                # if args.chunk_ctx > max_len:
                #     max_len = args.chunk_ctx
                # #print(f'max attention len = {max_len}')

                # input_ids = input_ids[:, :max_len]
                # top_k_values = top_k_values[:, :max_len]
                # top_k_indices = top_k_indices[:, :max_len, :]
                # attention_mask = attention_mask[:, :max_len]
                target = input_ids[:,1:]
                #input_ids = input_ids[:, :-1]
                #top_k_values = top_k_values[:,:-1]
                #top_k_indices = top_k_indices[:, :-1, :]
                #a#ttention_mask = attention_mask#[:,1:]


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
                    #student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids[:, :-1],last_shift_states, last_wkv_states)
                    

                    # Prepare targets and mask
                    #targets = chunk_input_ids[:, 1:].contiguous().view(-1)
                    #mask = chunk_attention_mask[:, 1:].contiguous().view(-1)
                    targets = chunk_target_ids.contiguous().view(-1)
                    mask = chunk_attention_mask.contiguous().view(-1)
                    sum_mask = torch.sum(mask).item()

                    #print(f'sum_mask = {sum_mask}')

                    if sum_mask == 0:
                        #print('summask return')
                        return prev_loss,prev_smooth_loss,prev_kl_loss, last_shift_states, last_wkv_states, prev_token_amount
                        #return prev_loss, new_shift_states, new_wkv_states, prev_token_amount
                    
                    student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)

                    # Label Smoothing Loss
                    label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                    student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                    #smooth_loss = label_smoothing_loss(student_logits_shifted, targets)
                    smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                    # Top-k teacher logits KL-divergence loss
                    teacher_probs = chunk_top_k_values#[:, :-1]
                    teacher_indices = chunk_top_k_indices#[:, :-1]
                    student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)
                    kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_probs, args.temperature)

                    

                    current_token_amount = chunk_input_ids.shape[1]#sum_mask

                    # Combine losses
                    if sum_mask == mask.shape[0]:
                        loss = args.alpha * smooth_loss.mean() + (1 - args.alpha) * kl_loss.mean()
                        smooth_loss = smooth_loss.mean()
                        kl_loss = kl_loss.mean()
                        #loss = smooth_loss.mean()
                        loss = L2Wrap.apply(loss, student_logits, current_token_amount)
                        #print(f'smooth_loss={float(smooth_loss.mean())} kl_loss={float(kl_loss.mean())} nomask')
                    else:
                        smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                        loss = smooth_loss
                        #print(f'before mask = {(kl_loss)}')
                        kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                        #print(f'klloss = {float(kl_loss)}')
                        loss = args.alpha * smooth_loss + (1 - args.alpha) * kl_loss
                        loss = L2Wrap.apply(loss, student_logits, current_token_amount)
                        #print(f'smooth_loss={float(smooth_loss)} kl_loss={float(kl_loss)}')

                    
                    new_token_amount = prev_token_amount + current_token_amount
                    if new_token_amount > 0:
                        new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                        new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
                        new_kl_loss = prev_kl_loss * (prev_token_amount / new_token_amount) + kl_loss * (current_token_amount / new_token_amount)
                    else:
                        new_loss = prev_loss
                        new_smooth_loss = smooth_loss
                        new_kl_loss = kl_loss

                    return new_loss, new_smooth_loss, new_kl_loss, new_shift_states, new_wkv_states, new_token_amount
                
                # def checkpointed_step(idx, targets, attention_mask,prev_loss, last_shift_states,
                #                 last_wkv_states, prev_token_amount):
                #     logits, new_shift_states, new_wkv_states = self(idx, last_shift_states, last_wkv_states)
                #     current_token_amount = (targets!=-100).sum() #这样是不是更合适？
                #     current_token_amount = idx.shape[1]
                #     mask = attention_mask.contiguous().view(-1)
                #     sum_mask = torch.sum(mask).item()
                #     print(f'sum_mask = {sum_mask}')
                #     print(f'mask = {mask}')
                #     if sum_mask == 0:
                #         return prev_loss, new_shift_states, new_wkv_states, prev_token_amount
                #     if current_token_amount == 0:
                #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),reduction='sum')
                #     else:
                #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),reduction='none')
                #         print(f'loss before mask = {loss}')
                #         loss = torch.sum(loss * mask) / sum_mask
                #         print(f'loss after mask = {loss}')
                #         loss = L2Wrap.apply(loss, logits, current_token_amount)
                #     new_token_amount = prev_token_amount+current_token_amount
                #     if new_token_amount>0:
                #         new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                #             current_token_amount / new_token_amount)
                #     else:
                #         new_loss = prev_loss

                #     return new_loss, new_shift_states, new_wkv_states, new_token_amount
                
                #print(f'FLA Infctx Mode T={T}')
                
                for i in range(math.ceil(T / T_train)):
                    chunk_start = i * T_train
                    chunk_end = min((i + 1) * T_train, T)
                    #print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')
                    
                    total_loss, smooth_loss,kl_loss, new_shift_states, new_wkv_states, token_amount = torch_checkpoint(
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
                    states = BlockStateList(new_shift_states, new_wkv_states)

                #print('End')
                # num_chunks = max(1, math.ceil(T / T_train))
                # for i in range(num_chunks):
                #     print(f'loop i={i}')
                #     chunk_start = i * T_train
                #     chunk_end = min((i + 1) * T_train, T)
                    
                #     total_loss, new_shift_states, new_wkv_states, token_amount = torch_checkpoint(
                #         checkpointed_step,
                #         input_ids[:, chunk_start:chunk_end],
                #         top_k_values[:, chunk_start:chunk_end],
                #         top_k_indices[:, chunk_start:chunk_end],
                #         attention_mask[:, chunk_start:chunk_end],
                #         total_loss,
                #         states.shift_states,
                #         states.wkv_states,
                #         token_amount,
                #         use_reentrant=False
                #     )
                #     states = BlockStateList(new_shift_states, new_wkv_states)
                
                self.trainer.smooth_loss = float(smooth_loss)
                self.trainer.kl_loss = float(kl_loss)
                self.trainer.realproceedtokens =float(input_ids.shape[1])

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
            for i in range(math.ceil(T / T_train)):
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
            if args.tiny_att_dim > 0:
                for block in self.blocks:
                    if args.grad_cp == 1:
                        layer_mode = LAYER_CONFIG[f'{str(block.layer_id)}']['mode']
                        if layer_mode == 'full' or layer_mode == 'freeze':
                            x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                        else:
                            x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                    else:
                        x = block(x, x_emb)
            else:
                for block in self.blocks:
                    if args.grad_cp == 1:
                        layer_mode = LAYER_CONFIG[f'{str(block.layer_id)}']['mode']
                        if layer_mode == 'full' or layer_mode == 'freeze':
                            x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                        else:
                            x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
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
                top_k_values = batch['top_k_values']
                top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']

                max_len = int(attention_mask.sum(dim=1).max().item())
                #print(f'max attention len = {max_len}')

                input_ids = input_ids[:, :max_len]
                top_k_values = top_k_values[:, :max_len]
                top_k_indices = top_k_indices[:, :max_len, :]
                attention_mask = attention_mask[:, :max_len]

                # Forward: input_ids[:, :-1]を使用
                student_logits = self(input_ids[:, :-1])

                # 評価: input_ids[:, 1:]を使用
                targets = input_ids[:, 1:].contiguous().view(-1) #
                #del input_ids

                # マスクの調整
                mask = attention_mask[:, 1:].contiguous().view(-1) #.contiguous()
                #del attention_mask
                sum_mask = torch.sum(mask).item()

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)

                # Label Smoothing Loss
                label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                # Top-k teacher logitsを使用したKL-divergence loss
                teacher_probs = top_k_values[:, :-1]
                teacher_indices = top_k_indices[:, :-1]

                
                # 学生モデルのlogitsからTop-k値を取得
                student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)
                
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





            


            if args.orpo:
                batch_general, batch_orpo = batch

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


            elif args.dpo:
                batch_general, batch_dpo = batch
                idx, targets = batch_general

                loss1 = 0.0
                self.trainer.loss_1_general_or_sft = float(loss1) # for logging
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_dpo)
                loss2 = 0.0
                for s in range(bsz):
                    chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob = batch_dpo[s]

                    chosen_input_len = len(chosen_input)
                    chosen_output_len = len(chosen_output)

                    reject_input_len = len(reject_input)
                    reject_output_len=len(reject_output)

                    if self.inputtalkmax < chosen_input_len:
                        self.inputtalkmax = chosen_input_len
                    if self.inputtalkmax < chosen_output_len:
                        self.inputtalkmax = chosen_output_len
                    if self.inputtalkmax < reject_input_len:
                        self.inputtalkmax = reject_input_len
                    if self.inputtalkmax < reject_output_len:
                        self.inputtalkmax = reject_output_len
                    
                    chosen_logits = self(chosen_input)
                    loss_chosen = F.cross_entropy(chosen_logits.view(-1, chosen_logits.size(-1)), chosen_output.view(-1), reduction='none') # .squeeze()
                    del chosen_logits
                    gc.collect()
                    torch.cuda.empty_cache()
                    chosen_prob = -torch.sum(loss_chosen[-length_chosen:])
                    reject_logits = self(reject_input)
                    loss_reject = F.cross_entropy(reject_logits.view(-1, reject_logits.size(-1)), reject_output.view(-1), reduction='none') # .squeeze()
                    del reject_logits
                    gc.collect()
                    torch.cuda.empty_cache()
                    reject_prob = -torch.sum(loss_reject[-length_reject:])
                    pref_ratio = args.dpo_beta * (chosen_prob - reject_prob - chosen_ref_prob + reject_ref_prob)
                    pref_matches += (pref_ratio > 0)
                    loss2 = loss2 - F.logsigmoid(pref_ratio)
                loss2 = loss2 / bsz
                self.trainer.loss_2_dpo = float(loss2)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                
                

                #return args.dpo_general_corpus_ratio * loss1 + (1-args.dpo_general_corpus_ratio) * loss2
                return loss2
            ################################################################################################################# dpo

            #basically already returned if dpo mode



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
