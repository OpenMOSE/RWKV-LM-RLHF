#Linear Matmul Arena What is fastest?

#Entrylist
#BF16 Native
#FP16 Native
#Bitsandbytes NF4,Int8
#Torchao FP6 in coding
#Torchao FP5 in coding

from .config import LAYER_CONFIG

import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear


import torch
from torch.nn import functional as F
import functools
import torch.nn as nn
from einops import rearrange
import os, math, gc, importlib
from torch._lowrank import svd_lowrank
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState
from bitsandbytes.autograd._functions import matmul as i8matmul




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

def fp8_hybrid_matmul_noreshape(a,b):
    #with torch.no_grad():
        if b.dtype == torch.float8_e4m3fn:
                xg = a
                x, output_amax = torch._scaled_mm(
                    xg.to(torch.float8_e4m3fn).contiguous(),
                    b,
                    bias=None,
                    out_dtype=a.dtype,
                    scale_a=torch.tensor(1.0, device='cuda'),
                    scale_b=torch.tensor(1.0, device='cuda')
                )
                return x
        else:
                return a.to(dtype=b.dtype) @ b

#if 'cuda' in os.environ["RWKV_MY_ARCHITECTURE"]:
@torch.jit.script #FP8 Experiment Matmul
def fp8_hybrid_matmul(a,b): # shape3 @ shape2 only
    with torch.no_grad():
        if b.dtype == torch.float8_e4m3fn:
                xg = a
                #print(f'xg shape = {xg.shape}')

                if len(xg.shape) == 2:
                    S0=xg.shape[0]
                    if xg.dtype != torch.float8_e4m3fn:
                        xg = torch.clamp(xg, min=-448.0, max=448.0) # for avoid NaN
                    #in torch2.5+ deleted absmax 
                    x = torch._scaled_mm(
                        xg.view(S0,xg.shape[1]).to(torch.float8_e4m3fn).contiguous(),
                        b,
                        bias=None,
                        out_dtype=a.dtype,
                        scale_a=torch.tensor(1.0, device='cuda'),
                        scale_b=torch.tensor(1.0, device='cuda')
                    )
                    #x.requires_grad = False
                    return x.view(S0, -1)
                else:
                    S0=xg.shape[0]
                    S1=xg.shape[1]
                    if xg.dtype != torch.float8_e4m3fn:
                        xg = torch.clamp(xg, min=-448.0, max=448.0) # for avoid NaN
                    #in torch2.5+ deleted absmax 
                    x = torch._scaled_mm(
                        xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn).contiguous(),
                        b,
                        bias=None,
                        out_dtype=a.dtype,
                        scale_a=torch.tensor(1.0, device='cuda'),
                        scale_b=torch.tensor(1.0, device='cuda')
                    )
                    #x.requires_grad = False
                    return x.view(S0, S1, -1)
        else:
                return a.to(dtype=b.dtype) @ b
        
class FP8HybridMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(weight)
        # fp8_hybrid_matmulを呼び出し、出力はBF16
        output = fp8_hybrid_matmul(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output): #これやらないとBackwardでLossが右肩上がりになる
        weight, = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.to(dtype=torch.bfloat16).t())
        del weight
        # 重みは完全に凍結されているので、grad_weightは不要
        return grad_input, None
    
fp8_matmul = FP8HybridMatmul.apply

@torch.jit.script
def fp16_matmul_(a,b):
    with torch.no_grad():
        #a = torch.clamp(a, min=-448.0, max=448.0) # for avoid NaN
        return (a.to(dtype=b.dtype) @ b ).to(dtype=a.dtype)
    
class FP16HybridMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(weight)
        # fp8_hybrid_matmulを呼び出し、出力はBF16
        output = fp16_matmul_(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output): #これやらないとBackwardでLossが右肩上がりになる
        weight, = ctx.saved_tensors
        #grad_input = torch.matmul(grad_output, weight.to(dtype=torch.bfloat16).t())
        #try FP16 Grad
        grad_input = torch.matmul(grad_output.to(dtype=torch.float16), weight.t()).to(dtype=torch.bfloat16)
        del weight
        # 重みは完全に凍結されているので、grad_weightは不要
        return grad_input, None
    
fp16_matmul = FP16HybridMatmul.apply


def fp6ao_matmul_(a,b,scale):
    with torch.no_grad():
        #a = torch.clamp(a, min=-448.0, max=448.0) # for avoid NaN
        #return (a.to(dtype=b.dtype) @ b ).to(dtype=a.dtype)
        S0=a.shape[0]
        S1=a.shape[1]
        a = a.to(dtype=torch.float16).view(-1,a.shape[2])  
        out = quant_llm_linear(3, 2, a, b, scale).view(S0,S1,-1)
        return out
    
class FP6aoHybridMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight ,Scale):
        ctx.save_for_backward(weight,Scale)
        output = fp6ao_matmul_(input, weight,Scale)
        return output

    @staticmethod
    def backward(ctx, grad_output): #これやらないとBackwardでLossが右肩上がりになる
        weight,Scale, = ctx.saved_tensors
        grad_input = fp6ao_matmul_(grad_output, weight.t(), Scale).to(dtype=torch.bfloat16)
        # 重みは完全に凍結されているので、grad_weightは不要
        return grad_input, None
    
fp6ao_matmul = FP6aoHybridMatmul.apply

class Int8_16HybridMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight ,Scale):
        fp16weight = rwkv_dequantize('int8fp16',weight,Scale)
        ctx.save_for_backward(fp16weight)
        output = input.to(dtype=torch.float16) @ fp16weight.t()
        return output#.to(dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output): #これやらないとBackwardでLossが右肩上がりになる
        fp16weight, = ctx.saved_tensors
        # 重みは完全に凍結されているので、grad_weightは不要
        grad_input = (grad_output.to(dtype=torch.float16) @ fp16weight).to(dtype=torch.bfloat16)
        return grad_input, None
    
int8_16_matmul = Int8_16HybridMatmul.apply



    
        

        
#@#torch.jit.script 
def BoneProcessing(weight,bone,r:int):
    # 'weight'の形状を取得
    # H, W = weight.shape
    # assert H % r == 0 and W % r == 0
    # w = weight.reshape(H // r, r, W // r, r)
    # w = w.permute(0, 2, 1, 3)
    # w = torch.matmul(w, bone) + bone
    # w = w.reshape(H, W)

    H, W = weight.shape
    assert H % r == 0 and W % r == 0
    w = weight.view(H // r, r, W // r, r)
    w = w.permute(0, 2, 1, 3)
    w = torch.matmul(w, bone) + bone
    w = w.view(H, W)

    return w

        

def rwkv_quantize(quant_type, weight):
    global NowCurrentlyGPUNo
    if quant_type=='4bit':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_4bit((weight.data))
    elif quant_type=='nf4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_nf4((weight.data))
    elif quant_type=='nf4fp16':
        weight = weight.to(dtype=torch.float16)
        print(f'before quant weight datatype to {weight.dtype}')
        qweight, qstate= bnb.functional.quantize_nf4((weight.data))
    elif quant_type=='fp4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_fp4((weight.data))
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize((weight.data))
    elif quant_type=='int8fp16':
        print(f'before quant weight datatype to {weight.dtype}')
        qweight, qstate= bnb.functional.quantize((weight.data))
    elif quant_type=='fp8':
        qweight = weight.data.to(dtype = torch.float8_e4m3fn)
        qstate = None
    elif quant_type=='fp6ao':
        weight = weight.to(dtype=torch.float16)
        qweight,qstate = to_scaled_tc_floatx(weight.data, 3 , 2)
        #qstate = None
    elif quant_type=='fp16':
        qweight = weight.data.to(dtype = torch.float16) # faster in MI100
        qstate = None
    return qweight, qstate


def rwkv_dequantize(quant_type, weight, qstate):
    #with torch.no_grad():
        #device = weight.device
        if quant_type=='4bit':
            deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='nf4':
            deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='nf4fp16':
            deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate).to(torch.float16)
        elif quant_type=='fp4':
            deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='int8':
            deweight= bnb.functional.dequantize(weight.data,state=qstate).to(torch.bfloat16)
        elif quant_type=='int8fp16':
            deweight= bnb.functional.dequantize(weight.data,state=qstate).to(torch.float16)
        elif quant_type=='fp8':
            deweight= weight.data.to(torch.bfloat16)
        elif quant_type=='fp16':
            deweight= weight.data
        return deweight#.to(torch.bfloat16).contiguous()
@torch.compile
def LinearForward(self,x,passthrough = False):
    #print(f'passthgough = {passthrough}')
    if self.is_quant:
            if self.pissa:
                if self.quant_type == 'fp8': #native
                    if passthrough:
                        return fp8_matmul(x,self.Qweight.t()) #Inference without lora
                    return (
                    fp8_matmul(x,self.Qweight.t()) + 
                    F.linear(F.linear(x, self.lora_A), self.lora_B))
                elif self.quant_type == 'fp16': #FP16 PISSA
                    return (
                        fp16_matmul(x,self.Qweight.t()) + 
                        F.linear(F.linear(x, self.lora_A), self.lora_B))
                elif self.quant_type == 'fp6ao': #FP6 TorchAO
                    return (
                        fp6ao_matmul(x,self.Qweight,self.qstate) + 
                        F.linear(F.linear(x, self.lora_A), self.lora_B))
                else: #Bitsandbytes NF4 INT8, FP16
                    temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                    if passthrough:
                        return F.linear(x, temporal_weight) #Inference without lora
                    if temporal_weight.dtype == torch.float16:
                        return (
                            fp16_matmul(x,temporal_weight.t()) + 
                            F.linear(F.linear(x, self.lora_A), self.lora_B))
                    return (
                        F.linear(x, temporal_weight) + 
                        F.linear(F.linear(x, self.lora_A), self.lora_B))
            
            elif self.bonemode: # Covered All quantize method. currently slow implementation
                temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)

                if passthrough:
                    return F.linear(x, temporal_weight)
                w = rearrange(temporal_weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
                w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
                #w = w + temporal_weight
                return x @ (w + temporal_weight).t()
            
            else: #LoRA
                if self.quant_type == 'fp8': #native
                    if passthrough:
                        return fp8_matmul(x,self.Qweight.t())
                    return (fp8_matmul(x,self.Qweight.t()) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                    )
                
                elif self.quant_type == 'fp16': #FP16 LoRA
                    return (fp16_matmul(x,self.Qweight.t()) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                    )
                
                elif self.quant_type == 'fp6ao': #FP6 TorchAO
                    return (fp6ao_matmul(x,self.Qweight.t(),self.qstate) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                    )
                else:
                    w = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                    if passthrough:
                        return F.linear(x, temporal_weight)
                    #print(f'lora mode dtype = {w.dtype}')
                    if w.dtype == torch.float16:
                        return (fp16_matmul(x,w.t()) + self.scaling *
                                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                        )
                    return ( 
                            F.linear(x, w) + self.scaling *
                            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 
                
    else: # Non Quant mode
        if self.pissa:
            if passthrough:
                return F.linear(x, self.weight)
            return (
                F.linear(x, self.weight) + 
                F.linear(F.linear(x, self.lora_A), self.lora_B))
        elif self.bonemode:
            if passthrough:
                F.linear(x, self.weight)
            w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
            return F.linear(x,self.weight+w)
        if passthrough:
                F.linear(x, self.weight)
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 
    

def LinearForward_Experts(self,adapter,x):
    if self.is_quant:
            if adapter.bonemode: # Covered All quantize method. currently slow implementation
                temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                bone = getattr(adapter,adapter.prefix)
                w = rearrange(temporal_weight, '(a r1) (b r2) -> a b r1 r2', r1 = adapter.r, r2 = adapter.r)@bone+bone
                w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
                return x @ (w + temporal_weight).t()
            
            else: #LoRA
                lora_A = getattr(adapter,adapter.prefix_A)
                lora_B = getattr(adapter,adapter.prefix_B)
                if self.quant_type == 'fp8': #native
                    return (fp8_matmul(x,self.Qweight.t()) + adapter.scaling *
                        F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)
                    )
                
                elif self.quant_type == 'fp16': #FP16 LoRA
                    return (fp16_matmul(x,self.Qweight.t()) + adapter.scaling *
                        F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)
                    )
                
                elif self.quant_type == 'fp6ao': #FP6 TorchAO
                    return (fp6ao_matmul(x,self.Qweight.t(),self.qstate) + adapter.scaling *
                        F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)
                    )
                else:
                    w = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                    #print(f'lora mode dtype = {w.dtype}')
                    if w.dtype == torch.float16:
                        return (fp16_matmul(x,w.t()) + adapter.scaling *
                                F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)
                        )
                    return ( 
                            F.linear(x, w) + adapter.scaling *
                            F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)) 
                
    else: # Non Quant mode
        if adapter.bonemode:
            bone = getattr(adapter,adapter.prefix)
            w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = adapter.r, r2 = adapter.r)@bone+bone
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
            return F.linear(x,self.weight+w)
        
        lora_A = getattr(adapter,adapter.prefix_A)
        lora_B = getattr(adapter,adapter.prefix_B)
        return (
            F.linear(x, self.weight) + adapter.scaling *
            F.linear(F.linear(adapter.lora_dropout(x), lora_A), lora_B)) 

class HeadLoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        if LAYER_CONFIG[f'head']['mode'] == 'bone':
            print('bone mode')
            r = int(LAYER_CONFIG[f'head']['rank'])
            self.r = r
            self.bone = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
            self.bonemode = True

        else:
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
            self.bonemode = False
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
        self.Qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to(target_gpu))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete
    #@torch.jit.ignore
    def forward(self, x,passthrough=False):
        return LinearForward(self,x,passthrough)

        
@torch.jit.ignore
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
    @torch.jit.ignore
    def forward(self, x):
        # Regular embedding
        embedded = self.weight(x)
        
        # LoRA path
        lora_embedded = self.dropout(F.embedding(x, self.lora_A.T))  # [batch_size, seq_len, r]
        lora_embedded = lora_embedded @ self.lora_B.T  # [batch_size, seq_len, embedding_dim]
        
        return embedded + self.scaling * lora_embedded

class LoraLinear(nn.Module): # from RWKV-PEFT @JL-er Thanks :) Chaos Modified
    #@torch.jit.unused
    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int, pname=''):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        if LAYER_CONFIG[f'{str(n_layer)}']['mode'] == 'bone':
            print('bone mode')
            r = int(LAYER_CONFIG[f'{str(n_layer)}']['rank'])
            self.r = r
            self.bone = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
            self.bonemode = True
        else:
            r = int(LAYER_CONFIG[f'{str(n_layer)}']['rank'])
            alpha = int(LAYER_CONFIG[f'{str(n_layer)}']['alpha'])
            d = LAYER_CONFIG[f'{str(n_layer)}']

            dropout = float(LAYER_CONFIG[f'{str(n_layer)}']['dropout'])

            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(dropout)
            self.scaling = alpha / r
            self.r = r
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.bonemode = False
        self.pissa = False
        self.is_quant = False
    #@torch.jit.ignore
    def pissa_load(self, init_A, init_B):
        self.pissa = True
        self.weight.data = self.weight.data - init_B @ init_A
    #@torch.jit.ignore
    def pissa_init(self, svd_niter):

        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A
    #@torch.jit.ignore
    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        self.Qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight).to(device=target_gpu))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete
    #@torch.jit.ignore
    def forward(self, x,passthrough = False):
        return LinearForward(self,x,passthrough)
    
class Shared_LoraLinear(nn.Module): # for Mixture of LoRA Experts

    def __init__(self,in_features: int, out_features: int, shared_weight, bias: bool, n_layer: int, ExpertNo: int, pname = '' ):
        super().__init__()

        self.ExpertNo = ExpertNo

        self.weight = shared_weight
        assert bias == False, "Biased LoraLinear not supported"

        if LAYER_CONFIG[f'{str(n_layer)}']['mode'] == 'bone':
            print('bone mode')
            r = int(LAYER_CONFIG[f'{str(n_layer)}']['rank'])
            self.r = r
            self.prefix = f"bone_expert_{ExpertNo}"
            setattr(self, self.prefix, nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r)))
            self.bonemode = True
        else:
            r = int(LAYER_CONFIG[f'{str(n_layer)}']['rank'])
            alpha = int(LAYER_CONFIG[f'{str(n_layer)}']['alpha'])
            d = LAYER_CONFIG[f'{str(n_layer)}']
            dropout = float(LAYER_CONFIG[f'{str(n_layer)}']['dropout'])

            self.prefix_A = f"lora_A_expert_{ExpertNo}"
            self.prefix_B = f"lora_B_expert_{ExpertNo}"

            setattr(self, self.prefix_A, nn.Parameter(torch.empty(r, in_features)))
            setattr(self, self.prefix_B, nn.Parameter(torch.empty(out_features, r)))

            self.lora_dropout = nn.Dropout(dropout)
            self.scaling = alpha / r
            self.r = r

            nn.init.kaiming_uniform_(getattr(self,self.prefix_A), a=math.sqrt(5))
            nn.init.zeros_(getattr(self,self.prefix_B))
            self.bonemode = False
        self.pissa = False
        self.is_quant = False

    def forward(self, x):
        return LinearForward_Experts(self.weight,self,x)
    

class QuantLinear(nn.Module): # from RWKV-PEFT @JL-er Thanks :)
    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int,pname = ''):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False
    #@torch.jit.ignore
    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        self.Qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight).to(device=target_gpu))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete
    #@torch.jit.ignore
    def forward(self, x,passthrough=False):

        if self.is_quant:
            if self.quant_type == 'fp8':
                return fp8_matmul(x,self.Qweight.t())
            elif self.quant_type == 'fp6ao':
                return fp6ao_matmul(x,self.Qweight.t(),self.qstate)
            return F.linear(x, rwkv_dequantize(self.quant_type, self.Qweight, self.qstate))
        else:
            return F.linear(x, self.weight)
        
@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    layer_id = kwargs.get('n_layer')
    pname = kwargs.get('pname')
    Reject = False
    if any(word in pname for word in LAYER_CONFIG[f'{str(layer_id)}']['RejectParts']) and LAYER_CONFIG[f'{str(layer_id)}']['RejectParts'][0] != '':
        Reject = True
        print(f'reject pname {pname}')
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze' or Reject == True:
        return QuantLinear(*args, **kwargs)
    else:
        return LoraLinear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    layer_id = kwargs.get('n_layer')
    pname = kwargs.get('pname')
    Reject = False
    if any(word in pname for word in LAYER_CONFIG[f'{str(layer_id)}']['RejectParts']) and LAYER_CONFIG[f'{str(layer_id)}']['RejectParts'][0] != '':
        Reject = True
        print(f'reject pname {pname}')
    #print(f'ffn layerid = {layer_id}')
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze' or Reject == True:
        return QuantLinear(*args, **kwargs)
    else:
        return LoraLinear(*args, **kwargs)
    
@functools.wraps(Shared_LoraLinear)
def make_linear_ffn_experts(*args, **kwargs):
    layer_id = kwargs.get('n_layer')
    pname = kwargs.get('pname')
    Reject = False
    if any(word in pname for word in LAYER_CONFIG[f'{str(layer_id)}']['RejectParts']) and LAYER_CONFIG[f'{str(layer_id)}']['RejectParts'][0] != '':
        Reject = True
        print(f'reject pname {pname}')
    #print(f'ffn layerid = {layer_id}')
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze' or Reject == True:
        return QuantLinear(*args, **kwargs)
    else:
        return Shared_LoraLinear(*args, **kwargs)
    

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