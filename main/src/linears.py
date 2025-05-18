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
        output = fp16_matmul_(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output): #これやらないとBackwardでLossが右肩上がりになる
        weight, = ctx.saved_tensors
        #try FP16 Grad
        grad_input = torch.matmul(grad_output.to(dtype=torch.float16), weight.t()).to(dtype=torch.bfloat16)
        del weight
        return grad_input, None
    
fp16_matmul = FP16HybridMatmul.apply


def fp6ao_matmul_(a,b,scale):
    with torch.no_grad():
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
    def backward(ctx, grad_output): 
        weight,Scale, = ctx.saved_tensors
        grad_input = fp6ao_matmul_(grad_output, weight.t(), Scale).to(dtype=torch.bfloat16)
        return grad_input, None
    
fp6ao_matmul = FP6aoHybridMatmul.apply


def rwkv_quantize(quant_type, weight):
    global NowCurrentlyGPUNo
    if quant_type=='4bit':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_4bit((weight.data))
    elif quant_type=='nf4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_nf4((weight.data))
    elif quant_type=='fp4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_fp4((weight.data))
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize((weight.data))
    elif quant_type=='fp8':
        qweight = weight.data.to(dtype = torch.float8_e4m3fn)
        qstate = None
    elif quant_type=='fp6ao':
        weight = weight.to(dtype=torch.float16)
        qweight,qstate = to_scaled_tc_floatx(weight.data, 3 , 2)
    return qweight, qstate


def rwkv_dequantize(quant_type, weight, qstate):
    #with torch.no_grad():
        #device = weight.device
        if quant_type=='4bit':
            deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='nf4':
            deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='fp4':
            deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='int8':
            deweight= bnb.functional.dequantize(weight.data,state=qstate).to(torch.bfloat16)
        elif quant_type=='fp8':
            deweight= weight.data.to(torch.bfloat16)
        elif quant_type=='fp16':
            deweight= weight.data
        return deweight#.to(torch.bfloat16).contiguous()
#@torch.compile
def LinearForward(self,x,passthrough = False):
    if self.is_quant:
            if self.bonemode: # Covered All quantize method. currently slow implementation
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

                    if self.doramode:
                        lora_weight = self.lora_B @ self.lora_A
                        weight_combined = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate) + self.scaling * lora_weight
                        norm = weight_combined.norm(dim=0, keepdim=True) + 1e-6
                        norm = norm.detach()
                        W_eff = (self.lora_M * weight_combined) / norm  
                        out = F.linear(x, W_eff)
                        #print(out)
                        return out
                
                    return (fp8_matmul(x,self.Qweight.t()) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                    )
                
                
                elif self.quant_type == 'fp6ao': #FP6 TorchAO
                    return (fp6ao_matmul(x,self.Qweight.t(),self.qstate) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                    )
                else:
                    w = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                    if passthrough:
                        return F.linear(x, w)
                    
                    if self.doramode:
                        lora_weight = self.lora_B @ self.lora_A
                        weight_combined = w + self.scaling * lora_weight
                        norm = weight_combined.norm(dim=0, keepdim=True) + 1e-6
                        norm = norm.detach() 
                        W_eff = (self.lora_M * weight_combined) / norm  # shape: (out_features, in_features)
                        out = F.linear(x, W_eff)
                        return out
                    
                    if w.dtype == torch.float16:
                        return (fp16_matmul(x,w.t()) + self.scaling *
                                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                        )
                    return ( 
                            F.linear(x, w) + self.scaling *
                            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 
                
    else: # Non Quant mode
        if passthrough:
                return F.linear(x, self.weight)
        if self.bonemode:
            if passthrough:
                return F.linear(x, self.weight)
            w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
            return F.linear(x,self.weight+w)
        if self.doramode:
            lora_weight = self.lora_B @ self.lora_A
            weight_combined = self.weight + self.scaling * lora_weight
            norm = weight_combined.norm(dim=0, keepdim=True) + 1e-6
            norm = norm.detach()  
            W_eff = (self.lora_M * weight_combined) / norm 
            out = F.linear(x, W_eff)
            return out
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


    
class NormalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int, pname=''):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        #assert bias == False, "Biased LoraLinear not supported"
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
            self.biasmode = True
        else:
            self.biasmode = False
    def forward(self, x,passthrough = False):
        if self.biasmode == True:
            return F.linear(x,self.weight) + self.bias
        else:
            return F.linear(x,self.weight)
    

class LoraLinear(nn.Module): # from RWKV-PEFT @JL-er Thanks :) Chaos Modified
    #@torch.jit.unused
    def __init__(self, in_features: int, out_features: int, bias: bool, n_layer: int=-1, pname=''):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        #assert bias == False, "Biased LoraLinear not supported"
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
            self.biasmode = True
        else:
            self.biasmode = False

        TargetName = f'{str(n_layer)}'
        if n_layer < 0:
            TargetName = 'head'

        if LAYER_CONFIG[TargetName]['mode'] == 'bone':
            print('bone mode')
            r = int(LAYER_CONFIG[TargetName]['rank'])
            self.r = r
            self.bone = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
            self.bonemode = True
        else:
            self.doramode = False
            if LAYER_CONFIG[TargetName]['mode'] == 'dora':
                #DoRA: Weight-Decomposed Low-Rank Adaptation
                with torch.no_grad():
                    # self.weight は shape (out_features, in_features)
                    # 各列の L2 ノルムを計算（shape: (1, in_features)）
                    m_init = self.weight.norm(dim=0, keepdim=True)
                self.lora_M = nn.Parameter(m_init) #momemtum
                
                self.doramode = True
            r = int(LAYER_CONFIG[TargetName]['rank'])
            alpha = int(LAYER_CONFIG[TargetName]['alpha'])
            d = LAYER_CONFIG[TargetName]

            dropout = float(LAYER_CONFIG[TargetName]['dropout'])

            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(dropout)
            self.scaling = alpha / r
            self.r = r
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            #nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
            self.bonemode = False
        self.pissa = False
        self.is_quant = False
    #@torch.jit.ignore
    def dora_init(self):
        self.doramode = True
        if self.is_quant:
            temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
            with torch.no_grad():
                # self.weight は shape (out_features, in_features)
                # 各列の L2 ノルムを計算（shape: (1, in_features)）
                m_init = temporal_weight.norm(dim=0, keepdim=True) #temporal_weight#.norm(p=2, dim=0, keepdim=True)
            self.lora_M = nn.Parameter(m_init) #momemtum
            print(self.lora_M)
        else:
            with torch.no_grad():
                    # self.weight は shape (out_features, in_features)
                    # 各列の L2 ノルムを計算（shape: (1, in_features)）
                    m_init = self.weight.norm(dim=0, keepdim=True)
            self.lora_M = nn.Parameter(m_init) #momemtum

 
    #@torch.jit.ignore
    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        self.Qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight).to(device=target_gpu))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete
    #@torch.jit.ignore
    def forward(self, x,passthrough = False):
        if self.biasmode == True:
            return LinearForward(self,x,passthrough) + self.bias
        else:
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
        #assert bias == False, "Biased QuantLinear not supported"
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
            self.biasmode = True
        else:
            self.biasmode = False
            
        self.is_quant = False
    #@torch.jit.ignore
    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        self.Qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight).to(device=target_gpu))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete
    #@torch.jit.ignore
    def forward(self, x,passthrough=False):

        if self.biasmode == True:
            if self.is_quant:
                if self.quant_type == 'fp8':
                    return fp8_matmul(x,self.Qweight.t()) + self.bias
                elif self.quant_type == 'fp6ao':
                    return fp6ao_matmul(x,self.Qweight.t(),self.qstate) + self.bias
                return F.linear(x, rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)) + self.bias
            else:
                return F.linear(x, self.weight) + self.bias
        else:
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
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'full' and Reject == False:
        return NormalLinear(*args, **kwargs)    
    elif LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze' or Reject == True:
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
    if LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'full' and Reject == False:
        return NormalLinear(*args, **kwargs)    
    elif LAYER_CONFIG[f'{str(layer_id)}']['mode'] == 'freeze' or Reject == True:
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
    

@functools.wraps(LoraLinear)
def make_linear_head(*args, **kwargs):
    if LAYER_CONFIG[f'head']['mode'] == 'full':
        return NormalLinear(*args, **kwargs)    
    elif LAYER_CONFIG[f'head']['mode'] == 'freeze':
        return NormalLinear(*args, **kwargs)
    else:
        return LoraLinear(*args, **kwargs)
    

#@functools.wraps(LoraEmbedding)
def make_emb(*args, **kwargs):
    if LAYER_CONFIG[f'emb']['mode'] == 'full' or LAYER_CONFIG[f'emb']['mode'] == 'freeze':
        return nn.Embedding(*args, **kwargs)
    else:
        return nn.Embedding(*args, **kwargs)
        #return LoraEmbedding(*args, **kwargs)