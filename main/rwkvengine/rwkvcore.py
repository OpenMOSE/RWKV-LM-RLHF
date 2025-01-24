#Refactoring RWKV x060,x070 Inference Engine with Flash Linear Attention
# Experimental Implement x070
#2024 OpenMOSE

#Test Torchao
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
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script


from rwkvengine.rwkv6 import RWKV_6, fused_recurrent_rwkv6_torch
from rwkvengine.rwkv7 import RWKV_7



class RWKV_x(nn.Module):

    def __init__(self,load_model: str,base_precision: str = 'int8',adapter_model:str = '', adapter_mode:str = '', adapter_scale:float=2.0,fully_fusedrecurrent:bool=True):

        #print('Helloworld RWKV v060 :) Initializing')
        print('RWKV-Infer RWKVCore Initializing')

        super().__init__()
        self.transfer_stream = torch.cuda.Stream()

        #GANBATTE CODE KAKOU
        self.bit8quant = False
        self.bit4quant = False
        self.bitfp8quant = False
        self.bitfp6quant = False

        self.fully_fusedrecurrent = fully_fusedrecurrent

        self.ExtremeCPUOffload = False

        self.debug = False

        self.eval()

        if base_precision == 'fp16':
            self.base_precision = torch.half
        elif base_precision == 'int8':
            print('int8 Duplicated Automatic Change to NF4')
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = True
        elif base_precision == 'fp16int8':
            print('int8 Duplicated Automatic Change to NF4')
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = True
        elif base_precision == 'nf4':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = True
        elif base_precision == 'fp8':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = True
        elif base_precision == 'fp6':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = False
            self.bitfp6quant = True
            self.bitfp5quant = False
        elif base_precision == 'fp5':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = False
            self.bitfp6quant = True
            self.bitfp5quant = True
        else:
            self.base_precision = torch.bfloat16
        
        modelpath = load_model

        z = torch.load(modelpath,map_location="cpu",mmap=True)
        z_adapter_keys = None
        self.ModeMode = 'standard'
        if adapter_model != '' and adapter_mode != '':
            print('Adapter LoadMode')
            if 'lora' in adapter_mode or 'LoRA' in adapter_mode:
                print('LoRA Mode Lets Merge!')
                self.ModeMode = 'lora'
                z_adapter = torch.load(adapter_model,map_location="cpu",mmap=True)
                z_adapter_keys = list(z_adapter.keys())
                for zkeys in z_adapter_keys:
                    z[zkeys] = z_adapter[zkeys]

            elif 'bone' in adapter_mode or 'Bone' in adapter_mode:
                print('Bone(Block Affine Transformation) Mode Lets Merge!')
                self.ModeMode = 'bone'
                z_adapter = torch.load(adapter_model,map_location="cpu",mmap=True)
                z_adapter_keys = list(z_adapter.keys())
                for zkeys in z_adapter_keys:
                    z[zkeys] = z_adapter[zkeys]
                print(f'adapter keys = {z_adapter_keys}')
                #exit()

        def Attach_Adapter(keyname,weight,adapter,mode,scaling=2.0,device='cuda'): #from JL-er lora merge inspired
            
            print(f'AttachAdapter = {keyname}')
            if keyname.endswith('.weight') or keyname.endswith('head'):
                adapterkeys = list(adapter.keys())
                #print(adapterkeys)
#                for k in adapterkeys:
                if mode == 'lora':
                    print(f'scaling = {scaling}')
                    prefix = keyname[:-len('.weight')]
                    lora_A = prefix + '.lora_A'
                    lora_B = prefix + '.lora_B'
                    if lora_A in adapterkeys:
                        w=adapter
                        assert lora_B in adapterkeys
                        print(f'lora merging {lora_A} and {lora_B} into {k}')
                        
                        assert w[lora_B].shape[1] == w[lora_A].shape[0]

                        w[lora_A] = w[lora_A].to(device=device)
                        
                        w[lora_B] = w[lora_B].to(device=device)
                        
                        weight = weight + w[lora_B] @ w[lora_A] * scaling
                        del w[lora_A]
                        del w[lora_B]
                        return weight
                    for key in adapterkeys:
                        if key == keyname:
                            weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                            print(f'key = {key} is swapped from Adapter')
                    return weight
                elif mode == 'bone':
                    prefix = keyname[:-len('.weight')]
                    gbmm = prefix + '.bone'
                    print(f'gbmm target = {gbmm}')
                    if gbmm in adapterkeys:
                        w=adapter
                        print(f'bone merging {gbmm} into {k}')
                        w[gbmm] = w[gbmm].to(device=device)
                        b,r,_ = w[gbmm].shape
                        bone = rearrange(weight, '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                        weight += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')
                        print(weight)
                        del w[gbmm]
                        return weight

                    for key in adapterkeys:
                        if key == keyname:
                            weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                            print(f'key = {key} is swapped from Adapter')
                    return weight
                else:
                    return weight
            else:
                adapterkeys = list(adapter.keys())
                for key in adapterkeys:
                    if key == keyname:
                        weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                        print(f'key = {key} is swapped from Adapter')

                return weight
            
                   

        keys = list(z.keys())
        print("keys", keys)



        RWKVMode = 6 #default RWKV 6

        for key in keys:
            if 'blocks.0.att.r_k' in key:
                print("RWKV x070 Mode :) with Native Pytorch Implementation")
                RWKVMode = 7
                break

        if z_adapter_keys is not None:
            for key in z_adapter_keys:
                if 'blocks.0.att.r_k' in key:
                    print("RWKV x070 Mode :) with Native Pytorch Implementation")
                    RWKVMode = 7
                    break

        if RWKVMode == 6:
            print('RWKV x060 Mode :) with Flash-Linear-Attention')

        self.RWKVMode = RWKVMode
     

        

        # detect model details
        vocab_size, n_embd = z["emb.weight"].shape
        print(f'vocab = {vocab_size}')
        print(f'n_embd = {n_embd}')

        self.n_embd = n_embd
        self.vocab_size = vocab_size

        

        n_layer = 0
        for key in keys:
            if key.startswith("blocks."):
                layer = int(key.split(".")[1])
                if layer > n_layer:
                    n_layer = layer

        n_layer = n_layer + 1
        print("n_layer", n_layer)

        if self.RWKVMode == 7:
            self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
            print(self.head_size)
            z['emb.weight'] = F.layer_norm(z['emb.weight'], (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
            z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
            z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
            z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored
        else:
            dim_ffn = z[f"blocks.0.ffn.value.weight"].shape[1]
            print(f'dim_ffn = {dim_ffn}')
            n_head = z[f"blocks.0.att.time_faaaa"].shape[0]
            print("n_head", n_head)
            self.head_size = n_embd // n_head
            self.dim_ffn = dim_ffn
            self.n_head = n_head
            self.ctx = 1024 #FLA Window

        
        self.n_layer = n_layer
        
        

        keys = list(z.keys())

        self.requires_grad_(False)

        QuantList = ['.receptance.weight','.key.weight','.value.weight','.gate.weight','.output.weight','head.weight']
        QuantListFP8 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight'] #, ,
        QuantListFP6 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight'] #, ,
 
        # 4bit Quantize Mode via Bitsandbytes NF4
        if self.bit4quant == True:
            for k in keys:

                if self.ModeMode != 'standard':
                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')

                QuantKeyFound = False
                for QuantKey in QuantList:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to NF4')
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.bfloat16) 
                        z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if self.RWKVMode == 6:
                        if k.endswith('.time_decay'): z[k] = z[k].float()
                        elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                        elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        else:
                            z[k] = z[k].to(dtype = self.base_precision)
                    elif self.RWKVMode == 7:
                        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                            print(f'target = {k} shape = {z[k].shape}')
                            z[k] = z[k].t()
                        if k.endswith('att.r_k'): z[k] = z[k].flatten()
                        z[k] = z[k].squeeze().to(dtype=self.base_precision)

        # FP8 Transformer Engine Quantize Mode 
        elif self.bitfp8quant == True:
            for k in keys:
                print(f' k = {k} shape = {z[k].shape}' )
                if self.ModeMode != 'standard':
                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')

                if self.ModeMode != 'standard':
                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')

                QuantKeyFound = False
                for QuantKey in QuantListFP8:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to torch.float8_e4m3fn')
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.float8_e4m3fn).contiguous() 
                       
                if QuantKeyFound == False:
                    for QuantKey in QuantList:
                        if k.endswith(QuantKey):
                            print(f'Quant {k} PassThrough')
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if self.RWKVMode == 6:
                        if k.endswith('.time_decay'): z[k] = z[k].float().contiguous() 
                        elif k.endswith('.time_faaaa'): z[k] = z[k].float().contiguous() 
                        elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        else:
                            z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif self.RWKVMode == 7:
                        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                            print(f'target = {k} shape = {z[k].shape}')
                            z[k] = z[k].t()
                        if k.endswith('att.r_k'): z[k] = z[k].flatten()
                        z[k] = z[k].squeeze().to(dtype=self.base_precision)

        # FP6 Quantize Mode via Torch.AO
        elif self.bitfp6quant == True:
            if self.bitfp5quant:
                self.ebits, self.mbits = 2, 2
            else:
                self.ebits, self.mbits = 3, 2
            for k in keys:
                if self.ModeMode != 'standard':
                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')
                QuantKeyFound = False
                for QuantKey in QuantListFP6:
                    if k.endswith(QuantKey):
                        if self.bitfp5quant:
                            print(f'Quant {k} to FP5 shape = {z[k].shape}' )
                        else:
                            print(f'Quant {k} to FP6 shape = {z[k].shape}' )
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.float16)#.t() 

                        # pre-process the weight. this will quantize the weight to FP6 and pack it in a special
                        # layout for tensor cores. refer to paper for more details.
                        z[k], z[k+'.qstate'] = to_scaled_tc_floatx(z[k], self.ebits, self.mbits)

                        if self.ExtremeCPUOffload:
                            z[k] = z[k].to(device='cpu')
                            z[k+'.qstate'] = z[k+'.qstate'].to(device='cpu')

                        #z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])

                if QuantKeyFound == False:
                    for QuantKey in QuantList:
                        if k.endswith(QuantKey):
                            print(f'Quant {k} PassThrough')
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                            z[k+'.qstate'] = torch.randn(1)
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if self.RWKVMode == 6:
                        if k.endswith('.time_decay'): z[k] = z[k].float()
                        elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                        elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                        elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                        else:
                            z[k] = z[k].to(dtype = self.base_precision)
                    elif self.RWKVMode == 7:
                        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                            print(f'target = {k} shape = {z[k].shape}')
                            z[k] = z[k].t()
                        if k.endswith('att.r_k'): z[k] = z[k].flatten()
                        z[k] = z[k].squeeze().to(dtype=self.base_precision)


        # Non Quantize Mode FP16 or BF16
        else:
            for k in keys:
                if self.ModeMode != 'standard':
                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')
                z[k] = z[k].to(device='cuda')

                if self.RWKVMode == 6:
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    if k.endswith('.time_faaaa'): z[k] = z[k].float()
                    elif k.endswith('.receptance.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('.key.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('.value.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('.gate.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('.output.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('head.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    else:
                        z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif self.RWKVMode == 7:
                    if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                        print(f'target = {k} shape = {z[k].shape}')
                        z[k] = z[k].t()
                    if k.endswith('att.r_k'): z[k] = z[k].flatten()

                    z[k] = z[k].squeeze().to(dtype=self.base_precision)

        if self.RWKVMode == 6:
            for i in range(n_layer):
                z[f'blocks.{i}.att.time_maa_wkvrg'] = torch.stack([z[f'blocks.{i}.att.time_maa_w'], z[f'blocks.{i}.att.time_maa_k'], z[f'blocks.{i}.att.time_maa_v'], z[f'blocks.{i}.att.time_maa_r'], z[f'blocks.{i}.att.time_maa_g']], dim=0).contiguous()

        self.z = z
        self.device = z['emb.weight'].device
        self.dtype = z['emb.weight'].dtype

        if self.ModeMode != 'standard':
            del z_adapter

        keys = list(z.keys())
        for key in keys:
            print(f'{key} {z[key].shape}')
            if '.bone' in key or '.lora' in key:
                z[key] = None
                print(f'{key} deleted')

        
    
        gc.collect()
        torch.cuda.empty_cache()


    def new_state(self, B):
         if self.RWKVMode == 6:
            return BlockStateList.create(
                    self.n_layer, B, self.n_embd, 
                    self.n_head,# self.head_size,
                    self.device, self.dtype
                )
         elif self.RWKVMode == 7:
            return BlockStateList.x070_create(self.n_layer,
                                              B,
                                              self.n_embd,
                                              self.head_size,
                                              self.device,
                                              self.dtype)
             
    
    
    
    
        
    #@torch.compile
    def x060_forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
        # if self.bit8quant == True:
        #     StrategyMode = 1
        if self.bit4quant == True:
            StrategyMode = 2
        elif self.bitfp6quant == True:
            StrategyMode = 3
            
        with torch.no_grad():
            #
            z = self.z
            H = self.n_head

            x = RWKV_6.x060_First(z['emb.weight'],idx,self.n_embd,z['blocks.0.ln0.weight'],z['blocks.0.ln0.bias'])
            x=x.to(dtype=self.base_precision)
            B, T, C = x.size()

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'



                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                if StrategyMode == 0:
                    r,k,v,g,w,xx = RWKV_6.x060_TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:
                        att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = RWKV_6.x060_TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight']
                                               )
                    

                    x = x + att1

                    ffn1, channel_mix_state = RWKV_6.x060_ChannelMix_FC_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'],
                                                                     z[ffn+'key.weight'],
                                                                     z[ffn+'value.weight']
                                                                     )
                    
                    x = x + ffn1



                elif StrategyMode == 2: #NF4 Mode

                    r,k,v,g,w,xx = RWKV_6.x060_TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],

                                                    bnb.functional.dequantize_4bit(z[att+'receptance.weight'],
                                                                                   quant_state=z[att+'receptance.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'key.weight'],
                                                                                   quant_state=z[att+'key.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'value.weight'],
                                                                                   quant_state=z[att+'value.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'gate.weight'],
                                                                                   quant_state=z[att+'gate.weight.qstate']) ,



                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:

                        att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:

                        if B < 16:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = RWKV_6.x060_TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               bnb.functional.dequantize_4bit(z[att+'output.weight'],
                                                                                   quant_state=z[att+'output.weight.qstate'])
                                               )
                    
                    x = x + att1

                    ffn1, channel_mix_state = RWKV_6.x060_ChannelMix_FC_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     bnb.functional.dequantize_4bit(z[ffn+'receptance.weight'],
                                                                                   quant_state=z[ffn+'receptance.weight.qstate']),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'key.weight'],
                                                                                   quant_state=z[ffn+'key.weight.qstate']),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'value.weight'],
                                                                                   quant_state=z[ffn+'value.weight.qstate']),
                                                                     )
                    
                    x = x + ffn1


                elif StrategyMode == 3: #FP6
                    r,k,v,g,w,xx = RWKV_6.x060_TimeMix_FC_FP6_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'].to(device='cuda'),
                                                    z[att+'receptance.weight.qstate'].to(device='cuda'),
                                                    z[att+'key.weight'].to(device='cuda'),
                                                    z[att+'key.weight.qstate'].to(device='cuda'),
                                                    z[att+'value.weight'].to(device='cuda'),
                                                    z[att+'value.weight.qstate'].to(device='cuda'),
                                                    z[att+'gate.weight'].to(device='cuda'),
                                                    z[att+'gate.weight.qstate'].to(device='cuda'),
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias'],
                                                    self.ebits, self.mbits
                                                    )
                    if T>1:
                        att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = RWKV_6.x060_TimeMix_FC_FP6_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight'].to(device='cuda'),
                                               z[att+'output.weight.qstate'].to(device='cuda'),
                                               self.ebits, self.mbits
                                               )
                    # att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                    #                            z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                    #                            z[att+'output.weight']
                    #                            )
                    

                    x = x + att1

                    ffn1, channel_mix_state = RWKV_6.x060_ChannelMix_FC_FP6_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'].to(device='cuda'),
                                                                     z[ffn+'receptance.weight.qstate'].to(device='cuda'),
                                                                     z[ffn+'key.weight'].to(device='cuda'),
                                                                     z[ffn+'key.weight.qstate'].to(device='cuda'),
                                                                     z[ffn+'value.weight'].to(device='cuda'),
                                                                     z[ffn+'value.weight.qstate'].to(device='cuda'),
                                                                     self.ebits, self.mbits
                                                                     )
                    # ffn1, channel_mix_state = self.ChannelMix_FC_Step1(x,channel_mix_state,
                    #                                                  z[bbb+'ln2.weight'],
                    #                                                  z[bbb+'ln2.bias'],
                    #                                                  int(self.n_embd),
                    #                                                  z[ffn+'time_maa_k'],
                    #                                                  z[ffn+'time_maa_r'],
                    #                                                  z[ffn+'receptance.weight'],
                    #                                                  z[ffn+'key.weight'],
                    #                                                  z[ffn+'value.weight']
                    #                                                  )
                    
                    x = x + ffn1
                
                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state


            if self.bit4quant:
                x = RWKV_6.x060_Final(x,bnb.functional.dequantize_4bit(z['head.weight'],quant_state=z['head.weight.qstate']),
                               self.n_embd,z['ln_out.weight'],z['ln_out.bias'])
                # x = self.Final_NF4(x,z['head.weight'],z['head.weight.qstate'],
                #                self.n_embd,z['ln_out.weight'],z['ln_out.bias'])
            elif self.bitfp6quant:
                x = RWKV_6.x060_Final_FP6(x,z['head.weight'].to(device='cuda'),
                                   z['head.weight.qstate'].to(device='cuda'),
                            self.n_embd,z['ln_out.weight'],z['ln_out.bias'],
                            self.ebits, self.mbits
                            )
            else:
                x = RWKV_6.x060_Final(x,z['head.weight'],self.n_embd,z['ln_out.weight'],z['ln_out.bias'])            

            return x, last_shift_states, last_wkv_states
        
    def load_state(self,state_filename):
        try:
            state_raw = torch.load(state_filename, map_location="cpu")
        except Exception as e:
            print(e)
            return "error"
        state_raw_shape = next(iter(state_raw.values())).shape

        #args = model.args
        self.debug = 1
        if self.debug:
            print(f"{len(state_raw)} != {self.n_layer}")
            print(f"{state_raw_shape[0] * state_raw_shape[1]} != {self.n_embd}")

        if (
            len(state_raw) != self.n_layer
            or state_raw_shape[0] * state_raw_shape[1] != self.n_embd
        ):
            print("state failed to load")
            return "error"

        #strategy = model.strategy

        model_current_statetuned = [None] * self.n_layer * 3

        dev = 'cpu'

        for i in range(self.n_layer):
            #dd = strategy[i]
            #dd.device
            atype = torch.bfloat16 #dd.atype
            model_current_statetuned[i * 3 + 0] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()
            model_current_statetuned[i * 3 + 1] = (
                state_raw[f"blocks.{i}.att.time_state"]
                .transpose(1, 2)
                .to(dtype=torch.float, device=dev)
                .requires_grad_(False)
                .contiguous()
            )
            model_current_statetuned[i * 3 + 2] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()

        wkv_states = torch.empty((self.n_layer, self.n_head, self.n_embd//self.n_head, self.n_embd//self.n_head),
                                 device=dev,
                                 dtype=torch.bfloat16)
        
        for i in range(self.n_layer):
            wkv_states[i] = model_current_statetuned[i*3 + 1]

        return wkv_states#.to(dtype=torch.float16)
    


    
    
    def x070_forward_one(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor] ):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, time_mix_shift, time_mix_state, v_first = self.x070_TimeMix_one(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                #print(f'Before ChannelMix LayerNorm x.shape = {x.shape}')

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, channel_mix_state = self.x070_ChannelMix_one(xx, channel_mix_state, z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

                last_shift_states[i*2] = time_mix_shift.view(time_mix_shift.shape[0],-1)
                last_shift_states[i*2+1] = channel_mix_state.view(channel_mix_state.shape[0],-1)
                
                last_wkv_states[i] = time_mix_state
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            #exit()
            return x, last_shift_states, last_wkv_states
        
    def x070_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)

            StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
            if self.bit4quant == True:
                StrategyMode = 2
            elif self.bitfp6quant == True:
                StrategyMode = 3





            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                B, T, X = xx.shape

                if StrategyMode == 0:
                    # r1,w1,k1,v1,g1,aa1,bb1,xx_step11 = RWKV_7.x070_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                    #                                                     z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    #                                                     z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    #                                                     z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    #                                                     z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    #                                                     z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                    if B<4 and T == 1:
                        #x070_TimeMix_one_hybrid
                        xx, time_mix_shift, time_mix_state, v_first = RWKV_7.x070_TimeMix_one_hybrid(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                                                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                                                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                                                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                                                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                                                        z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                    else:
                        r,w,k,v,g,aa,bb,xx_step1,v_first = RWKV_7.x070_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                
                        xx_step2, time_mix_state = RWKV_7.x070_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = RWKV_7.x070_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,g,z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first,z[att+'ln_x.weight'], z[att+'ln_x.bias'])










                # xx, time_mix_shift, time_mix_state, v_first = RWKV_7.x070_TimeMix_fla_combined(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                #     z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                #     z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                #     z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                #     z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                #     z[att+'ln_x.weight'], z[att+'ln_x.bias'])

                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, channel_mix_state = RWKV_7.x070_ChannelMix_seq(xx, channel_mix_state, z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = hybrid_matmul(x , z['head.weight'])
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states
        
    def x070_forward(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor], full_output=False,one_mode=False,KernelMode = 0):
        #if one_mode:
        #    return self.x070_forward_one(idx, last_shift_states, last_wkv_states)
        return self.x070_forward_seq(idx, last_shift_states,last_wkv_states, full_output,KernelMode)
    

    def forward(self, idx, last_shift_states , last_wkv_states, one_mode = False, KernelMode = 0,full_output=False):
        if self.RWKVMode == 6:
            return self.x060_forward(idx,last_shift_states,last_wkv_states)
        elif self.RWKVMode == 7:
            return self.x070_forward(idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output)
    













    