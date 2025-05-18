########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import functools
import os, math, gc, importlib
import torch
import time
import requests
import json
import time
import threading

from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

from einops import rearrange

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.ops.lion import DeepSpeedCPULion, FusedLion

from .infctx_module import *
from .trainutils import *

from .mudamw import MuAdamW
from bitsandbytes.optim import Adam8bit,AdamW8bit

from .trainers.zerocot import *
from .trainers.grpo import grpo_init,training_step_grpo
from .trainers.sft import training_step_sft, training_step_sft_infctx, training_sft_infctx_init
from .trainers.simpo import training_step_simpo
from .trainers.wpo import training_step_wpo
from .trainers.dpo import training_step_dpo
from .trainers.orpo import training_step_orpo



if 'x070' in os.environ["RWKV_MY_TESTING"]:
    from .models.rwkv7 import LAYER_CONFIG,RWKV_Tmix_x070,RWKV_Tmix_x070_state,RWKV_Tmix_x070_infctx,RWKV_CMix_x070,RWKV_CMix_x070_MoLE,RWKV_CMix_x070_infctx,RWKV_Tmix_x070m,make_linear_head,make_emb
if 'xa07' in os.environ["RWKV_MY_TESTING"]:
    from .models.arwkv7 import LAYER_CONFIG,ARWKV_Tmix_x070,ARWKV_Tmix_x070_state,ARWKV_Tmix_x070_infctx,Qwen2MLP,Qwen2MLP_infctx,Qwen2RMSNorm,Phi35MLP,Phi35MLP_infctx,make_linear_head,make_emb
    from .models.prwkv7 import LAYER_CONFIG,PRWKV_Tmix_cxa075,PRWKV_Tmix_cxa075_infctx,PRWKV_Tmix_cxa076,PRWKV_Tmix_cxa076_infctx
elif 'x060' in os.environ["RWKV_MY_TESTING"]:
    from .models.rwkv6 import LAYER_CONFIG,RWKV_Tmix_x060,RWKV_Tmix_x060_state,RWKV_Tmix_x060_infctx,RWKV_CMix_x060,RWKV_CMix_x060_infctx,make_linear_head,make_emb
else:
    assert "Unsupported RWKV Architecture. please set xa070 or x070 or x060"
try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

NowCurrentlyGPUNo = 0

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

            if 'x070m' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x070m(args, layer_id)  
            else:
                if os.environ["RWKV_TRAIN_TYPE"] == 'state':
                    self.att = RWKV_Tmix_x070_state(args, layer_id) 
                elif os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.att = RWKV_Tmix_x070_infctx(args, layer_id) 
                else:
                    self.att = RWKV_Tmix_x070(args, layer_id)  

            if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                self.ffn = RWKV_CMix_x070_infctx(args, layer_id)
            else:
                if os.environ["CustomModel"] == 'MoE':
                    self.ffn = RWKV_CMix_x070_MoLE(args,layer_id,self.args.moe_experts)
                else:
                    self.ffn = RWKV_CMix_x070(args, layer_id)


        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def forward(self, x, v_first,last_state: BlockState, x_emb=None):
                if self.layer_id == 0:
                    x = self.ln0(x)

                x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state)
                x = x + x_attn

                ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)

                x = x + ffn_out
                return x, v_first ,BlockState(att_state, ffn_state)
        elif os.environ["CustomModel"] == 'MoE':
            def forward(self, x, v_first,input_ids):
                if self.layer_id == 0:
                    x = self.ln0(x)

                x_attn, v_first = self.att(self.ln1(x), v_first)
                x = x + x_attn

                ffn_out ,moe_router_loss = self.ffn(self.ln2(x),input_ids)

                x  = x + ffn_out
                return x, v_first,moe_router_loss
        else:
            def forward(self, x, v_first,passthrough = False,x_emb=None):
                if self.layer_id == 0:
                    x = self.ln0(x)

                # x_attn, v_first = self.att(self.ln1(x), v_first, passthrough)
                if self.args.state:
                    x_attn, v_first, out_state = self.att(self.ln1(x), v_first, passthrough)
                else:
                    x_attn, v_first = self.att(self.ln1(x), v_first, passthrough)

                x = x + x_attn

                x = x + self.ffn(self.ln2(x),passthrough)
                if self.args.state:
                    return x, v_first, out_state
                else:
                    return x, v_first
                # return x, v_first
            @torch.no_grad()
            def forward_rnn(self, x, v_first,last_state: BlockState,passthrough=False):
                if self.layer_id == 0:
                    x = self.ln0(x)

                x_attn, v_first, att_state = self.att.forward_rnn(self.ln1(x), v_first, last_state.time_mix_state,passthrough)
                x = x + x_attn

                ffn_out ,ffn_state = self.ffn.forward_rnn(self.ln2(x), last_state.channel_mix_state,passthrough)

                x = x + ffn_out
                return x, v_first ,BlockState(att_state, ffn_state)
if 'xa07' in os.environ["RWKV_MY_TESTING"]:
    class Block(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.ln1 = Qwen2RMSNorm(args.n_embd,args.rms_norm_eps)
            self.ln2 = Qwen2RMSNorm(args.n_embd,args.rms_norm_eps)

            if os.environ["RWKV_TRAIN_TYPE"] == 'state':
                self.att = ARWKV_Tmix_x070_state(args, layer_id) 
            elif os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                if 'cxa075' in os.environ["RWKV_MY_TESTING"]:
                    self.att = PRWKV_Tmix_cxa075_infctx(args, layer_id) 
                elif 'cxa076' in os.environ["RWKV_MY_TESTING"]:
                    self.att = PRWKV_Tmix_cxa076_infctx(args, layer_id) 
                else:
                    self.att = ARWKV_Tmix_x070_infctx(args, layer_id) 
            else:
                if 'cxa075' in os.environ["RWKV_MY_TESTING"]:
                    self.att = PRWKV_Tmix_cxa075(args, layer_id)  
                elif 'cxa076' in os.environ["RWKV_MY_TESTING"]:
                    self.att = PRWKV_Tmix_cxa076(args, layer_id)  
                else:
                    self.att = ARWKV_Tmix_x070(args, layer_id)  

            if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                if 'pxa070' in os.environ["RWKV_MY_TESTING"]:
                    self.ffn = Phi35MLP_infctx(args,layer_id)
                else:
                    self.ffn = Qwen2MLP_infctx(args, layer_id)
            else:
                if 'pxa070' in os.environ["RWKV_MY_TESTING"]:
                    self.ffn = Phi35MLP(args,layer_id)
                else:
                    self.ffn = Qwen2MLP(args, layer_id)


        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def forward(self, x, v_first,last_state: BlockState, x_emb=None):

                x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state)

                x = x + x_attn

                ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)

                x = x + ffn_out
                return x, v_first ,BlockState(att_state, ffn_state)
        else:
            def forward(self, x, v_first,passthrough = False,x_emb=None):
      
                if self.args.state:
                    x_attn, v_first, out_state = self.att(self.ln1(x), v_first, passthrough)
                else:
                    x_attn, v_first = self.att(self.ln1(x), v_first, passthrough)
                x = x + x_attn

                x = x + self.ffn(self.ln2(x),passthrough)

                if self.args.state:
                    return x, v_first, out_state
                else:
                    return x, v_first
            @torch.no_grad()
            def forward_rnn(self, x, v_first,last_state: BlockState,passthrough=False):

                x_attn, v_first, att_state = self.att.forward_rnn(self.ln1(x), v_first, last_state.time_mix_state,passthrough)
                x = x + x_attn

                ffn_out ,ffn_state = self.ffn.forward_rnn(self.ln2(x), last_state.channel_mix_state,passthrough)

                x = x + ffn_out
                return x, v_first ,BlockState(att_state, ffn_state)
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

            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.att = RWKV_Tmix_x060_infctx(args, layer_id)
                elif os.environ["RWKV_TRAIN_TYPE"] == 'state':
                    self.att = RWKV_Tmix_x060_state(args, layer_id)
                else:
                    self.att = RWKV_Tmix_x060(args, layer_id)


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

                return x, BlockState(att_state, fnn_state)
        else:
            def forward(self, x, x_emb=None):
                args = self.args
                B, T, C = x.size()
                if self.layer_id == 0:
                    x = self.ln0(x)

                if self.args.dropout == 0:
                    x = x + self.att(self.ln1(x))
                    x = x + self.ffn(self.ln2(x))
                else:
                    x = self.drop0(x + self.att(self.ln1(x)))
                    x = self.drop1(x + self.ffn(self.ln2(x)))

    
                return x
        
 



class RWKV(pl.LightningModule):
    def __init__(self, args,load_dict=None,cold_adapter_dict=None,realtime_quant=False):
        super().__init__()

        device = self.device
        print(f"Device in __init__: {device}")


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

        if cold_adapter_dict is not None:
            cold_adapter_keys = list(cold_adapter_dict.keys())
            mode = 'none'
            for ckeys in cold_adapter_keys:
                if 'lora' in ckeys:
                    mode = 'lora'
                elif 'bone' in ckeys:
                    mode = 'bone'
                load_dict[ckeys] = cold_adapter_dict[ckeys]

            def Attach_Adapter(keyname,weight,adapter,mode,scaling=2.0,device='cuda'): #from JL-er lora merge inspired
                beforeDevice = str(weight.device)
                if beforeDevice == 'cpu':
                    weight = weight.to(device=device)      
        
                print(f'AttachAdapter = {keyname}')
                if keyname.endswith('.weight') or keyname.endswith('head'):
                    adapterkeys = list(adapter.keys())

                    if mode == 'lora':
                        print(f'scaling = {scaling}')
                        prefix = keyname[:-len('.weight')]
                        lora_A = prefix + '.lora_A'
                        lora_B = prefix + '.lora_B'
                        if lora_A in adapterkeys:
                            w=adapter
                            assert lora_B in adapterkeys
                            print(f'lora merging {lora_A} and {lora_B} into {keyname}')
                            
                            assert w[lora_B].shape[1] == w[lora_A].shape[0]
                            
                            lora_r = w[lora_B].shape[1]

                            w[lora_A] = w[lora_A].to(device=device)
                            
                            w[lora_B] = w[lora_B].to(device=device)
                            
                            weight = weight + w[lora_B] @ w[lora_A] * scaling
                            del w[lora_A]
                            del w[lora_B]
                            if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                            return weight
                        for key in adapterkeys:
                            if key == keyname:
                                weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                                print(f'key = {key} is swapped from Adapter')
                        if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                        return weight
                    elif mode == 'bone':
                        prefix = keyname[:-len('.weight')]
                        gbmm = prefix + '.bone'
                        print(f'gbmm target = {gbmm}')
                        if gbmm in adapterkeys:
                            w=adapter
                            print(f'bone merging {gbmm} into {keyname}')
                            w[gbmm] = w[gbmm].to(device=device)
                            b,r,_ = w[gbmm].shape
                            bone = rearrange(weight, '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                            weight += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')
                            print(weight)
                            del w[gbmm]
                            if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                            return weight
                        #adapterkeys = list(adapter.keys())
                        for key in adapterkeys:
                            if key == keyname:
                                weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                                print(f'key = {key} is swapped from Adapter')
                        if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                        return weight
                    else:
                        if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                        return weight
                else:
                    adapterkeys = list(adapter.keys())
                    for key in adapterkeys:
                        if key == keyname:
                            weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                            print(f'key = {key} is swapped from Adapter')
                    #print('no target bone merge')
                    if beforeDevice == 'cpu':
                                weight = weight.to(device='cpu')
                    return weight
            if mode == 'lora' or mode == 'bone':
                print('Cold Adapter Merging Mode')
                load_dict_keys = list(load_dict.keys())
                for key in load_dict_keys:
                    load_dict[key] = Attach_Adapter(key,load_dict[key],cold_adapter_dict,mode)

            if mode != 'none':
                del cold_adapter_dict


        
        if args.prefix_tuning == 1:
            print('Prefix Softtoken tuning enabled')
            self.prefix_token = nn.Parameter(torch.zeros(self.args.prefix_token_len, args.n_embd))


        self.emb = make_emb(args.vocab_size, args.n_embd)


        if 'xa07' in os.environ["RWKV_MY_TESTING"]:
            self.ln_out = Qwen2RMSNorm(args.n_embd,args.rms_norm_eps)
        else:
            self.ln_out = nn.LayerNorm(args.n_embd)



        self.head = make_linear_head(args.n_embd, args.vocab_size, bias=False)

        if args.zerocot:
            self.clitic_head = nn.Linear(args.n_embd, 1, bias=False)

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
                        #print(name)
                        #exit()
                        if hasattr(m, "quant") and callable(getattr(m, "quant")) and f'{str(i)}.' in name:
                                m.quant(args.quant_mode,target_gpu)
                                print(f'{name} Quant')

                if LAYER_CONFIG[f'{str(i)}']['mode'] == 'dora':
                    print('DoRA Mode Norm')
                    for name, m in self.blocks.named_modules():
                        if hasattr(m, "dora_init") and callable(getattr(m, "dora_init")) and f'{str(i)}.' in name:
                                m.dora_init()
                                print(f'{name} dora_init')
                                #exit()

            self.load_element_weights(self,'emb',load_dict)
            self.load_element_weights(self,'ln_out',load_dict)
            self.load_element_weights(self,'head',load_dict)
            if realtime_quant:
                for name, m in self.named_modules():
                    #print(f'pname = {name}')
                    
                    if hasattr(m, "quant") and callable(getattr(m, "quant")) and f'head' in name:
                            m.quant(args.quant_mode,target_gpu)
                            print(f'{name} Quant')

            if LAYER_CONFIG[f'head']['mode'] == 'dora':
                    print('DoRA Mode Norm head')
                    for name, m in self.named_modules():
                        if hasattr(m, "dora_init") and callable(getattr(m, "dora_init")) and f'head' in name:
                                m.dora_init()
                                print(f'{name} dora_init')
            #exit()
        else:
            self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        global NowCurrentlyGPUNo
        self.CurrentCudaNo = NowCurrentlyGPUNo

        print('finish blocks')

        if args.zerocot:
            zerocot_init(self)
        if args.grpo:
            grpo_init(self)
        if args.infctx and args.sft:
            training_sft_infctx_init(self)
        

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
            if key.startswith(block_prefix):
                new_key = key[len(block_prefix):] 
                block_state_dict[new_key] = value


        block.load_state_dict(block_state_dict, strict=False)

    def load_element_weights(self,element,element_name, load_dict):
        block_prefix = element_name
        block_state_dict = {}
        keys_to_delete = []

        for key, value in load_dict.items():
            if key.startswith(block_prefix):
                new_key = key
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
                if ('emb' in n  or 'ln0' in n) and LAYER_CONFIG['emb']['mode'] == 'full':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['emb']['lr_init']), 
                                            'lr_final':float(LAYER_CONFIG['emb']['lr_final']) , 
                                            'weight_decay':float(LAYER_CONFIG['emb']['weight_decay']), 
                                            'pname':'emb'})

                elif ('head' in n or 'ln_out' in n) and LAYER_CONFIG['head']['mode'] != 'freeze':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['head']['lr_init']),
                                            'lr_final':float(LAYER_CONFIG['head']['lr_final']),
                                            'weight_decay':float(LAYER_CONFIG['head']['weight_decay']) ,
                                            'pname':'head'})
                elif ('prefix' in n):
                    if p.requires_grad:
                        print(f"Prefix-tuning {n} Set lr_init {float(LAYER_CONFIG['emb']['lr_init'])} lr_final {float(LAYER_CONFIG['emb']['lr_final'])}")
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['emb']['lr_init']),
                                            'lr_final':float(LAYER_CONFIG['emb']['lr_final']),
                                            'weight_decay':float(LAYER_CONFIG['emb']['weight_decay']) ,
                                            'pname':'prefix'})
                else:
                    print('Layer Check')
                    Found = False
                    for i in range(args.n_layer):
                        blockname = f'blocks.{i}.'
                        if blockname in n:
                            print(n)
                        if blockname in n and ('time_state' in n or 'time_offset' in n) and args.state:
                            if p.requires_grad:
                                print(f"State-tuning {n} Set lr_init {float(LAYER_CONFIG[f'{str(i)}']['lr_init_state'])} lr_final {float(LAYER_CONFIG[f'{str(i)}']['lr_final_state'])}")
                            
                                optim_groups.append({"params":[param_dict[n]], "weight_decay": 0.0,
                                                    'lr_init':float(LAYER_CONFIG[f'{str(i)}']['lr_init_state']), 
                                                    'lr_final':float(LAYER_CONFIG[f'{str(i)}']['lr_final_state']),  
                                                    'pname':n
                                                    })
                                Found = True
                            break
                        elif blockname in n and LAYER_CONFIG[f'{str(i)}']['mode'] != 'freeze':
                            if any(word in n for word in LAYER_CONFIG[f'{str(i)}']['RejectParts']) and LAYER_CONFIG[f'{str(i)}']['RejectParts'][0] != '':
                                print(f'Rejected {n}')
                                Found = True
                                break

                            lr_x = 1.0
                            if 'time_decay' in n: # for x060
                                lr_x = 2.0

                            if p.requires_grad:
                                print(f"WeightParameter {n} Set lr_init {float(LAYER_CONFIG[f'{str(i)}']['lr_init'])} lr_final {float(LAYER_CONFIG[f'{str(i)}']['lr_final'])}")
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
                if args.optim == 'lion':
                    print('Deepspeed CPULion Mode')
                    return DeepSpeedCPULion(optim_groups, betas=self.args.betas)
                else:
                    return DeepSpeedCPUAdam(optim_groups, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            if args.optim == 'Adam8bit':
                print('Bitsandbytes Adam8bit Mode')
                return Adam8bit(optim_groups,  betas=self.args.betas, eps=self.args.adam_eps)
            elif args.optim == 'AdamW8bit':
                print('Bitsandbytes AdamW8bit Mode')
                return AdamW8bit(optim_groups,  betas=self.args.betas, eps=self.args.adam_eps)
            elif args.optim == 'lion':
                print('Deepspeed Lion Mode')
                return FusedLion(optim_groups, betas=self.args.betas)
            elif args.optim == 'muon':
                print('Muon')
                return MuAdamW(optim_groups, betas=self.args.betas)
            else:
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
                        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    ]
                    print(optim_groups)
                    #exit()
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

            if 'x070' in os.environ["RWKV_MY_TESTING"] or '07' in os.environ["RWKV_MY_TESTING"]:
                v_first = torch.empty_like(x)
                
                for i, (block, block_state) in enumerate(zip(self.blocks,
                    BlockStateList(last_shift_states, last_wkv_states))):
                    if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                        x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, x_emb, use_reentrant=False)
    
                    else:
                        #x, new_block_state = torch_checkpoint(block, x, block_state,x_emb, use_reentrant=False)
                        x, v_first, new_block_state = deepspeed.checkpointing.checkpoint(block, x,v_first,block_state, x_emb)
            
                    new_states[i] = new_block_state 
            else:
                for i, (block, block_state) in enumerate(zip(self.blocks,
                    BlockStateList(last_shift_states, last_wkv_states))):
                    if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                        x, new_block_state = torch_checkpoint(block, x, block_state, x_emb, use_reentrant=False)
    
                    else:
                        #x, new_block_state = torch_checkpoint(block, x, block_state,x_emb, use_reentrant=False)
                        x, new_block_state = deepspeed.checkpointing.checkpoint(block, x,block_state, x_emb)
            
                    new_states[i] = new_block_state 

            x = self.ln_out(x)

            x = self.head(x)

            return x, new_states.shift_states, new_states.wkv_states
        
    
        def training_step(self, batch,batch_idx): #
            args = self.args
            T_train = args.chunk_ctx  #chunk size

            if args.distillation or args.sft:
                return training_step_sft_infctx(self,batch,batch_idx)


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

            return total_loss
        

    
    else: #Normal Trianing Mode = have limit context size 


        def forward_rnn(self, idx,  last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor,passthrough=False):
            args = self.args
            B, T = idx.size()
            #assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."  
            #Autoregressive
            C = args.n_embd
            H =  args.dim_att // args.head_size_a
            assert C==H*args.head_size_a
            
            x = self.emb(idx)
            x_emb = x
            new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                            x.device, x.dtype)
            if args.dropout > 0:
                x = self.drop0(x)

            if 'x070' in os.environ["RWKV_MY_TESTING"] or 'xa07' in os.environ["RWKV_MY_TESTING"]:
                v_first = torch.empty_like(x)
                
                for i, (block, block_state) in enumerate(zip(self.blocks,
                    BlockStateList(last_shift_states, last_wkv_states))):
                  
                    x, v_first, new_block_state = block.forward_rnn(x,v_first, block_state,passthrough)
            
                    new_states[i] = new_block_state 
            else:
                assert "currently only supported v7"

            x = self.ln_out(x)

            x = self.head(x,passthrough)

            return x, new_states.shift_states, new_states.wkv_states
        


        def forward(self, idx,passthrough=False,frozen=False,clitic=False):
            args = self.args
            StatePack = torch.zeros(self.args.n_layer,self.args.n_embd // self.args.head_size_a, self.args.head_size_a,self.args.head_size_a )
            if idx is None:
                B = 1
                T = self.args.prefix_token_len
                x = self.prefix_token.unsqueeze(0).expand(B, -1, -1)
                StatePack = torch.zeros(self.args.n_layer,self.args.n_embd // self.args.head_size_a, self.args.head_size_a,self.args.head_size_a )
            else:
                B, T = idx.size()
                assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
                x = self.emb(idx)
                if self.args.state and self.args.prefix_tuning:
                    Prefix_expanded = self.prefix_token.unsqueeze(0).repeat(B, 1, 1)
                    x = torch.cat([Prefix_expanded, x], dim=1)

            x_emb = x

            if args.dropout > 0:
                x = self.drop0(x)
            if 'x070' in os.environ["RWKV_MY_TESTING"] or 'xa07' in os.environ["RWKV_MY_TESTING"]:
                    v_first = torch.empty_like(x)
                    moe_total_loss = 0
                    i = 0
                    for block in self.blocks:
                        if frozen:
                            x, v_first = block(x, v_first,passthrough)

                        elif args.grad_cp == 1:
                            layer_mode = LAYER_CONFIG[f'{str(block.layer_id)}']['mode']
                            if layer_mode == 'full' or layer_mode == 'freeze':
                                #x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
                                if os.environ["CustomModel"] == 'MoE':
                                    x, v_first,moe_router_loss = torch_checkpoint(block, x, v_first, idx, use_reentrant=False)
                                    moe_total_loss += (moe_router_loss+0.001) / float(args.n_layer)
                                else:
                                    if self.args.state:
                                        x, v_first,out_state = torch_checkpoint(block, x, v_first,passthrough,x_emb,use_reentrant=False)
                                        StatePack[i] = out_state[0]
                                    else:
                                        x, v_first = torch_checkpoint(block, x, v_first,passthrough,x_emb,use_reentrant=False)
                            else:
                                if os.environ["CustomModel"] == 'MoE':
                                    x, v_first ,moe_router_loss = torch_checkpoint(block, x, v_first,idx,use_reentrant=False)
                                    moe_total_loss += (moe_router_loss+0.001) / float(args.n_layer)
                                else:
                                    if self.args.state:
                                        x, v_first,out_state = torch_checkpoint(block, x, v_first,passthrough,x_emb,use_reentrant=False)
                                        StatePack[i] = out_state[0]
                                    else:
                                        x, v_first = torch_checkpoint(block, x, v_first,passthrough,x_emb,use_reentrant=False)
                                #x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first )
                        else:
                            if os.environ["CustomModel"] == 'MoE':
                                x, v_first,moe_router_loss = block(x, v_first,idx)
                                moe_total_loss += (moe_router_loss+0.001) / float(args.n_layer)
                            else:
                                x, v_first = block(x, v_first)
                        i = i + 1
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

            if clitic:
                x= self.clitic_head(x)
            else:
                x = self.head(x)

            if os.environ["CustomModel"] == 'MoE':
                #print(f'Moe Router_loss = {moe_router_loss}')
                return x, moe_router_loss
            if idx is None:
                return StatePack
            else:
                return x, 0.0
        

        def forward_head2(self, idx,passthrough=False,frozen=False,clitic=False):
            
            with torch.no_grad():
                args = self.args
                B, T = idx.size()
                assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

                x = self.emb(idx)
                x_emb = x
                if 'x070' in os.environ["RWKV_MY_TESTING"]:
                        v_first = torch.empty_like(x)
                        moe_total_loss = 0
                        for block in self.blocks:
                            if frozen:
                                x, v_first = block(x, v_first,passthrough)

                x = self.ln_out(x)

            x= self.clitic_head(x)

            return x, 0.0
    
        

        

        def training_step(self, batch, batch_idx):
            
            args = self.args

            if args.zerocot:
                return training_step_zerocot(self,batch,batch_idx)
            if args.grpo:
                return training_step_grpo(self,batch,batch_idx)
            if args.distillation or args.sft:
                return training_step_sft(self,batch,batch_idx)

            if args.simpo:  
                return training_step_simpo(self,batch,batch_idx)
            if self.args.wpo:
                return training_step_wpo(self,batch,batch_idx)
            if args.dpo:
                return training_step_dpo(self,batch,batch_idx)
            if args.orpo:
                return training_step_orpo(self,batch,batch_idx)
                
            if args.my_qa_mask != 1:
                idx, targets = batch
                logits = self(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                idx, targets, mask = batch
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()

                logits = self(idx)
                if sum_mask == mask.shape[0]:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                    loss = torch.sum(loss * mask) / sum_mask

            return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

