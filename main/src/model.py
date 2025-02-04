########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from .adopt import ADOPT
import functools
import os, math, gc, importlib
import torch
import time
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

from .adam_mini import Adam_mini

from .zerocot import *


from bitsandbytes.optim import Adam8bit,AdamW8bit



if 'x070' in os.environ["RWKV_MY_TESTING"]:
    from .rwkv7 import LAYER_CONFIG,RWKV_Tmix_x070,RWKV_Tmix_x070_state,RWKV_Tmix_x070_infctx,RWKV_CMix_x070,RWKV_CMix_x070_MoE,RWKV_CMix_x070_infctx,make_linear_head,make_emb
elif 'x060' in os.environ["RWKV_MY_TESTING"]:
    from .rwkv6 import LAYER_CONFIG,RWKV_Tmix_x060,RWKV_Tmix_x060_state,RWKV_Tmix_x060_infctx,RWKV_CMix_x060,RWKV_CMix_x060_infctx,make_linear_head,make_emb
else:
    assert "Unsupported RWKV Architecture. please set x070 or x060"
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
                    self.ffn = RWKV_CMix_x070_MoE(args,layer_id,self.args.moe_experts)
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
            def forward(self, x, v_first,passthrough = False):
                if self.layer_id == 0:
                    x = self.ln0(x)

                x_attn, v_first = self.att(self.ln1(x), v_first, passthrough)
                x = x + x_attn

                x = x + self.ffn(self.ln2(x),passthrough)
                return x, v_first
            @torch.no_grad()
            def forward_rnn(self, x, v_first,last_state: BlockState,passthrough=False):
                if self.layer_id == 0:
                    x = self.ln0(x)

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



        self.emb = make_emb(args.vocab_size, args.n_embd)

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

        if args.zerocot:
            zerocot_init(self)
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
                if ('emb' in n  or 'ln0' in n):# and LAYER_CONFIG['emb']['mode'] == 'full':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['emb']['lr_init']), 
                                            'lr_final':float(LAYER_CONFIG['emb']['lr_final']) , 
                                            'weight_decay':float(LAYER_CONFIG['emb']['weight_decay']), 
                                            'pname':'emb'})
                        #print(optim_groups)
                    #exit()
                elif ('head' in n or 'ln_out' in n) and LAYER_CONFIG['head']['mode']:# != 'freeze':
                    if p.requires_grad:
                        optim_groups.append({"params":[param_dict[n]],
                                            'lr_init':float(LAYER_CONFIG['head']['lr_init']),
                                            'lr_final':float(LAYER_CONFIG['head']['lr_final']),
                                            'weight_decay':float(LAYER_CONFIG['head']['weight_decay']) ,
                                            'pname':'head'})
                        #print(optim_groups)
                else:
                    print('Layer Check')
                    Found = False
                    for i in range(args.n_layer):
                        blockname = f'blocks.{i}.'
                        if blockname in n:
                            print(n)
                        if blockname in n and 'time_state' in n and args.state:
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
                            #if n in  LAYER_CONFIG[f'{str(i)}']['RejectParts'] and len(LAYER_CONFIG[f'{str(i)}']['RejectParts']) > 0:
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

            if 'x070' in os.environ["RWKV_MY_TESTING"]:
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

            
            #print()
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

            if 'x070' in os.environ["RWKV_MY_TESTING"]:
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
            B, T = idx.size()
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

            x = self.emb(idx)
            x_emb = x

            if args.dropout > 0:
                x = self.drop0(x)
            if 'x070' in os.environ["RWKV_MY_TESTING"]:
                    v_first = torch.empty_like(x)
                    moe_total_loss = 0
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
                                    x, v_first = torch_checkpoint(block, x, v_first,passthrough,use_reentrant=False)
                            else:
                                if os.environ["CustomModel"] == 'MoE':
                                    x, v_first ,moe_router_loss = torch_checkpoint(block, x, v_first,idx,use_reentrant=False)
                                    moe_total_loss += (moe_router_loss+0.001) / float(args.n_layer)
                                else:
                                    x, v_first = torch_checkpoint(block, x, v_first,passthrough, use_reentrant=False)
                                #x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first )
                        else:
                            if os.environ["CustomModel"] == 'MoE':
                                x, v_first,moe_router_loss = block(x, v_first,idx)
                                moe_total_loss += (moe_router_loss+0.001) / float(args.n_layer)
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

            if clitic:
                x= self.clitic_head(x)
            else:
                x = self.head(x)

            if os.environ["CustomModel"] == 'MoE':
                #print(f'Moe Router_loss = {moe_router_loss}')
                return x, moe_router_loss
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
    
        def compute_logps_simple_mask(self, chosen_inputs, logits, attention_mask=None):

            log_probs = torch.log_softmax(logits[:, :-1, :], dim=2)

            gathered_log_probs = torch.gather(log_probs, dim=2, index=chosen_inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1]
            else:
                attention_mask = torch.ones_like(gathered_log_probs)

            masked_log_probs = gathered_log_probs * attention_mask
            
            sequence_logps = masked_log_probs.sum(dim=1)
            
            effective_lengths = attention_mask.sum(dim=1)
            
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

            if args.zerocot:
                return training_step_zerocot(self,batch,batch_idx)

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
                student_logits,moe_loss = self(input_ids)

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

                def find_next_128_multiple(n):
                    remainder = n % 128
                    if remainder == 0:
                        return n
                    return n + (128 - remainder)

                if 'x070' in os.environ["RWKV_MY_TESTING"]:
                    max_len = find_next_128_multiple(max_len)
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]




                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]

                # Forward: input_ids[:, :-1]を使用
                student_logits,moe_loss = self(input_ids)

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
                #label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")
                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)


                # Lossの計算
                if sum_mask == mask.shape[0]:
                    loss = smooth_loss.mean()# + (1 - alpha) * kl_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    #kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                    loss = smooth_loss# + (1 - alpha) * kl_loss

                if os.environ["CustomModel"] == "MoE":
                    loss = loss + args.moe_balance_alpha * moe_loss
                    self.trainer.moe_router_loss = moe_loss

                self.trainer.smooth_loss = float(smooth_loss.mean())

                #self.trainer.kl_loss = float(kl_loss.mean())
                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)
            


            
            




            if args.simpo:  
                batch_orpo = batch

                loss1 = 0.0
                loss_simpo_only = 0.0

                # 参考統計用
                try:
                    self.trainer.pref_match_percentage
                except (NameError, AttributeError):
                    self.trainer.pref_match_percentage = 0.0
                pref_matches = 0

                bsz = len(batch_orpo)
                loss2 = 0.0

                for s in range(bsz):
                    if os.environ["H5_MODE"] == "1":
                        chosen_input = batch_orpo[s]['chosen_input']
                        chosen_output = batch_orpo[s]['chosen_target']
                        length_chosen = batch_orpo[s]['chosen_token_len']
                        # ↓SimPOなので参照モデル由来の確率は使わない
                        # chosen_ref_prob = batch_orpo[s]['chosen_base_prob']
                        reject_input = batch_orpo[s]['reject_input']
                        reject_output = batch_orpo[s]['reject_target']
                        length_reject = batch_orpo[s]['reject_token_len']
                        # reject_ref_prob = batch_orpo[s]['reject_base_prob']
                        chosen_token = batch_orpo[s]['chosentoken']
                        reject_token = batch_orpo[s]['rejecttoken']
                    else:
                        # unpackしたときの要素数を合わせる（参照モデル確率は削除）
                        chosen_input, chosen_output, length_chosen, _, reject_input, reject_output, length_reject, _ = batch_orpo[s]

                    # パディング用のマスク作成
                    chosen_mask = (chosen_output != 0).float()
                    reject_mask = (reject_output != 0).float()

                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)
                    max_len = max(len1, len2)

                    # 必要に応じてパディング
                    if 'x070' in os.environ.get("RWKV_MY_TESTING", ""):
                        max_len = args.ctx_len

                    if len1 < max_len:
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))

                    # 選択応答(chosen)と却下応答(reject)をまとめてバッチ化してモデルに通す
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0)
                    RT ,moe_loss= self(SFT_idx)
                    outputs_pos = RT[0].unsqueeze(0)  # chosen応答用
                    outputs_neg = RT[1].unsqueeze(0)  # reject応答用
                    del SFT_idx

                    # クロスエントロピー（SFTロス）用の関数
                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()

                    # SFTロス（chosen応答だけに対して計算する例）
                    l2_pos_loss = masked_cross_entropy(outputs_pos, chosen_output, chosen_mask)

                    # 実際に各トークンの対数確率を取り出す
                    chosen_logits = outputs_pos[:, len1 - length_chosen : len1].squeeze(0)
                    reject_logits = outputs_neg[:, len2 - length_reject : len2].squeeze(0)

                    # トークンインデックスに対応する log-softmax を取得し、平均をとる
                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_token)), chosen_token]
                    chosen_prob = chosen_loss.sum().float() / float(len(chosen_token))  # 平均対数確率
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[torch.arange(len(reject_token)), reject_token]
                    reject_prob = reject_loss.sum().float() / float(len(reject_token))

                    # ====== SimPOの主部分 ======
                    # 報酬: r(chosen) = β * chosen_prob, r(reject) = β * reject_prob
                    # 差分: [r(chosen) - r(reject) - γ] をシグモイドの対数に通して損失化
                    # chosenが勝ち応答なので、この差分を正にするよう学習 → -log( sigmoid( 差分 ) )
                    diff = args.simpo_beta * (chosen_prob - reject_prob) - args.simpo_gamma

                    # 統計用: 差分が正なら「正しい勝ち」になっているとみなす
                    pref_matches += (diff > 0)

                    # ロス計算
                    pref_ratio = -F.logsigmoid(diff)

                    # 最終損失 = (SFTロス × スケール) + SimPOロス
                    final_loss = (l2_pos_loss * args.simpo_alpha) + pref_ratio*(1.0*args.simpo_alpha)

                    # L2Wrapは元コードのHook等で必要なら
                    final_loss = L2Wrap2.apply(final_loss, outputs_pos,outputs_neg)

                    loss2 += final_loss
                    loss1 += l2_pos_loss
                    loss_simpo_only += pref_ratio

                
                loss2 /= bsz
                loss1 /= bsz
                loss_simpo_only /= bsz

                
                self.trainer.loss_2_dpo = float(loss_simpo_only)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.99 * self.trainer.pref_match_percentage + 0.01 * (pref_matches / bsz)

                return loss2
            

            if self.args.wpo:
                # ===== ここからWPO実装 =====
                batch_wpo = batch

                loss1 = 0.0  # いわゆるSFT部分の損失 (l2_pos_loss 累積)
                loss_pref_part = 0.0  # pref_ratio（ペア間の好み）部分
                total_loss = 0.0

                try:
                    self.trainer.pref_match_percentage
                except (NameError, AttributeError):
                    self.trainer.pref_match_percentage = 0.0

                pref_matches = 0
                bsz = len(batch_wpo)

                for s in range(bsz):
                    # データ取り出し
                    if os.environ.get("H5_MODE", "0") == "1":
                        chosen_input = batch_wpo[s]['chosen_input']
                        chosen_output = batch_wpo[s]['chosen_target']
                        length_chosen = batch_wpo[s]['chosen_token_len']
                        chosen_ref_prob = batch_wpo[s]['chosen_base_prob']
                        reject_input = batch_wpo[s]['reject_input']
                        reject_output = batch_wpo[s]['reject_target']
                        length_reject = batch_wpo[s]['reject_token_len']
                        reject_ref_prob = batch_wpo[s]['reject_base_prob']
                        chosen_token = batch_wpo[s]['chosentoken']
                        reject_token = batch_wpo[s]['rejecttoken']
                    else:
                        chosen_input, chosen_output, length_chosen, chosen_ref_prob, \
                            reject_input, reject_output, length_reject, reject_ref_prob = batch_wpo[s]

                    # マスク作成（0をパディングとみなす）
                    chosen_mask = (chosen_output != 0).float()
                    reject_mask = (reject_output != 0).float()

                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)
                    max_len = max(len1, len2)

                    # 必要に応じてパディング
                    if 'x070' in os.environ.get("RWKV_MY_TESTING", ""):
                        max_len = args.ctx_len

                    # パディング処理（右側0埋め）
                    if len1 < max_len:
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))

                    # 一度にまとめて推論: chosenとrejectをバッチで入れる
                    SFT_idx = torch.stack([chosen_input, reject_input], dim=0)  # shape [2, max_len]
                    # ログits計算 [2, max_len, vocab_size]
                    RT ,moe_loss= self(SFT_idx)

                    outputs_pos = RT[0].unsqueeze(0)  # chosen
                    outputs_neg = RT[1].unsqueeze(0)  # reject
                    del SFT_idx


                    def masked_cross_entropy(pred, target, mask):
                        ce_loss = F.cross_entropy(
                            pred.view(-1, pred.size(-1)),
                            target.view(-1),
                            reduction='none'
                        )
                        ce_loss = ce_loss * mask.view(-1)
                        return ce_loss.sum() / (mask.sum() + 1e-9)

                    l2_pos_loss = masked_cross_entropy(outputs_pos, chosen_output, chosen_mask)

                    # (2) 好みの差分 (chosen vs reject) の対数確率を計算
                    # logitsから実際に生成されたtokenだけ抜き出し
                    chosen_logits = outputs_pos[:, len1 - length_chosen:len1].squeeze(0)
                    reject_logits = outputs_neg[:, len2 - length_reject:len2].squeeze(0)

                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[
                        torch.arange(len(chosen_token)),
                        chosen_token
                    ]
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[
                        torch.arange(len(reject_token)),
                        reject_token
                    ]

                    # ここでは平均対数尤度の形
                    chosen_prob = torch.sum(chosen_loss)/float(len(chosen_token))  # 平均log p(y_w)
                    reject_prob = torch.sum(reject_loss)/float(len(reject_token))   # 平均log p(y_l)

                    # (3) DPOタイプのpref_ratio（ただしWPOでも流用）
                    pref_ratio = self.args.wpo_beta * (
                        chosen_prob - reject_prob
                        #- chosen_ref_prob + reject_ref_prob
                    )
                    # 好みを正にするときのシグモイド
                    pref_matches += (pref_ratio > 0)
                    pref_ratio = -F.logsigmoid(pref_ratio)

                    # ============== ここが WPO のカギ ========================
                    # WPO重み計算：
                    # chosen_prob, reject_prob はそれぞれ「平均対数確率」なので
                    # シーケンス全体の対数確率にするため length_chosen, length_reject を掛ける
                    chosen_seq_logprob = chosen_prob# * float(length_chosen)
                    reject_seq_logprob = reject_prob# * float(length_reject)

                    print(f'chosen_seq_logprob = {chosen_seq_logprob}, reject_seq_logprob = {reject_seq_logprob}')

                    # w_chosen, w_reject はそれぞれ e^(シーケンス全体のlog p)
                    w_chosen = torch.exp(chosen_seq_logprob)
                    w_reject = torch.exp(reject_seq_logprob)
                    # 二つを掛け合わせたものを最終的なペアの重みとする
                    w_pair = w_chosen * w_reject

                    # (4) 最終loss: 通常の (SFT部分 + pref部分) にWPOの重み w_pair を掛ける
                    print(f'w_pair = {w_pair}')
                    final_loss = w_pair * (l2_pos_loss * self.args.wpo_alpha + pref_ratio)
                    # ======================================================

                    # もし勾配に対して特殊な処理(L2Wrap等)が必要ならここで
                    final_loss = L2Wrap.apply(final_loss, outputs_pos)

                    loss1 += l2_pos_loss
                    loss_pref_part += pref_ratio
                    total_loss += final_loss

                # バッチサイズで割って平均
                total_loss = total_loss / bsz
                loss1 = loss1 / bsz
                loss_pref_part = loss_pref_part / bsz

                self.trainer.loss_2_dpo = float(loss_pref_part)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = (
                    0.9 * self.trainer.pref_match_percentage
                    + 0.1 * (pref_matches / bsz)
                )

                return total_loss

            


            if args.dpo:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                #print(batch_orpo)


                for s in range(bsz):
                    if os.environ["H5_MODE"] == "1":
                        chosen_input = batch_orpo[s]['chosen_input']
                        chosen_output = batch_orpo[s]['chosen_target']
                        length_chosen = batch_orpo[s]['chosen_token_len']
                        chosen_ref_prob = batch_orpo[s]['chosen_base_prob']
                        reject_input = batch_orpo[s]['reject_input']
                        reject_output = batch_orpo[s]['reject_target']
                        length_reject = batch_orpo[s]['reject_token_len']
                        reject_ref_prob = batch_orpo[s]['reject_base_prob']
                        chosen_token = batch_orpo[s]['chosentoken']
                        reject_token = batch_orpo[s]['rejecttoken']
                    else:
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

                    RT ,moe_loss= self(SFT_idx)


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

                    #loss_chosen = cross_entropy(outputs_pos,chosen_output)
                    #loss_reject = cross_entropy(outputs_neg,reject_output)

                    chosen_logits = outputs_pos[:,len1-length_chosen:len1].squeeze(0)
                    reject_logits = outputs_neg[:,len2-length_reject:len2].squeeze(0)

                    print(f'chosen logits shape = {chosen_logits.shape}')



                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_token)), chosen_token]
                    chosen_prob = (torch.sum(chosen_loss.view(-1))).float() / float(len(chosen_token))
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[torch.arange(len(reject_token)), reject_token]
                    reject_prob = (torch.sum(reject_loss.view(-1))).float() / float(len(reject_token))

                    #chosen_prob = -torch.sum(loss_chosen[len1-length_chosen:len1])/float(length_chosen)
                    #reject_prob = -torch.sum(loss_reject[len2-length_reject:len2])/float(length_reject)

                    print(f'chosen_prob ={chosen_prob} reject_prob={reject_prob}')


                    #reject_prob = -torch.sum(loss_reject[-length_reject:])
                    pref_ratio = args.dpo_beta * (chosen_prob - reject_prob - chosen_ref_prob + reject_ref_prob)
                    pref_matches += (pref_ratio > 0)
                    pref_ratio = - F.logsigmoid(pref_ratio)
                    #pref_ratio = F.softplus(-pref_ratio)
                    

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

                    RT ,moe_loss= self(SFT_idx)


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

                    RT ,moe_loss= self(SFT_idx)


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
