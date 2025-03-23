########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
import requests
import json

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl
    from src.layerprofiler import LayerProfiler,v7_additional_parameters
    #from lightning.pytorch.strategies import SingleDeviceStrategy, FSDPStrategy, DDPStrategy, DeepSpeedStrategy
    #from lightning.pytorch.accelerators.accelerator import Accelerator

    #LayerProfiler l_profile
    



    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--load_adapter", default="", type=str)  # full path, with .pth
    parser.add_argument("--load_cold_adapter", default="", type=str)  # full path, with .pth
    parser.add_argument("--load_adapter_pissa_init", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    #parser.add_argument("--data_file", default="default_text_document", type=str)
    #parser.add_argument("--data_type", default="binidx", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=2048, type=int) #maximum context size
    parser.add_argument("--infctx", default=0, type=int) #from RWKV-PEFT :)
    parser.add_argument("--infctx_dataset_multiplier", default=100, type=int) #from RWKV-PEFT :)
    parser.add_argument("--chunk_ctx", default=512, type=int)

    parser.add_argument("--state", default=0, type=int) #for state-tuning x060
    parser.add_argument("--suffix_offset", default=0, type=int) #for offset state-tuning on x070 
    parser.add_argument("--prefix_tuning", default=0, type=int) #for offset state-tuning on x070 
    parser.add_argument("--state_output_mode", default=1, type=int) #0: state in MainAdapter, 1: Separate MainAdapter,State 2: Separate state in MainAdapter, State

    parser.add_argument("--fla", default=0, type=int)

    parser.add_argument("--moe", default=0, type=int)
    parser.add_argument("--moe_experts", default=8, type=int)
    parser.add_argument("--moe_active", default=2, type=int)
    parser.add_argument("--moe_shared", default=1, type=int)
    parser.add_argument("--moe_balance_alpha", default=0.01, type=float)

    parser.add_argument("--zerocot", default=0, type=int)

    parser.add_argument("--grpo", default=0, type=int)
    parser.add_argument("--grpo_debug", default=1, type=int)
    parser.add_argument("--grpo_gen_count", default=4, type=int)
    parser.add_argument("--grpo_gen_length", default=1024, type=int)
    parser.add_argument("--grpo_gen_temperature", default=1.0, type=float)
    parser.add_argument("--grpo_gen_topp", default=0.7, type=float)
    parser.add_argument("--grpo_kl_beta", default=0.1, type=float)


    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"
    #parser.add_argument("--max_epochs", default=500, type=int) 
    parser.add_argument("--micro_bsz", default=1, type=int)  # micro batch size (batch size per GPU) maybe not working on lisa
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=1e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    
    parser.add_argument("--lr_advanced", default=1, type=int) #Schedule from CustomLR LayerProfile

    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=1, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)


    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=64, type=int)  # deepspeed bucket size in MB. 200 seems enough

    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)

    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x060', type=str) # if RWKV x070, set 'x070'
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser.add_argument("--gpu_arch",default="cuda",type=str)# if CUDA set cuda, but if rocm and 4bit need custom bitsandbytes for rocm
    parser.add_argument("--layer_profile",default='layerprofile/24_test_bone.csv',type=str)
    parser.add_argument("--quant", default=1, type=int) #Quantize NF4 on LoRA Layers
    parser.add_argument("--quant_mode", default='nf4', type=str) #Quantize NF4 on LoRA Layers or freezing

    parser.add_argument("--limited_lora", default=0, type=int)

    parser.add_argument("--svd_niter", default=4, type=int) # for PIZZA
    


    parser.add_argument("--dpo", default=0, type=int)
    parser.add_argument("--dpo_alpha", default=0.1, type=float) 
    parser.add_argument("--dpo_beta", default=0.01, type=float)

    parser.add_argument("--orpo", default=0, type=int) #orpo
    parser.add_argument("--orpo_mode", default=0, type=int) #orpo
    parser.add_argument("--orpo_alpha", default=0.01, type=float) #orpo

    parser.add_argument("--simpo", default=0, type=int)
    parser.add_argument("--simpo_alpha", default=0.3, type=float) 
    parser.add_argument("--simpo_beta", default=0.01, type=float)
    parser.add_argument("--simpo_gamma", default=0.00, type=float)

    parser.add_argument("--wpo", default=0, type=int)
    parser.add_argument("--wpo_alpha", default=0.3, type=float) 
    parser.add_argument("--wpo_beta", default=0.01, type=float)


    parser.add_argument("--rlhf_max_corpus_len", default=1024, type=int) #limit maximum dpo dataset token per dpo item. if avoid OoM decrease this value
    parser.add_argument("--rlhf_train_file", default="trainset.save", type=str)#need pytorch tensor type input 

    #Hyper Parameters Distillation
    parser.add_argument("--distillation", default=0, type=int)
    parser.add_argument("--temperature", default=2.0, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--smoothing", default=0.001, type=float)
    parser.add_argument("--top_k", default=100, type=int)

    


    #Hyper Parameters SFT(masked)
    parser.add_argument("--sft", default=0, type=int)
    parser.add_argument("--sft_jsonmode", default=0, type=int)
    parser.add_argument("--sft_jsonmode_overlap_tokenshift", default=1, type=int)
    parser.add_argument("--sft_jsonmode_tokenizermode", default='world', type=str)
    parser.add_argument("--train_data_file", default='datasets/test_jp_logits.h5', type=str)
    parser.add_argument("--random_mode", default=1, type=int)

    # QRWKV,ARWKV,PRWKV Stage2 Distillation Compatible 
    # for more Anarchy Training :)
    parser.add_argument("--sft_kl_mode", default=0, type=int)
    parser.add_argument("--sft_kl_accesspoint", default='http://localhost:10000', type=str)
    parser.add_argument("--sft_kl_targetmodel", default='myfolder/Phi-4-mini-instruct', type=str)
    parser.add_argument("--sft_kl_loadin4bit", default=1, type=int)
    parser.add_argument("--sft_kl_temperature", default=2.0, type=float)
    parser.add_argument("--sft_kl_alpha", default=0.5, type=float)
    parser.add_argument("--sft_kl_topk", default=2000, type=int)

    #new optim
    parser.add_argument("--optim", default="", type=str)

    #parser.add_argument("--accelerator", default="gpu", type=str)

    parser.add_argument("--rms_norm_eps", default=1e-6, type=float)

    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_MY_ARCHITECTURE"] = args.gpu_arch
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)

    if args.infctx:
        os.environ["RWKV_TRAIN_TYPE"]='infctx'
        os.environ["RWKV_CTXLEN"] = str(args.chunk_ctx)
    elif args.state:
        os.environ["RWKV_TRAIN_TYPE"]='state'
        print('State-tuning Mode')
    else:
        os.environ["RWKV_TRAIN_TYPE"]='normal'

    if args.fla:
        os.environ["FLA_MODE"] = "1"
    else:
        os.environ["FLA_MODE"] = "0"

    if args.moe:
        os.environ["CustomModel"] = "MoE"
    else:
        os.environ["CustomModel"] = ""


    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        if 'x070' in args.my_testing:
            args.dim_ffn = int((args.n_embd * 4.0))

     
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)



    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-6 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.train_data_file}, ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    #assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    #assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16-true"

    ########################################################################################################
    from src.config import LAYER_CONFIG,update_layer_config

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset
    
    
    if args.dpo or args.orpo or args.simpo or args.wpo:
        
        if '.save' in args.rlhf_train_file:
            from src.dpodataset import DPODataset
            dpo_train_data = DPODataset(args)
            os.environ["H5_MODE"] = "0"
        else:
            print('h5 RLHF file mode')
            os.environ["H5_MODE"] = "1"
            from src.rlhfdataset import RLHFDataset
            dpo_train_data = RLHFDataset(args,args.rlhf_train_file,args.ctx_len)

    if args.zerocot or args.grpo:
        print('h5 CoT-RL file mode')
        os.environ["H5_MODE"] = "1"
        from src.cotdataset import RLHFDataset
        dpo_train_data = RLHFDataset(args,args.rlhf_train_file,args.ctx_len)

    

    if args.distillation:
        from src.distillationdataset import HDF5TopKTensorDataset,collate_fn
        distillation_data = HDF5TopKTensorDataset(args,args.train_data_file,args.top_k,args.ctx_len)
    elif args.sft:

        if args.sft_jsonmode:
            from src.jsonldataset import JSONLOnDemandOffsetDataset,collate_fn
            sft_data = JSONLOnDemandOffsetDataset(args,args.train_data_file,args.sft_jsonmode_tokenizermode,args.ctx_len)
        else:
            from src.sftdataset import HDF5TopKTensorDataset,collate_fn
            filename = ''
            if os.path.isfile(args.train_data_file):
                filename = args.train_data_file
            elif os.path.isfile(args.rlhf_train_file) and '.h5' in args.rlhf_train_file:
                filename = args.rlhf_train_file
            sft_data = HDF5TopKTensorDataset(args,filename,args.ctx_len)



    #else:
    #    train_data = MyDataset(args)




    #args.vocab_size = train_data.vocab_size
    
    from src.model import RWKV #, LoraLinear
    

    
    
    

    quant_mode = "none"
    if args.quant:
        quant_mode = args.quant_mode

    l_profile = LayerProfiler(args.layer_profile)
    LAYER_CONFIG = l_profile.make_layer_config(args.n_layer,quant_mode)
    update_layer_config(LAYER_CONFIG)

    print(LAYER_CONFIG)
    


    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu", mmap=True)
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        raise "Please check model correctly"
    

    AdapterMethod = 'lora'
    for i in range(args.n_layer):
        if LAYER_CONFIG[f'{str(i)}']['mode']=='pissa':
            AdapterMethod = 'pissa'
            break
        if LAYER_CONFIG[f'{str(i)}']['mode']=='bone':
            AdapterMethod = 'bone'
            break


    Realtime_Quant = False
    if (AdapterMethod == 'lora' or AdapterMethod == 'bone') and args.quant:
        Realtime_Quant = True

    print(Realtime_Quant)
 

    #os.environ["RWKV_GLOBAL_NO"] = "0"

    @rank_zero_only
    def FirstProcess():
        with open('internaltemp.dat', 'w') as f:
            f.write('0')
        time.sleep(0.5)

    FirstProcess()

    ColdAdapter = None
    if os.path.isfile(args.load_cold_adapter):
        print('Found ColdAdapter')
        ColdAdapter = torch.load(args.load_cold_adapter, map_location="cpu")
    

    model = RWKV(args,load_dict,ColdAdapter,realtime_quant=Realtime_Quant)
    #model = RWKV(args,load_dict=None,realtime_quant=Realtime_Quant)

    #exit()

    

    #Full Freeze 
    model.requires_grad_(False)


    for name, module in model.named_modules():
        
        if any(n.startswith("emb.") for n, _ in module.named_parameters()) and LAYER_CONFIG['emb']['mode']=='full':
            for pname, param in module.named_parameters():
                if 'emb.weight'==pname:
                    print(f'  EMB additionally training module {pname}')
                    param.requires_grad = True
        if any(n.startswith("head.") for n, _ in module.named_parameters()) and LAYER_CONFIG['head']['mode']=='full':
            for pname, param in module.named_parameters():
                if 'head.weight'==pname:
                    print(f'  head additionally training module {pname}')
                    param.requires_grad = True



                    
        if any(n.startswith("lora_") for n, _ in module.named_parameters()):
            print(f'  LoRA additionally training module {name}')
            for pname, param in module.named_parameters():
                param.requires_grad = 'lora_' in pname
                print(f'LoRA Parts Enabled Training :{pname}')

        elif any(n.startswith("bone") for n, _ in module.named_parameters()):
            print(f'  bone additionally training module {name}')
            for pname, param in module.named_parameters():
                param.requires_grad = 'bone' in pname
                print(f'bone Parts Enabled Training :{pname}')

 
        
        elif ('ln0' in name or 'ln_out' in name) and args.limited_lora == 0:
            for param in module.parameters():
                print(f'  additionally training module {name}')
                param.requires_grad = True


        
        if 'x070' in os.environ["RWKV_MY_TESTING"] or 'xa070' in os.environ["RWKV_MY_TESTING"]:# and args.limited_lora == 0:
            #print('x070 Additional Parameters')
            for i in range(args.n_layer):
                text = f'blocks.{str(i)}.'

                for pname, param in module.named_parameters():
                    for targetname in v7_additional_parameters:
                        if (targetname in pname and targetname != '' and pname != '') and text in pname and (LAYER_CONFIG[f'{str(i)}']['mode'] != 'freeze' and args.limited_lora == 0):
                            print(f'x070 Additional Parameters {pname}')
                            param.requires_grad = True
                            break

                    

        for i in range(args.n_layer):
            text = f'blocks.{str(i)}.'
            for pname, param in module.named_parameters():
                #print(f'pname = {pname}')
                if LAYER_CONFIG[f'{str(i)}']['mode'] == 'full' and text in pname:
                    print(f'  FullParameter additionally training parameter {pname}')
                    param.requires_grad = True
                elif LAYER_CONFIG[f'{str(i)}']['mode'] == 'freeze' and text in pname and ('time_state' in pname or 'time_offset' in pname) :
                    print(f'  State-tuning additionally training parameter {pname}')
                    param.requires_grad = True
                elif LAYER_CONFIG[f'{str(i)}']['mode'] == 'freeze' and text in pname:
                    print(f'frozen {pname}')
                else:
                    #print(len(LAYER_CONFIG[f'{str(i)}']['RejectParts']))
                    if any(word in pname for word in LAYER_CONFIG[f'{str(i)}']['RejectParts']) and LAYER_CONFIG[f'{str(i)}']['RejectParts'][0] != "" and text in pname:
                        print(f'{pname} train rejected')
                    elif ('ln_x' in pname or 'ln1' in pname or 'ln2' in pname or 'time' in pname or (i == 0 and 'ln' in pname)) and text in pname and (LAYER_CONFIG[f'{str(i)}']['mode'] != 'freeze' and args.limited_lora == 0):
                        print(f'Additional training FullParameter {pname}')
                        param.requires_grad = True

    #exit()
                





    if os.path.isfile(args.load_adapter) and AdapterMethod == 'lora':
        model.load_state_dict(torch.load(args.load_adapter, map_location="cpu"),
                              strict=False)
        
    if os.path.isfile(args.load_adapter) and AdapterMethod == 'bone':
        model.load_state_dict(torch.load(args.load_adapter, map_location="cpu"),
                              strict=False)
        


    if args.quant and Realtime_Quant == False:
        rank_zero_info(f"########## Quant... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                    m.quant(args.quant_mode,model.target_gpu)
                    print(f'{name} Quant on {model.target_gpu}')
                    #rank_zero_info(f'{name} Quant')
    
     

    if pl.__version__[0]=='2':
        trainer = Trainer(sync_batchnorm=True,accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
        logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
        )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")


        if args.sft_kl_mode == 1:
            LOAD_MODEL_URL = f"{args.sft_kl_accesspoint}/LoadModel"
            #PROCESS_LOGITS_URL = f"{args.sft_kl_accesspoint}/ProcessLogits"

            def test_load_model(LOAD_MODEL_URL,model_name="myfolder/Phi-4-mini-instruct", use_4bit=True, use_cuda=True):
                """LoadModelエンドポイントをテスト"""
                payload = {
                    "modelname": model_name,
                    "use_4bit": use_4bit,
                    "use_cuda": use_cuda
                }
                response = requests.post(LOAD_MODEL_URL, json=payload)
                print(f"LoadModelレスポンス (ステータス: {response.status_code}):")
                print(json.dumps(response.json(), indent=2))
                return response.status_code == 200
            
            use_4bit = False
            if args.sft_kl_loadin4bit:
                use_4bit = True
            model_name = args.sft_kl_targetmodel

            test_load_model(LOAD_MODEL_URL,model_name,use_4bit,True) # Load in GPU0 CUDA

    print(trainer.strategy.config)
    #exit()

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["overlap_comm"] = False
        trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    # if args.precision == 32:
    #     print('fp32 mode')
    #     for pname, param in model.named_modules():
    #             print(f'{pname} is changed to fp32')
    #             param = param.to(dtype=torch.float32)

    print(os.environ["RWKV_TRAIN_TYPE"])


    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    if args.orpo or args.dpo or args.simpo or args.wpo or args.zerocot or args.grpo:
       # if args.dpo == 1:
       #     args.dpo = 0
        print("RLHF Mode") 
        #from pytorch_lightning.trainer.supporters import CombinedLoader
        #data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
        dpo_loader = DataLoader(dpo_train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True, collate_fn=lambda x:x)

        #combined_loader = CombinedLoader([data_loader, dpo_loader], "min_size")
        trainer.fit(model, dpo_loader)
    if args.distillation:
        print('Distillation Training Mode')
        print('This feature is still in experiment')
        print('')
        data_loader = DataLoader(distillation_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True, collate_fn=collate_fn)
        #print('')
        trainer.fit(model, data_loader)
    if args.sft:
        print('SFT Training Mode')
        print('This feature is still in experiment')
        print('')
        data_loader = DataLoader(sft_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True, collate_fn=collate_fn)
        #print('')
        trainer.fit(model, data_loader)
    #if args.dpo:
    #    print("Direct Preference Optimization Mode Enabled.") 
    #    from pytorch_lightning.trainer.supporters import CombinedLoader
    #    dpo_loader = DataLoader(dpo_train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True, collate_fn=lambda x:x)
    #    combined_loader = CombinedLoader([data_loader, dpo_loader], "min_size")
    #    trainer.fit(model, combined_loader)
    else:
        data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
        trainer.fit(model, data_loader)
