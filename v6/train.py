########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl
    from src.layerprofiler import LayerProfiler
    #from lightning.pytorch.strategies import SingleDeviceStrategy, FSDPStrategy, DDPStrategy, DeepSpeedStrategy
    #from lightning.pytorch.accelerators.accelerator import Accelerator

    #LayerProfiler l_profile
    



    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--load_adapter", default="", type=str)  # full path, with .pth
    parser.add_argument("--load_adapter_pissa_init", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    #parser.add_argument("--data_file", default="default_text_document", type=str)
    #parser.add_argument("--data_type", default="binidx", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=2048, type=int) #maximum context size
    parser.add_argument("--infctx", default=1, type=int) #from RWKV-PEFT :)
    parser.add_argument("--infctx_dataset_multiplier", default=100, type=int) #from RWKV-PEFT :)
    parser.add_argument("--chunk_ctx", default=512, type=int)



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
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=1, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    #parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    #parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    #parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=64, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    #parser.add_argument("--my_sample_len", default=0, type=int)
    #parser.add_argument("--my_ffn_shift", default=1, type=int)
    #parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    #parser.add_argument("--my_pos_emb", default=0, type=int)
    #parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    #parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x060', type=str)
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
    parser.add_argument("--orpo_mode", default=1, type=int) #orpo
    parser.add_argument("--orpo_alpha", default=0.01, type=float) #orpo


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
    parser.add_argument("--train_data_file", default='datasets/test_jp_logits.h5', type=str)
    parser.add_argument("--random_mode", default=1, type=int)

    #new optim
    parser.add_argument("--optim", default="", type=str)

    #parser.add_argument("--accelerator", default="gpu", type=str)








    

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
    else:
        os.environ["RWKV_TRAIN_TYPE"]='normal'



    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

     
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    # if args.my_pile_stage > 0:
    #     magic_prime_bak = args.magic_prime

    #     if args.my_pile_shift < 0:
    #         args.my_pile_shift = 0

    #     if magic_prime_bak > 0:
    #         args.magic_prime = magic_prime_bak
    #     if args.my_qa_mask == 2:
    #         args.epoch_count = 2 * args.magic_prime // 40320
    #     else:
    #         args.epoch_count = args.magic_prime // 40320

    #     args.epoch_steps = 40320 // args.real_bsz
    #     assert args.epoch_steps * args.real_bsz == 40320
    #     # if args.my_pile_stage == 2:
    #     #     assert args.lr_final == args.lr_init
    #     if args.my_pile_stage >= 2:  # find latest saved model
    #         list_p = []
    #         for p in os.listdir(args.proj_dir):
    #             if p.startswith("rwkv") and p.endswith(".pth"):
    #                 p = ((p.split("-"))[1].split("."))[0]
    #                 if p != "final":
    #                     if p == "init":
    #                         p = -1
    #                     else:
    #                         p = int(p)
    #                     list_p += [p]
    #         list_p.sort()
    #         max_p = list_p[-1]
    #         if len(list_p) > 1:
    #             args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
    #         if max_p == -1:
    #             args.load_model = f"{args.proj_dir}/rwkv-init.pth"
    #         else:
    #             args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
    #             if args.warmup_steps < 0:
    #                 if args.my_pile_stage == 2:
    #                     args.warmup_steps = 10
    #                 else:
    #                     args.warmup_steps = 30
    #         args.epoch_begin = max_p + 1

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
    
    
    if args.dpo or args.orpo:
        from src.dpodataset import DPODataset
        dpo_train_data = DPODataset(args)

    

    if args.distillation:
        from src.distillationdataset import HDF5TopKTensorDataset,collate_fn
        distillation_data = HDF5TopKTensorDataset(args,args.train_data_file,args.top_k,args.ctx_len)
    elif args.sft:
        from src.sftdataset import HDF5TopKTensorDataset,collate_fn
        sft_data = HDF5TopKTensorDataset(args,args.train_data_file,args.ctx_len)
    #else:
    #    train_data = MyDataset(args)




    #args.vocab_size = train_data.vocab_size
    
    from src.model import RWKV , LoraLinear
    

    
    
    

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

    #os.environ["RWKV_GLOBAL_NO"] = "0"

    @rank_zero_only
    def FirstProcess():
        with open('internaltemp.dat', 'w') as f:
            f.write('0')
        time.sleep(0.5)

    FirstProcess()
    

    model = RWKV(args,load_dict,realtime_quant=Realtime_Quant)
    #model = RWKV(args,load_dict=None,realtime_quant=Realtime_Quant)

    #exit()

    

    #Full Freeze 
    model.requires_grad_(False)


    for name, module in model.named_modules():
        # if len(args.load_model) == 0:
        #     if any(n.startswith("emb.") for n, _ in module.named_parameters()) and LAYER_CONFIG['emb']['mode']=='full':
        #         for pname, param in module.named_parameters():
        #             if 'emb.weight'==pname:
        #                 print(f'  EMB additionally training module {pname}')
        #                 param.requires_grad = True
        #     if any(n.startswith("head.") for n, _ in module.named_parameters()) and LAYER_CONFIG['head']['mode']=='full':
        #         for pname, param in module.named_parameters():
        #             if 'head.weight'==pname:
        #                 print(f'  head additionally training module {pname}')
        #                 param.requires_grad = True
        #     if 'ln' in name:
        #         print(f'  LoRA additionally training module {name}')
        #         for param in module.parameters():
        #             param.requires_grad = True
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

        #elif enable_ln_finetune and '.ln' in name:
        elif '.ln_x' in name:# and args.limited_lora == 0:
            for param in module.parameters():
                print(f'  additionally training module {name}')
                param.requires_grad = True
        elif ('ln_in' in name or 'ln_out' in name) and args.limited_lora == 0:
            for param in module.parameters():
                print(f'  additionally training module {name}')
                param.requires_grad = True
        #elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
        elif (any(n.startswith("time") for n, _ in module.named_parameters())) and args.limited_lora == 0:
            for pname, param in module.named_parameters():
                if pname.startswith("time"):
                    print(f'  LoRA additionally training parameter {pname}')
                    param.requires_grad = True
                    

        for i in range(args.n_layer):
            text = f'blocks.{str(i)}.'
            for pname, param in module.named_parameters():
                #print(f'pname = {pname}')
                if LAYER_CONFIG[f'{str(i)}']['mode'] == 'full' and text in name:
                    print(f'  FullParameter additionally training parameter {name}')
                    param.requires_grad = True


    #if args.lisa:
    #    model.requires_grad_(True)
    #    for name, param in model.named_parameters():
    #        if 'blocks' in name and str(args.n_layer-1) not in name:
    #            param.requires_grad = False
    #            print(f"Freezed: {name}")  # 凍結したパラメータの名前を表示
    #        elif 'blocks' in name and ('ffn' in name or(('att') in name and ('receptance'in name or 'key' in name or 'value' in name) )) and str(args.n_layer-1) in name:
    #            param.requires_grad = False
    #            print(f"Freezed: {name}")  # 凍結したパラメータの名前を表示

    

    #if args.load_partial == 1:
    #    load_keys = load_dict.keys()
    ###    for k in model.state_dict():
    #        if k not in load_keys:
    #            load_dict[k] = model.state_dict()[k]
    #if args.anarchy_mode:
    #    model.load_state_dict(load_dict,strict=False)
    #else:
    #    model.load_state_dict(load_dict,strict=False)

    if AdapterMethod == 'pissa':
        init_dict = {}
        rank_zero_info(f"########## Init PISSA... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                m.pissa_init(args.svd_niter)
                init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                init_dict[f'{name}.init_lora_B'] = m.lora_B.data
        torch.save(init_dict, f'{args.proj_dir}/init_pissa.pth')

    if os.path.isfile(args.load_adapter) and AdapterMethod == 'lora':
        model.load_state_dict(torch.load(args.load_adapter, map_location="cpu"),
                              strict=False)
        
    if os.path.isfile(args.load_adapter) and AdapterMethod == 'bone':
        model.load_state_dict(torch.load(args.load_adapter, map_location="cpu"),
                              strict=False)
        
    if os.path.isfile(args.load_adapter) and AdapterMethod == 'pissa':
        model.load_state_dict(torch.load(args.load_adapter, map_location="cpu"),
                            strict=False)
        pissa_init = torch.load(args.load_adapter_pissa_init, map_location="cpu")
        rank_zero_info(f"########## Load PISSA... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                m.pissa_load(pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

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

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["overlap_comm"] = False

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    if args.orpo or args.dpo:
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
