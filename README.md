# RWKV-LM-RLHF

# WARNING: This repository is under development
This repo is forked from RWKV-LM

Reinforcement Learning Toolkit for RWKV

Enables ORPO and DPO training for RWKV x060 "finch" architecture LLMs

CUDA and Rocm Supported.(Tested on MI100 Rocm6.1.2 + Bitsandbytes(multi backend branch))

MultiGPU with quantization Supported.(Monkey implement) 

## Refactoring of the ORPO algorithm
A key feature of ORPO is that it allows for simultaneous SFT and aligning. By adjusting the orpo_alpha value, you can control the ratio between SFT and aligning.When the alignment training ratio is high, it seems beneficial to lower the learning rate to around 1e-6.

## Training backend
With layer_profile, you can configure full parameter layers, frozen layers, and adapter layers (LoRA, PISSA) on a per-layer basis.
Frozen layers or adapter layers are compatible with Bitsandbytes NF4 quantization.
This aims to maximize training performance within limited VRAM capacity.

14B L61D4096,NF4,LoRA(A=8,R=16,Blocks only, no head,emb) on Single RTX4090 


## Orpo Usages
1. Prepare Orpo Dataset
   - now only support UTF-8 CSV(keys 'prompt','chosen','reject')
   - if you wanna add reject, 
   - ```python rlhf_generate_reject.py --load_model YOURMODEL --input_csv YOURCSV --output_csv OUTPUTDEST ```
   - Tokenize using RWKV models
   - ```python PrepareOrpoDataset.py --load_model YOURMODEL --input_csv YOURCSV --output_save OUTPUTDEST --target_pair_count 1000 ```
2. Run `train.py`:
   - Configure layer_profile (its better select full layers)
   ![layerconfig.png](layerconfig.png)
   - set --orpo 1 
   - set --orpo_alpha 0.0004 (coefficient while observing the balance between OddsRatio Loss and SFT Loss (e.g., 1:1))
   - set --rlhf_max_corpus_len 1024 Maximum Token limit each prompt,chosen,reject for avoid OoM
   - set --rlhf_train_file 'YOUR TOKENIZED DATA FILENAME'
## DPO Usages
1. This is under development.

I wish performance close to full parameter learning


## Orpo Mode
My orpo training command is provided as follows:
```
python train.py --load_model "models/rwkv-x060-14b-world-v2.1-81%trained-20240527-ctx4k.pth"\
 --wandb "RWKV-LM-RLHF 14b Rocm Test" --proj_dir "14brocm-bf16"\
 --vocab_size 65536 --ctx_len 2048 \
 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 1 --n_layer 61 --n_embd 4096\
 --lr_init 5e-6 --lr_final 1e-6 \
 --warmup_steps 100 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 2 --precision bf16 \
 --grad_cp 1 --my_testing "x060" \
 --strategy deepspeed_stage_1 \
 --layer_profile 'layerprofile/61_TEST_head_emb.csv' \
 --quant 1 \
 --quant_mode 'nf4'\
 --gpu_arch 'rocm' \
 --orpo 1 \
 --orpo_alpha 0.0004 \
 --rlhf_train_file dataset_3b.save \
 --rlhf_max_corpus_len 1024
```

## Model Merge
My model merge command is provided as follows:
```
base_model='models/rwkv-x060-14b-world-v2.1-81%trained-20240527-ctx4k.pth'
lora_checkpoint='14bwo-head-emb/rwkv-1.pth'
output='14bwo-head-emb/rwkv-1-merged.pth'
QUANT='nf4' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type '$TYPE' \
--lora_scaling $Lora_scaling
```

## Todo
   - 1. Re-engineering DPO Algorithm with Gradient Checkpointing
   - 2. SimPO research
   - 3. Self-Play ORPO research
   - 4. Re-engineering LISA+
   - 5. support FLA backend(help me...)


# And Thanks to:
   - RWKV-LM @BlinkDL
   - RWKV-LM-RLHF-DPO @Triang-jyed-driung
   - RWKV-PEFT @Jl-er
   - LMFlow @OptimalScale
   - Orpo @xfactlab




# License
same with RWKV-LM

Apache 2.0


@ 2024 OpenMOSE
