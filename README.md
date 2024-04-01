# RWKV-LM-LISA

# WARNING: Eternal Debugging, pre-release.
This repo is forked from RWKV-LM

Test implement of Layerwise Importance Sampled AdamW

## Usages
1. Prepare SFT Dataset:
   - Compatible with Original RWKV-LM dataset format
3. Run `train.py`:
   - Currently RWKV-5 and 6 are supported;
   - Maybe can train RWKV5-7b on 32GB VRAM with 3k ctx

## This repo works
   - 1. Freeze all layers
   - 2. Choose active layers(lisa_active_layer) randomly every lisa_interval_steps
   - 3. Foward and backward

I wish performance close to full parameter learning
 
My training command is provided as follows:
```
python train.py --load_model "RWKV-5-World-0.4B-v2-20231113-ctx4096.pth"\
 --wandb "RWKV-LM-LISA 04b" --proj_dir "04b-output2"\
 --data_file "output"\
 --data_type "binidx" --vocab_size 65536 --ctx_len 4096 \
 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 1 --n_layer 24 --n_embd 1024\
 --lr_init 1e-4 --lr_final 1e-6 \
 --warmup_steps 100 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 2 --precision bf16 \
 --grad_cp 1 --my_testing "r2r3r4" \
 --strategy deepspeed_stage_2_offload \
 --lisa 1 \
 --lisa_active_layer 2 \
 --lisa_interval_steps 10 \
 --lisa_debug 1 \
 --lisa_rand_seed 0 \
 --gpu_arch rocm
```

## Todo
   - 1. Make a layer selection probability profile


# And Thanks to:
RWKV-LM @BlinkDL
LMFlow @OptimalScale


# License
same with RWKV-LM

Apache 2.0


@ 2024 OpenMOSE
