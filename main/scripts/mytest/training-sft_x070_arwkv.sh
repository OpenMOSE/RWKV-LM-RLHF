python train.py --load_model "/home/client/Projects/RWKV-Infer/models/ARWKV-7B-Preview-0.1.pth" \
 --wandb "RWKV-LM-RLHF xa070 ARWKV-7B SFT CJE5 test" --proj_dir "myfolder/Outputs/arwkv5cjetest" \
 --vocab_size 152064 --ctx_len 512 \
 --chunk_ctx 1024 \
 --epoch_steps 100 --epoch_count 200 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 1 --n_layer 28 --n_embd 3584 --dim_ffn 18944 \
 --warmup_steps 100 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision 'bf16' \
 --grad_cp 1 --my_testing "xa070" \
 --strategy deepspeed_stage_2_offload \
 --layer_profile 'layerprofile/28_TEST_dora.csv' \
 --fla 1 \
 --infctx 0 \
 --state 0 \
 --quant 1 \
 --quant_mode 'fp8'\
 --gpu_arch 'tritonfla' \
 --limited_lora 0 \
 --sft 1 \
 --smoothing 0.005 \
 --random_mode 1 \
 --infctx_dataset_multiplier 50 \
 --optim 'AdamW8bit' \
 --train_data_file 'myfolder/2024_dataset/may-qwen-tokenizer.h5' \
 --accumulate_grad_batches 1
