python train.py --load_model "models/rwkv-x060-14b-world-v2.1-81%trained-20240527-ctx4k.pth"\
 --wandb "RWKV-LM-RLHF 14b cuda Test" --proj_dir "14bcuda"\
 --vocab_size 65536 --ctx_len 2048 \
 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 1 --n_layer 61 --n_embd 4096\
 --lr_init 5e-6 --lr_final 1e-6 \
 --warmup_steps 100 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision bf16 \
 --grad_cp 1 --my_testing "x060" \
 --strategy deepspeed_stage_1 \
 --layer_profile 'layerprofile/61_TEST_head_emb.csv' \
 --quant 1 \
 --quant_mode 'nf4'\
 --gpu_arch 'cuda' \
 --orpo 1 \
 --orpo_alpha 0.0004 \
 --rlhf_train_file dataset_3b.save \
 --rlhf_max_corpus_len 1024
