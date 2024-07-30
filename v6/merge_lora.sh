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
