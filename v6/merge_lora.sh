base_model='models/RWKV-6-World-0.4B-v2-20231113-ctx4096.pth'
lora_checkpoint='0B4-Distillation-lgtm/rwkv-2.pth'
output='0B4-Distillation-lgtm/rwkv0b4-lgtm-2.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type '$TYPE' \
--lora_scaling $Lora_scaling
