base_model='models/RWKV-x060-Jpn-6B.pth'
lora_checkpoint='6B-distillation1/rwkv-0.pth'
output='6B-distillation1/rwkv-6b-jpn-1-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type '$TYPE' \
--lora_scaling $Lora_scaling
