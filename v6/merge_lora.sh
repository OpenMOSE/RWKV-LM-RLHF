base_model='models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth'
lora_checkpoint='7b-Code/rwkv-2.pth'
output='7b-Code/rwkv-2-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type '$TYPE' \
--lora_scaling $Lora_scaling
