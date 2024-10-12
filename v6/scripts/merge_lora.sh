base_model='/home/client/Projects/RWKV-Infer-FLA/RWKV-Infer/models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_checkpoint='Outputs/1B6-Code/rwkv-0.pth'
output='Outputs/1B6-Code/rwkv-0-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
