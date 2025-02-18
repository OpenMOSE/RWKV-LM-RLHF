base_model='myfolder/models/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth'
lora_checkpoint='myfolder/Outputs/x070-0B4-moe-cjev4/rwkv-28.pth'
output='myfolder/models/RWKV-x070-Potato-0.6B-MoLE-20250208-ctx4096.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
