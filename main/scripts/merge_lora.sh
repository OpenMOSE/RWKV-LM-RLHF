base_model='myfolder/models/RWKV7-G1-2.9B-87%25trained-20250511-ctx4k.pth'
lora_checkpoint='myfolder/Outputs/x070-tbptt-maykirihara-2b9/rwkv-2.pth'
output='myfolder/Outputs/x070-tbptt-maykirihara/2b9-may.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
