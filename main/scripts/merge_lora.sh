base_model='myfolder/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth'
lora_checkpoint='myfolder/cft3.pth'
output='myfolder/models/cft1b5-3.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
