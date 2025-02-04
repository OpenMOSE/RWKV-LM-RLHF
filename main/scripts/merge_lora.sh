base_model='myfolder/models/rwkv-x070-2b9-world-v3-82%trained-20250203-ctx4k.pth'
lora_checkpoint='myfolder/Outputs/ZeroCoT2/rwkv-3.pth'
output='myfolder/models/2b9-reinforce.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
