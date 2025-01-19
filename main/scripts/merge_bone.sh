base_model='myfolder/models/rwkv-x070-2b9-world-v3-40%trained-20250113-ctx4k.pth'
lora_checkpoint='myfolder/Outputs/rwkv7-2b9-40p-gen3.pth'
output='myfolder/Outputs/rwkv-x070-2b9-cje-instruct-1.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
