base_model='myfolder/models/rwkv-x070-450m-world-v2.9-65%trained-20241223-ctx4k.pth'
lora_checkpoint='myfolder/Outputs/0b4-x070-test/rwkv-23.pth'
output='myfolder/Outputs/0b4-x070-test/rwkv-23-merged.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
