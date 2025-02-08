base_model='myfolder/models/rwkv-x070-2b9-world-v3-82%trained-20250203-ctx4k.pth'
lora_checkpoint='../rwkv-5.pth'
output='myfolder/models/rwkv-x070-2b9-cjev4-instruct-e5.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
