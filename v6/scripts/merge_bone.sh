base_model='myfolder/models/RWKV-x060-Jpn-14B-20240819-ctx4096.pth'
lora_checkpoint='myfolder/14b-sft/rwkv-0.pth'
output='myfolder/14b-sft/rwkv-0-merged.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
