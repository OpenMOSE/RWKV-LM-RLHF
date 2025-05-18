base_model='myfolder/models/rwkv7-g1-1.5b-20250429-ctx4096.pth'
lora_checkpoint='myfolder/Outputs/x070-tbptt-maykirihara/rwkv-1.pth'
output='myfolder/Outputs/x070-tbptt-maykirihara/1b5-may.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
