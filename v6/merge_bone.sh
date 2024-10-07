base_model='models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_checkpoint='Outputs/1b6-Code-bone/rwkv-0.pth'
output='Outputs/1b6-Code-bone/rwkv-0-merged.pth'
QUANT='' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
