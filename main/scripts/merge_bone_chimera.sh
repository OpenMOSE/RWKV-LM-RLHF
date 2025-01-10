base_model='myfolder/converted/x060-deleted.pth'
lora_checkpoint='myfolder/Outputs/x070-Convertv6/rwkv-2.pth'
output='myfolder/converted/RWKV-x070-JPN.pth'
QUANT='int8' #follow train

python merge_bone.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--quant $QUANT \
--output $output \
