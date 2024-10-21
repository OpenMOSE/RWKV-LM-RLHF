base_model='myfolder/Outputs/3BDistillation/rwkv-2-merged.pth'
lora_checkpoint='myfolder/Outputs/3B-RLHF-ORPO/rwkv-8.pth'
output='myfolder/Outputs/3B-RLHF-ORPO/rwkv-8-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
