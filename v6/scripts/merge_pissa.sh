base_model='myfolder/models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth'
lora_init='myfolder/Outputs/3BDistillation/init_pissa.pth'
lora_checkpoint='myfolder/Outputs/3BDistillation/rwkv-2.pth'
output='myfolder/Outputs/3BDistillation/rwkv-2-merged.pth'
QUANT='' #follow train
TYPE='pissa'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--lora_init $lora_init \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
