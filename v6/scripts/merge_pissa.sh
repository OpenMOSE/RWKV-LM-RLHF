base_model='/home/client/Projects/RWKV-Infer-FLA/RWKV-Infer/models/RWKV-x060-Jpn-14B-20240819-ctx4096.pth'
lora_init='Outputs/14b-Code/init_pissa.pth'
lora_checkpoint='Outputs/14b-Code/rwkv-1.pth'
output='Outputs/14b-Code/rwkv-1-merged.pth'
QUANT='' #follow train
TYPE='pissa'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--lora_init $lora_init \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
