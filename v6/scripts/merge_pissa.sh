base_model='myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth'
lora_init='myfolder/Outputs/7BDist/init_pissa.pth'
lora_checkpoint='myfolder/Outputs/7BDist/rwkv-1.pth'
output='myfolder/Outputs/7BDist/rwkv-1-merged.pth'
QUANT='' #follow train
TYPE='pissa'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--lora_init $lora_init \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
