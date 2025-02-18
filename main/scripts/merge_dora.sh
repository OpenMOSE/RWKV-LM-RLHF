base_model='myfolder/models/ARWKV-7B-Preview-0.1.pth'
lora_checkpoint='myfolder/adapters/arwkv-cje5-9.pth'
output='myfolder/models/ARWKV-7B-CJE-30%.pth'
QUANT='' #follow train
TYPE='dora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
