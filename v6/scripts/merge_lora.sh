base_model='myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth'
lora_checkpoint='myfolder/Outputs/7B-RLHF-DPO/rwkv-0.pth'
output='myfolder/Outputs/7B-RLHF-DPO/rwkv-0-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
