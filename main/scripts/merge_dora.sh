base_model='myfolder/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth'
lora_checkpoint='myfolder/Outputs/x070-r1-infctx2/rwkv-5.pth'
output='myfolder/models/RWKV-x070-1.5B-R1-SFT-20250219-ctx32768.pth'
QUANT='' #follow train
TYPE='dora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
