base_model='/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/7b-sft-code2/rwkv-2-merged.pth'
lora_init='myfolder/Outputs/7B-RLHF-DPO/init_pissa.pth'
lora_checkpoint='myfolder/Outputs/7B-RLHF-DPO/rwkv-9.pth'
output='myfolder/Outputs/7B-RLHF-DPO/rwkv-9-merged.pth'
QUANT='' #follow train
TYPE='pissa'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--lora_init $lora_init \
--output $output \
--type $TYPE \
--lora_scaling $Lora_scaling
