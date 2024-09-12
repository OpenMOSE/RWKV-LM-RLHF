base_model='models/x060-6B-prune.pth'
lora_checkpoint='6B-SRD-nsfw2/rwkv-10.pth'
output='6B-SRD-nsfw2/rwkv-6b-nsfw-10-merged.pth'
QUANT='' #follow train
TYPE='lora'
Lora_scaling=2.0

python merge.py --base_model $base_model \
--lora_checkpoint $lora_checkpoint \
--output $output \
--type '$TYPE' \
--lora_scaling $Lora_scaling
