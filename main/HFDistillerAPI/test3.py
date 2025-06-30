import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "/home/client/Projects/llm/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, use_fast=False)
inputs = tokenizer('Today is', return_tensors='pt').to(device)

model_eager = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="sdpa").cuda(device)
#model_ckFAv2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention").cuda(device)

print("eager GQA: ", tokenizer.decode(model_eager.generate(**inputs, max_new_tokens=8192)[0], skip_special_tokens=True))
#print("ckFAv2 GQA: ", tokenizer.decode(model_ckFAv2.generate(**inputs, max_new_tokens=10)[0], skip_special_tokens=True))

#  eager GQA:  Today is the day of the Lord, and we are the
# ckFAv2 GQA: Today is the day of the Lord, and we are the