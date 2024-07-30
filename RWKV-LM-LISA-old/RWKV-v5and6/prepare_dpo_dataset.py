import pandas as pd
df = pd.read_parquet("train_prefs-00000-of-00001.parquet")

print(df)

DPO_pair_number = 1000 # from 2 to 61965

import numpy as np
from rwkv.utils import PIPELINE
from rwkv.model import RWKV

model = RWKV("RWKV-5-World-0.4B-v2-20231113-ctx4096.pth", "cuda bf16")

tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

tokenizer.encode("\n\nUser:"), tokenizer.encode("\n\nResponse:")

import random
random.seed(100)
train_percent = 0.8
train_valid_percent = 0.9

trainset = []
validset = []
testset = []

import torch
import torch.nn.functional as F

def compute_base_prob(prompt_tokens, chosen_tokens):
    global tokenizer
    full_logits_chosen, _ = tokenizer.model.forward(prompt_tokens + chosen_tokens[:-1], None, full_output=True)
    chosen_logits = full_logits_chosen[-len(chosen_tokens):]
    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_tokens)), chosen_tokens]
    return float(torch.sum(chosen_loss))


for i in range(DPO_pair_number):
    prompt_str = str(df.iloc[i].prompt).strip()
    print(prompt_str)
    print(df.iloc[i].chosen)
    chosen_str = str(df.iloc[i].chosen[1]["content"]).strip().replace("\n\n", "\n")
    
    print(chosen_str)
    reject_str = str(df.iloc[i].rejected[1]["content"]).strip().replace("\n\n", "\n")
    h = random.random()
    if h < 0.8:
        prompt_str = "User: " + prompt_str + "\n\nAssistant:" # Helpfulness optimization
        chosen_str = ' ' + chosen_str + "\n\n"
        reject_str = ' ' + reject_str + "\n\n"
    elif h<0.9:
        prompt_str = "Question: " + prompt_str + "\n\nAnswer:" # Factuality
        chosen_str = ' ' + chosen_str + "\n\n"
        reject_str = ' ' + reject_str + "\n\n"
    else:
        prompt_str = "Input: " + prompt_str + "\n\nResponse:" # Instruction-following
        chosen_str = ' ' + chosen_str + "\n\n"
        reject_str = ' ' + reject_str + "\n\n"
    prompt_tokens = tokenizer.encode(prompt_str)
    chosen_tokens = tokenizer.encode(chosen_str)
    reject_tokens = tokenizer.encode(reject_str)

    chosen_base_prob = compute_base_prob(prompt_tokens, chosen_tokens)
    reject_base_prob = compute_base_prob(prompt_tokens, reject_tokens)

    print(chosen_base_prob, reject_base_prob)
    # prompt_chosen_mask = [0] * (len(prompt_tokens)-1) + [1] * len(chosen_tokens)
    # prompt_reject_mask = [0] * (len(prompt_tokens)-1) + [1] * len(reject_tokens)
    h = random.random()
    if h < train_percent:
        trainset.append((prompt_tokens, chosen_tokens, reject_tokens, chosen_base_prob, reject_base_prob))
    elif h < train_valid_percent:
        validset.append((prompt_tokens, chosen_tokens, reject_tokens, chosen_base_prob, reject_base_prob))
    else:
        testset.append((prompt_tokens, chosen_tokens, reject_tokens, chosen_base_prob, reject_base_prob))

import torch
torch.save(trainset, "trainset.save")
torch.save(validset, "validset.save")
torch.save(testset, "testset.save")