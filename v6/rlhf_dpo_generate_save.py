#Prepare DPO Dataset from CSV  2024 OpenMOSE
import csv
import json
from argparse import ArgumentParser
parser = ArgumentParser()




print('Prepare RWKV ')
import numpy as np
from rwkv.utils import PIPELINE
from rwkv.model import RWKV

parser.add_argument("--load_model", default="base_model/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth", type=str)
parser.add_argument("--input_csv", default="example.csv", type=str)
parser.add_argument("--output_save", default="orpotest.save", type=str)
parser.add_argument("--target_pair_count", default=1000, type=int)
args = parser.parse_args()

model = RWKV(args.load_model, "cuda bf16")
tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer.encode("\n\nUser:"), tokenizer.encode("\n\nResponse:")

DPO_pair_number = args.target_pair_count # from 2 to 61965

# CSVファイルを読み込み、各項目を抽出する
RawData = []
NowRawPosition = 0
with open(args.input_csv, "r",encoding='utf-8-sig', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        #print(row)
        ToRawData = {}
        prompt = row["prompt"]
        chosen = row["chosen"]
        reject = row["reject"]

        ToRawData["prompt"] = prompt
        ToRawData["chosen"] = chosen
        ToRawData["reject"] = reject

        RawData.append(ToRawData)

        #print(f"Prompt: {prompt}, Chosen: {chosen}, Reject: {reject}")

print(RawData)

def GetRawData():
    global RawData
    global NowRawPosition
    if NowRawPosition == len(RawData):
        NowRawPosition = 0
    NowRaw = RawData[NowRawPosition]
    NowRawPosition = NowRawPosition+1
    return NowRaw["prompt"], NowRaw["chosen"], NowRaw["reject"]



import random
random.seed(100)
train_percent = 1.0
train_valid_percent = 0.9

trainset = []
validset = []
testset = []

import torch
import torch.nn.functional as F

def masked_cross_entropy(pred, target, mask):
    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
    loss = loss * mask.view(-1)
    return loss.sum() / mask.sum()

def cross_entropy(pred, target):
    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
    #loss = loss * mask.view(-1)
    return #loss.sum() / mask.sum()

def compute_base_prob(prompt_tokens, chosen_tokens):
    global tokenizer
    full_logits_chosen, _ = tokenizer.model.forward(prompt_tokens + chosen_tokens[:-1], None, full_output=True)
    chosen_logits = full_logits_chosen[-len(chosen_tokens):]
    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_tokens)), chosen_tokens]
    return float(torch.sum(chosen_loss))


with open("output.jsonl", "w", encoding="utf-8") as file:
    
    for i in range(int(DPO_pair_number)):
        #prompt_str = str(df.iloc[i].prompt).strip()
        #print(prompt_str)
        #print(df.iloc[i].chosen)
        #chosen_str = str(df.iloc[i].chosen[1]["content"]).strip().replace("\n\n", "\n")
        
        #print(chosen_str)
        #reject_str = str(df.iloc[i].rejected[1]["content"]).strip().replace("\n\n", "\n")

        prompt_str, chosen_str, reject_str = GetRawData()
        
        prompt_str = prompt_str.strip().replace("\r\n", "\n").replace("\n\n", "\n")
        chosen_str = chosen_str.strip().replace("\r\n", "\n").replace("\n\n", "\n")
        reject_str = reject_str.strip().replace("\r\n", "\n").replace("\n\n", "\n")




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

        json_object = {"text": prompt_str+chosen_str}
        file.write(json.dumps(json_object, ensure_ascii=False) + "\n")

        chosen_base_prob = compute_base_prob(prompt_tokens, chosen_tokens)
        reject_base_prob = compute_base_prob(prompt_tokens, reject_tokens)

        print(prompt_str)
        print(chosen_str)
        print(reject_str)

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
# リストをランダムに入れ替える
random.shuffle(trainset)
torch.save(trainset, args.output_save)
