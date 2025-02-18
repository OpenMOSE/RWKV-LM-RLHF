########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os, json, datetime, random
from tqdm import tqdm
import numpy as np
import torch
import csv
import json
import copy
import codecs

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from datasets import load_dataset, load_from_disk, Dataset

# HF_MODE = True # this is currently insanely slow. please wait for fix

HF_MODE = False # you will get 44.87% for RWKV-x070-World-1.5B-v3-20250127-ctx4096



########################################################################################################

if not HF_MODE:
    # download from https://huggingface.co/BlinkDL/rwkv-7-world
    #MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x060-World-7B-v3-20241112-ctx4096"
    #MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096"
    MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-2b9-world-v3-82%trained-20250203-ctx4k"
    print(f"Loading model - {MODEL_NAME}")

    os.environ["RWKV_V7_ON"] = '1'
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    model = RWKV(model=MODEL_NAME, strategy="cuda fp16")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokenizer = pipeline.tokenizer

    initialstate = None

    #initialstate = load_state(model,'rwkv-4-state.pth')
else:
    MODEL_NAME = 'fla-hub/rwkv7-1.5B-world'
    print(f"Loading model - {MODEL_NAME}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda:0", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    

########################################################################################################

mmlu_test = Dataset.from_parquet("mmlu/test-00000-of-00001.parquet")
#mmlu_dev = load_from_disk("mmlu_dev_dataset")

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''

TEMPLATE1 = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Thinking:'''

TEMPLATE2 = '''
Assistant: The answer is'''


# TEMPLATE = '''Question: You are a very talented school teacher in <SUBJECT>. Answer this question:
# <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>
# Answer: The answer is'''


# TEMPLATE = (
#     'System: You are a school teacher in <SUBJECT>.\n\n\x17'+
#     'User: Answer this question:' + '\n' +
#     '<Q>' + '\n' +
#     'A. <|A|>' + '\n' +
#     'B. <|B|>' + '\n' +
#     'C. <|C|>' + '\n' +
#     'D. <|D|>' + '\n\n\x17' +
#     'Assistant: The answer is'
# )

# TEMPLATE = (
#     #'System: You are a school teacher in <SUBJECT>.\n\n\x17'+
#     'User: '+# Answer this question:' + '\n' +
#     '<Q>' + '\n' +
#     'A. <|A|>' + '\n' +
#     'B. <|B|>' + '\n' +
#     'C. <|C|>' + '\n' +
#     'D. <|D|>' + '\n\n\x17' +
#     'Assistant: Correct answer is'
# )

CHOICES = [" A", " B", " C", " D"]

SHUFFLE = False
SEED = 42

CoT_Tokens = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

correct = 0
total = 0
correct_few = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token]), "Choices are not single token, use rwkv_mmlu.py instead"
choices_token = [x[0] for x in choices_token]

eachdata = {}
for idx, sample in enumerate(mmlu_test):
    subject = sample["subject"]
    eachdata[subject] = {}
    eachdata[subject]['correct'] = 0
    eachdata[subject]['total'] = 0
    eachdata[subject]['score'] = 0
    eachdata[subject]['subject'] = subject
GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.5
GEN_alpha_frequency = 0.5
GEN_penalty_decay = 0.996


with codecs.open('mmlu-cheat.jsonl', 'w', encoding='utf-8') as f:
    with open('cft1b5.csv', 'w', newline='', encoding='utf-8') as file:
        # CSVライターを作成
        writer = csv.writer(file)

        for idx, sample in enumerate(mmlu_test):
            #if idx> 10 :
            #    break
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]
            gt = sample["answer"]

            if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
                original_gt_text = choices[gt]
                np.random.shuffle(choices)
                gt = choices.index(original_gt_text)

            all_prefix = (
                TEMPLATE.replace("<Q>", question)
                .replace("<|A|>", choices[0])
                .replace("<|B|>", choices[1])
                .replace("<|C|>", choices[2])
                .replace("<|D|>", choices[3])
                .replace("<SUBJECT>", subject.replace("_", " "))
            )

            fordataset = all_prefix + CHOICES[gt] + '.\n\n'

            fordataset = fordataset.replace('\r\n','\n').strip()

            json_line = json.dumps({"text": fordataset}, ensure_ascii=False)
            # 1行ずつ書き込み、改行を追加
            f.write(json_line + '\n')

            #print(subject)

            if idx == 0:
                print(f"Format example:")
                print("-" * 100)
                print(all_prefix)
                print("-" * 100)
                format_example = all_prefix

            all_prefix_ids = [0] + tokenizer.encode(all_prefix.replace('\r\n','\n').strip())

            Correct = False

            out, model_state = model.forward(all_prefix_ids, copy.deepcopy(initialstate), full_output=False)

            occurrence = {}
            out_tokens = []
            out_last = 0

            model_tokens = []

            for i in range(CoT_Tokens):
                for n in occurrence:
                    out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
                out[0] -= 1e10  # disable END_OF_TEXT

                token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)

                out, model_state = model.forward([token], model_state)
                model_tokens += [token]

                out_tokens += [token]

                for xxx in occurrence:
                    occurrence[xxx] *= GEN_penalty_decay
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

                tmp = pipeline.decode(out_tokens[out_last:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                    #print(tmp, end="", flush=True)
                    out_last = i + 1

                if "\n\n" in tmp:
                    #print(tmp, end="", flush=True)
                    break

            all_prefix2 = TEMPLATE2
            all_prefix_ids = [0] + tokenizer.encode(all_prefix2.replace('\r\n','\n').strip())
            logits, model_state = model.forward(all_prefix_ids, copy.deepcopy(model_state), full_output=False)

            for i in range(1):
                correct_few += 1

                logits, _ = model.forward(all_prefix_ids, copy.deepcopy(model_state), full_output=False)
                
                neg_log_prob = F.log_softmax(logits, dim=-1)
                target_prob = neg_log_prob[choices_token]
            
                if torch.argmax(target_prob).item() == gt:
                    Correct = True
                    break

                random.seed(SEED+i)
                np.random.seed(SEED+i)
                torch.manual_seed(SEED+i)
                torch.cuda.manual_seed(SEED+i)

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            if Correct:
                correct += 1
                eachdata[subject]['correct']+=1
            total += 1
            eachdata[subject]['total']+=1

            eachdata[subject]['score'] = float(eachdata[subject]['correct']) / float(eachdata[subject]['total'])
            pbar.set_description(f"{subject} Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
            #print(eachdata)
            pbar.update(1)

        writer.writerow(['subject','correct','total','score'])

        for data in eachdata:
            Subject = eachdata[data]['subject']
            Correct = eachdata[data]['correct']
            Total = eachdata[data]['total']
            Score = eachdata[data]['score']

            writer.writerow([Subject,Correct,Total,Score])

        


        pbar.close()