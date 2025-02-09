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

HF_MODE = False #



########################################################################################################

if not HF_MODE:
    # download from https://huggingface.co/BlinkDL/rwkv-7-world
    #MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x060-Jpn-14B-20240819-ctx4096.pth"
    #MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-2b9-world-v3-82%trained-20250203-ctx4k"
    #MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096"
    MODEL_NAME = "/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k"
    print(f"Loading model - {MODEL_NAME}")

    os.environ["RWKV_V7_ON"] = '1'
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"
    
    x17mode = False
    RetryCount = 1

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

mmlu_test = Dataset.from_parquet("jmmlu/output.parquet")
#mmlu_dev = load_from_disk("mmlu_dev_dataset")

TEMPLATE = '''User: あなたは非常に有能な<SUBJECT>の専門家です。この質問に対する答えを以下から選択してください:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: 答えは'''

TEMPLATE_x17 = '''User: あなたは非常に有能な<SUBJECT>の専門家です。この質問に対する答えを以下から選択してください:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

''' + '\x17' + 'Assistant: 答えは'

if x17mode:
    TEMPLATE = TEMPLATE_x17



TEMPLATE_prompt = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>'''

TEMPLATE_chosen = '''The answer is'''



CHOICES = [" A", " B", " C", " D"]

SHUFFLE = False
SEED = 42

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
    print(sample)
    #exit()
    subject = sample["subject"]
    eachdata[subject] = {}
    eachdata[subject]['correct'] = 0
    eachdata[subject]['total'] = 0
    eachdata[subject]['score'] = 0
    eachdata[subject]['subject'] = subject


with codecs.open('mmlu-cheat.jsonl', 'w', encoding='utf-8') as f:
    with open('v7-2b9-preview.csv', 'w', newline='', encoding='utf-8') as file:
        # CSVライターを作成
        writer = csv.writer(file)

        for idx, sample in enumerate(mmlu_test):
            #if idx> 10 :
            #    break
            question = sample["question"]
            choices = []
            choices.append(sample['A'])
            choices.append(sample['B'])
            choices.append(sample['C'])
            choices.append(sample['D'])
            subject = sample["subject"]
            #gt = sample["answer"]
            if sample["answer"] == 'A':
                gt = 0
            elif sample["answer"] == 'B':
                gt = 1
            elif sample["answer"] == 'C':
                gt = 2
            elif sample["answer"] == 'D':
                gt = 3


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

            all_prefix_prompt = (
                TEMPLATE_prompt.replace("<Q>", question)
                .replace("<|A|>", choices[0])
                .replace("<|B|>", choices[1])
                .replace("<|C|>", choices[2])
                .replace("<|D|>", choices[3])
                .replace("<SUBJECT>", subject.replace("_", " "))
            )

            all_prefix_chosen = TEMPLATE_chosen + CHOICES[gt] + '.\n\n'

            fordataset = all_prefix + CHOICES[gt] + '.\n\n'

            fordataset = fordataset.replace('\r\n','\n').strip()

            json_line = json.dumps({"text": fordataset,'prompt':all_prefix_prompt, 'chosen':all_prefix_chosen}, ensure_ascii=False)
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

            for i in range(RetryCount):
                correct_few += 1
                if HF_MODE:
                    logits = model.forward(torch.tensor([all_prefix_ids]).cuda())[0][0][-1]
                else:
                    logits, _ = model.forward(all_prefix_ids, copy.deepcopy(initialstate), full_output=False)
                
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