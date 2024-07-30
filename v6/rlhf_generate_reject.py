import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
from argparse import ArgumentParser
import csv

parser = ArgumentParser()

parser.add_argument("--load_model", default="base_model/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth", type=str)
parser.add_argument("--input_csv", default="example.csv", type=str)
parser.add_argument("--output_csv", default="rlhfoutput.csv", type=str)
args2 = parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()

args.strategy = "cuda fp16"  # use CUDA, fp16

args.MODEL_NAME = args2.load_model

GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.0
GEN_alpha_frequency = 1.0
GEN_penalty_decay = 0.996
GEN_MAX_COUNT = 1000

CHUNK_LEN = 1024  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None

def run_rnn(ctx):
    global model_tokens, model_state

    ctx = ctx.replace("\r\n", "\n")

    tokens = pipeline.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    # print(f"### model ###\n{model_tokens}\n[{pipeline.decode(model_tokens)}]")  # debug

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out

#init_ctx = "User: hi" + "\n\n"
#init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"

#run_rnn(init_ctx)



input_file_path = args2.input_csv # 入力CSVファイル名
output_file_path = args2.output_csv  # 出力CSVファイル名

# 入力ファイルを開いて処理
with open(input_file_path, mode='r', encoding='utf-8', newline='') as input_file:
    # 出力ファイルを開いて書き込み
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as output_file:
        csv_reader = csv.DictReader(input_file)
        fieldnames = ['prompt', 'chosen', 'reject']  # 出力するキー
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        
        csv_writer.writeheader()  # ヘッダーの書き込み
        
        # 各行を読み込みながら処理
        for row in csv_reader:
            prompt = row['prompt']
            chosen = row['chosen']
            reject = row['reject']

            if len(prompt) > 0:
                occurrence = {}
                out_tokens = []
                out_last = 0

                model_state = None

                output_text = ''

                input_prompt = "User: " + prompt + "\n\nAssistant:"

                out = run_rnn(input_prompt)
                print(input_prompt)
                for i in range(GEN_MAX_COUNT):
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
                        output_text = output_text + tmp
                        out_last = i + 1

                    if "\n\n" in tmp:
                        #print(tmp, end="", flush=True)
                        output_text = output_text + tmp
                        break
                reject = output_text
                print(reject)
            
            # ここで必要な処理を行う
            # 例: prompt, chosen, reject の内容をそのまま出力ファイルに書き込む
            csv_writer.writerow({'prompt': prompt, 'chosen': chosen, 'reject': reject})

# 処理が完了した旨を通知
print('処理が完了しました。結果は Output.csv ファイルに保存されています。')

