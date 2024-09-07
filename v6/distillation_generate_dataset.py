#RWKV x060 Distillation Dataset Generator
#2024 OpenMOSE
import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
from argparse import ArgumentParser
import csv
import random
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import random
from typing import List, Tuple
import os
import concurrent.futures
import h5py
from torch.utils.data import Dataset, DataLoader

parser = ArgumentParser()

parser.add_argument("--load_model", default="models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth", type=str)
parser.add_argument("--input_csv", default="datasets/test_jp.csv", type=str)
parser.add_argument("--output_parquet", default="datasets/test_jp_logits.h5", type=str)
parser.add_argument("--endtoken", default="\n\n\x17", type=str)
parser.add_argument("--target_pair_count", default=30000, type=int)
#\x17
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
GEN_TOP_P = 0.4
GEN_alpha_presence = 0.0
GEN_alpha_frequency = 1.0
GEN_penalty_decay = 0.996
GEN_MAX_COUNT = 1000

CHUNK_LEN = 2048  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None

DPO_pair_number = args2.target_pair_count # from 2 to 61965

trainset = []

# CSVファイルを読み込み、各項目を抽出する
RawData = []
NowRawPosition = 0
with open(args2.input_csv, "r",encoding='utf-8-sig', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        #print(row)
        ToRawData = {}
        prompt = row["prompt"]
        chosen = row["chosen"]

        ToRawData["prompt"] = prompt
        ToRawData["chosen"] = chosen

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
    return NowRaw["prompt"], NowRaw["chosen"]

 

def run_rnn_logits(intoken):
    global model_tokens, model_state

    #ctx = ctx.replace("\r\n", "\n")

    tokens = copy.deepcopy(intoken)#tokenizer.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    accumulated_logits = []
    model_state = None

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state,full_output=True)
        accumulated_logits.append(out)
        tokens = tokens[CHUNK_LEN:]

    # Concatenate all accumulated logits along the sequence dimension
    final_logits = torch.cat(accumulated_logits, dim=0)

    del accumulated_logits

    # If you want to flatten the tensor completely
    #flattened_logits = final_logits.view(-1)

    return final_logits.to(dtype=torch.bfloat16).to(device='cpu')

#init_ctx = "User: hi" + "\n\n"
#init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"

#run_rnn(init_ctx)


# for i in range(int(DPO_pair_number)):


#     prompt_str, chosen_str  = GetRawData()
    
#     prompt_str = prompt_str.strip().replace("\r\n", "\n")#.replace("\n\n", "\n")
#     chosen_str = chosen_str.strip().replace("\r\n", "\n")#.replace("\n\n", "\n")

#     h = random.random()
#     if h < 0.8:
#         prompt_str = "User: " + prompt_str + "\n\nAssistant:" # Helpfulness optimization
#         chosen_str = ' ' + chosen_str + args2.endtoken
#     elif h<0.9:
#         prompt_str = "Question: " + prompt_str + "\n\nAnswer:" # Factuality
#         chosen_str = ' ' + chosen_str + args2.endtoken
#     else:
#         prompt_str = "Input: " + prompt_str + "\n\nResponse:" # Instruction-following
#         chosen_str = ' ' + chosen_str + args2.endtoken
#     #prompt_tokens = tokenizer.encode(prompt_str)
#     #chosen_tokens = tokenizer.encode(chosen_str)
#     tokens = tokenizer.encode(prompt_str + chosen_str)

#     print(prompt_str)
#     print(chosen_str)

#     model_state = None
#     logits = run_rnn_logits(tokens)
#     print(logits)
#     print(f'logits shape = {logits.shape}')
#     trainset.append((tokens, logits))
#     print(f'\n\n {i} / {int(DPO_pair_number)}')


# import torch
# # リストをランダムに入れ替える
# random.shuffle(trainset)
# torch.save(trainset, args2.output_dist)


# class StreamingParquetDatasetBuilder:
#     def __init__(self, base_path: str, schema: pa.Schema, max_rows_per_file: int = 10000):
#         self.base_path = base_path
#         self.schema = schema
#         self.max_rows_per_file = max_rows_per_file
#         self.current_file = None
#         self.current_writer = None
#         self.rows_in_current_file = 0
#         self.file_count = 0

#     def _create_new_file(self):
#         if self.current_writer:
#             self.current_writer.close()
#         file_path = f"{self.base_path}_{self.file_count:04d}.parquet"
#         self.current_file = pa.OSFile(file_path, 'wb')
#         self.current_writer = pq.ParquetWriter(self.current_file, self.schema, compression='ZSTD', compression_level=3)
#         self.rows_in_current_file = 0
#         self.file_count += 1

#     def append(self, tokens: torch.Tensor, logits: torch.Tensor):
#         if self.current_writer is None or self.rows_in_current_file >= self.max_rows_per_file:
#             self._create_new_file()

#         # Convert tensors to PyArrow arrays
#         tokens_array = pa.array(tokens.numpy().tolist(), type=pa.int64())
#         if logits.dtype == torch.bfloat16:
#             logits_array = pa.array(logits.to(torch.float32).numpy().tolist(), type=pa.list_(pa.float32()))
#         else:
#             logits_array = pa.array(logits.numpy().tolist(), type=pa.list_(pa.float32()))

#         # Create a record batch and write it
#         batch = pa.RecordBatch.from_arrays([tokens_array, logits_array], ['tokens', 'logits'])
#         self.current_writer.write_batch(batch)
#         self.rows_in_current_file += 1

#     def close(self):
#         if self.current_writer:
#             self.current_writer.close()

#     def get_file_paths(self) -> List[str]:
#         return [f"{self.base_path}_{i:04d}.parquet" for i in range(self.file_count)]

class HDF5TopKTensorDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            tokens = torch.from_numpy(f['tokens'][idx][:]).long()
            top_k_values = torch.from_numpy(f['top_k_values'][idx][:]).float()
            top_k_indices = torch.from_numpy(f['top_k_indices'][idx][:]).long()
        
        if self.transform:
            tokens = self.transform(tokens)
            top_k_values = self.transform(top_k_values)
            top_k_indices = self.transform(top_k_indices)
        
        return tokens, top_k_values, top_k_indices

def get_top_k_logits(logits, k=100):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    return top_k_values, top_k_indices

def create_sample_data(
    file_path: str,
    DPO_pair_number: int,
    tokenizer,
    run_rnn_logits,
    GetRawData,
    args2,
    max_rows_per_file: int = 10000,
    top_k: int = 100
):
    with h5py.File(file_path, 'w') as f:
        tokens_dataset = f.create_dataset('tokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                          compression="gzip", compression_opts=9)
        top_k_values_dataset = f.create_dataset('top_k_values', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.float32),
                                                compression="gzip", compression_opts=9)
        top_k_indices_dataset = f.create_dataset('top_k_indices', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                                 compression="gzip", compression_opts=9)
        
        for i in range(int(DPO_pair_number)):
            prompt_str, chosen_str = GetRawData()
            
            prompt_str = prompt_str.strip().replace("\r\n", "\n").replace("\n\n", "\n")
            chosen_str = chosen_str.strip().replace("\r\n", "\n").replace("\n\n", "\n")

            h = random.random()
            if h < 0.8:
                prompt_str = "User: " + prompt_str + "\n\nAssistant:"
                chosen_str = ' ' + chosen_str + args2.endtoken
            elif h < 0.9:
                prompt_str = "Question: " + prompt_str + "\n\nAnswer:"
                chosen_str = ' ' + chosen_str + args2.endtoken
            else:
                prompt_str = "Input: " + prompt_str + "\n\nResponse:"
                chosen_str = ' ' + chosen_str + args2.endtoken

            print(f'{prompt_str + chosen_str}')

            print(f'\n\n {i+1} / {int(DPO_pair_number)}')

            tokens = torch.tensor(tokenizer.encode(prompt_str + chosen_str))
            logits = run_rnn_logits(tokens)
            
            # Top-K Logitsの計算
            top_k_values, top_k_indices = get_top_k_logits(logits, k=top_k)

            # データセットのサイズを拡張
            tokens_dataset.resize((i+1,))
            top_k_values_dataset.resize((i+1,))
            top_k_indices_dataset.resize((i+1,))

            # PyTorch TensorをNumPy配列に変換してから保存
            tokens_dataset[i] = tokens.numpy().astype(np.int64)
            
            # Top-K値とインデックスを1次元に変換して保存
            top_k_values_dataset[i] = top_k_values.to(dtype=torch.float32).flatten().numpy().astype(np.float32)
            top_k_indices_dataset[i] = top_k_indices.flatten().numpy().astype(np.int64)


            #print(f'tokens_dataset[i] = {tokens_dataset[i]}')
            #print(f'top_k_values_dataset[i] = {top_k_values_dataset[i]}')
            #print(f'top_k_indices_dataset[i] = {top_k_indices_dataset[i]}')

            print(f'count = {len(tokens_dataset[i])} {len(top_k_values_dataset[i])} {len(top_k_indices_dataset[i])}')

            

            if (i + 1) % max_rows_per_file == 0:
                print(f"Saved {i + 1} rows.")

    print(f"Dataset saved to {file_path}")
    return file_path
create_sample_data(
        args2.output_parquet,
        DPO_pair_number=args2.target_pair_count,
        tokenizer=tokenizer,
        run_rnn_logits=run_rnn_logits,
        GetRawData=GetRawData,
        args2=args2,
        max_rows_per_file=50
    )



exit()



























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

                out = run_rnn_logits(input_prompt)
                print(input_prompt)
                # for i in range(GEN_MAX_COUNT):
                #     for n in occurrence:
                #         out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
                #     out[0] -= 1e10  # disable END_OF_TEXT
                #     token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)
                #     out, model_state = model.forward([token], model_state)
                #     model_tokens += [token]
                #     out_tokens += [token]

                #     for xxx in occurrence:
                #         occurrence[xxx] *= GEN_penalty_decay
                #     occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

                #     tmp = pipeline.decode(out_tokens[out_last:])
                #     if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                #         #print(tmp, end="", flush=True)
                #         output_text = output_text + tmp
                #         out_last = i + 1

                #     if "\n\n" in tmp:
                #         #print(tmp, end="", flush=True)
                #         output_text = output_text + tmp
                #         break
                # reject = output_text
                # print(reject)
            
            # ここで必要な処理を行う
            # 例: prompt, chosen, reject の内容をそのまま出力ファイルに書き込む
            csv_writer.writerow({'prompt': prompt, 'chosen': chosen, 'chosen_logits': reject})

# 処理が完了した旨を通知
print('処理が完了しました。結果は Output.csv ファイルに保存されています。')

