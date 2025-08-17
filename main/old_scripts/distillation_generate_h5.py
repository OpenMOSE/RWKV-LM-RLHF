#RWKV x060 Distillation Dataset Generator
#2024 OpenMOSE
import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
from argparse import ArgumentParser
import csv
import random
#import pyarrow as pa
#import pyarrow.parquet as pq
import numpy as np
import random
from typing import List, Tuple
import os
import concurrent.futures
import h5py
from torch.utils.data import Dataset, DataLoader
import glob
import json



parser = ArgumentParser()

parser.add_argument("--load_model", default="models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth", type=str)
parser.add_argument("--load_initial_state", default="", type=str)
parser.add_argument("--input_folder", default="datasets_box/ClaudeOutput", type=str)
parser.add_argument("--output_parquet", default="datasets/Claude.h5", type=str)
parser.add_argument("--strategy", default="cuda fp16", type=str)
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

args.strategy = args2.strategy#"cuda fp16"  # use CUDA, fp16

args.MODEL_NAME = args2.load_model

GEN_TEMP = 1.0
GEN_TOP_P = 0.4
GEN_alpha_presence = 0.0
GEN_alpha_frequency = 1.0
GEN_penalty_decay = 0.996
GEN_MAX_COUNT = 1000

CHUNK_LEN = 4096  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None

#DPO_pair_number = args2.target_pair_count # from 2 to 61965

trainset = []

model_current_statetuned = None




def load_state(state_filename):
        global model
        global model_current_statetuned
        debug = 1
        if state_filename != "":
            try:
                state_raw = torch.load(state_filename, map_location="cpu")
            except Exception as e:
                print(e)
                return "state file failed to load"
            state_raw_shape = next(iter(state_raw.values())).shape

            args = model.args
            if debug:
                print(f"{len(state_raw)} != {args.n_layer}")
                print(f"{state_raw_shape[0] * state_raw_shape[1]} != {args.n_embd}")

            if (
                len(state_raw) != args.n_layer
                or state_raw_shape[0] * state_raw_shape[1] != args.n_embd
            ):
                print("state failed to load")
                return "state shape mismatch"

            strategy = model.strategy

            model_current_statetuned = [None] * args.n_layer * 3

            for i in range(args.n_layer):
                dd = strategy[i]
                dev = dd.device
                atype = dd.atype
                model_current_statetuned[i * 3 + 0] = torch.zeros(
                    args.n_embd, dtype=atype, requires_grad=False, device=dev
                ).contiguous()
                model_current_statetuned[i * 3 + 1] = (
                    state_raw[f"blocks.{i}.att.time_state"]
                    .transpose(1, 2)
                    .to(dtype=torch.float, device=dev)
                    .requires_grad_(False)
                    .contiguous()
                )
                model_current_statetuned[i * 3 + 2] = torch.zeros(
                    args.n_embd, dtype=atype, requires_grad=False, device=dev
                ).contiguous()

            #self.model_state = None
            #self.model_current_statetuned_filename = state_filename
            if debug:
                print(f"State-tune model loaded:{state_filename}")
        elif state_filename == "":
            print('state reset')
            model_current_statetuned = None
            gc.collect()
            torch.cuda.empty_cache()

load_state(args2.load_initial_state)

 

 

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
        if out.dim() == 1:
            out = out.unsqueeze(0)
        accumulated_logits.append(out.to(dtype=torch.bfloat16).to(device='cpu'))
        tokens = tokens[CHUNK_LEN:]

    # Concatenate all accumulated logits along the sequence dimension
    final_logits = torch.cat(accumulated_logits, dim=0)

    del accumulated_logits

    # If you want to flatten the tensor completely
    #flattened_logits = final_logits.view(-1)

    return final_logits
 

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
    
# Inputフォルダ内のJSONLファイル一覧を取得
input_folder = args2.input_folder
jsonl_files = glob.glob(os.path.join(input_folder, '*.jsonl'))

def get_top_k_logits(logits, k=100):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    return top_k_values, top_k_indices

top_k = 50
i = 0


with h5py.File(args2.output_parquet, 'w') as f:
        tokens_dataset = f.create_dataset('tokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                          compression="gzip", compression_opts=9)
        top_k_values_dataset = f.create_dataset('top_k_values', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.float32),
                                                compression="gzip", compression_opts=9)
        top_k_indices_dataset = f.create_dataset('top_k_indices', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                                 compression="gzip", compression_opts=9)
        NowProcessing = 0
        # 各JSONLファイルを処理
        for jsonl_file in jsonl_files:
            print(f"Processing file: {jsonl_file}")
            NowProcessing = NowProcessing + 1
            
            # JSONLファイルを開いて各行を処理
            with open(jsonl_file, 'r', encoding='utf-8') as file:
                for line in file:
                    # 各行をJSONとしてパース
                    try:
                        json_data = json.loads(line.strip())
                        # 'text'キーの値を取得
                        if 'text' in json_data:
                            text_value = json_data['text']

                            print(f'filename = {jsonl_file} text = {text_value}')

                            tokens = torch.tensor(tokenizer.encode(text_value))
                            model_state = None
                            if model_current_statetuned is not None:
                                model_state = copy.deepcopy(model_current_statetuned)
                                print('initial state deepcopy')

                            logits = run_rnn_logits(tokens)

                            print('finished RNN Processing')
                            
                            # Top-K Logitsの計算
                            top_k_values, top_k_indices = get_top_k_logits(logits, k=top_k)

                            # データセットのサイズを拡張
                            tokens_dataset.resize((i+1,))
                            top_k_values_dataset.resize((i+1,))
                            top_k_indices_dataset.resize((i+1,))

                            print('logits compute finished')

                            # PyTorch TensorをNumPy配列に変換してから保存
                            tokens_dataset[i] = tokens.numpy().astype(np.int64)
                            
                            # Top-K値とインデックスを1次元に変換して保存
                            top_k_values_dataset[i] = top_k_values.to(dtype=torch.float32).flatten().numpy().astype(np.float32)
                            top_k_indices_dataset[i] = top_k_indices.flatten().numpy().astype(np.int64)
                            


                            i=i+1


                            print(f"Text value: {text_value}")

                            print(f'NowProcessing {NowProcessing}/{len(jsonl_files)}')
                        else:
                            print("No 'text' key found in this line")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {jsonl_file}")
            

            print("-------------------------")














exit()



