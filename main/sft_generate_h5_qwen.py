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

#parser.add_argument("--load_initial_state", default="", type=str)
parser.add_argument("--input_folder", default="myfolder/2024_dataset/General3_qwen", type=str)
parser.add_argument("--output_parquet", default="myfolder/2024_dataset/General-jpencnv5-qwen-tokenizer-clean.h5", type=str)
#\x17
args2 = parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
#os.environ["RWKV_JIT_ON"] = "1"
#os.environ["RWKV_CUDA_ON"] = "1"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

#from rwkv.model import RWKV
#from rwkv.utils import PIPELINE
from tokenizer.rwkv_tokenizer import *

########################################################################################################

args = types.SimpleNamespace()


print(f'Qwen Tokenizer')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/tokenizer/qwen")


########################################################################################################

#model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
#tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
#tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
model_tokens = []
model_state = None

#DPO_pair_number = args2.target_pair_count # from 2 to 61965

trainset = []

model_current_statetuned = None




# class HDF5TopKTensorDataset(Dataset):
#     def __init__(self, file_path, transform=None):
#         self.file_path = file_path
#         self.transform = transform
        
#         with h5py.File(self.file_path, 'r') as f:
#             self.dataset_length = len(f['tokens'])
    
#     def __len__(self):
#         return self.dataset_length
    
#     def __getitem__(self, idx):
#         with h5py.File(self.file_path, 'r') as f:
#             tokens = torch.from_numpy(f['tokens'][idx][:]).long()
#             top_k_values = torch.from_numpy(f['top_k_values'][idx][:]).float()
#             top_k_indices = torch.from_numpy(f['top_k_indices'][idx][:]).long()
        
#         if self.transform:
#             tokens = self.transform(tokens)
#             top_k_values = self.transform(top_k_values)
#             top_k_indices = self.transform(top_k_indices)
        
#         return tokens, top_k_values, top_k_indices
    
# Inputフォルダ内のJSONLファイル一覧を取得
input_folder = args2.input_folder
jsonl_files = glob.glob(os.path.join(input_folder, '*.jsonl'))

def get_top_k_logits(logits, k=100):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    return top_k_values, top_k_indices

top_k = 100
i = 0


with h5py.File(args2.output_parquet, 'w') as f:
        tokens_dataset = f.create_dataset('tokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                          )

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

                            #print(f'filename = {jsonl_file} text = {text_value}')

                            tokens = torch.tensor(tokenizer.encode(text_value))
                            model_state = None


                            # データセットのサイズを拡張
                            tokens_dataset.resize((i+1,))


                            print('logits compute finished')

                            # PyTorch TensorをNumPy配列に変換してから保存
                            tokens_dataset[i] = tokens.numpy().astype(np.int64)

                            # for token in tokens_dataset:
                            #     if 65530 in token or 65536 in token:
                            #         print(tokens_dataset)
                            #         #exit()



                            i=i+1


                            #print(f"Text value: {text_value}")

                            print(f'NowProcessing {NowProcessing}/{len(jsonl_files)}')
                        else:
                            print("No 'text' key found in this line")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {jsonl_file}")
            

            print("-------------------------")














exit()

