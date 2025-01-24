#RWKV RLHF Dataset Converter
#2025 OpenMOSE
import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
from argparse import ArgumentParser
import csv
import random
import numpy as np
import random
from typing import List, Tuple
import os
import concurrent.futures
import h5py
from torch.utils.data import Dataset, DataLoader
import glob
import json
import torch.nn.functional as F

from rwkvengine.rwkvcore import RWKV_x, PIPELINE
pipeline = PIPELINE(mode='world')



parser = ArgumentParser()

parser.add_argument("--load_model", default="myfolder/models/x070-1b5-cje-17.pth", type=str)
parser.add_argument("--model_strategy", default="bf16", type=str)
parser.add_argument("--rlhf_input_folder", default="myfolder/RLHF/RLHF", type=str)
parser.add_argument("--rlhf_output_h5", default="myfolder/RLHF/qwqtest.h5", type=str)
parser.add_argument("--endtoken", default="\n\n\x17", type=str)
#\x17
args2 = parser.parse_args()



########################################################################################################



GEN_TEMP = 1.0
GEN_TOP_P = 0.4
GEN_alpha_presence = 0.0
GEN_alpha_frequency = 1.0
GEN_penalty_decay = 0.996
GEN_MAX_COUNT = 1000

CHUNK_LEN = 4096  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args2.load_model}")

model = RWKV_x(args2.load_model,args2.model_strategy)

#model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
#tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None

#DPO_pair_number = args2.target_pair_count # from 2 to 61965

trainset = []

model_current_statetuned = None





def run_rnn_logits(prompttoken,chosentoken):

    with torch.no_grad(): 

        tokens = torch.cat((prompttoken,chosentoken))

        tokens = [int(x) for x in tokens]

        accumulated_logits = []

        i = 0

        States = model.new_state(1)
        shift_states = States.shift_states
        wkv_states = States.wkv_states

        CHUNK_LEN = 4096

        while len(tokens) > 0:
            prompts = []
            #print('append tensor')
            prompts.append(torch.tensor(tokens[:CHUNK_LEN]).unsqueeze(0).to('cuda'))

            idx = torch.cat(prompts, dim=0)

            #print(f'idx shape = {idx.shape}')
            #p#rint(idx.view(-1))
            i+=1
            #print(f'Running RNN Logits = {i}')
            out, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states,full_output=True)
    
            out = out.view(-1,65536)
            #print(f'out shape = {out.shape}')

            accumulated_logits.append(copy.deepcopy(out))

            tokens = tokens[CHUNK_LEN:]


    
        final_logits = torch.cat(accumulated_logits, dim=0)
        #del accumulated_logits

        chosen_logits = final_logits[-len(chosentoken):]
        #del final_logits
        print(f'chosen_logits shape = {chosen_logits.shape}')
        seq_length = chosen_logits.size(0)
        #chosen_loss = F.log_softmax(chosen_logits, dim=-1)[torch.arange(seq_length), chosen_tokens[:seq_length]]
        chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosentoken)), chosentoken]

        chosen_prob = (torch.sum(chosen_loss)).cpu().float() / float(len(chosentoken))

        print(f'chosen_logits {chosen_prob}')

        return copy.deepcopy(chosen_prob)






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
input_folder = args2.rlhf_input_folder
jsonl_files = glob.glob(os.path.join(input_folder, '*.jsonl'))


i = 0


with h5py.File(args2.rlhf_output_h5, 'w') as f:
        tokens_dataset = f.create_dataset('tokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64))

        prompttokens_dataset = f.create_dataset('prompttokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64))
        chosentokens_dataset = f.create_dataset('chosentokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64))
        rejecttokens_dataset = f.create_dataset('rejecttokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64))

        chosenprob_dataset = f.create_dataset('chosenprob', (0,), maxshape=(None,), dtype='float32')
        rejectprob_dataset = f.create_dataset('rejectprob', (0,), maxshape=(None,), dtype='float32')
       
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
                        if 'prompt' in json_data:
                            #text_value = json_data['text']

                            prompt_raw = json_data['prompt']
                            chosen_raw = json_data['chosen']
                            reject_raw = json_data['reject']

                            Token = f"User: {prompt_raw}{args2.endtoken}Assistant: {chosen_raw}{args2.endtoken}"
                            Prompt = f"User: {prompt_raw}{args2.endtoken}Assistant: "
                            Chosen = f"{chosen_raw}{args2.endtoken}"
                            Reject = f"{reject_raw}{args2.endtoken}"



                            print(f'filename = {jsonl_file} text = {Token}')

                            tokens = torch.tensor(pipeline.encode(Token))

                            prompt_tokens = torch.tensor(pipeline.encode(Prompt))
                            chosen_tokens = torch.tensor(pipeline.encode(Chosen))
                            reject_tokens = torch.tensor(pipeline.encode(Reject))
                            model_state = None

                            chosen_probs = run_rnn_logits(prompt_tokens,chosen_tokens)
                            print('ugyu')
                            reject_probs = run_rnn_logits(prompt_tokens,reject_tokens)
                          
                          

                            # データセットのサイズを拡張
                            tokens_dataset.resize((i+1,))

                            prompttokens_dataset.resize((i+1,))
                            chosentokens_dataset.resize((i+1,))
                            rejecttokens_dataset.resize((i+1,))
                            chosenprob_dataset.resize((i+1,))
                            rejectprob_dataset.resize((i+1,))
                          

                            print('logits compute finished')

                            # PyTorch TensorをNumPy配列に変換してから保存
                            tokens_dataset[i] = tokens.numpy().astype(np.int64)
                            print('ugyu')

                            prompttokens_dataset[i] = prompt_tokens.numpy().astype(np.int64)
                            chosentokens_dataset[i] = chosen_tokens.numpy().astype(np.int64)
                            rejecttokens_dataset[i] = reject_tokens.numpy().astype(np.int64)
                            print('ugyu2')
                            chosenprob_dataset[i] = chosen_probs.numpy().astype(np.float32)
                            print('ugyu2.5')
                            rejectprob_dataset[i] = reject_probs.numpy().astype(np.float32)
                            print('ugyu3')

                            #print(prompt_raw)
                            print(f'chosen prob={chosenprob_dataset[i]} reject prob={rejectprob_dataset[i]}')
                            


                            i=i+1


                            #print(f"Text value: {text_value}")

                            print(f'NowProcessing {NowProcessing}/{len(jsonl_files)}')
                        else:
                            print("No 'text' key found in this line")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {jsonl_file}")
            

            print("-------------------------")














exit()



