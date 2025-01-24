import torch
import h5py
import numpy as np
import random
import copy
from torch.utils.data import Dataset, DataLoader
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

class RLHFDataset(Dataset):
    def __init__(self,args, file_path, max_seq_length=4096):
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.args = args
        self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

        self.tokenshift = True
        self.CurrentExcessToken = None
        #self.overlap_length = int(max_seq_length / overlap_div)

        self.debug = False
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])

        self.Count = 0
    
    def __len__(self):
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.dataset_length
    
    def __getitem__(self, idx):

        N = 1

        
        RetryCount = self.dataset_length

        for i in range(RetryCount):
            random_indices = [random.randint(0, self.dataset_length - 1) for _ in range(N)]

            if self.args.random_mode == 0:
                idx = self.Count
                self.Count = self.Count + 1
                if self.dataset_length - 1 < self.Count:
                    self.Count = 0
                    idx = 0
                random_indices = [idx]



            if self.debug:
                print('GetNewDataset')
            with h5py.File(self.file_path, 'r') as f:
                tokens_list = []
                prompttokens_list = []
                chosentokens_list = []
                rejecttokens_list = []

                chosenprob_list = []
                rejectprob_list = []



                for idx in random_indices:
                    tokens_list.append(f['tokens'][idx][:])

                    prompttokens_list.append(f['prompttokens'][idx][:]) #dict
                    chosentokens_list.append(f['chosentokens'][idx][:]) #dict
                    rejecttokens_list.append(f['rejecttokens'][idx][:]) #dict

                    chosenprob_list.append(f['chosenprob'][idx]) #Scaler
                    rejectprob_list.append(f['rejectprob'][idx]) #Scaler
                # Concatenate all data
                tokens = np.concatenate(tokens_list)

                prompttokens = np.concatenate(prompttokens_list)
                chosentokens = np.concatenate(chosentokens_list)
                rejecttokens = np.concatenate(rejecttokens_list)

                chosenprob = chosenprob_list[0]#np.concatenate(chosenprob_list)
                rejectprob = rejectprob_list[0]#np.concatenate(rejectprob_list)


            # limit sequence length
            seq_len = min(len(tokens), self.max_seq_length)

            #print(chosentokens)
            merged_chosen_tokens = np.concatenate([prompttokens ,chosentokens])
            merged_reject_tokens = np.concatenate([prompttokens ,rejecttokens])

            chosen_len = (len(chosentokens))
            reject_len = (len(rejecttokens))

            chosen_start_point = len(merged_chosen_tokens) - (chosen_len)
            reject_start_point = len(merged_reject_tokens) - (reject_len)

            real_chosen_token_len = min(len(merged_chosen_tokens), self.max_seq_length) - chosen_start_point
            real_reject_token_len = min(len(merged_reject_tokens), self.max_seq_length) - reject_start_point

            # print(f'len(tokens) = {len(tokens)},chosen_len={chosen_len} ,reject_len={reject_len} ')
            # print(f'real_chosen_token_len = {real_chosen_token_len},real_reject_token_len={real_reject_token_len}   ')
            if real_chosen_token_len < 16 or real_reject_token_len < 16:
                print(f'realchosentoken below < 16. will choose another pairs. if you want solve set ctxlen higher')
                #exit()
                continue

            chosentokens = merged_chosen_tokens[chosen_start_point:chosen_start_point+real_chosen_token_len]
            rejecttokens = merged_reject_tokens[reject_start_point:reject_start_point+real_reject_token_len]


            prompt_chosen_input = np.concatenate([prompttokens , chosentokens[:-1]])
            prompt_chosen_target = np.concatenate([prompttokens[1:] , chosentokens])

            prompt_reject_input = np.concatenate([prompttokens , rejecttokens[:-1]])
            prompt_reject_target= np.concatenate([prompttokens[1:] , rejecttokens] )


            #print(f'clen = {len(prompt_chosen_input)} rlen = {len(prompt_reject_input)}')

            #self.tokenizer.printTokens(chosentokens)




            


            # tokens = tokens[:seq_len]

            # # Padding and mask
            # padded_tokens_input = np.zeros(self.max_seq_length, dtype=np.int64)
            # padded_tokens_input[:seq_len-1] = tokens[:seq_len-1]

            # padded_tokens_target = np.zeros(self.max_seq_length, dtype=np.int64)
            # padded_tokens_target[0:seq_len-1] = tokens[1:seq_len]

            # attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
            # attention_mask[0:seq_len-1] = 1.0
            
            # Prepare input and target
            #input_tokens = padded_tokens_input#padded_tokens[:-1]
            #target_tokens = padded_tokens_target#padded_tokens[1:]


            # print(f'choseninput len = {len(prompt_chosen_input)}')
            # print(f'realchosen len = {(real_chosen_token_len)}')
            # print(f'chosenprob  = {(chosenprob)}')
            # print(f'rejectprob  = {(rejectprob)}')







            
            return {
                'chosen_input': torch.from_numpy(prompt_chosen_input),
                'chosen_target': torch.from_numpy(prompt_chosen_target),
                'chosen_token_len': int(real_chosen_token_len),
                'chosen_base_prob': float(chosenprob),
                'reject_input': torch.from_numpy(prompt_reject_input),
                'reject_target': torch.from_numpy(prompt_reject_target),
                'reject_token_len': int(real_reject_token_len),
                'reject_base_prob': float(rejectprob),
                'chosentoken':torch.from_numpy(chosentokens),
                'rejecttoken':torch.from_numpy(rejecttokens)
            }

# def collate_fn(batch):
#     return {
#         'input_ids': torch.stack([item['input_ids'] for item in batch]),

#     }
        
def collate_fn(batch):
    return {
        torch.stack([item['chosen_input'] for item in batch]),
        torch.stack([item['chosen_target'] for item in batch]),
        torch.stack([item['chosen_token_len'] for item in batch]),
        torch.stack([item['chosen_base_prob'] for item in batch]),
        torch.stack([item['reject_input'] for item in batch]),
        torch.stack([item['reject_target'] for item in batch]),
        torch.stack([item['reject_token_len'] for item in batch]),
        torch.stack([item['chosentoken'] for item in batch]),
        torch.stack([item['rejecttoken'] for item in batch]),
    }


