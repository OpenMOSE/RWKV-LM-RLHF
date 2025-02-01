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
                    #rejecttokens_list.append(f['rejecttokens'][idx][:]) #dict

                    #chosenprob_list.append(f['chosenprob'][idx]) #Scaler
                    #rejectprob_list.append(f['rejectprob'][idx]) #Scaler
                # Concatenate all data
                #tokens = np.concatenate(tokens_list)

                prompttokens = np.concatenate(prompttokens_list)
                chosentokens = np.concatenate(chosentokens_list)
                #rejecttokens = np.concatenate(rejecttokens_list)




            

            






            
            return {
                'prompttoken':torch.from_numpy(prompttokens),
                'chosentoken':torch.from_numpy(chosentokens)
            }

# def collate_fn(batch):
#     return {
#         'input_ids': torch.stack([item['input_ids'] for item in batch]),

#     }
        
def collate_fn(batch):
    return {
        torch.stack([item['prompttoken'] for item in batch]),
        torch.stack([item['chosentoken'] for item in batch]),
    }


