import torch
import h5py
import numpy as np
import random
import copy
from torch.utils.data import Dataset, DataLoader

class HDF5TopKTensorDataset(Dataset):
    def __init__(self,args, file_path, max_seq_length=4096,overlap_div=4):
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.args = args

        self.tokenshift = True
        self.CurrentExcessToken = None
        self.overlap_length = int(max_seq_length / overlap_div)

        self.debug = False
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])

        self.Count = 0
    
    def __len__(self):
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.dataset_length
    
    def __getitem__(self, idx):

        N = self.args.infctx_dataset_multiplier

        random_indices = [random.randint(0, self.dataset_length - 1) for _ in range(N)]

        

        if self.args.random_mode == 0:
            idx = self.Count
            self.Count = self.Count + 1
            if self.dataset_length - 1 < self.Count:
                self.Count = 0
                idx = 0
            random_indices = [idx]

        #print(f'idx = {idx}')
        LastTokens = None
        tokens = None
        if self.CurrentExcessToken is not None:
            if len(self.CurrentExcessToken) > 0:
                LastTokens = self.CurrentExcessToken

        GetNewDataset = False

        if LastTokens is not None:
            if len(LastTokens) < self.max_seq_length:
                GetNewDataset = True
            else:
                GetNewDataset = False
        else: 
            GetNewDataset = True

        if self.tokenshift == False:
            GetNewDataset = True

        if GetNewDataset:
            if self.debug:
                print('GetNewDataset')
            with h5py.File(self.file_path, 'r') as f:
                tokens_list = []
                for idx in random_indices:
                    tokens_list.append(f['tokens'][idx][:])
                # Concatenate all data
                tokens = np.concatenate(tokens_list)

        if self.tokenshift:
            if LastTokens is not None and tokens is not None:
                if self.debug:
                    print(f'Combined with old tokens {len(LastTokens)} {len(tokens)}')
                tokens = np.hstack((LastTokens,tokens))
            elif LastTokens is not None:
                if self.debug:
                    print(f'Load from old tokens {len(LastTokens)}')
                tokens = LastTokens
        
        # limit sequence length
        seq_len = min(len(tokens), self.max_seq_length)

        if self.tokenshift:
            if len(tokens) > self.max_seq_length:
                #Oversize tokens, reserve next token
                if self.debug:
                    print(f'tokens= {tokens}')
                    print(f'tokens cut [{self.max_seq_length-self.overlap_length} : {len(tokens)-1}]')
                self.CurrentExcessToken = tokens[ self.max_seq_length-self.overlap_length : len(tokens)-1]
                if self.debug:
                    print(f'Stored to ExcessToken {len(self.CurrentExcessToken)}')
            else:
                self.CurrentExcessToken = None



        tokens = tokens[:seq_len]

        # Padding and mask
        padded_tokens_input = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_input[:seq_len-1] = tokens[:seq_len-1]

        padded_tokens_target = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_target[0:seq_len-1] = tokens[1:seq_len]

        
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        attention_mask[0:seq_len-1] = 1.0

        
        # Prepare input and target
        input_tokens = padded_tokens_input#padded_tokens[:-1]
        target_tokens = padded_tokens_target#padded_tokens[1:]

        
        return {
            'input_ids': torch.from_numpy(input_tokens),
            'target_ids': torch.from_numpy(target_tokens),
            'attention_mask': torch.from_numpy(attention_mask)#.to(dtype=torch.bfloat16)#[1:]
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }
        


