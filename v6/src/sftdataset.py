import torch
import h5py
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class HDF5TopKTensorDataset(Dataset):
    def __init__(self,args, file_path, max_seq_length=4096):
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.args = args
        
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
            if self.max_seq_length - 1 < self.Count:
                self.Count = 0
                idx = 0
            random_indices = [idx]

        #print(f'idx = {idx}')

        with h5py.File(self.file_path, 'r') as f:

            tokens_list = []
 
            # # Read data for each random index
            for idx in random_indices:
                tokens_list.append(f['tokens'][idx][:])
            # Concatenate all data
            tokens = np.concatenate(tokens_list)
 
 
        
        # limit sequence length
        seq_len = min(len(tokens), self.max_seq_length)
        tokens = tokens[:seq_len]

        #print(f'tokens count = {len(tokens)}')

        tokens_n1 = tokens[:seq_len-1]
        #print(f'tokens_n1 count = {len(tokens_n1)}')

        # Padding and mask
        padded_tokens_input = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_input[:seq_len-1] = tokens[:seq_len-1]

        padded_tokens_target = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_target[0:seq_len-1] = tokens[1:seq_len]

        #print(padded_tokens_input)
        #print(padded_tokens_target)
        
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        attention_mask[0:seq_len-1] = 1.0

        #print(f'dataloader sum attentionmask = {np.sum(attention_mask)}')
        
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
        


