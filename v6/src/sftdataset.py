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
    
    def __len__(self):
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.dataset_length
    
    def __getitem__(self, idx):

        N = 1

        random_indices = [random.randint(0, self.dataset_length - 1) for _ in range(N)]

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
        
        # padding and mask
        padded_tokens = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens[:seq_len] = tokens
        
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        attention_mask[:seq_len] = 1.0
        
        return {
            'input_ids': torch.from_numpy(padded_tokens),
            'attention_mask': torch.from_numpy(attention_mask).to(dtype=torch.bfloat16)
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }
        


