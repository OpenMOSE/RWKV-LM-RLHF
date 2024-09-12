import torch
import h5py
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class HDF5TopKTensorDataset(Dataset):
    def __init__(self,args, file_path, top_k=100, max_seq_length=4096):
        self.file_path = file_path
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.args = args
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])
    
    def __len__(self):
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.dataset_length
    
    def __getitem__(self, idx):

        if self.args.random_mode:
            idx = random.randint(0, self.dataset_length - 1)



        with h5py.File(self.file_path, 'r') as f:
            tokens = f['tokens'][idx][:]
            top_k_values = f['top_k_values'][idx][:]
            top_k_indices = f['top_k_indices'][idx][:]

        #print(f'idx = {idx}')
        
        # limit sequence length
        seq_len = min(len(tokens), self.max_seq_length)
        tokens = tokens[:seq_len]
        
        # Top-K値とインデックスを2次元に戻す
        top_k_values = top_k_values[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        top_k_indices = top_k_indices[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        
        # padding and mask
        padded_tokens = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens[:seq_len] = tokens
        
        padded_top_k_values = np.zeros((self.max_seq_length, self.top_k), dtype=np.float32)
        padded_top_k_values[:seq_len] = top_k_values
        
        padded_top_k_indices = np.zeros((self.max_seq_length, self.top_k), dtype=np.int64)
        padded_top_k_indices[:seq_len] = top_k_indices
        
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        attention_mask[:seq_len] = 1.0
        
        return {
            'input_ids': torch.from_numpy(padded_tokens),
            'top_k_values': torch.from_numpy(padded_top_k_values).to(dtype=torch.bfloat16),
            'top_k_indices': torch.from_numpy(padded_top_k_indices),
            'attention_mask': torch.from_numpy(attention_mask).to(dtype=torch.bfloat16)
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'top_k_values': torch.stack([item['top_k_values'] for item in batch]),
        'top_k_indices': torch.stack([item['top_k_indices'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }
        


