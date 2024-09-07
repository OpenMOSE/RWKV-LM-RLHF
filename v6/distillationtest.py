import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HDF5TopKTensorDataset(Dataset):
    def __init__(self, file_path, top_k=100, max_seq_length=4096):
        self.file_path = file_path
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            tokens = f['tokens'][idx][:]
            top_k_values = f['top_k_values'][idx][:]
            top_k_indices = f['top_k_indices'][idx][:]
        
        # シーケンス長を制限
        seq_len = min(len(tokens), self.max_seq_length)
        tokens = tokens[:seq_len]
        
        # Top-K値とインデックスを2次元に戻す
        top_k_values = top_k_values[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        top_k_indices = top_k_indices[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        
        # パディングとマスクの作成
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
            'top_k_values': torch.from_numpy(padded_top_k_values),
            'top_k_indices': torch.from_numpy(padded_top_k_indices),
            'attention_mask': torch.from_numpy(attention_mask)
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'top_k_values': torch.stack([item['top_k_values'] for item in batch]),
        'top_k_indices': torch.stack([item['top_k_indices'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }

# 使用例
file_path = "datasets/test_jp_logits.h5py"
dataset = HDF5TopKTensorDataset(file_path, top_k=100, max_seq_length=4096)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)

# データローダーの使用例
for batch in dataloader:
    input_ids = batch['input_ids']
    top_k_values = batch['top_k_values']
    top_k_indices = batch['top_k_indices']
    attention_mask = batch['attention_mask']

    print(f'input_ids shape = {input_ids.shape}')
    print(f'top_k_values shape = {top_k_values.shape}')
    print(f'top_k_indices shape = {top_k_indices.shape}')
    print(f'attention_mask shape = {attention_mask.shape}')

    #print(attention_mask)