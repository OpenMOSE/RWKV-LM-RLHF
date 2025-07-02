import torch
import h5py
import numpy as np
import random
import copy
from torch.utils.data import Dataset, DataLoader

class HDF5TopKTensorDataset(Dataset):
    def __init__(self,args, file_path, top_k=100, max_seq_length=4096,overlap_div=4):
        self.file_path = file_path
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.args = args

        self.tokenshift = True
        self.CurrentExcessToken = None
        self.CurrentExcessTopKValue = None
        self.CurrentExcessTopKIndice = None
        self.overlap_length = int(max_seq_length / overlap_div)

        self.debug = False
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])
    
    def __len__(self):
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.dataset_length
    
    def __getitem__(self, idx):

        N = self.args.infctx_dataset_multiplier#1

        random_indices = [random.randint(0, self.dataset_length - 1) for _ in range(N)]

        #if self.args.random_mode:
        #    idx = random.randint(0, self.dataset_length - 1)
        #    idx2 = random.randint(0, self.dataset_length - 1)



        #with h5py.File(self.file_path, 'r') as f:
        #    tokens = f['tokens'][idx][:]
        #    top_k_values = f['top_k_values'][idx][:]
        #    top_k_indices = f['top_k_indices'][idx][:]


        LastTokens = None
        LastTopKValue = None
        LastTopKIndice = None


        #tokens = None
        tokens = None
        top_k_values = None
        top_k_indices = None


        if self.CurrentExcessToken is not None:
            if len(self.CurrentExcessToken) > 0:
                LastTokens = self.CurrentExcessToken
                LastTopKValue = (self.CurrentExcessTopKValue)
                LastTopKIndice = (self.CurrentExcessTopKIndice)

        GetNewDataset = False

        

        if LastTokens is not None:
            if len(LastTokens) < self.max_seq_length:
                GetNewDataset = True
                if self.debug: print('LastTokens is below max seq length')
                #print(f'lasttoken = {len(LastTokens)}')
            else:
                GetNewDataset = False
                #print(f'lasttoken = {len(LastTokens)}')
        else: 
            #if self.debug: print('LastTokens is None')
            GetNewDataset = True

        if self.tokenshift == False:
            GetNewDataset = True



        if GetNewDataset:
            if self.debug:
                print('GetNewDataset')
            with h5py.File(self.file_path, 'r') as f:

                tokens_list = []
                top_k_values_list = []
                top_k_indices_list = []
            
                for idx in random_indices:
                    tokens_list.append(f['tokens'][idx][:])
                    top_k_values_list.append(f['top_k_values'][idx][:])
                    top_k_indices_list.append(f['top_k_indices'][idx][:])
                    
                
                # Concatenate all data
                tokens = np.concatenate(tokens_list)
                top_k_values = np.concatenate(top_k_values_list)
                top_k_indices = np.concatenate(top_k_indices_list) 

                top_k_values = top_k_values.reshape(len(tokens),-1)
                top_k_indices = top_k_indices.reshape(len(tokens),-1)

                #print(f'top_k_values = {top_k_values.shape}')

            #print(f'idx = {idx}')
        if self.tokenshift:
            if LastTokens is not None and tokens is not None:
                if self.debug:
                    print(f'Combined with old tokens {len(LastTokens)} {len(tokens)}')
                tokens = np.hstack((LastTokens,tokens))
                top_k_values = np.vstack((LastTopKValue,top_k_values))
                top_k_indices = np.vstack((LastTopKIndice,top_k_indices))
                #print(f'top_k_values = {top_k_values.shape}')


                #print(f'tokens = {len(tokens)} topkvalues = {len(top_k_values)} topkindices = {len(top_k_indices)}')
            elif LastTokens is not None:
                if self.debug:
                    print(f'Load from old tokens {len(LastTokens)}')
                tokens = LastTokens
                top_k_values = LastTopKValue
                top_k_indices = LastTopKIndice
        
        # limit sequence length
        seq_len = min(len(tokens), self.max_seq_length)

        if self.tokenshift:
            if len(tokens) > self.max_seq_length:
                #Oversize tokens, reserve next token
                if self.debug:
                    print(f'tokens= {tokens}')
                    print(f'tokens cut [{self.max_seq_length-self.overlap_length} : {len(tokens)-1}]')
                #print(f'tokens = {len(tokens)} topkvalues = {len(top_k_values)} topkindices = {len(top_k_indices)}')
                self.CurrentExcessToken = tokens[ self.max_seq_length-self.overlap_length : len(tokens)-1]
                self.CurrentExcessTopKValue= top_k_values[ self.max_seq_length-self.overlap_length : len(tokens)-1]#[:]
                self.CurrentExcessTopKIndice= top_k_indices[ self.max_seq_length-self.overlap_length : len(tokens)-1]#[:]
                #print(f'CurrentExcessToken = {len(self.CurrentExcessToken)} CurrentExcessTopKValue = {len(self.CurrentExcessTopKValue)} CurrentExcessTopKIndice = {len(self.CurrentExcessTopKIndice)}')
                if self.debug:
                    print(f'Stored to ExcessToken {len(self.CurrentExcessToken)}')
            else:
                self.CurrentExcessToken = None
                self.CurrentExcessTopKValue = None
                self.CurrentExcessTopKIndice = None






        tokens = tokens[:seq_len]
        #top_k_values = top_k_values[:seq_len][:]
        #top_k_indices = top_k_indices[:seq_len][:]

        #print(len(tokens))
        
        # Top-K値とインデックスを2次元に戻す
        #top_k_values = top_k_values[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        #top_k_indices = top_k_indices[:seq_len * self.top_k].reshape(seq_len, self.top_k)
        
        # padding and mask
        # padded_tokens = np.zeros(self.max_seq_length, dtype=np.int64)
        # padded_tokens[:seq_len] = tokens
        
        padded_top_k_values = np.zeros((self.max_seq_length, self.top_k), dtype=np.float32)
        padded_top_k_values[:seq_len-1] = top_k_values[:seq_len-1]
        
        padded_top_k_indices = np.zeros((self.max_seq_length, self.top_k), dtype=np.int64)
        padded_top_k_indices[:seq_len-1] = top_k_indices[:seq_len-1]



        # padding and mask
        padded_tokens_input = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_input[:seq_len-1] = tokens[:seq_len-1]

        padded_tokens_target = np.zeros(self.max_seq_length, dtype=np.int64)
        #padded_tokens_target[:seq_len] = tokens[:seq_len]
        #padded_tokens_target = padded_tokens_target[1:]
        padded_tokens_target[0:seq_len-1] = tokens[1:seq_len]
        
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        attention_mask[:seq_len-1] = 1.0



        
        # attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)
        # attention_mask[:seq_len] = 1.0
        
        return {
            'input_ids': torch.from_numpy(padded_tokens_input),
            'target_ids': torch.from_numpy(padded_tokens_target),
            'top_k_values': torch.from_numpy(padded_top_k_values).to(dtype=torch.bfloat16),
            'top_k_indices': torch.from_numpy(padded_top_k_indices),
            'attention_mask': torch.from_numpy(attention_mask).to(dtype=torch.bfloat16)
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'top_k_values': torch.stack([item['top_k_values'] for item in batch]),
        'top_k_indices': torch.stack([item['top_k_indices'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }
        


