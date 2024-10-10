import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
#from pytorch_lightning.trainer.supporters import CombinedLoader
import os, sys, random

class DPODataset(Dataset):
    def __init__(self, args):
        self.args = args
        # TODO: to args.dpo_train_file
        self.data = torch.load(args.rlhf_train_file)
        # TODO: to 
        self.precision = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[args.precision]
        # self.precision = torch.bfloat16
        # self.data1, self.data2 = data

    def __len__(self):
        # return len(self.data)
        return self.args.epoch_steps * self.args.micro_bsz
        # return len(self.data1)

    def __getitem__(self, idx):
        idx = random.randrange(len(self.data))
        prompt_tokens, chosen_tokens, reject_tokens, chosen_base_prob, reject_base_prob = self.data[idx]
        if len(prompt_tokens) > self.args.rlhf_max_corpus_len:
            prompt_tokens = prompt_tokens[:self.args.rlhf_max_corpus_len]

        if len(chosen_tokens) > self.args.rlhf_max_corpus_len:
            chosen_tokens = chosen_tokens[:self.args.rlhf_max_corpus_len]
            

        if len(reject_tokens) > self.args.rlhf_max_corpus_len:
            reject_tokens = reject_tokens[:self.args.rlhf_max_corpus_len]
            
        #print(f'prompt tokens {len(prompt_tokens) }')
        #print(f'chosen_tokens {len(chosen_tokens) }')
        #print(f'reject_tokens {len(reject_tokens) }')
        
        return (
            # chosen_input, chosen_output
            torch.tensor(prompt_tokens + chosen_tokens[:-1], dtype=torch.long),#.unsqueeze(0),
            torch.tensor(prompt_tokens[1:] + chosen_tokens, dtype=torch.long),
            len(chosen_tokens),
            chosen_base_prob,
            # torch.tensor([0] * (len(prompt_tokens)-1) + [1] * len(chosen_tokens), dtype=self.precision),
            # reject_input, reject_output
            torch.tensor(prompt_tokens + reject_tokens[:-1], dtype=torch.long),#.unsqueeze(0),
            torch.tensor(prompt_tokens[1:] + reject_tokens, dtype=torch.long),
            len(reject_tokens),
            reject_base_prob,
            # torch.tensor([0] * (len(prompt_tokens)-1) + [1] * len(reject_tokens), dtype=self.precision),
        )
        
# dpo_dataset = DPODataset("validset.save")
# data_loader = DataLoader(dpo_dataset, batch_size=2, shuffle=True, collate_fn=lambda x:x)

# for batch in data_loader:
#     print(batch)

# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data1, self.data2 = data

#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
#         return self.data1[idx], self.data2[idx]

# my_data = [[torch.randn(5) for _ in range(20)], [torch.randn(1) for _ in range(20)]]
# custom_dataset_1 = CustomDataset(my_data)
# data_loader_1 = DataLoader(custom_dataset_1, batch_size=4, shuffle=True, collate_fn= lambda x: x)

# my_data2 = [[torch.randn(4) for _ in range(20)], [torch.randn(1) for _ in range(20)]]
# custom_dataset_2 = CustomDataset(my_data2)
# data_loader_2 = DataLoader(custom_dataset_2, batch_size=3, shuffle=True)

# loaders = CombinedLoader([data_loader_1, data_loader_2], "max_size_cycle")

# for batch in loaders:
#     print(batch)



