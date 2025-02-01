# CoT Training Method がんばって実現しよう

# Test Reinforce Implement with LoRA
# Firsttime Online RL :)

# 2025 OpenMOSE, always need help

# My Approches(for gpu poors like me 4090)
# Most target is v7 0B4 World v2.9

#RNN Forward Use Flash-Linear-Attention Fused_Recurrent
#Chunk Forward Use Flash-Linear-Attention or CUDA or Triton.


# Basemodel = frozen basemodel
# Action model = basemodel + LoRA(only linears, trainable)
# Clitic = basemodel + clitic_head(only linears trainable)

# Important
# Enable Linears for train only.(if included other params. im not sure what happens)


# First Approach

# 0. Configure LayerProfile (for my testing, use fulllayer + LoRA Rank=16 Scale=2.0) my experiment, its better lora than bone for rlhf.
# 1. SFT InstructPars with "<cot>" tokens
# 2. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import numpy as np
import os
from tokenizer.rwkv_tokenizer import *
from .trainutils import *
from .infctx_module import *

def BaseModel_Forward_NoGrad(self,idx): #can call anytime.
    with torch.no_grad():
        return self.forward(idx,frozen=True,passthrough=True)
def BaseModel_Forward_Grad(self,idx):  # basically dont use
        return self.forward(idx,frozen=False,passthrough=True)

def ActorModel_Forward_NoGrad(self,idx): #This is for forward test
    with torch.no_grad():
        return self.forward(idx,frozen=True,passthrough=False)
    
def ActorModel_Forward_NoGrad(self,idx): #This is for forward test

    return self.forward(idx,frozen=True,passthrough=False)


def ActorModel_Forward_Grad(self,idx): # can call 1time per step. because of gradient checkpointing
        return self.forward(idx,frozen=False,passthrough=False)

def CliticModel_Forward_Grad(self,idx): 
        return self.forward_head2(idx,frozen=True,passthrough=True,clitic=True)

def sampling_multibatch(self,logits, temperature, top_p): # This is From RWKV-Infer
        # you can change any sampler
        device = logits.device

        temperature = temperature.view(-1, 1).to(device=device,dtype=logits.dtype)
        temperature[temperature == 0.0] = 1.0
        p = top_p.view(-1, 1).to(device=device,dtype=logits.dtype)
        probs = F.softmax(logits.to(dtype=torch.bfloat16), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        shifted = torch.zeros_like(sorted_indices_to_remove)
        shifted[:, 1:] = sorted_indices_to_remove[:, :-1]
        sorted_indices_to_remove = shifted
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove = indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)

        if not torch.all(temperature == 1.0):
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum(dim=-1, keepdim=True)
        
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return samples#


def GenerateForwardTokens(self,prompt,stoplength=50,additionaltoken=None,stoptoken='\n\n',temp=1.0,topp = 0.5): # from RWKV-Infer Methods. TSUKUTTETE YOKATTA-

    with torch.no_grad():
        args = self.args

        H =  args.dim_att // args.head_size_a

        B = 1 #Single Batch
        new_states = BlockStateList.create(args.n_layer, B, args.n_embd, H,
                                            self.emb.weight.device, self.emb.weight.dtype)
        
        temperature = torch.full((B,), temp)
        top_p = torch.full((B,), topp)
        GEN_alpha_presence = 0.5
        GEN_alpha_frequency = 0.5
        GEN_penalty_decay = 0.996

        occurrence = {}
        out_tokens = [[] for _ in range(B)]
        out_last = [0 for _ in range(B)]
        output_text = ['' for _ in range(B)]

        idx = prompt

        shift_states = new_states.shift_states
        wkv_states = new_states.wkv_states

        x, shift_states, wkv_states = self.forward_rnn(idx, shift_states, wkv_states,passthrough=True) #FLA

        tokenaddress = {}

        tokenaddress['prompt'] = x.shape[1]#(B,T,-1)

        out_x = x.clone()

    
        
        for i in range(stoplength):

            for n in occurrence:
                x[:,-1, n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
            
            x[:, -1, 0] -= 1e10

            otokens_tensor = sampling_multibatch(self,x[:, -1], temperature=temperature, top_p=top_p)
            otokens = otokens_tensor.tolist()

            tokens = []
            for j in range(B):
                tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

            idx = torch.cat(tokens, dim=0)

            for j in range(B):
                out_tokens[j] += [otokens[j]]
                try:
                    tmp = self.tokenizer.decode(out_tokens[j][out_last[j]:])
                    if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                            #if j == Target_batch - 1:
                            print(tmp,end="", flush=True)
                            output_text[j] = output_text[j] + tmp
                            out_last[j] = i + 1
                    if stoptoken in tmp:
                            print(f'Endtoken = {repr(tmp)}')
                            tmp = tmp.replace(stoptoken,'')
                            break
                            
                except:
                    pass


            x, shift_states, wkv_states = self.forward_rnn(idx, shift_states.clone(), wkv_states.clone(),passthrough=True)
            out_x = torch.cat([out_x, x], dim=1)

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
                occurrence[x[0,-1]] = 1 + (occurrence[x[0,-1]] if x[0,-1] in occurrence else 0)

        tokenaddress['generated'] = out_x.shape[1] - tokenaddress['prompt']

        if additionaltoken != None:
            x, shift_states, wkv_states = self.forward_rnn(additionaltoken, shift_states, wkv_states,passthrough=True) #FLA
            out_x = torch.cat([out_x, x], dim=1)
            tokenaddress['additionalprompt'] = x.shape[1]
                 
        return out_x, torch.tensor(out_tokens).to(device=self.emb.weight.device), tokenaddress

        
     

def zerocot_init(self):
    print('Zero CoT Initialize')
    self.CoTDebug = True

    self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

def training_step_zerocot(self, batch, batch_idx):
    print('Training Step ZeroCoT')
    batch_orpo = batch

    # 損失集計用
    loss_sft_all = 0.0   # 例: 通常の教師強制Loss(もし使うなら)
    loss_rl_all  = 0.0   # REINFORCE目的のLoss
    
    try: self.trainer.pref_match_percentage
    except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
    pref_matches = 0
    bsz = len(batch_orpo)
    loss2 = 0.0

    #print(batch_orpo)

    bsz = 1 # for test


    GenerateCount = 3
    GenerateTokens = 30


    for s in range(bsz):

        prompttoken = batch_orpo[s]['prompttoken']#.unsqueeze(0)
        chosentoken = batch_orpo[s]['chosentoken']#.unsqueeze(0)

        #both tokenized. but, MENDOKUSAI NODE ITTAN TEXT Ni SURU

        prompt_text = self.tokenizer.decode(prompttoken.tolist())
        chosen_text = self.tokenizer.decode(chosentoken.tolist())
        
        #Instruct Rules
        SplitText = '\n\n'
        UserText = 'User: '
        AssistantText = "Assistant: "
        ThinkingText = "Thinking:"

        #Make Prompt for Forward
        GeneratePrompt = UserText + prompt_text + SplitText
        GeneratePrompt += ThinkingText

        TempToken = self.tokenizer.encode(GeneratePrompt) 
        prompts = [] 
        prompts.append(torch.tensor(TempToken).unsqueeze(0).to(self.emb.weight.device))
        forward_idx = torch.cat(prompts, dim=0)

        TempToken = self.tokenizer.encode(SplitText + AssistantText + chosen_text) 
        prompts = [] 
        prompts.append(torch.tensor(TempToken).unsqueeze(0).to(self.emb.weight.device))
        chosen_idx = torch.cat(prompts, dim=0)

        print(GeneratePrompt)

        generated_logit = []
        generated_token = []
        generated_tokenaddress = []
        basemodel_logit = []

        for i in range(GenerateCount):
            generated_x, generated_tokens, tokenaddress = GenerateForwardTokens(self,
                                                                                prompt=forward_idx,
                                                                                additionaltoken=chosen_idx,
                                                                                stoplength=GenerateTokens,
                                                                                stoptoken=SplitText)
            # tokenaddress = {'prompt':prompt token size,'generated':generated token size, 'additionalprompt' : 'additional prompt token size'}

            #check basemodel logits
            prompts = []
            BaseModelIdx = torch.cat([forward_idx, generated_tokens,chosen_idx], dim=1)

            print(self.tokenizer.decode(BaseModelIdx.tolist()[0]))

            BaseModelLogits, _ = BaseModel_Forward_NoGrad(self,BaseModelIdx)
            

            generated_logit.append(generated_x)
            generated_token.append(generated_tokens)
            generated_tokenaddress.append(tokenaddress)
            basemodel_logit.append(BaseModelLogits)


            print(generated_x)
            print(f'generated_x shape = {generated_x.shape}')

        print(basemodel_logit)

        #generated_logit and generated_token contain prompt + thinking + chosen  tokens,logits, maybe can compute probs

        








        exit()

        #for Gradient Checkpointing, i can only forward 1 time with grad.
        logits = self.forward(idx)


        
        final_loss = L2Wrap.apply(final_loss, outputs_pos)

        loss2 = loss2 + final_loss # FinalLoss
        loss1 = loss1 + l2_pos_loss # SFT Loss
        lossorpoonly = lossorpoonly + pref_ratio #Pref-ratio
        

    
    loss2 = loss2 / bsz
    loss1 = loss1 / bsz
    lossorpoonly = lossorpoonly / bsz


    self.trainer.loss_2_dpo = float(lossorpoonly)
    self.trainer.loss_1_general_or_sft = float(loss1)
    self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

    return loss2



    return loss

