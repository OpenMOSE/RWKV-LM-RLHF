# First Implementation
# GRPO (Group Relative Policy Optimization) ぐるぽ

# 2025 OpenMOSE

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
import re

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

#@torch.jit.script
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

#Padding [B][T]-> [B][target]
def pad_to_size_2d(tensor, target_size=2048):
    current_size = tensor.size(1)
    pad_size = target_size - current_size
    # (L, 右, 上, 下)のパディングを指定
    padding = (0, pad_size, 0, 0)
    
    return F.pad(tensor, padding, mode='constant', value=0)
#Padding [B][T][]-> [B][target][]
def pad_to_size_3d(tensor, target_size=2048):
    current_size = tensor.size(1)
    pad_size = target_size - current_size
    padding = (0, 0,      # 最後の次元（65536）
              0, pad_size,# 真ん中の次元（648→2048など）
              0, 0)       # 最初の次元（1）
    return F.pad(tensor, padding, mode='constant', value=0)

@torch.compile
def GenerateForwardTokens(self,prompt,batchcount = 3,stoplength=50,additionaltoken=None,stoptoken='\n\n',temp=1.0,topp = 0.9): # from RWKV-Infer Methods. TSUKUTTETE YOKATTA-

    with torch.no_grad():
        args = self.args

        H =  args.dim_att // args.head_size_a

        B = batchcount

        

        
        temperature = torch.full((1,), temp)
        top_p = torch.full((1,), topp)
        GEN_alpha_presence = 0.3
        GEN_alpha_frequency = 0.3
        GEN_penalty_decay = 0.996

        #occurrence = {}
        occurrences = [{} for _ in range(B)]
        out_tokens = [[] for _ in range(B)]
        out_last = [0 for _ in range(B)]
        output_text = ['' for _ in range(B)]
        batch_status = [False for _ in range(B)] # if finished True
        batch_finishpos = [0 for _ in range(B)] # if finished True

        idx = prompt

        new_states = BlockStateList.create(args.n_layer, B, args.n_embd, H,
                                            self.emb.weight.device, self.emb.weight.dtype)
        

        batch_idx = idx.repeat(B, 1)

        shift_states = new_states.shift_states
        wkv_states = new_states.wkv_states
        x, shift_states, wkv_states = self.forward_rnn(batch_idx, shift_states, wkv_states,passthrough=False) #FLA

        #tokenaddress = {}

        #tokenaddress['prompt'] = x.shape[1]#(B,T,-1)

        #out_x = x.clone()

        #print('///////////////////////////////////////////////////////////////////////////////////////////////\n')
        
        for i in range(stoplength):

            for j in range(B):
                for n in occurrences[j]:
                    x[j,-1, n] -= GEN_alpha_presence + occurrences[j][n] * GEN_alpha_frequency # repetition penalty
            tokens = []
            for j in range(B):
                otokens_tensor = sampling_multibatch(self,x[j, -1].unsqueeze(0), temperature=temperature, top_p=top_p)
                otokens = otokens_tensor.tolist()
                tokens.append(torch.tensor(otokens).unsqueeze(0).to('cuda'))
                if batch_status[j] == False:
                   batch_finishpos[j] = batch_finishpos[j]+1
                out_tokens[j] += otokens#[otokens]

            idx = torch.cat(tokens, dim=0)

            #print(idx.shape)
            #exit()

            breaks = False

            for j in range(B):
                #out_tokens[j] += [otokens[j]]
                try:
                    tmp = self.tokenizer.decode(out_tokens[j][out_last[j]:])
                    
                    if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                            if batch_status[j] == False:
                                output_text[j] = output_text[j] + tmp
                                out_last[j] = i + 1
                    if batch_status[j] == False and stoptoken in tmp:
                            #print(f'\nEndtoken\n')
                            tmp = tmp.replace(stoptoken,'')

                            if len(output_text[j]) != 0:
                                 batch_status[j] = True
                                #  if batch_finishpos[j] == 0:
                                #     batch_finishpos[j] = i  
                    
                    if batch_status[j] == False and stoptoken in output_text[j]:
                            #break
                            breaks = True
                            batch_status[j] = True

                            output_text[j] = output_text[j].replace(stoptoken,'')

                            #if batch_finishpos[j] == 0:
                            #   batch_finishpos[j] = i  
                            #print('ENDTOKENDETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
          
                except Exception as e:
                    #print('exceptions')
                    #print(f"エラーが発生しました: {type(e).__name__}")
                    #print()
                    pass

            # if breaks:
            #      break
            breaks = True
            for j in range(B):
                if batch_status[j] == False:
                     breaks = False

            if breaks:
                 break
                 


            x, shift_states, wkv_states = self.forward_rnn(idx, shift_states, wkv_states,passthrough=False)
            #out_x = torch.cat([out_x, x], dim=1)

            #x=x.cpu()
            for j in range(B):
                for xxx in occurrences[j]:
                    occurrences[j][xxx] *= GEN_penalty_decay
                    occurrences[j][x[j,-1]] = 1 + (occurrences[j][x[j,-1]] if x[j,-1] in occurrences[j] else 0)

        #tokenaddress['generated'] = out_x.shape[1] - tokenaddress['prompt']

    #print(output_text)

    #print('\n///////////////////////////////////////////////////////////////////////////////////////////////\n')
    #tensor_tokens =  torch.tensor(out_tokens).to(device=self.emb.weight.device).view(B,-1)   
            
    return output_text# , batch_finishpos


def grpo_init(self):
    #print('Zero CoT Initialize')
    self.CoTDebug = self.args.grpo_debug

    self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

    self.grpo_gen_count = self.args.grpo_gen_count
    self.grpo_gen_length = self.args.grpo_gen_length
    self.grpo_gen_temperature = self.args.grpo_gen_temperature
    self.grpo_gen_topp = self.args.grpo_gen_topp
    self.grpo_kl_beta = self.args.grpo_kl_beta

##############################################################################
# Load and prep dataset
SYSTEM_PROMPT = """Respond in the following format:
<think>
Thinking Text (up to 100 words)
</think>
<answer>
Answer Text (up to 100 words)
</answer>"""

PENALTY_OUTPUT = '''<think>
Thinking Text (up to 100 words)
</think>
<answer>
Answer Text (up to 100 words)
</answer>'''

XML_COT_FORMAT = """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


##############################################################################################################################################
#define Rewards
#modified from unsloth method


# Reward functions
def correctness_reward_func(prompt, completions, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    q = prompt
    extracted_responses = [extract_xml_answer(r) for r in responses]

    #print(f'Extracted Response = {extracted_responses}')
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def repetition_penalty_reward_func(completions, **kwargs) -> list[float]:
    matches = []
    for text in completions:
        if PENALTY_OUTPUT in text:
              matches.append(0.0)  #if contained input prompt.
        else:
              matches.append(1.0) 
    return matches

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]

def my_rulebase_reward_func(prompt, completions: List[str],answer) -> List[float]:
    """
    Reward Function

    """
    rewards1 = correctness_reward_func(prompt,completions,answer)
    rewards2 = int_reward_func(completions)
    rewards3 = strict_format_reward_func(completions)
    rewards4 = soft_format_reward_func(completions)
    rewards5 = xmlcount_reward_func(completions)


    penalty = repetition_penalty_reward_func(completions)

    # Rewards = list((
    #                np.array(rewards1) + 
    #                np.array(rewards2) +
    #                np.array(rewards3) +
    #                np.array(rewards4) +
    #                np.array(rewards5)
    #                 ) * np.array(penalty)
                   
    #                )
    # すべての報酬配列が同じ長さであることを確認
    rewards = np.array(rewards1) + \
            np.array(rewards2) + \
            np.array(rewards3) + \
            np.array(rewards4) + \
            np.array(rewards5)

    # penaltyとrewardsの形状を確認
    final_rewards = rewards * np.array(penalty)

    # 必要に応じてリストに変換
    Rewards = final_rewards.tolist()
    
    #print(f'Rewards = {Rewards}')

    return Rewards


###########################################################################################################################################
'''
Seriously, I implemented this code by trial and error,
so I'm sure there are lots of mistakes.
Please help...........
'''

def training_step_grpo(self, batch, batch_idx):
    """
    GRPO (Group Relative Policy Optimization) 
    
      - Sampling G times (GenerateCount times) per batch => generated token sequence o_i
      - For each sample i:
    Calculate reward r_i and advantage A_i = (r_i - mean(r)) / std(r)
    Gather log πθ and ratio=πθ/πref for each generated token
    L_{GRPO} = - 1/G \sum_i [ 1/|o_i| \sum_t( ratio_no_grad * log πθ * A_i ) ] + β KL
    """
    args = self.args

    bsz = len(batch)
    assert bsz == 1, "currently can run only 1bsz"

    GenerateCount = self.grpo_gen_count  # G: Generate Count each batch hardcoded.

    # Get prompt and chosen from dataset
    prompttoken = batch[0]['prompttoken'][: args.ctx_len // 2]
    chosentoken = batch[0]['chosentoken'][: args.ctx_len // 2]

    prompt_text = self.tokenizer.decode(prompttoken.tolist()).replace('\n\n','\n')
    final_answer_text = self.tokenizer.decode(chosentoken.tolist()).replace('\n\n','\n')

    # Prompt format for RWKV. now hard coding....
    system_prefix = 'System: ' + SYSTEM_PROMPT 
    user_prefix   = system_prefix + "\n\nUser: "
    user_prefix_without_system = "User: "
    asst_prefix   = "\n\nAssistant:"


    prompt_prefix = user_prefix + prompt_text + asst_prefix

    # tokenize text-> idx
    prompt_idx = torch.tensor(
        self.tokenizer.encode(prompt_prefix),
        device=self.emb.weight.device
    ).unsqueeze(0)

    # 2) Generate Tokens(without Grad) maybe be we have better another option
    generated_completions_all = []
    combined_idxs_all = []
    token_addresses_list = []


    gen_texts= GenerateForwardTokens(
            self,
            prompt=prompt_idx,
            batchcount=GenerateCount,
            stoplength=self.grpo_gen_length,  #Maximum Generate Length
            additionaltoken=None,
            stoptoken='User:',  # StopToken
            temp=self.grpo_gen_temperature,
            topp=self.grpo_gen_topp
        )
    

    
    for g_i in range(GenerateCount):
        # if gen_tokens.shape[1] == batch_finishpos[g_i]:
        #     single_gen_tokens = gen_tokens[g_i]
        # else:
        #     single_gen_tokens = gen_tokens[g_i][:batch_finishpos[g_i]]
        # combined_idx = torch.cat([prompt_idx, single_gen_tokens.unsqueeze(0)], dim=1)  # shape [1, T_total]

        # Decode for debug
        # all_text = self.tokenizer.decode(combined_idx[0].tolist())
        # gen_text = self.tokenizer.decode(single_gen_tokens.unsqueeze(0).tolist())

        all_text = user_prefix_without_system + gen_texts[g_i]
        gen_text = gen_texts[g_i]

        combined_idx = torch.tensor(
                    self.tokenizer.encode(all_text),
                    device=self.emb.weight.device
                    ).unsqueeze(0)

        if self.CoTDebug:
            print(f"[DEBUG] Sample {g_i}:  {all_text}")

        generated_completions_all.append(gen_text)  # use for reward
        combined_idxs_all.append(combined_idx)


    # 3) Compute Reward and normalize
    rewards_list = my_rulebase_reward_func(
        prompt=user_prefix, 
        completions=generated_completions_all, 
        answer=final_answer_text
    )  

    # for debugging
    if self.CoTDebug:
        print("[DEBUG] rewards_list:", rewards_list)

    # normalize rewards (A_i = (r_i - mean)/std)
    rewards_tensor = torch.tensor(rewards_list, device=self.emb.weight.device, dtype=torch.float)
    mean_r = rewards_tensor.mean()
    std_r  = rewards_tensor.std(unbiased=False) + 1e-8
    advantages_tensor = (rewards_tensor - mean_r) / std_r  # shape [G]

    #Padding for fixed chunksize. and forward BaseModel and ActorModel
    #this is for avoid FLA Kernels error.(maybe variable len is buggy?)
    max_len = max(t.shape[1] for t in combined_idxs_all)
    for i in range(GenerateCount):
        combined_idxs_all[i] = pad_to_size_2d(combined_idxs_all[i], max_len)

    all_combined_idx = torch.cat(combined_idxs_all, dim=0)  # shape [G, max_len]

    with torch.no_grad():
        base_logits, _ = BaseModel_Forward_NoGrad(self, pad_to_size_2d(all_combined_idx, args.ctx_len))
    actor_logits, _   = ActorModel_Forward_Grad(self, pad_to_size_2d(all_combined_idx, args.ctx_len))

    # shape整理
    base_logits  = base_logits[:, :max_len, :]
    actor_logits = actor_logits[:, :max_len, :]

    device = self.emb.weight.device
    sum_loss_reinforce_torch = torch.zeros([], device=device, requires_grad=False)
    sum_kl_torch = torch.zeros([], device=device, requires_grad=False)

    # Gather ratio=πθ/πref and log(πθ)
    # → Loop "1/|o_i| sum_t(...) according to the formula" and multiply by 1/G at the end
    G = GenerateCount
 

    for i in range(G):
        # each shape [1, max_len, vocab]
        actor_i = actor_logits[i:i+1]
        base_i  = base_logits[i:i+1]
        tokens_i = combined_idxs_all[i]  # shape [1, max_len]

        # An example of extracting the next token after one step shift as the correct answer (or gather)
        # [1, max_len-1, vocab], [1, max_len-1]
        # Create a mask to avoid padding
        shifted_actor = actor_i[:, :-1, :]
        shifted_base  = base_i[:, :-1, :]
        shifted_tokens = tokens_i[:, 1:]  # shape [1, max_len-1]

        # padding mask
        flat_tokens = shifted_tokens.reshape(-1)  # [max_len-1]
        valid_mask  = (flat_tokens != 0)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # 1D index

        # gather
        actor_flat = shifted_actor.reshape(-1, actor_i.size(-1))  # [max_len-1, vocab]
        base_flat  = shifted_base.reshape(-1, actor_i.size(-1))

        actor_valid = actor_flat[valid_indices]  # [n_valid, vocab]
        base_valid  = base_flat[valid_indices]
        tokens_valid = flat_tokens[valid_indices]  # [n_valid]

        if tokens_valid.numel() == 0:
            continue

        # log_probs
        log_p_actor_valid = F.log_softmax(actor_valid, dim=-1)  # [n_valid, vocab]
        log_p_ref_valid   = F.log_softmax(base_valid,  dim=-1)

        # ratio = exp(log(πθ) - log(πref))
        ratio_full = torch.exp(log_p_actor_valid - log_p_ref_valid)  # [n_valid, vocab]

        # gather chosen tokens
        row_idx = torch.arange(ratio_full.size(0), device=ratio_full.device)
        chosen_toks = tokens_valid  # [n_valid]
        ratio_chosen = ratio_full[row_idx, chosen_toks]  # [n_valid]
        lp_chosen    = log_p_actor_valid[row_idx, chosen_toks]

        # KL term (approximation: ratio - log(ratio) - 1)
        # using chosen token only
        kl_per_tok = ratio_chosen - torch.log(ratio_chosen + 1e-10) - 1.0  # [n_valid]
        kl_mean_i  = kl_per_tok.mean()

        # Advantage is same each sample
        adv_i = advantages_tensor[i]

        # ratio は no_grad
        ratio_chosen_nd = ratio_chosen.detach()

        # REINFORCE term: ( ratio_no_grad * log πθ * A_i )
        # sum/average over t ∈ i
        # Here we multiply by "1 / |o_i|", so (1 / n_valid) corresponds approximately
        # but strictly speaking "n_valid" ≒ "|o_i|-1" may be true.
        # Here we consider (1 / n_valid) to be equivalent to "1 / |o_i|".
        # Or we can strictly count "1 / (number of tokens_i generated)".
        reinforce_per_tok = ratio_chosen_nd * lp_chosen * adv_i

        reinforce_mean_i = reinforce_per_tok.mean()

        sum_loss_reinforce_torch += reinforce_mean_i
        sum_kl_torch += kl_mean_i



    # サンプル i の平均 => / G
    reinforce_loss = - (sum_loss_reinforce_torch / G)
    kl_value_torch = (sum_kl_torch / G)

    beta_kl = self.grpo_kl_beta #beta hardcoded.

    kl_loss = beta_kl * kl_value_torch

    total_loss = reinforce_loss + kl_loss

    total_loss = L2Wrap.apply(total_loss,actor_logits)

    # for debug...
    self.log("rewards_mean", float(mean_r), prog_bar=True)
    # self.log("rewards_std",  float(std_r),  prog_bar=True)
    # self.log("kl_value",     float(kl_value_torch.item()), prog_bar=True)
    # self.log("loss_reinforce", float(reinforce_loss), prog_bar=True)

    self.trainer.rewards_mean = float(mean_r)
    self.trainer.rewards_std = float(std_r)
    self.trainer.kl_value = float(kl_value_torch.item())
    self.trainer.loss_reinforce = float(reinforce_loss)

    return total_loss

