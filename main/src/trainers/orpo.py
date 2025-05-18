import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import numpy as np
import os
from tokenizer.rwkv_tokenizer import *
from ..trainutils import *
from ..infctx_module import *
import requests
import time
def compute_logps_simple_mask(chosen_inputs, logits, attention_mask=None):

            log_probs = torch.log_softmax(logits[:, :-1, :], dim=2)

            gathered_log_probs = torch.gather(log_probs, dim=2, index=chosen_inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1]
            else:
                attention_mask = torch.ones_like(gathered_log_probs)

            masked_log_probs = gathered_log_probs * attention_mask
            
            sequence_logps = masked_log_probs.sum(dim=1)
            
            effective_lengths = attention_mask.sum(dim=1)
            
            #normalized_sequence_logps = sequence_logps / effective_lengths
            
            return sequence_logps
def training_step_orpo(self, batch, batch_idx):
            
            args = self.args


            if args.orpo and args.orpo_mode == 0:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0 
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                
                #SFT_targets = []
                for s in range(bsz):
                    if os.environ["H5_MODE"] == "1":
                        chosen_input = batch_orpo[s]['chosen_input']
                        chosen_output = batch_orpo[s]['chosen_target']
                        length_chosen = batch_orpo[s]['chosen_token_len']
                        chosen_ref_prob = batch_orpo[s]['chosen_base_prob']
                        reject_input = batch_orpo[s]['reject_input']
                        reject_output = batch_orpo[s]['reject_target']
                        length_reject = batch_orpo[s]['reject_token_len']
                        reject_ref_prob = batch_orpo[s]['reject_base_prob']
                        chosen_token = batch_orpo[s]['chosentoken']
                        reject_token = batch_orpo[s]['rejecttoken']
                    else:
                        chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob,chosen_token,reject_token = batch_orpo[s]

                    # マスクの作成
                    chosen_mask = (chosen_output != 0).float()  # パディングは0と仮定
                    reject_mask = (reject_output != 0).float()

                    
                    # 両方のテンソルの長さを取得
                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)

                    #print(f'len1 = {len1}')
                    #print(f'len2 = {len2}')

                    # 最大長を計算
                    max_len = max(len1, len2)

                    if '07' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    # 長さが異なる場合、短いテンソルをパディング
                    if len1 < max_len:
                        # len1がmax_lenになるようにパディングを追加 (右側にパディング)
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        # len2がmax_lenになるようにパディングを追加 (右側にパディング)
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))
    

                    SFT_idx = []
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0) # make batch with Chosen and Reject  

                    RT ,moe_loss= self(SFT_idx)


                    #print(RT)
                    outputs_pos = RT[0].unsqueeze(0)
                    outputs_neg = RT[1].unsqueeze(0)

                    #del RT
                    del SFT_idx

                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()

                    l2_pos_loss = masked_cross_entropy(outputs_pos,chosen_output,chosen_mask)


                    pos_prob = compute_logps_simple_mask(chosen_output.unsqueeze(0),outputs_pos,chosen_mask.unsqueeze(0))
                    neg_prob = compute_logps_simple_mask(reject_output.unsqueeze(0),outputs_neg,reject_mask.unsqueeze(0))


                    orpo_ratio = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
                    
                    pref_matches += (orpo_ratio > 0)

                    orpo_ratio = F.logsigmoid(orpo_ratio)

                    orpo_ratio = -orpo_ratio*args.orpo_alpha

                    orpo_loss = torch.mean(((l2_pos_loss*(1.0-args.orpo_alpha))+orpo_ratio)) #maybe no need torch.mean
                    loss1 = loss1 + l2_pos_loss

                    orpo_loss = L2Wrap.apply(orpo_loss, RT) #im not sure is this correct? outputs_pos or RT ? 

                    loss2 = loss2 + orpo_loss
                    lossorpoonly = lossorpoonly + orpo_ratio

                
                loss2 = loss2 / bsz
                loss1 = loss1 / bsz
                lossorpoonly = lossorpoonly / bsz


                self.trainer.loss_2_orpo = float(lossorpoonly)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                return loss2
            

            elif args.orpo and args.orpo_mode == 1:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                
                #SFT_targets = []
                for s in range(bsz):
                    if os.environ["H5_MODE"] == "1":
                        chosen_input = batch_orpo[s]['chosen_input']
                        chosen_output = batch_orpo[s]['chosen_target']
                        length_chosen = batch_orpo[s]['chosen_token_len']
                        chosen_ref_prob = batch_orpo[s]['chosen_base_prob']
                        reject_input = batch_orpo[s]['reject_input']
                        reject_output = batch_orpo[s]['reject_target']
                        length_reject = batch_orpo[s]['reject_token_len']
                        reject_ref_prob = batch_orpo[s]['reject_base_prob']
                        chosen_token = batch_orpo[s]['chosentoken']
                        reject_token = batch_orpo[s]['rejecttoken']
                    else:
                        chosen_input,chosen_output,length_chosen,chosen_ref_prob, reject_input,reject_output,length_reject,reject_ref_prob,chosen_token,reject_token = batch_orpo[s]

                    # マスクの作成
                    chosen_mask = (chosen_output != 0).float()  # パディングは0と仮定
                    reject_mask = (reject_output != 0).float()

                    
                    # 両方のテンソルの長さを取得
                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)

                    #print(f'len1 = {len1}')
                    #print(f'len2 = {len2}')

                    # 最大長を計算
                    max_len = max(len1, len2)

                    if '07' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    # 長さが異なる場合、短いテンソルをパディング
                    if len1 < max_len:
                        # len1がmax_lenになるようにパディングを追加 (右側にパディング)
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        # len2がmax_lenになるようにパディングを追加 (右側にパディング)
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))
    

                    SFT_idx = []
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0) # make batch with Chosen and Reject  

                    RT ,moe_loss= self(SFT_idx)


                    #print(RT)
                    outputs_pos = RT[0].unsqueeze(0)
                    outputs_neg = RT[1].unsqueeze(0)

                    #del RT
                    del SFT_idx

                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()
                    
                  
                    def cross_entropy(pred, target):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        #loss = loss * mask.view(-1)
                        return loss#loss.sum() / mask.sum()

                    l2_pos_loss = masked_cross_entropy(outputs_pos,chosen_output,chosen_mask)

                    loss_chosen = cross_entropy(outputs_pos,chosen_output)
                    loss_reject = cross_entropy(outputs_neg,reject_output)

                    pos_prob = -torch.sum(loss_chosen[len1-length_chosen:len1])
                    neg_prob = -torch.sum(loss_reject[len2-length_reject:len2])


                    #pos_prob = self.compute_logps_simple_mask(chosen_output.unsqueeze(0),outputs_pos,chosen_mask.unsqueeze(0))
                    #neg_prob = self.compute_logps_simple_mask(reject_output.unsqueeze(0),outputs_neg,reject_mask.unsqueeze(0))


                    orpo_ratio = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
                    
                    pref_matches += (orpo_ratio > 0)

                    orpo_ratio = F.logsigmoid(orpo_ratio)

                    orpo_ratio = -orpo_ratio*args.orpo_alpha

                    orpo_loss = torch.mean(((l2_pos_loss*(1.0-args.orpo_alpha))+orpo_ratio)) #maybe no need torch.mean
                    loss1 = loss1 + l2_pos_loss

                    orpo_loss = L2Wrap.apply(orpo_loss, RT) #im not sure is this correct? outputs_pos or RT ? 

                    loss2 = loss2 + orpo_loss
                    lossorpoonly = lossorpoonly + orpo_ratio

                
                loss2 = loss2 / bsz
                loss1 = loss1 / bsz
                lossorpoonly = lossorpoonly / bsz


                self.trainer.loss_2_orpo = float(lossorpoonly)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.9 * self.trainer.pref_match_percentage + 0.1 * (pref_matches / bsz)

                return loss2