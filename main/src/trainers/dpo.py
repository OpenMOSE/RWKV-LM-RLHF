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
def training_step_dpo(self, batch, batch_idx):
            
            args = self.args


            if args.dpo:
                batch_orpo = batch

                loss1 = 0.0
                lossorpoonly = 0.0
                
                try: self.trainer.pref_match_percentage
                except (NameError, AttributeError): self.trainer.pref_match_percentage = 0.5
                pref_matches = 0
                bsz = len(batch_orpo)
                loss2 = 0.0

                #print(batch_orpo)


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


                    # 最大長を計算
                    max_len = max(len1, len2)

                    #if max_len < 512:# GOMI CODE
                    #    max_len = 512 
                    chosen_output2 = chosen_output
                    reject_output2 = reject_output

                    if '07' in os.environ["RWKV_MY_TESTING"]:
                        max_len = args.ctx_len

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

                    #loss_chosen = cross_entropy(outputs_pos,chosen_output)
                    #loss_reject = cross_entropy(outputs_neg,reject_output)

                    chosen_logits = outputs_pos[:,len1-length_chosen:len1].squeeze(0)
                    reject_logits = outputs_neg[:,len2-length_reject:len2].squeeze(0)

                    print(f'chosen logits shape = {chosen_logits.shape}')



                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_token)), chosen_token]
                    chosen_prob = (torch.sum(chosen_loss.view(-1))).float() / float(len(chosen_token))
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[torch.arange(len(reject_token)), reject_token]
                    reject_prob = (torch.sum(reject_loss.view(-1))).float() / float(len(reject_token))

                    #chosen_prob = -torch.sum(loss_chosen[len1-length_chosen:len1])/float(length_chosen)
                    #reject_prob = -torch.sum(loss_reject[len2-length_reject:len2])/float(length_reject)

                    print(f'chosen_prob ={chosen_prob} reject_prob={reject_prob}')


                    #reject_prob = -torch.sum(loss_reject[-length_reject:])
                    pref_ratio = args.dpo_beta * (chosen_prob - reject_prob - chosen_ref_prob + reject_ref_prob)
                    pref_matches += (pref_ratio > 0)
                    pref_ratio = - F.logsigmoid(pref_ratio)
                    #pref_ratio = F.softplus(-pref_ratio)
                    

                    final_loss = (l2_pos_loss*args.dpo_alpha) + pref_ratio
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