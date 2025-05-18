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
def training_step_simpo(self, batch, batch_idx):
            
            args = self.args


            if args.simpo:  
                batch_orpo = batch

                loss1 = 0.0
                loss_simpo_only = 0.0

                # 参考統計用
                try:
                    self.trainer.pref_match_percentage
                except (NameError, AttributeError):
                    self.trainer.pref_match_percentage = 0.0
                pref_matches = 0

                bsz = len(batch_orpo)
                loss2 = 0.0

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


                    # パディング用のマスク作成
                    chosen_mask = (chosen_output != 0).float()
                    reject_mask = (reject_output != 0).float()

                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)
                    max_len = max(len1, len2)

                    # 必要に応じてパディング
                    if '07' in os.environ.get("RWKV_MY_TESTING", "") or 'xa07' in os.environ.get("RWKV_MY_TESTING", ""):
                        max_len = args.ctx_len

                    if len1 < max_len:
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))

                    # 選択応答(chosen)と却下応答(reject)をまとめてバッチ化してモデルに通す
                    SFT_idx = torch.cat([chosen_input.unsqueeze(0), reject_input.unsqueeze(0)], dim=0)
                    RT ,moe_loss= self(SFT_idx)
                    outputs_pos = RT[0].unsqueeze(0)  # chosen応答用
                    outputs_neg = RT[1].unsqueeze(0)  # reject応答用
                    del SFT_idx

                    # クロスエントロピー（SFTロス）用の関数
                    def masked_cross_entropy(pred, target, mask):
                        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='none')
                        loss = loss * mask.view(-1)
                        return loss.sum() / mask.sum()

                    # SFTロス（chosen応答だけに対して計算する例）
                    l2_pos_loss = masked_cross_entropy(outputs_pos, chosen_output, chosen_mask)

                    # 実際に各トークンの対数確率を取り出す
                    chosen_logits = outputs_pos[:, len1 - length_chosen : len1].squeeze(0)
                    reject_logits = outputs_neg[:, len2 - length_reject : len2].squeeze(0)

                    # トークンインデックスに対応する log-softmax を取得し、平均をとる
                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[torch.arange(len(chosen_token)), chosen_token]
                    chosen_prob = chosen_loss.sum().float() / float(len(chosen_token))  # 平均対数確率
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[torch.arange(len(reject_token)), reject_token]
                    reject_prob = reject_loss.sum().float() / float(len(reject_token))

                    # ====== SimPOの主部分 ======
                    # 報酬: r(chosen) = β * chosen_prob, r(reject) = β * reject_prob
                    # 差分: [r(chosen) - r(reject) - γ] をシグモイドの対数に通して損失化
                    # chosenが勝ち応答なので、この差分を正にするよう学習 → -log( sigmoid( 差分 ) )
                    diff = args.simpo_beta * (chosen_prob - reject_prob) - args.simpo_gamma

                    # 統計用: 差分が正なら「正しい勝ち」になっているとみなす
                    pref_matches += (diff > 0)

                    # ロス計算
                    pref_ratio = -F.logsigmoid(diff)

                    # 最終損失 = (SFTロス × スケール) + SimPOロス
                    final_loss = (l2_pos_loss * args.simpo_alpha) + pref_ratio*(1.0*args.simpo_alpha)

                    # L2Wrapは元コードのHook等で必要なら
                    final_loss = L2Wrap2.apply(final_loss, outputs_pos,outputs_neg)

                    loss2 += final_loss
                    loss1 += l2_pos_loss
                    loss_simpo_only += pref_ratio

                
                loss2 /= bsz
                loss1 /= bsz
                loss_simpo_only /= bsz

                
                self.trainer.loss_2_dpo = float(loss_simpo_only)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = 0.99 * self.trainer.pref_match_percentage + 0.01 * (pref_matches / bsz)

                return loss2