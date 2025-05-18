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
def training_step_wpo(self, batch, batch_idx):
            
            args = self.args


            if self.args.wpo:
                # ===== ここからWPO実装 =====
                batch_wpo = batch

                loss1 = 0.0  # いわゆるSFT部分の損失 (l2_pos_loss 累積)
                loss_pref_part = 0.0  # pref_ratio（ペア間の好み）部分
                total_loss = 0.0

                try:
                    self.trainer.pref_match_percentage
                except (NameError, AttributeError):
                    self.trainer.pref_match_percentage = 0.0

                pref_matches = 0
                bsz = len(batch_wpo)

                for s in range(bsz):
                    # データ取り出し
                    if os.environ.get("H5_MODE", "0") == "1":
                        chosen_input = batch_wpo[s]['chosen_input']
                        chosen_output = batch_wpo[s]['chosen_target']
                        length_chosen = batch_wpo[s]['chosen_token_len']
                        chosen_ref_prob = batch_wpo[s]['chosen_base_prob']
                        reject_input = batch_wpo[s]['reject_input']
                        reject_output = batch_wpo[s]['reject_target']
                        length_reject = batch_wpo[s]['reject_token_len']
                        reject_ref_prob = batch_wpo[s]['reject_base_prob']
                        chosen_token = batch_wpo[s]['chosentoken']
                        reject_token = batch_wpo[s]['rejecttoken']
                    else:
                        chosen_input, chosen_output, length_chosen, chosen_ref_prob, \
                            reject_input, reject_output, length_reject, reject_ref_prob = batch_wpo[s]

                    # マスク作成（0をパディングとみなす）
                    chosen_mask = (chosen_output != 0).float()
                    reject_mask = (reject_output != 0).float()

                    len1 = chosen_input.size(0)
                    len2 = reject_input.size(0)
                    max_len = max(len1, len2)

                    # 必要に応じてパディング
                    if '07' in os.environ.get("RWKV_MY_TESTING", "")or 'xa07' in os.environ.get("RWKV_MY_TESTING", ""):
                        max_len = args.ctx_len

                    # パディング処理（右側0埋め）
                    if len1 < max_len:
                        chosen_input = F.pad(chosen_input, (0, max_len - len1))
                        chosen_output = F.pad(chosen_output, (0, max_len - len1))
                        chosen_mask = F.pad(chosen_mask, (0, max_len - len1))
                    if len2 < max_len:
                        reject_input = F.pad(reject_input, (0, max_len - len2))
                        reject_output = F.pad(reject_output, (0, max_len - len2))
                        reject_mask = F.pad(reject_mask, (0, max_len - len2))

                    # 一度にまとめて推論: chosenとrejectをバッチで入れる
                    SFT_idx = torch.stack([chosen_input, reject_input], dim=0)  # shape [2, max_len]
                    # ログits計算 [2, max_len, vocab_size]
                    RT ,moe_loss= self(SFT_idx)

                    outputs_pos = RT[0].unsqueeze(0)  # chosen
                    outputs_neg = RT[1].unsqueeze(0)  # reject
                    del SFT_idx


                    def masked_cross_entropy(pred, target, mask):
                        ce_loss = F.cross_entropy(
                            pred.view(-1, pred.size(-1)),
                            target.view(-1),
                            reduction='none'
                        )
                        ce_loss = ce_loss * mask.view(-1)
                        return ce_loss.sum() / (mask.sum() + 1e-9)

                    l2_pos_loss = masked_cross_entropy(outputs_pos, chosen_output, chosen_mask)

                    # (2) 好みの差分 (chosen vs reject) の対数確率を計算
                    # logitsから実際に生成されたtokenだけ抜き出し
                    chosen_logits = outputs_pos[:, len1 - length_chosen:len1].squeeze(0)
                    reject_logits = outputs_neg[:, len2 - length_reject:len2].squeeze(0)

                    chosen_loss = (F.log_softmax(chosen_logits, dim=-1))[
                        torch.arange(len(chosen_token)),
                        chosen_token
                    ]
                    reject_loss = (F.log_softmax(reject_logits, dim=-1))[
                        torch.arange(len(reject_token)),
                        reject_token
                    ]

                    # ここでは平均対数尤度の形
                    chosen_prob = torch.sum(chosen_loss)/float(len(chosen_token))  # 平均log p(y_w)
                    reject_prob = torch.sum(reject_loss)/float(len(reject_token))   # 平均log p(y_l)

                    # (3) DPOタイプのpref_ratio（ただしWPOでも流用）
                    pref_ratio = self.args.wpo_beta * (
                        chosen_prob - reject_prob
                        #- chosen_ref_prob + reject_ref_prob
                    )
                    # 好みを正にするときのシグモイド
                    pref_matches += (pref_ratio > 0)
                    pref_ratio = -F.logsigmoid(pref_ratio)

                    # ============== ここが WPO のカギ ========================
                    # WPO重み計算：
                    # chosen_prob, reject_prob はそれぞれ「平均対数確率」なので
                    # シーケンス全体の対数確率にするため length_chosen, length_reject を掛ける
                    chosen_seq_logprob = chosen_prob# * float(length_chosen)
                    reject_seq_logprob = reject_prob# * float(length_reject)

                    print(f'chosen_seq_logprob = {chosen_seq_logprob}, reject_seq_logprob = {reject_seq_logprob}')

                    # w_chosen, w_reject はそれぞれ e^(シーケンス全体のlog p)
                    w_chosen = torch.exp(chosen_seq_logprob)
                    w_reject = torch.exp(reject_seq_logprob)
                    # 二つを掛け合わせたものを最終的なペアの重みとする
                    w_pair = w_chosen * w_reject

                    # (4) 最終loss: 通常の (SFT部分 + pref部分) にWPOの重み w_pair を掛ける
                    print(f'w_pair = {w_pair}')
                    final_loss = w_pair * (l2_pos_loss * self.args.wpo_alpha + pref_ratio)
                    # ======================================================

                    # もし勾配に対して特殊な処理(L2Wrap等)が必要ならここで
                    final_loss = L2Wrap.apply(final_loss, outputs_pos)

                    loss1 += l2_pos_loss
                    loss_pref_part += pref_ratio
                    total_loss += final_loss

                # バッチサイズで割って平均
                total_loss = total_loss / bsz
                loss1 = loss1 / bsz
                loss_pref_part = loss_pref_part / bsz

                self.trainer.loss_2_dpo = float(loss_pref_part)
                self.trainer.loss_1_general_or_sft = float(loss1)
                self.trainer.pref_match_percentage = (
                    0.9 * self.trainer.pref_match_percentage
                    + 0.1 * (pref_matches / bsz)
                )

                return total_loss