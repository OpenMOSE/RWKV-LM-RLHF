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
import functools
import os, math, gc, importlib
import torch
import time
import requests
import json
import time
import threading
from torch.utils.checkpoint import checkpoint as torch_checkpoint



def distillation_loss(student_logits, teacher_probs, temperature):
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
def kl_divergence_loss(student_logits, teacher_probs, temperature):
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
    return F.kl_div(student_probs.log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)

def training_step_sft(self, batch, batch_idx):
            
            args = self.args

            if args.distillation:
                temperature = args.temperature
                alpha = args.alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                top_k_values = batch['top_k_values']
                top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']

                #max_len = int(input_ids.shape[1])#int(attention_mask.sum(dim=1).max().item())

                max_len = int(attention_mask.sum(dim=1).max().item())
                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    top_k_values = top_k_values[:, :max_len]
                    top_k_indices = top_k_indices[:, :max_len, :]
                    attention_mask = attention_mask[:, :max_len]

                student_logits,moe_loss = self(input_ids)

                targets = target.contiguous().view(-1)

                mask = attention_mask.contiguous().view(-1)

                sum_mask = torch.sum(mask).item()

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)


                label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")



                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                teacher_probs = top_k_values#[:, :-1]
                teacher_indices = top_k_indices#[:, :-1]

                student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)

                kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_probs, temperature)

                if sum_mask == mask.shape[0]:
                    loss = alpha * smooth_loss.mean() + (1 - alpha) * kl_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                    loss = alpha * smooth_loss + (1 - alpha) * kl_loss

                self.trainer.smooth_loss = float(smooth_loss.mean())
                self.trainer.kl_loss = float(kl_loss.mean())
                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)
            

            if args.sft and args.sft_kl_mode == 0:
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                attention_mask = batch['attention_mask']


                max_len = int(attention_mask.sum(dim=1).max().item())

                def find_next_128_multiple(n):
                    remainder = n % 128
                    if remainder == 0:
                        return n
                    return n + (128 - remainder)

                if 'x070' in os.environ["RWKV_MY_TESTING"] or 'xa07' in os.environ["RWKV_MY_TESTING"]:
                    max_len = find_next_128_multiple(max_len)
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]


                if 'x060' in os.environ["RWKV_MY_TESTING"]:
                    input_ids = input_ids[:, :max_len]
                    target = target[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]

                student_logits,moe_loss = self(input_ids)

                if args.state and args.prefix_tuning:
                    student_logits = student_logits[:, args.prefix_token_len:, :]



                targets = target.contiguous().view(-1)

                mask = attention_mask.contiguous().view(-1)

                sum_mask = torch.sum(mask).item()

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)

                label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")

                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)


                # Lossの計算
                if sum_mask == mask.shape[0]:
                    loss = smooth_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    loss = smooth_loss

                if os.environ["CustomModel"] == "MoE":
                    loss = loss + args.moe_balance_alpha * moe_loss
                    self.trainer.moe_router_loss = moe_loss

                self.trainer.smooth_loss = float(smooth_loss.mean())

                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)
            


            # Hybrid Distillation for QRWKV,ARWKV,PRWKV Stage2(Token Distillation)
            #
            #
            def convert_to_array_with_mask(input_ids, attention_mask):
                """
                input_idsをattention_maskに基づいて変換する関数
                
                Parameters:
                -----------
                input_ids : torch.Tensor or list
                    変換したい入力ID
                attention_mask : torch.Tensor or list
                    マスク情報（1は保持、0は削除）
                
                Returns:
                --------
                list
                    attention_maskで1の部分だけを保持したinput_idsの配列
                """
                # Tensorの場合はnumpy配列に変換
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.cpu().numpy().tolist()
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.float().cpu().numpy().tolist()
                
                # バッチごとに処理
                result = []
                for i in range(len(input_ids)):
                    # attention_maskが1の部分だけを抽出
                    masked_ids = [input_ids[i][j] for j in range(len(input_ids[i])) if j < len(attention_mask[i]) and attention_mask[i][j] == 1]
                    result.append(masked_ids)
                
                return result
            def pad_kl_loss_to_match_attention_mask(kl_loss, attention_mask):
                """
                KL損失をアテンションマスクと同じサイズにパディングする関数
                
                Parameters:
                -----------
                kl_loss : torch.Tensor
                    KL損失テンソル [Batch, Size1]
                attention_mask : torch.Tensor
                    アテンションマスクテンソル [Batch, Size2]
                    
                Returns:
                --------
                torch.Tensor
                    パディングされたKL損失テンソル [Batch, Size2]
                """
                batch_size = kl_loss.size(0)
                kl_size = kl_loss.size(1)
                mask_size = attention_mask.size(1)
                
                # KL損失のサイズがマスクより小さい場合、パディングが必要
                if kl_size < mask_size:
                    # 必要なパディングサイズを計算
                    padding_size = mask_size - kl_size
                    
                    # 0パディングを作成（バッチサイズ × パディングサイズ）
                    padding = torch.zeros(batch_size, padding_size, device=kl_loss.device, dtype=kl_loss.dtype)
                    
                    # KL損失と0パディングを結合
                    padded_kl_loss = torch.cat([kl_loss, padding], dim=1)
                    
                    return padded_kl_loss
                
                # KL損失のサイズがマスクより大きい場合、切り取りが必要
                elif kl_size > mask_size:
                    return kl_loss[:, :mask_size]
                
                # サイズが同じ場合はそのまま返す
                else:
                    return kl_loss

            # Hybrid Distillation for QRWKV,ARWKV,PRWKV Stage2(Token Distillation)
            #
            #
            def convert_to_array_with_mask(input_ids, attention_mask):
                """
                input_idsをattention_maskに基づいて変換する関数
                
                Parameters:
                -----------
                input_ids : torch.Tensor or list
                    変換したい入力ID
                attention_mask : torch.Tensor or list
                    マスク情報（1は保持、0は削除）
                
                Returns:
                --------
                list
                    attention_maskで1の部分だけを保持したinput_idsの配列
                """
                # Tensorの場合はnumpy配列に変換
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.cpu().numpy().tolist()
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.int().cpu().numpy().tolist()
                
                # バッチごとに処理
                result = []
                for i in range(len(input_ids)):
                    # attention_maskが1の部分だけを抽出
                    masked_ids = [input_ids[i][j] for j in range(len(input_ids[i])) if j < len(attention_mask[i]) and attention_mask[i][j] == 1]
                    result.append(masked_ids)
                
                return result
            if args.sft and args.sft_kl_mode == 1:
                temperature = args.sft_kl_temperature
                alpha = args.sft_kl_alpha
                smoothing = args.smoothing

                input_ids = batch['input_ids']
                target = batch['target_ids']
                #top_k_values = batch['top_k_values']
                #top_k_indices = batch['top_k_indices']
                attention_mask = batch['attention_mask']

                with torch.no_grad():
                    arraywmask = convert_to_array_with_mask(input_ids, attention_mask)

                    #print(arraywmask)


                    payload = {
                        "input_ids": arraywmask,
                        "topk": args.sft_kl_topk
                    }
                    PROCESS_LOGITS_URL = f"{args.sft_kl_accesspoint}/ProcessLogits"
                    

                    while True:
                        try:
                            teacher_response = requests.post(PROCESS_LOGITS_URL, json=payload)
                            if teacher_response.status_code == 200:
                                teacher_result = teacher_response.json()
                                logits_numpy_array = np.array(teacher_result['logits'],dtype=np.float32)
                                indices_numpy_array = np.array(teacher_result['indices'],dtype=np.int64)
                                teacher_loss = teacher_result['loss']
                                # print(f"Loss: {teacher_result['loss']}")
                                # print(f"バッチ数: {len(teacher_result['indices'])}")
                                # print(f"シーケンス長: {[len(batch) for batch in teacher_result['indices']]}")
                                # print(f"Topkサイズ: {len(teacher_result['indices'][0][0])}")
                                break
                        except Exception as e:
                            print('retry')
                            print(f"エラーが発生しました: {e}")
                            print(f"エラーの型: {type(e).__name__}")
                            time.sleep(1)

                    top_k_values = torch.tensor(logits_numpy_array, dtype=torch.bfloat16).to(device=input_ids.device)
                    top_k_indices = torch.tensor(indices_numpy_array, dtype=torch.int64).to(device=input_ids.device)

                    

 



                #max_len = int(input_ids.shape[1])#int(attention_mask.sum(dim=1).max().item())

                max_len = int(attention_mask.sum(dim=1).max().item())
               

                student_logits,moe_loss = self(input_ids)
                targets = target.contiguous().view(-1)
                kl_mask = attention_mask.contiguous().view(-1) #[:,:-1]
                sum_kl_mask = torch.sum(kl_mask).item()
                mask = attention_mask.contiguous().view(-1)
                sum_mask = torch.sum(mask).item()

                #print(f'mask = {mask}')

                if sum_mask == 0:
                    return torch.tensor([0.0], requires_grad=True)


                label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")



                student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
                smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

                teacher_logits = top_k_values#[:, :-1]
                teacher_indices = top_k_indices#[:, :-1]

                student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)

                kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_logits, temperature)

                kl_loss = pad_kl_loss_to_match_attention_mask(kl_loss,attention_mask)

                if sum_mask == mask.shape[0]:
                    loss = alpha * smooth_loss.mean() + (1 - alpha) * kl_loss.mean()
                else:
                    smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                    kl_loss = torch.sum(kl_loss.view(-1) * kl_mask) / sum_kl_mask
                    loss = alpha * smooth_loss + (1 - alpha) * kl_loss

                self.trainer.teacher_loss = float(teacher_loss)
                self.trainer.smooth_loss = float(smooth_loss.mean())
                self.trainer.kl_loss = float(kl_loss.mean())
                self.trainer.realproceedtokens =float(max_len)

                return L2Wrap.apply(loss, student_logits)


def training_sft_infctx_init(self):
    args = self.args
    if args.infctx and args.sft and args.infctx_truncated_bptt:
        self.automatic_optimization = False


def training_step_sft_infctx(self, batch,batch_idx):
    args = self.args
    def distillation_loss(student_logits, teacher_probs, temperature):
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
            return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
    def kl_divergence_loss(student_logits, teacher_probs, temperature):
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
        return F.kl_div((student_probs + 1e-8).log(), teacher_probs, reduction='none').sum(-1) * (temperature ** 2)
    
    if args.distillation:
        #temperature = args.temperature
        #alpha = args.alpha
        smoothing = args.smoothing

        input_ids = batch['input_ids']
        target = batch['target_ids']
        top_k_values = batch['top_k_values']
        top_k_indices = batch['top_k_indices']
        attention_mask = batch['attention_mask']


        #target = input_ids[:,1:]
        #input_ids = input_ids[:,:-1]
        

        B, T = input_ids.shape
        total_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
        kl_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
        smooth_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
            self.emb.weight.dtype)

        def checkpointed_step2(chunk_input_ids,chunk_target_ids, chunk_top_k_values, chunk_top_k_indices, 
                            chunk_attention_mask, prev_loss, prev_smooth_loss, prev_kl_loss, last_shift_states,last_wkv_states, prev_token_amount):
            # Forward pass
            targets = chunk_target_ids.contiguous().view(-1)
            mask = chunk_attention_mask.contiguous().view(-1)
            sum_mask = torch.sum(mask).item()
            if sum_mask == 0:
                status = 'skip'
                return prev_loss,prev_smooth_loss,prev_kl_loss, last_shift_states, last_wkv_states, prev_token_amount, status
            
            student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)

            # Label Smoothing Loss

            #label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)
            # if 'xa070' in os.environ["RWKV_MY_TESTING"]:
            #     label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")
            # else:
            #     label_smoothing_loss = LabelSmoothingLoss(smoothing=smoothing)

            label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")



            student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1))
            smooth_loss = label_smoothing_loss(student_logits_shifted, targets)

            # Top-k teacher logits KL-divergence loss
            teacher_probs = chunk_top_k_values#[:, :-1]
            teacher_indices = chunk_top_k_indices#[:, :-1]
            student_top_k_logits = torch.gather(student_logits, -1, teacher_indices)
            kl_loss = self.kl_divergence_loss(student_top_k_logits, teacher_probs, args.temperature)

            current_token_amount = chunk_input_ids.shape[1]#sum_mask

            # Combine losses
            #print(f'summask = {sum_mask} maskshape = {mask.shape[0]}')
            if sum_mask == mask.shape[0]:
                loss = args.alpha * smooth_loss.mean() + (1 - args.alpha) * kl_loss.mean()
                smooth_loss = smooth_loss.mean()
                kl_loss = kl_loss.mean()
                loss = L2Wrap.apply(loss, student_logits, current_token_amount)
            else:
                smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
                loss = smooth_loss
                kl_loss = torch.sum(kl_loss.view(-1) * mask) / sum_mask
                loss = args.alpha * smooth_loss + (1 - args.alpha) * kl_loss
                loss = L2Wrap.apply(loss, student_logits, current_token_amount)

            
            new_token_amount = prev_token_amount + current_token_amount
            if new_token_amount > 0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
                new_kl_loss = prev_kl_loss * (prev_token_amount / new_token_amount) + kl_loss * (current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss
                new_smooth_loss = smooth_loss
                new_kl_loss = kl_loss

            status = 'proceed'
            return new_loss, new_smooth_loss, new_kl_loss, new_shift_states, new_wkv_states, new_token_amount, status
        
        proceedtokens = 0
        
        for i in range(math.ceil(T / T_train)):
            chunk_start = i * T_train
            chunk_end = (i + 1) * T_train#min((i + 1) * T_train, T)

            
            total_loss, smooth_loss,kl_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(
                checkpointed_step2,
                input_ids[:, chunk_start:chunk_end],
                target[:, chunk_start:chunk_end],
                top_k_values[:, chunk_start:chunk_end],
                top_k_indices[:, chunk_start:chunk_end],
                attention_mask[:, chunk_start:chunk_end],
                total_loss,
                smooth_loss,
                kl_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                use_reentrant=False
            )
            #states = BlockStateList(new_shift_states, new_wkv_states)
            states.shift_states = new_shift_states
            states.wkv_states = new_wkv_states

            if status == 'skip':
                break

            if status == 'proceed':
                proceedtokens = proceedtokens + (chunk_end-chunk_start)

        
        self.trainer.smooth_loss = float(smooth_loss)
        self.trainer.kl_loss = float(kl_loss)
        self.trainer.realproceedtokens =float(proceedtokens)

        return total_loss.float()#, states
    
    if args.sft and args.infctx_truncated_bptt:
        optimizer = self.optimizers()

        input_ids = batch['input_ids']         # [B, T]
        target_ids = batch['target_ids']       # [B, T]
        attention_mask = batch['attention_mask']  # [B, T]（1 = 有効, 0 = 無視）
        max_len = int(attention_mask.sum(dim=1).max().item())

        B, T = input_ids.shape
        C = args.n_embd
        H =  args.dim_att // args.head_size_a

        states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
            self.emb.weight.dtype) # make new state
        
        T = max_len

        chunk_size = int(args.chunk_ctx) #256
        total_loss = 0.0
        total_tokens = 0   # Accumulate Effective Tokens

        for i in range(0, T, chunk_size):
            input_chunk = input_ids[:, i:i+chunk_size]           # [B, T(chunk)]
            target_chunk = target_ids[:, i:i+chunk_size]         # [B, T(chunk)]
            mask_chunk = attention_mask[:, i:i+chunk_size]       # [B, T(chunk)]
            #print(f'Chunk {i} {i+chunk_size}')
            #print(f'idx = {batch_idx}')
            logits,new_shift_states, new_wkv_states = self.forward(input_chunk, states.shift_states,states.wkv_states)   # logits: [B, C, V]

            Bc, C, V = logits.shape
            logits = logits.reshape(-1, V)                          # [B*T(chunk), hidden]
            targets = target_chunk.reshape(-1)                      # [B*T(chunk)]
            mask = mask_chunk.reshape(-1).float()                   # [B*T(chunk)]

            losses = F.cross_entropy(logits, targets, reduction='none')  # [B*T(chunk)]
            masked_loss = torch.sum(losses * mask)                         # Mask loss
            num_tokens = torch.sum(mask)

            if num_tokens > 0:
                final_loss = masked_loss / num_tokens
                final_loss = L2Wrap.apply(final_loss, logits, num_tokens)
                #print(final_loss)
                self.manual_backward(final_loss) # backward each chunk
                total_loss += masked_loss.item()
                total_tokens += num_tokens.item()

            states.shift_states = new_shift_states.detach()
            states.wkv_states = new_wkv_states.detach()

        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0: # for simulate gradient accumulation
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()


        self.trainer.smooth_loss = float(total_loss / max(total_tokens, 1.0)) / (float(self.args.accumulate_grad_batches)+1e-8)
        self.trainer.realproceedtokens =float(max_len)

        return torch.tensor(total_loss / max(total_tokens, 1.0), device=self.device)
    


    elif args.sft:

        smoothing = args.smoothing

        input_ids = batch['input_ids']
        
        target = batch['target_ids']
        attention_mask = batch['attention_mask'].float()
        max_len = int(attention_mask.sum(dim=1).max().item())

        B, T = input_ids.shape
        total_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
        smooth_loss = torch.tensor(0., dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        states = BlockStateList.create(args.n_layer, B, C, H, self.emb.weight.device,
            self.emb.weight.dtype)
        

        def checkpointed_step2(chunk_input_ids,chunk_target_ids,
                            chunk_attention_mask,
                                prev_loss,
                                prev_smooth_loss,
                                    last_shift_states,last_wkv_states, prev_token_amount):
            # Forward pass
            targets = chunk_target_ids.contiguous().view(-1)
            mask = chunk_attention_mask.contiguous().view(-1)
            batchsize = chunk_attention_mask.shape[0]
            #print(f'mask = {mask}')
            sum_mask = torch.sum(mask).item()

            avg_mask_sum = torch.sum(mask) / batchsize
            L2Wrap_factor = 1e-4 / avg_mask_sum


            #print(sum_mask)
            
            if sum_mask == 0:
                status = 'skip'
                #print('skip')
                return prev_loss,prev_smooth_loss,last_shift_states, last_wkv_states, prev_token_amount, status
            
            student_logits,new_shift_states, new_wkv_states = self(chunk_input_ids,last_shift_states, last_wkv_states)
    

            label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=smoothing,reduction="none")


            student_logits_shifted = student_logits.contiguous().view(-1, student_logits.size(-1)).float()

            smooth_loss = label_smoothing_loss(student_logits_shifted.float(), targets)

            current_token_amount = chunk_input_ids.shape[1]#sum_mask

            #print(f'current token amount = {current_token_amount}')


            smooth_loss = torch.sum(smooth_loss * mask) / sum_mask
            #smooth_loss = robust_masked_mean(smooth_loss, mask)


            loss = smooth_loss
            loss = L2Wrap.apply(loss, student_logits, current_token_amount)

            #MemoryEfficientL2Wrap
            #if loss <= 0.0:
            #    loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
            #loss = L2Wrap_infctx.apply(loss, student_logits,L2Wrap_factor, mask)

            #print(f'checkpoint loss = {loss}')
            
            new_token_amount = prev_token_amount + current_token_amount
            if new_token_amount > 0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (current_token_amount / new_token_amount)
                new_smooth_loss = prev_smooth_loss * (prev_token_amount / new_token_amount) + smooth_loss * (current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss
                new_smooth_loss = smooth_loss


                

            status = 'proceed'
            #print(f'new_loss loss = {new_loss}')
            return new_loss, new_smooth_loss,new_shift_states, new_wkv_states, new_token_amount, status
        
        proceedtokens = 0

        batchmax_tokens = max_len

        # remainder = max_len % 1024
        # if remainder == 0:
        #     max_len = max_len
        # else:
        #     padding = 1024 - remainder
        #     max_len = max_len + padding
        
        # if max_len < 1024:
        #     max_len = 1024


        #print(f'T = {T}')

        T_train = args.chunk_ctx#min(args.chunk_ctx,max_len)
        #print(f'T_train = {T_train}')

        #print(f'math.ceil(T / T_train) = {math.ceil(T / T_train)}')

        chunk_start = 0
        while True:
            chunk_end = chunk_start + T_train
            if chunk_end > T:
                chunk_end = T

            #print(f'chunk start = {chunk_start} chunk end = {chunk_end} diff = {chunk_end-chunk_start}')
            total_loss, smooth_loss, new_shift_states, new_wkv_states, token_amount , status = torch_checkpoint(

                checkpointed_step2,
                input_ids[:, chunk_start:chunk_end],
                target[:, chunk_start:chunk_end],
                attention_mask[:, chunk_start:chunk_end],
                total_loss,
                smooth_loss,
                states.shift_states,#.clone().detach(),
                states.wkv_states,#.clone().detach(),
                token_amount,
                use_reentrant=False
            )
            if status == 'skip':
                break
            #states = BlockStateList(new_shift_states, new_wkv_states)
            states = BlockStateList(new_shift_states.clone().detach(), new_wkv_states.clone().detach())
            #print('nu')


            if status == 'proceed':
                proceedtokens = proceedtokens + (chunk_end-chunk_start)

            chunk_start = chunk_end
            if chunk_start >= batchmax_tokens:
                break

        self.trainer.smooth_loss = float(smooth_loss)
        self.trainer.realproceedtokens =float(proceedtokens)

    

        return total_loss#, states