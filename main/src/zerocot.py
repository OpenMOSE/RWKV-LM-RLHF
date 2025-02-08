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

# パディング関数の定義
def pad_to_size_2d(tensor, target_size=2048):
    current_size = tensor.size(1)
    pad_size = target_size - current_size
    
    # (左, 右, 上, 下)のパディングを指定
    padding = (0, pad_size, 0, 0)
    
    return F.pad(tensor, padding, mode='constant', value=0)

def pad_to_size_3d(tensor, target_size=2048):
    current_size = tensor.size(1)
    pad_size = target_size - current_size
    
    # (前, 後, 上, 下, 左, 右)のパディングを指定
    # 3次元テンソルの場合、パディングは後ろから指定する
    padding = (0, 0,      # 最後の次元（65536）
              0, pad_size,# 真ん中の次元（648→2048など）
              0, 0)       # 最初の次元（1）
    return F.pad(tensor, padding, mode='constant', value=0)


def GenerateForwardTokens(self,prompt,stoplength=50,additionaltoken=None,stoptoken='\n\n',temp=1.0,topp = 0.9): # from RWKV-Infer Methods. TSUKUTTETE YOKATTA-
    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #         ],
    #         record_shapes=True
    #     ) as prof:
    with torch.no_grad():
        args = self.args

        H =  args.dim_att // args.head_size_a

        B = 1 #Single Batch
        new_states = BlockStateList.create(args.n_layer, B, args.n_embd, H,
                                            self.emb.weight.device, self.emb.weight.dtype)
        
        temperature = torch.full((B,), temp)
        top_p = torch.full((B,), topp)
        GEN_alpha_presence = 0.3
        GEN_alpha_frequency = 0.3
        GEN_penalty_decay = 0.996

        occurrence = {}
        out_tokens = [[] for _ in range(B)]
        out_last = [0 for _ in range(B)]
        output_text = ['' for _ in range(B)]

        idx = prompt

        shift_states = new_states.shift_states
        wkv_states = new_states.wkv_states
        #with torch.profiler.record_function("Initial forward_rnn"):
        x, shift_states, wkv_states = self.forward_rnn(idx, shift_states, wkv_states,passthrough=False) #FLA

        tokenaddress = {}

        tokenaddress['prompt'] = x.shape[1]#(B,T,-1)

        out_x = x.clone()

        print('///////////////////////////////////////////////////////////////////////////////////////////////\n')
        
        for i in range(stoplength):

            for n in occurrence:
                x[:,-1, n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
            
            x[:, -1, 0] -= 1e10

           # x=x.to(self.emb.weight.device)

            otokens_tensor = sampling_multibatch(self,x[:, -1], temperature=temperature, top_p=top_p)
            otokens = otokens_tensor.tolist()

            tokens = []
            for j in range(B):
                tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

            idx = torch.cat(tokens, dim=0)

            breaks = False

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
                            #print(f'\nEndtoken\n')
                            tmp = tmp.replace(stoptoken,'')
                            breaks = True

                            if len(output_text[j]) == 0:
                                 #print('output text not detected try again')
                                 breaks = False
                            break
                    
                            
                except:
                    pass

            if breaks:
                 break


            x, shift_states, wkv_states = self.forward_rnn(idx, shift_states, wkv_states,passthrough=False)
            out_x = torch.cat([out_x, x], dim=1)

            #x=x.cpu()

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
                occurrence[x[0,-1]] = 1 + (occurrence[x[0,-1]] if x[0,-1] in occurrence else 0)

        tokenaddress['generated'] = out_x.shape[1] - tokenaddress['prompt']

        if additionaltoken != None:
            x, shift_states, wkv_states = self.forward_rnn(additionaltoken, shift_states, wkv_states,passthrough=False) #FLA
            out_x = torch.cat([out_x, x], dim=1)
            tokenaddress['additionalprompt'] = x.shape[1]

            # プロファイラ結果を表示（関数の最後で一度出力）

    #exit()

    print('\n///////////////////////////////////////////////////////////////////////////////////////////////\n')
        
            
    return out_x, torch.tensor(out_tokens).to(device=self.emb.weight.device), tokenaddress

        
     

def zerocot_init(self):
    #print('Zero CoT Initialize')
    self.CoTDebug = True

    self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")



def compute_nll_and_masked_tokens(
    logits: torch.Tensor, 
    tokens: torch.Tensor,
    pad_token_id: int = 0,
    start_idx: int = 0,
    end_idx: int = None
) -> Tuple[torch.Tensor, int]:
    """
    logits: [B, T, V]
    tokens: [B, T]
    start_idx: この位置以降を NLL 計算対象とする
    end_idx:  この位置以前を NLL 計算対象とする (None の場合は T まで)
    pad_token_id: パディングに使うトークン ID
    
    返り値:
    - total_nll: バッチごとの合計負の対数尤度 (cross-entropy の合計)
    - count     : 有効トークン数 (paddingを除いた)
    """
    B, T, V = logits.shape
    if end_idx is None:
        end_idx = T
    
    # slice
    slice_logits = logits[:, start_idx:end_idx, :]    # [B, T_sliced, V]
    slice_tokens = tokens[:, start_idx:end_idx]       # [B, T_sliced]

    # flat へ変形
    # CrossEntropy で扱いやすいよう、(B*T_sliced, V) と (B*T_sliced) へ
    flat_logits = slice_logits.reshape(-1, V)         # [B*T_sliced, V]
    flat_tokens = slice_tokens.reshape(-1)            # [B*T_sliced]

    # パディング位置マスクを作成（pad_token_id や 0 を除外するなど）
    # ここでは単純に 'pad_token_id == 0 も無視' とした例
    valid_mask = (flat_tokens != pad_token_id) & (flat_tokens != 0) 
    
    # valid のみ抽出
    valid_logits = flat_logits[valid_mask]  # shape [N_valid, V]
    valid_tokens = flat_tokens[valid_mask]  # shape [N_valid]
    valid_count  = valid_logits.size(0) 

    if valid_count == 0:
        # 有効トークンがない場合は NLL=0 で返す
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0
    
    # 負の対数尤度 (nll) を計算
    # F.cross_entropy は mean なので、合計が欲しければ reduction='sum' を使う
    nll = F.cross_entropy(valid_logits, valid_tokens, reduction='mean')
    
    return nll, valid_count

def compute_nll_and_masked_tokens_shifted(
    logits: torch.Tensor, 
    tokens: torch.Tensor,
    pad_token_id: int = 0,
    start_idx: int = 0,
    end_idx: int = None
) -> Tuple[torch.Tensor, int]:
    """
    logits: [B, T, V]
    tokens: [B, T]
    start_idx: この位置以降を NLL 計算対象とする
    end_idx:  この位置以前を NLL 計算対象とする (None の場合は T まで)
    pad_token_id: パディングに使うトークン ID
    
    ※ 標準的なLM学習の挙動（1ステップシフト）を再現するために、
       ロジットは [start_idx, end_idx-1) を用い、
       対象トークンは [start_idx+1, end_idx) を用います。
    
    返り値:
    - total_nll: バッチごとの合計負の対数尤度 (cross-entropy の合計)
    - count     : 有効トークン数 (paddingを除いた)
    """
    B, T, V = logits.shape
    if end_idx is None:
        end_idx = T

    # シフト処理のため、区間長が少なくとも2でなければならない
    if end_idx - start_idx < 2:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0

    # シフトを反映したスライス
    # logits: [start_idx, end_idx-1) → 予測対象として使う
    # tokens: [start_idx+1, end_idx) → 正解（ターゲット）として使う
    slice_logits = logits[:, start_idx:end_idx - 1, :]    # [B, T_sliced-1, V]
    slice_tokens = tokens[:, start_idx + 1:end_idx]         # [B, T_sliced-1]

    # flat へ変形 (CrossEntropy の入力形式に合わせる)
    flat_logits = slice_logits.reshape(-1, V)         # [B*(T_sliced-1), V]
    flat_tokens = slice_tokens.reshape(-1)            # [B*(T_sliced-1)]

    # パディング位置マスクを作成
    valid_mask = (flat_tokens != pad_token_id) & (flat_tokens != 0)
    
    # 有効な位置のみ抽出
    valid_logits = flat_logits[valid_mask]  # shape [N_valid, V]
    valid_tokens = flat_tokens[valid_mask]  # shape [N_valid]
    valid_count  = valid_logits.size(0) 

    if valid_count == 0:
        # 有効トークンがない場合は NLL=0 で返す
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0
    
    # cross_entropy を使って負の対数尤度を計算（合計値）
    total_nll = F.cross_entropy(valid_logits, valid_tokens, reduction='sum')
    return total_nll, valid_count




def training_step_zerocot(self, batch, batch_idx):
    """
    RL (REINFORCE) を使って
      Step1: Prompt -> Thinking(10個) 
      Step2: (Prompt + Thinking) -> Output
      => Output部分のPPLを下げるように学習
      => Thinking部の対数尤度に報酬を掛けて更新
    ベースモデル(BaseModel)のNLLをbaselineとして Advantage を計算。
    """
    #print('Training Step ZeroCoT')

    args = self.args
    batch_orpo = batch
    bsz = len(batch_orpo)


    # 実験としてバッチサイズは1想定で進めるなら
    # デモ用に bsz=1 として話を進める
    bsz = 1

    # いくつ Thinking サンプルを生成するか
    # (実験で同じプロンプトに対して複数回 Thinking をサンプリングし、平均的に学習する等)
    GenerateCount = 3
    GenerateTokens = 1024  # Thinking の長さ(目安)
    # ここではさらに続けて final Output を書き足して Loss を見る例

    for s in range(bsz):
        prompttoken = batch_orpo[s]['prompttoken'][:args.ctx_len // 2 - GenerateTokens]
        chosentoken = batch_orpo[s]['chosentoken'][:args.ctx_len // 2]  # 最終回答など

        # テキストに戻してデモ。実際は token のまま使う方が望ましい
        prompt_text  = self.tokenizer.decode(prompttoken.tolist()).replace('\n\n','\n') 
        chosen_text  = self.tokenizer.decode(chosentoken.tolist()).replace('\n\n','\n')
        
        # プロンプト作成: 
        SplitText      = '\n\n'
        UserText       = 'User: '
        AssistantText  = "Assistant: "
        ThinkingText   = "<think> yeah. let me think about it. the question is"# Maybe "+ chosen_text.replace(SplitText,'') + ' let me think about it. the question is'# okay. Let's think about it. this question is"

        GeneratePrompt = UserText + prompt_text + SplitText + ThinkingText
        # ここで Thinking を生成
        prompt_idx = torch.tensor(self.tokenizer.encode(GeneratePrompt), 
                                  device=self.emb.weight.device).unsqueeze(0)

        # 最終回答を後ろに付ける
        final_answer_text = "</think>" + SplitText + AssistantText + chosen_text
        final_answer_idx  = torch.tensor(self.tokenizer.encode(final_answer_text), 
                                         device=self.emb.weight.device).unsqueeze(0)

        #print(f"-----\n[Debug] Prompt:\n{GeneratePrompt}")
        #print(f"[Debug] FinalAnswer:\n{final_answer_text}")

        # 生成結果を格納するバッファ
        generated_logits_list     = []
        generated_tokens_list     = []
        generated_tokenaddress_list = []
        base_model_logits_list    = []
        combined_idxs_list        = []
        
        # ============== Thinking を複数サンプリング ==============
        for i in range(GenerateCount):
            # まず ActorModel(あるいは self ) で Thinking + Output 生成
            # ここで GenerateForwardTokens は no_grad で書かれているので
            #  サンプルの取得のみ行う
            gen_x, gen_tokens, tokenaddr = GenerateForwardTokens(
                self,
                prompt=prompt_idx,
                stoplength=GenerateTokens,
                additionaltoken=final_answer_idx,
                stoptoken=SplitText
            )
            # gen_x: [B, (prompt + gen + additionaltoken), vocabLogits的次元] の時系列スタック
            # gen_tokens: 実際にサンプリングされた Thinking トークン
            # tokenaddr: { 'prompt':..., 'generated':..., 'additionalprompt':... }

            # (Thinking + Output)全部つなげた token 列を ActorModel / BaseModel で比較したい
            # => [prompt_idx, gen_tokens, final_answer_idx] を連結
            #    ただし gen_tokens は sampling のみ行われたので shape [B, #tokens]
            combined_idx = torch.cat([prompt_idx, gen_tokens, final_answer_idx], dim=1) # shape [1, T合計]

            print(f"[Debug] Combined text:\n{self.tokenizer.decode(combined_idx[0].tolist())}")

            # ベースライン (BaseModel) の logits
            #combined_idx_len = combined_idx.shape[1]
            #base_logits, _ = BaseModel_Forward_NoGrad(self, pad_to_size_2d(combined_idx,args.ctx_len))
            #base_logits = base_logits[:,:combined_idx_len]

            # リストに格納 (後で一括で pad & RL 計算)
            combined_idxs_list.append(combined_idx)
            generated_logits_list.append(gen_x)
            generated_tokens_list.append(gen_tokens)
            generated_tokenaddress_list.append(tokenaddr)
            #base_model_logits_list.append(base_logits)

        # それぞれ長さを取り出し、padding してまとめる
        max_len = max(x.shape[1] for x in combined_idxs_list)
        for k in range(len(combined_idxs_list)):
            combined_idxs_list[k]      = pad_to_size_2d(combined_idxs_list[k], max_len)
            #base_model_logits_list[k]  = pad_to_size_3d(base_model_logits_list[k], max_len)
        CombinedIdx     = torch.cat(combined_idxs_list, dim=0)       # [GenerateCount, max_len]
        #BaseModelLogits = torch.cat(base_model_logits_list, dim=0)   # [GenerateCount, max_len, vocab]


        #print(f'BaseModelLogits shape ={BaseModelLogits.shape}')
        print(f'CombinedIdx shape = {CombinedIdx.shape}')

        # ====== ActorModel の Forward(勾配あり) ======
        #   ここで1回だけ forward して logits を取得
        #   (Thinking部の対数確率を計算し、REINFORCEしたい)
        CombinedIdx_len = CombinedIdx.shape[1]
        base_logits, _ = BaseModel_Forward_NoGrad(self, pad_to_size_2d(CombinedIdx,args.ctx_len))
        actor_logits, _ = ActorModel_Forward_Grad(self, pad_to_size_2d(CombinedIdx,args.ctx_len))
        actor_logits=actor_logits[:,:CombinedIdx_len]
        base_logits=base_logits[:,:CombinedIdx_len]

        print(f'actor_logits = {actor_logits.shape}')

        print(f'diff = {torch.sum(base_logits - actor_logits)}')
        # actor_logits shape = [GenerateCount, max_len, vocab]

        # ====== いよいよ REINFORCE の計算 ======
        #   1) Thinking 部分の log_prob を取り出す
        #   2) 出力( final answer ) 部分の NLL を計算(ActorModel, BaseModel)
        #   3) Advantage = BaseNLL - ActorNLL
        #   4) loss = - Advantage * sum_{Thinking} log( p(Thinking_i) )

        # GenerateCount 回分をまとめて計算する
        # まず、有効なトークン長をそれぞれ記録しておく
        thinking_logprob_all = []
        advantage_all        = []
        entropy_all          = []  # [CHANGE: エントロピー計算用]
        
        for g_i in range(GenerateCount):
            # CombinedIdx[g_i]: [1, max_len]
            tokens_g = CombinedIdx[g_i:g_i+1, :]  # shape [1, max_len]
            actor_g  = actor_logits[g_i:g_i+1, :, :]      # [1, max_len, vocab]
            base_g   = base_logits[g_i:g_i+1, :, :]   # [1, max_len, vocab]

            # tokenaddress から区間を把握
            pad_len_prompt = generated_tokenaddress_list[g_i]['prompt']
            pad_len_thinking = generated_tokenaddress_list[g_i]['generated']
            # additionalprompt が無い場合も考慮し、getで取得
            pad_len_additional = generated_tokenaddress_list[g_i].get('additionalprompt', 0)

            # Thinking 部分は [prompt_end : prompt_end + thinking_len]
            # final output 部分は [prompt_end + thinking_len : prompt_end + thinking_len + additional_len]
            start_thinking = pad_len_prompt
            end_thinking   = pad_len_prompt + pad_len_thinking
            start_output   = end_thinking
            end_output     = end_thinking + pad_len_additional

            # === Thinking 部分の logprob を計算 ===
            #  サンプルされたトークンが tokens_g[:, start_thinking : end_thinking] にある
            #  その log(prob) を合計
            #  1ステップ後の logits で計算するケースや、標準的には tokens[ t+1 ] の確率を logits[t] から取る等
            #  実装に応じてオフセットをずらす必要あり。

            
            #thinking_slice_logits = actor_g[:, start_thinking : end_thinking, :]  # shape [1, #thinking, vocab]
            #thinking_slice_tokens = tokens_g[:, start_thinking : end_thinking]     # [1, #thinking]
            # --- 変更後 ---
            # 1ステップシフトを実施： logits[t] で次のトークン tokens[t+1] を予測
            if end_thinking - start_thinking < 2:
                # シーケンス長が短すぎる場合はシフトできないので、そのまま計算（またはスキップ）
                thinking_slice_logits = actor_g[:, start_thinking : end_thinking, :]
                thinking_slice_tokens = tokens_g[:, start_thinking : end_thinking]
            else:
                thinking_slice_logits = actor_g[:, start_thinking : end_thinking - 1, :]
                thinking_slice_tokens = tokens_g[:, start_thinking + 1 : end_thinking]

            # (B, T_think, V) => flat
            flat_th_logits = thinking_slice_logits.reshape(-1, thinking_slice_logits.size(-1)) # [#thinking, V]
            flat_th_tokens = thinking_slice_tokens.reshape(-1)                                  # [#thinking]

            # padding 無視
            valid_mask_th = (flat_th_tokens != 0)
            flat_th_logits = flat_th_logits[valid_mask_th]
            flat_th_tokens = flat_th_tokens[valid_mask_th]

            if flat_th_tokens.numel() == 0:
                # Thinking トークンが空っぽなら logprob=0
                sum_logprob_thinking = torch.zeros([], device=actor_g.device)
            else:
                logprob_th = F.log_softmax(flat_th_logits, dim=-1) # [N, V]
                # [N]
                chosen_lp = logprob_th[range(logprob_th.size(0)), flat_th_tokens]
                sum_logprob_thinking = chosen_lp.sum()

            # [CHANGE: エントロピー計算 (Thinking 部分の平均エントロピーを適当に入れる)]
            if flat_th_tokens.numel() > 0:
                p_th = logprob_th.exp()  # [N, V]
                ent_th = -(p_th * logprob_th).sum(dim=-1).mean()  # 平均エントロピー
            else:
                ent_th = torch.zeros([], device=actor_g.device)

            # === Output 部分の NLL(Actor) ===
            actor_nll, actor_count = compute_nll_and_masked_tokens_shifted(
                actor_g, tokens_g, pad_token_id=0, start_idx=start_output, end_idx=end_output
            )
            # === Output 部分の NLL(Base) ===
            base_nll, base_count = compute_nll_and_masked_tokens_shifted(
                base_g, tokens_g, pad_token_id=0, start_idx=start_output, end_idx=end_output
            )
            # actor_nll, base_nll はスカラー（合計）
            # advantage = baseline - actor => smaller actor_nll => advantage>0
            advantage = (base_nll - actor_nll).detach() # detach して定数に

            # thinking_logprob を張り付け
            thinking_logprob_all.append(sum_logprob_thinking.unsqueeze(0)) 
            advantage_all.append(advantage.unsqueeze(0))
            entropy_all.append(ent_th.unsqueeze(0))  # [CHANGE]

        # shape: [GenerateCount, 1]
        thinking_logprob_all = torch.cat(thinking_logprob_all, dim=0) 
        advantage_all        = torch.cat(advantage_all, dim=0)
        entropy_all          = torch.cat(entropy_all, dim=0)           # [CHANGE]

        mean_adv = advantage_all.mean()
        std_adv  = advantage_all.std() + 1e-9
        adv_normed = (advantage_all - mean_adv) / std_adv

        #print(f'mean_adv = {mean_adv}')
        #print(f'mean_normed = {adv_normed}')
        #print('')

        clip_val = 5.0
        adv_clipped = torch.clamp(adv_normed, -clip_val, clip_val)

        # [CHANGE: エントロピー正則化の重み]
        beta_entropy = 0.1
        # Thinking 部分の平均エントロピー
        mean_entropy = entropy_all.mean()

        # REINFORCE 損失
        # thinking_logprob_all: (GenerateCount,)
        # advantage_all       : (GenerateCount,)
        # => まとめて loss = - advantage * log_prob
        #loss_rl = -(1.0*thinking_logprob_all * advantage_all).mean()
        loss_rl = -((0.2 * thinking_logprob_all) * adv_clipped).mean()
        
        # [CHANGE: エントロピーボーナスをマイナス方向に加える (maximize entropy => minimize -entropy)]
        loss_entropy = - beta_entropy * mean_entropy

        actor_ppl = torch.exp(actor_nll / actor_count)
        base_ppl  = torch.exp(base_nll  / base_count)

        self.trainer.advantage = float(mean_adv)
        self.trainer.advantage_clipped = float(adv_clipped.mean())

        self.trainer.actor_ppl = float(actor_ppl)
        self.trainer.base_ppl = float(base_ppl)

        self.trainer.actor_nll = float(actor_nll)
        self.trainer.base_nll = float(base_nll)

        self.trainer.entropy = float(mean_entropy)

        self.trainer.loss_rl = float(loss_rl)
        self.trainer.loss_entropy = float(loss_entropy)

        # 合計 REINFORCE ロス（エントロピー正則化込み）
        total_loss_rl = loss_rl + loss_entropy


        # 必要に応じて SFT のクロスエントロピーなど別ロスを加えるならここで加算
        # total_loss = alpha * loss_rl + beta * loss_sft といった形に
        total_loss = total_loss_rl
        


        total_loss = L2Wrap.apply(total_loss, actor_logits)

        # Lightning では通常 "return total_loss" => backward => optimizer
        # ただしバッチループを自前でするなら都度 accumulate
        # (サンプルコードでは s=1 なので1回でOK)
        return total_loss

