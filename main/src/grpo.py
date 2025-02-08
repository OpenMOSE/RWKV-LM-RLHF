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
                            #break
                    
                    if stoptoken in output_text[j]:
                            #break
                            breaks = True
                            print('ENDTOKENDETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
                    

                    
                            
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

        
     

def grpo_init(self):
    #print('Zero CoT Initialize')
    self.CoTDebug = True

    self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

##############################################################################
# Load and prep dataset
SYSTEM_PROMPT = """Respond in the following format:
<think>
Thinking Text (up to 100 words)
</think>
<answer>
Answer Text (up to 100 words)
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
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


##############################################################################
#define Rewards
# 報酬関数の例 (ルールベース)
def my_rulebase_reward_func_example(texts: List[str]) -> List[float]:
    """
    例: テキストが数字だけだったら +1.0, そうでなければ 0.0 などの単純ルール。
    実際にはユーザ独自の評価ロジックを組み込んでください。
    """
    rewards = []
    for t in texts:
        # 例：数字だけなら 1.0, 文字含むなら 0.0
        if t.strip().isdigit():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def my_rulebase_reward_func(texts: List[str]) -> List[float]:
    """
    例: テキストが数字だけだったら +1.0, そうでなければ 0.0 などの単純ルール。
    実際にはユーザ独自の評価ロジックを組み込んでください。
    """
    rewards = []
    #pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r'(?s)\s*<think>\s*.*?\s*(?:<answer>.*?</answer>)?\s*</think>'
    
    # for t in texts:
    #     # 例：数字だけなら 1.0, 文字含むなら 0.0
    #     if t.strip().isdigit():
    #         rewards.append(1.0)
    #     else:
    #         rewards.append(0.0)
    matches = [re.match(pattern, r) for r in texts]
    print(f'matches = {matches}')
    return [0.5 if match else 0.0 for match in matches]
    #return rewards

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]



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





def training_step_grpo(self, batch, batch_idx):
    """
    GRPO (Group Relative Policy Optimization) を用いた学習ステップの例。

    - 1バッチにつき G 回(GenerateCount回)サンプリング
    - 各サンプリングに対して報酬を計算
    - 報酬を正規化して advantage を得る
    - モデル(Actor)とリファレンス(BaseModelなど)の KL 散逸を計算
    - 損失 L = - sum( ratio_no_grad * log_probs_actor * advantage ) + β * KL
      を最小化（=上式を最大化）する。

    ※あくまでサンプル実装のため、実際には用途に応じて整形を調整してください。
    """
    args = self.args

    # (簡易サンプル) バッチサイズを1と仮定
    # 複数バッチを同時に扱うなら forループを回すか、あるいは一括で扱う
    bsz = len(batch)
    assert bsz == 1, "このサンプルコードはバッチサイズ1想定です"

    GenerateCount = 3  # G: 1バッチあたり何回生成するか
    # Prompt, Chosen などはユーザのタスクごとに書き換えてください
    prompttoken = batch[0]['prompttoken'][: args.ctx_len // 2]  # 適宜切り取り
    chosentoken = batch[0]['chosentoken'][: args.ctx_len // 2]

    # テキストに戻す（実運用ならトークン列で使う）
    prompt_text = self.tokenizer.decode(prompttoken.tolist()).replace('\n\n','\n')
    final_answer_text = self.tokenizer.decode(chosentoken.tolist()).replace('\n\n','\n')

    # -----------------------------------------------------
    # 例: "User: <prompt>\nAssistant:" の後に生成させる形にする
    #     その後、最終回答などを付け足したい場合は追加
    # -----------------------------------------------------
    system_prefix = 'System: ' + SYSTEM_PROMPT 
    user_prefix = system_prefix + "\n\nUser: "
    asst_prefix = "\n\nAssistant:"
    prompt_idx = torch.tensor(
        self.tokenizer.encode(user_prefix + prompt_text + asst_prefix),
        device=self.emb.weight.device
    ).unsqueeze(0)


    # -----------------------------------------------------
    # G回 生成 (Actorモデル) - (no_gradでサンプリングのみ)
    # -----------------------------------------------------
    generated_completions = []   
    generated_completions_only = []       
    combined_idxs_list = []            # [Prompt + Generated + (Opt)Appended] のトークン列
    token_addresses_list = []

    for g_i in range(GenerateCount):
        # GenerateForwardTokens は既存コードを流用 (no_grad でサンプリング)
        gen_x, gen_tokens, tokenaddr = GenerateForwardTokens(
            self,
            prompt=prompt_idx,
            stoplength=1024,  # 生成ステップ数
            additionaltoken=None,
            stoptoken='User:',  # 適宜
            temp=1.0,
            topp=0.9
        )

        # 生成されたシーケンス (prompt_idx + gen_tokens + appended_answer_idx) をまとめてトークン列に
        combined_idx = torch.cat([prompt_idx, gen_tokens], dim=1)

        # 実際にテキスト表示して確認
        generated_text = self.tokenizer.decode(combined_idx[0].tolist())
        print(f"[DEBUG] g_i={g_i}, completion:\n", generated_text)

        generated_text_only = self.tokenizer.decode(gen_tokens[0].tolist())

        generated_completions.append(generated_text)
        generated_completions_only.append(generated_text_only)
        combined_idxs_list.append(combined_idx)
        token_addresses_list.append(tokenaddr)

    # -----------------------------------------------------
    # 報酬を計算（ルールベース or RewardModel）
    #   今回は単純に my_rulebase_reward_func の例を呼び出す
    # -----------------------------------------------------
    # 生成結果が "User: ... Assistant: <ここ>" 全部含まれるので、
    # 場合によっては "<ここ>" 以降だけ切り出して RewardModel に通す等を行う
    # ここでは全体を突っ込む簡単な例
    rewards_list = my_rulebase_reward_func(generated_completions_only)  # 長さG
    print(f'rewards_list = {rewards_list}')

    # 報酬を正規化 => advantage
    rewards_tensor = torch.tensor(rewards_list, device=self.emb.weight.device, dtype=torch.float)
    mean_r = rewards_tensor.mean()
    std_r  = rewards_tensor.std(unbiased=False) + 1e-8
    advantages_tensor = (rewards_tensor - mean_r) / std_r  # shape [G]

    # -----------------------------------------------------
    # まとめて ActorModel/ReferenceModel をForward
    #   - ActorModel は勾配あり
    #   - ReferenceModel(BaseModel) は勾配なし
    #
    #  まず、padding して連結 (dim=0がG個)
    # -----------------------------------------------------
    max_len = max(idx.shape[1] for idx in combined_idxs_list)
    for i in range(GenerateCount):
        combined_idxs_list[i] = pad_to_size_2d(combined_idxs_list[i], max_len)

    all_combined_idx = torch.cat(combined_idxs_list, dim=0)  # [G, max_len]

    CombinedIdx_len = all_combined_idx.shape[1]

    # Reference (Base) の logits => no_grad
    with torch.no_grad():
        base_logits, _ = BaseModel_Forward_NoGrad(self, pad_to_size_2d(all_combined_idx,args.ctx_len))

    # Actor の logits => 勾配あり
    actor_logits, _ = ActorModel_Forward_Grad(self, pad_to_size_2d(all_combined_idx,args.ctx_len))
    
    actor_logits=actor_logits[:,:CombinedIdx_len]
    base_logits=base_logits[:,:CombinedIdx_len]

    # softmax して対数を取っておく (Actor / Base)
    # shape [G, max_len, vocab]
    log_p_actor = F.log_softmax(actor_logits, dim=-1)
    log_p_ref   = F.log_softmax(base_logits,  dim=-1)

    # -----------------------------------------------------
    # KL散逸 D_KL(πθ||πref) の計算
    #   tokenごとに  D_KL = r - log(r) - 1,  r=exp(log_p_theta - log_p_ref)
    #   => ここでは全トークンで平均を取る (あるいは合計でも可)
    # -----------------------------------------------------
    # マスク(パディング0など)を無視して計算したい場合は下記のようにする:
    # ここでは簡単のため全トークンで計算
    ratio = torch.exp(log_p_actor - log_p_ref)  # [G, max_len, vocab]
    kl_element = ratio - torch.log(ratio + 1e-10) - 1.0  # [G, max_len, vocab]
    # 実際に選択されたトークンだけを見る場合は、以下のように gather する:
    #   chosen_token = all_combined_idx[:, 1:] (1ステップシフトなら要調整)
    #   ratio_chosen = ratio[range(G), t, chosen_token[t]] のように計算
    #  ただし近似式の "r - log(r) - 1" は πθ全体の分布を使う例が多いため、
    #  ここでは全次元を平均している例を示す。
    kl_value = kl_element.mean()  # スカラー

    # -----------------------------------------------------
    # GRPO の 損失計算
    #   L = - 1/G ∑_i ∑_t [ ratio_i(t).detach() * log_p_actor(i,t) * A_i ]
    #       + β * KL
    #
    #  ただし token単位のループを回す必要がある。
    #  (ここでは簡単のため "全トークンに同じ advantage_i" を掛ける例)
    # -----------------------------------------------------

    # Actor が実際に「生成したトークン」に対する log p を取り出し、
    # ratio.detach() を掛けたものを合計して -を取る。
    #
    # サンプリングした「実際のトークン」だけを抜き出すには 1ステップシフト等が必要ですが、
    # 例では「全トークンに対して 'ratio.detach() * log_p_actor(*, token) * advantage' を和」しており、
    # コードを簡単にするために gather せずに「正解トークン=all_combined_idx[:,1:]」を使う形にします。
    #
    # → 実際には「1ステップシフト」や「<BOS>禁止」などタスクに合わせて要実装。
    #

    # ========== ここでは簡単のため 1ステップシフトでトークンを取る ===========
    # shape [G, max_len-1]
    tokens_shifted = all_combined_idx[:, 1:].clone()
    logits_for_shifted = log_p_actor[:, :-1, :]  # [G, max_len-1, vocab]
    base_for_shifted   = log_p_ref[:,   :-1, :]  # 同様に使う場合

    # ratio_for_shifted = exp(log_p_actor - log_p_ref) も同様にシフト
    ratio_for_shifted  = ratio[:, :-1, :]

    # flatten
    B, T, V = logits_for_shifted.shape
    flat_logits_actor = logits_for_shifted.reshape(-1, V)          # [G*(max_len-1), V]
    flat_ratio        = ratio_for_shifted.reshape(-1, V)           # 同じshape
    flat_tokens       = tokens_shifted.reshape(-1)                 # [G*(max_len-1)]

    # padding除去
    valid_mask = (flat_tokens != 0)
    flat_logits_actor = flat_logits_actor[valid_mask]
    flat_ratio        = flat_ratio[valid_mask]
    flat_tokens       = flat_tokens[valid_mask]

    # log p_theta (chosen_token) と ratio(chosen_token) を取り出す
    log_probs_chosen = F.log_softmax(flat_logits_actor, dim=-1)  # 既にlog_softmax済みなら省略可
    # ratio_chosen     = exp(log_probs_chosen - log_p_ref_chosen) ... 既に flat_ratio があるなら gather
    row_idx = torch.arange(len(flat_tokens), device=flat_tokens.device)
    ratio_chosen = flat_ratio[row_idx, flat_tokens]  # [N_valid]
    lp_chosen    = log_probs_chosen[row_idx, flat_tokens]  # [N_valid]

    # 各シーケンス i の advantage を token単位に割り当てるため、
    # i を特定 (0..G-1) するインデックスを作る
    # ( max_len-1 ぶん連番で並んでいるので、 (0..G-1)を繰り返す形を作るなど)
    # shape [G*(T-1)]
    seq_indices = torch.arange(B * T, device=flat_tokens.device)
    # i = seq_indices // T (integer div), ただしvalid_maskで抜かれた後なのでずれる → うまく対応
    # 簡単のため再度マスク前の形に reshape してから gather するアプローチを示す
    seq_indices_2d = seq_indices.reshape(B, T)
    valid_mask_2d  = valid_mask.reshape(B, T)
    # 2次元で取り出す → i = 行インデックス
    # いったん flattenしたものを元に戻してから使う
    # (やや冗長だが、分かりやすさ重視で記述)
    i_list = []
    flat_idx_counter = 0
    for i_seq in range(B):
        for j_seq in range(T):
            if valid_mask_2d[i_seq, j_seq]:
                i_list.append(i_seq)
    i_tensor = torch.tensor(i_list, device=flat_tokens.device, dtype=torch.long)
    # i_tensor[k] ∈ [0..G-1]

    # advantage を対応づける
    # advantages_tensor: shape[G], i_tensor[k] が i なら advantage[i] を引っ張る
    adv_chosen = advantages_tensor[i_tensor]  # [N_valid], 各トークンに対して対応するシーケンスのadvantageを付与

    # ratio.detach() だけ取り出す
    ratio_chosen_nd = ratio_chosen.detach()  # 勾配を流さない
    # 最終的な REINFORCE 項: sum( ratio_no_grad * log_prob_theta * advantage )
    #  ↓ 符号は「maximize」したいので  "loss" に入れるときは負号
    reinforce_term = ratio_chosen_nd * lp_chosen * adv_chosen
    # ミニバッチ全体(かつ全トークン)で平均を取る(論文式だと 1/G とか 1/|o_i| がつく)
    # ここでは簡易的に全部の token / sample で平均
    reinforce_loss = - reinforce_term.mean()

    # KLペナルティ
    # すでに kl_value = kl_element.mean() でスカラー出してあるが、
    # "生成トークンだけ" などに限定するには上記 gather と同様の処理が必要。
    # ここでは「全トークンで平均」した kl_value を使い、β 係数を掛ける形。
    beta_kl = 0.1  # 適宜ハイパーパラメータ化
    kl_loss = beta_kl * kl_value  # こちらは "loss" に足し合わせる方向

    total_loss = reinforce_loss + kl_loss

    # 追加: ログなどを格納
    self.log("rewards_mean", float(mean_r), prog_bar=True)
    self.log("rewards_std",  float(std_r),  prog_bar=True)
    self.log("kl_value",     float(kl_value), prog_bar=True)
    self.log("loss_reinforce", float(reinforce_loss), prog_bar=True)

    self.trainer.rewards_mean = float(mean_r)
    self.trainer.rewards_std = float(std_r)
    self.trainer.kl_value = float(kl_value)
    self.trainer.loss_reinforce = float(reinforce_loss)

    # Lightningの都合で返す
    return total_loss

