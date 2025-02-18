

def training_step_zerocot____(self, batch, batch_idx):
    """
    RL (REINFORCE) を使って
      Step1: Prompt -> Thinking(10個) 
      Step2: (Prompt + Thinking) -> Output
      => Output部分のPPLを下げるように学習
      => Thinking部の対数尤度に報酬を掛けて更新
    ベースモデル(BaseModel)のNLLをbaselineとして Advantage を計算。
    """
    print('Training Step ZeroCoT')

    args = self.args
    batch_orpo = batch
    bsz = len(batch_orpo)

    loss_sft_all = 0.0   # (オプション)教師強制のLossがあれば加える
    loss_rl_all  = 0.0   # REINFORCE用

    # 実験としてバッチサイズは1想定で進めるなら
    # デモ用に bsz=1 として話を進める
    bsz = 1

    # いくつ Thinking サンプルを生成するか
    # (実験で同じプロンプトに対して複数回 Thinking をサンプリングし、平均的に学習する等)
    GenerateCount = 4
    GenerateTokens = 30  # Thinking の長さ(目安)
    # ここではさらに続けて final Output を書き足して Loss を見る例

    for s in range(bsz):
        prompttoken = batch_orpo[s]['prompttoken'] 
        chosentoken = batch_orpo[s]['chosentoken'] # 最終回答など

        # テキストに戻してデモ。実際は token のまま使う方が望ましい
        prompt_text  = self.tokenizer.decode(prompttoken.tolist())[: args.ctx_len // 2]
        chosen_text  = self.tokenizer.decode(chosentoken.tolist())[: args.ctx_len // 2]
        
        # プロンプト作成: 
        SplitText      = '\n\n'
        UserText       = 'User: '
        AssistantText  = "Assistant: "
        ThinkingText   = "Thinking: "

        GeneratePrompt = UserText + prompt_text + SplitText + ThinkingText
        # ここで Thinking を生成
        prompt_idx = torch.tensor(self.tokenizer.encode(GeneratePrompt), 
                                  device=self.emb.weight.device).unsqueeze(0)

        # 最終回答を後ろに付ける
        final_answer_text = SplitText + AssistantText + chosen_text
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

            #print(f"[Debug] Combined text:\n{self.tokenizer.decode(combined_idx[0].tolist())}")

            # ベースライン (BaseModel) の logits
            base_logits, _ = BaseModel_Forward_NoGrad(self, combined_idx)

            # リストに格納 (後で一括で pad & RL 計算)
            combined_idxs_list.append(combined_idx)
            generated_logits_list.append(gen_x)
            generated_tokens_list.append(gen_tokens)
            generated_tokenaddress_list.append(tokenaddr)
            base_model_logits_list.append(base_logits)

        # それぞれ長さを取り出し、padding してまとめる
        max_len = max(x.shape[1] for x in combined_idxs_list)
        for k in range(len(combined_idxs_list)):
            combined_idxs_list[k]      = pad_to_size_2d(combined_idxs_list[k], max_len)
            base_model_logits_list[k]  = pad_to_size_3d(base_model_logits_list[k], max_len)
        CombinedIdx     = torch.cat(combined_idxs_list, dim=0)       # [GenerateCount, max_len]
        BaseModelLogits = torch.cat(base_model_logits_list, dim=0)   # [GenerateCount, max_len, vocab]

        # ====== ActorModel の Forward(勾配あり) ======
        #   ここで1回だけ forward して logits を取得
        #   (Thinking部の対数確率を計算し、REINFORCEしたい)
        actor_logits, _ = ActorModel_Forward_Grad(self, CombinedIdx)
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
        
        for g_i in range(GenerateCount):
            # CombinedIdx[g_i]: [1, max_len]
            tokens_g = CombinedIdx[g_i:g_i+1, :]  # shape [1, max_len]
            actor_g  = actor_logits[g_i:g_i+1, :, :]      # [1, max_len, vocab]
            base_g   = BaseModelLogits[g_i:g_i+1, :, :]   # [1, max_len, vocab]

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
            #  ここでは簡易に "actor_g[:, t, tokens[t]] を使う" とする
            #  ただし cross_entropy のときは1 step シフトが必要。厳密にはご自身のモデル実装に合わせてください。
            
            # actor_g, tokens_g をシフトさせて cross_entropy 形式で計算
            # (Thinking を teacher forcing するわけではなく、REINFORCE用に対数確率を抽出)
            # 簡易に next_tokens = tokens_g[:, start_thinking+1 : end_thinking+1] で取り、logits = actor_g[:, start_thinking : end_thinking]
            # みたいにする等。ここでは最も単純に token[t] を logits[t] で計算してしまう例。
            # padding を除くためにマスクもかける
            thinking_slice_logits = actor_g[:, start_thinking : end_thinking, :]  # shape [1, #thinking, vocab]
            thinking_slice_tokens = tokens_g[:, start_thinking : end_thinking]     # [1, #thinking]

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

            # === Output 部分の NLL(Actor) ===
            actor_nll, actor_count = compute_nll_and_masked_tokens(
                actor_g, tokens_g, pad_token_id=0, start_idx=start_output, end_idx=end_output
            )
            # === Output 部分の NLL(Base) ===
            base_nll, base_count = compute_nll_and_masked_tokens(
                base_g, tokens_g, pad_token_id=0, start_idx=start_output, end_idx=end_output
            )
            # actor_nll, base_nll はスカラー（合計）
            # advantage = baseline - actor => smaller actor_nll => advantage>0
            advantage = (base_nll - actor_nll).detach() # detach して定数に

            # thinking_logprob を張り付け
            thinking_logprob_all.append(sum_logprob_thinking.unsqueeze(0)) 
            advantage_all.append(advantage.unsqueeze(0))

        # shape: [GenerateCount, 1]
        thinking_logprob_all = torch.cat(thinking_logprob_all, dim=0) 
        advantage_all        = torch.cat(advantage_all, dim=0)

        # REINFORCE 損失
        # thinking_logprob_all: (GenerateCount,)
        # advantage_all       : (GenerateCount,)
        # => まとめて loss = - advantage * log_prob
        loss_rl = -(thinking_logprob_all * advantage_all).mean()

        # 必要に応じて SFT のクロスエントロピーなど別ロスを加えるならここで加算
        # total_loss = alpha * loss_rl + beta * loss_sft といった形に
        total_loss = loss_rl
        
        # ロスとして積算 (バッチ外 for s in range(bsz) だがデモ用に1バッチのみ)
        loss_rl_all += float(loss_rl.item())

        # Lightning では通常 "return total_loss" => backward => optimizer
        # ただしバッチループを自前でするなら都度 accumulate
        # (サンプルコードでは s=1 なので1回でOK)
        return total_loss

    # もしバッチの最後にまとめてロス返すなら以下のように
    # avg_loss = loss_rl_all / bsz
    # return avg_loss



def training_step_zerocot_(self, batch, batch_idx):
    print('Training Step ZeroCoT')
    batch_orpo = batch

    args = self.args

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

        prompt_text = self.tokenizer.decode(prompttoken.tolist())[:args.ctx_len//2]
        chosen_text = self.tokenizer.decode(chosentoken.tolist())[:args.ctx_len//2]
        
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

        #print(GeneratePrompt)

        generated_logit = []
        generated_token = []
        generated_tokenaddress = []
        basemodel_logit = []
        combined_idxs = []

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
            
            combined_idxs.append(BaseModelIdx)
            generated_logit.append(generated_x)
            generated_token.append(generated_tokens)
            generated_tokenaddress.append(tokenaddress)
            basemodel_logit.append(BaseModelLogits)


            print(generated_x)
            print(f'generated_x shape = {generated_x.shape}')

        print(basemodel_logit)
        effective_ctxlen = []
        for k in range(len(combined_idxs)):
             effective_ctxlen.append((combined_idxs[k].shape[1]))
             combined_idxs[k] = pad_to_size_2d(combined_idxs[k],args.ctx_len)
             basemodel_logit[k] = pad_to_size_3d(basemodel_logit[k],args.ctx_len)

        
             

        CombinedIdx = torch.cat(combined_idxs,dim=0) # will [B,ctx_len]
        BaseModelLogits = torch.cat(basemodel_logit,dim=0) # will [B,ctx_len,65536]

        print(f'CombinedIdx = {CombinedIdx.shape}') 
        print(f'BaseModelLogits = {BaseModelLogits.shape}') 

        logits,_= self.forward(CombinedIdx)

        #diff = logits - BaseModelLogits

        #print(diff)




        #Memo

        #BaseModel Logits [B,ctx_len]
        #ActorModel Logits[B,ctx_len]

        #Reinforce BaseModel vs ActorModel in ChosenToken Perplexity



        #generated_logit and generated_token contain prompt + thinking + chosen  tokens,logits, maybe can compute probs

        #ここまでは動作しています








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



def compute_logprob_for_tokens(
    logits: torch.Tensor, 
    tokens: torch.Tensor,
    pad_token_id: int = 0,
    start_idx: int = 0,
    end_idx: int = None
) -> torch.Tensor:
    """
    [start_idx : end_idx) のトークンに対応する log p(token) の「合計」を返す。
    padding は無視する。
    
    具体的には cross_entropy 形式を使わずに、
       logp = log_softmax(logits) を取った後に tokens を index して合計。
    """
    B, T, V = logits.shape
    if end_idx is None:
        end_idx = T

    slice_logits = logits[:, start_idx:end_idx, :]   # [B, T_sliced, V]
    slice_tokens = tokens[:, start_idx:end_idx]      # [B, T_sliced]

    # フラット化
    flat_logits = slice_logits.reshape(-1, V)    # [B*T_sliced, V]
    flat_tokens = slice_tokens.reshape(-1)       # [B*T_sliced]

    # マスク
    valid_mask = (flat_tokens != pad_token_id) & (flat_tokens != 0)
    valid_logits = flat_logits[valid_mask]
    valid_tokens = flat_tokens[valid_mask]
    if valid_tokens.numel() == 0:
        return torch.zeros([], device=logits.device)

    logp = F.log_softmax(valid_logits, dim=-1)              # [N, V]
    chosen_logp = logp[range(len(valid_tokens)), valid_tokens]  # [N]
    sum_logp = chosen_logp.sum()  # 合計

    return sum_logp

def reinforce_loss_with_baseline(
    actor_logprob: torch.Tensor,
    advantage: torch.Tensor
) -> torch.Tensor:
    """
    REINFORCE の損失を計算:  - advantage * log_prob(actor)
    の形になるように合計。多バッチであれば平均化するなど好みに合わせて調整。

    actor_logprob: [B, n_thinking_tokens] のような形を想定
    advantage: [B] or [B, 1] のような形

    return: scalar
    """
    # advantage を thinking 次元へブロードキャスト
    # actor_logprob: (B, n_think) 
    # advantage     : (B,)  => (B,1)
    if advantage.dim() == 1:
        advantage = advantage.unsqueeze(1)  # (B,1)

    # 損失の計算
    # REINFORCE: loss = - E[ advantage * sum(logpi(a)) ]
    # shape => (B, n_think)
    loss_mat = - advantage * actor_logprob
    
    # バッチ / thinking トークン 合計
    loss = loss_mat.mean()
    return loss







def training_step_grpo_(self, batch, batch_idx):
    """
    GRPO (Group Relative Policy Optimization) Implementa

    - Sample G times (GenerateCount times) per batch
    - Calculate reward for each sampling
    - Normalize reward to get advantage
    - Calculate KL dissipation for model (Actor) and reference (BaseModel, etc.)

    - Minimize loss L = - sum( ratio_no_grad * log_probs_actor * advantage ) + β * KL
(= maximize the above formula).
    """
    args = self.args

    # (簡易サンプル) バッチサイズを1と仮定
    # 複数バッチを同時に扱うなら forループを回すか、あるいは一括で扱う
    bsz = len(batch)
    assert bsz == 1, "このサンプルコードはバッチサイズ1想定です"

    GenerateCount = 4  # G: 1バッチあたり何回生成するか
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
    rewards_list = my_rulebase_reward_func(prompt=user_prefix,completions=generated_completions_only,answer=final_answer_text)  # 長さG
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




# if you want to use. its suitable for reinforce.
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
    
    slice_logits = logits[:, start_idx:end_idx, :]    # [B, T_sliced, V]
    slice_tokens = tokens[:, start_idx:end_idx]       # [B, T_sliced]

    # CrossEntropy で扱いやすいよう、(B*T_sliced, V) と (B*T_sliced) へ
    flat_logits = slice_logits.reshape(-1, V)         # [B*T_sliced, V]
    flat_tokens = slice_tokens.reshape(-1)            # [B*T_sliced]

    # パディング位置マスクを作成（pad_token_id や 0 を除外するなど）
    valid_mask = (flat_tokens != pad_token_id) & (flat_tokens != 0) 
    
    valid_logits = flat_logits[valid_mask]  # shape [N_valid, V]
    valid_tokens = flat_tokens[valid_mask]  # shape [N_valid]
    valid_count  = valid_logits.size(0) 

    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0
    
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

    if end_idx - start_idx < 2:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0

    slice_logits = logits[:, start_idx:end_idx - 1, :]    # [B, T_sliced-1, V]
    slice_tokens = tokens[:, start_idx + 1:end_idx]         # [B, T_sliced-1]

    flat_logits = slice_logits.reshape(-1, V)         # [B*(T_sliced-1), V]
    flat_tokens = slice_tokens.reshape(-1)            # [B*(T_sliced-1)]

    valid_mask = (flat_tokens != pad_token_id) & (flat_tokens != 0)
    
    valid_logits = flat_logits[valid_mask]  # shape [N_valid, V]
    valid_tokens = flat_tokens[valid_mask]  # shape [N_valid]
    valid_count  = valid_logits.size(0) 

    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0
    
    total_nll = F.cross_entropy(valid_logits, valid_tokens, reduction='sum')
    return total_nll, valid_count

