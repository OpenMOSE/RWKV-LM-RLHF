

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