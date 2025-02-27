import os
import json
import random
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import json
# (標準ライブラリではない点に注意)
from jinja2 import Template

def load_tokenizer_config(config_path: str) -> dict:
    """
    TokenizerConfig (例: tokenizer_config.json) をロードして辞書として返す
    """
    with open(config_path + "/tokenizer_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_prompt_from_config(tokenizer_config: dict, messages: list, add_generation_prompt: bool = False) -> str:
    """
    tokenizer_config 内の chat_template を元に、messages (role, content) をまとめた文字列を生成する
    """
    # chat_template を取り出す
    chat_template_str = tokenizer_config.get("chat_template", "")
    if not chat_template_str:
        raise ValueError("chat_template が TokenizerConfig 内に存在しません。")

    # eos_token など必要なトークンも取り出す
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")

    # テンプレートを読み込む
    template = Template(chat_template_str)

    # テンプレート変数を指定してレンダリング
    rendered_text = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        eos_token=eos_token
    )
    return rendered_text

class JSONLOnDemandOffsetDataset(Dataset):
    def __init__(
        self,
        args,
        folder_path,
        tokenizer_mode='world',
        max_seq_length=4096,
        overlap_div=4,
        debug=False
    ):
        """
        JSONLファイル(各行が {"messages": [...]} 形式)において、
        ファイル先頭から各行のオフセットを事前に取得(初期化時)し、
        __getitem__で必要な行へ直接f.seek()でジャンプ→読み込み→トークナイズするクラス。

        Args:
            args: 学習時のパラメータやフラグなどを保持する引数クラス
            folder_path (str): JSONLファイルが置かれているフォルダパス
            max_seq_length (int): 1サンプルの最大トークン長
            overlap_div (int): tokenshift時の重複長を max_seq_length / overlap_div で決定
            debug (bool): デバッグ用出力フラグ
        """
        self.args = args
        self.folder_path = folder_path
        self.max_seq_length = max_seq_length
        self.overlap_length = int(max_seq_length / overlap_div)
        self.debug = debug

        self.tokenizer_config = load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/../tokenizer/" + tokenizer_mode)

        # 例: Qwenのトークナイザ (パスはご利用環境に合わせてください)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(os.path.abspath(__file__)) + "/../tokenizer/" + tokenizer_mode,
            trust_remote_code=True
        )

        # tokenshift が有効かどうか。元コードに合わせて True/False を設定
        self.tokenshift = True
        # tokenshiftで余ったトークンを次に持ち越す領域
        self.CurrentExcessToken = None

        # JSONLファイルの一覧を取得
        file_list = [
            fn for fn in os.listdir(folder_path) if fn.endswith(".jsonl")
        ]
        file_list.sort()  # ファイル名順などが必要ならソート

        # ファイルごとの情報を保持
        # self.files_line_offsets[i]: i番目のファイルの各行オフセットリスト
        # self.files_info[i] = (ファイルパス, 行数)
        # self.cumulative_lines[i] = i番目ファイルが全体で何行目から始まるか(累積行数)
        self.files_line_offsets = []
        self.files_info = []
        self.cumulative_lines = []

        current_cumulative = 0
        for fn in file_list:
            fullpath = os.path.join(folder_path, fn)
            offsets = self._build_offset_index(fullpath)
            line_count = len(offsets)

            self.files_line_offsets.append(offsets)
            self.files_info.append((fullpath, line_count))
            self.cumulative_lines.append(current_cumulative)

            current_cumulative += line_count

        # 全ファイルの合計行数
        self.total_lines = current_cumulative

        # random_mode=0 の場合に使うシーケンシャル読みのカウンタ
        self.Count = 0

        if self.debug:
            print(f"[DEBUG] Found {len(file_list)} jsonl files.")
            for (f, c), cumu in zip(self.files_info, self.cumulative_lines):
                print(f"  {f} : {c} lines (cumulative start={cumu})")
            print(f"[DEBUG] total_lines = {self.total_lines}")

    def _build_offset_index(self, filepath):
        """
        1つのファイルを開き、各行の先頭バイトオフセットをリストに収集。
        例: offsets[i] = i行目が始まるファイル内バイト位置
        """
        offsets = []
        current_offset = 0
        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                offsets.append(current_offset)
                # 行を読んだ後のファイルポインタ位置 (次行の開始バイト) に更新
                current_offset = f.tell()
        return offsets

    def __len__(self):
        """
        random_mode=1 の場合は epoch_steps*micro_bsz としてループ、
        random_mode=0 の場合はファイル全体の行数 total_lines を返す例。
        """
        if self.args.random_mode:
            return self.args.epoch_steps * self.args.micro_bsz
        return self.total_lines

    def __getitem__(self, idx):
        """
        1回の __getitem__ で infctx_dataset_multiplier(N) 回ぶん行インデックスを取得し、
        該当行をファイルから直接読み込んでトークナイズ→連結→tokenshift処理。
        """
        N = self.args.infctx_dataset_multiplier

        line_indices = []
        if self.args.random_mode == 0:
            # シーケンシャルアクセス
            for _ in range(N):
                line_indices.append(self.Count)
                self.Count += 1
                if self.Count >= self.total_lines:
                    self.Count = 0
        else:
            # ランダムアクセス
            for _ in range(N):
                line_indices.append(random.randint(0, self.total_lines - 1))

        # 今回分のトークンを連結するためのリスト
        tokens_list = []

        TextList = ''

        for line_idx in line_indices:
            # どのファイルに属する行か
            file_pos = bisect.bisect_right(self.cumulative_lines, line_idx) - 1
            if file_pos < 0:
                file_pos = 0

            filename, file_line_count = self.files_info[file_pos]
            start_line_idx = self.cumulative_lines[file_pos]      # ファイルが始まる全体行インデックス
            local_line_idx = line_idx - start_line_idx            # ファイル内での行インデックス
            offsets = self.files_line_offsets[file_pos]
            offset = offsets[local_line_idx]

            # ファイルを開いて offset にシークし、その行を取得
            with open(filename, "r", encoding="utf-8") as f:
                f.seek(offset)
                raw_line = f.readline().strip()
                if raw_line:
                    # JSONロード
                    data = json.loads(raw_line)
                    messages = data.get("messages", [])

                    # role/content をまとめてテキスト化
                    # 例: "user: 質問\nassistant: 回答" という形式で結合するなど
                    text_list = []

                    text_list.append(generate_prompt_from_config(self.tokenizer_config,messages,False))



                    # for msg in messages:
                    #     role = msg.get("role", "")
                    #     content = msg.get("content", "")
                    #     # 必要に応じて role 表示を入れる/入れないを調整してください
                    #     text_list.append(f"{role}: {content}")


                    # 1行ぶんのトークン化対象テキスト
                    # 例: user,assistant ロールを改行で区切る
                    text = "\n".join(text_list)

                    TextList = TextList + text

                    # トークナイズ
                    token_ids = self.tokenizer.encode(text, add_special_tokens=True)
                    tokens_list.append(np.array(token_ids, dtype=np.int64))


        #print(TextList)
        #print(tokens_list)
        #exit()

        # 連結
        if len(tokens_list) > 0:
            new_tokens = np.concatenate(tokens_list, axis=0)
        else:
            new_tokens = np.array([], dtype=np.int64)

        # tokenshift 処理
        LastTokens = self.CurrentExcessToken if self.CurrentExcessToken is not None else None
        # 実際に新規データを取得して連結するかどうかの判定(元コードにならう)
        GetNewDataset = True
        if LastTokens is not None and len(LastTokens) >= self.max_seq_length:
            GetNewDataset = False
        if not self.tokenshift:
            GetNewDataset = True

        if GetNewDataset:
            tokens = new_tokens
        else:
            tokens = LastTokens

        if self.tokenshift:
            if LastTokens is not None and GetNewDataset:
                tokens = np.hstack((LastTokens, tokens))

        if tokens is None:
            tokens = np.array([], dtype=np.int64)

        # 最大長を超えた分を持ち越し
        seq_len = min(len(tokens), self.max_seq_length)
        if self.tokenshift:
            if len(tokens) > self.max_seq_length:
                self.CurrentExcessToken = tokens[self.max_seq_length - self.overlap_length:]
            else:
                self.CurrentExcessToken = None

        tokens = tokens[:seq_len]

        # 入力/出力/attention_mask を作成
        padded_tokens_input = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_tokens_target = np.zeros(self.max_seq_length, dtype=np.int64)
        attention_mask = np.zeros(self.max_seq_length, dtype=np.float32)

        #

        if seq_len > 1:
            padded_tokens_input[:seq_len - 1] = tokens[:seq_len - 1]
            padded_tokens_target[:seq_len - 1] = tokens[1:seq_len]
            attention_mask[:seq_len - 1] = 1.0

        #print(padded_tokens_input)

        return {
            "input_ids": torch.from_numpy(padded_tokens_input),
            "target_ids": torch.from_numpy(padded_tokens_target),
            "attention_mask": torch.from_numpy(attention_mask),
        }

def collate_fn(batch):
    """
    DataLoader用のコラテ関数。バッチ単位でテンソルをまとめて返す。
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
