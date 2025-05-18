import os
import json

input_dir = "myfolder/kirihara/Input"
output_dir = "myfolder/kirihara/Output"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# Inputフォルダ内の全ての.txtファイルを取得
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", ".jsonl"))

        with open(input_path, "r", encoding="utf-8") as infile:
            text = infile.read().strip()  # 改行なども含めてまとめて取得

        json_obj = {"text": text}

        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")  # 1行のJSONLにする

print("変換が完了しました。")
