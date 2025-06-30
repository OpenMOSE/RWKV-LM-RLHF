import os
import json

from hiroshima_converter import DialectConverter

# 入出力フォルダのパス
input_dir = '/home/client/Projects/RWKV-LM-RLHF/main/myfolder/new_dataset_format_light'
output_dir = '/home/client/Projects/RWKV-LM-RLHF/main/myfolder/jsonl_dataset_hiroshima'

# 出力フォルダが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# DialectConverter のインスタンスを作成
converter = DialectConverter()

# Inputフォルダ内のすべてのjsonlファイルを処理
for filename in os.listdir(input_dir):
    if filename.endswith('.jsonl'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                # JSON形式でパース
                data = json.loads(line)

                # messages を確認して、assistantのcontentを変換
                for message in data.get('messages', []):
                    if message.get('role') == 'assistant' and 'content' in message:
                        original_content = message['content']
                        # 置換
                        converted_content = converter.convert_to_dialect(original_content)
                        message['content'] = converted_content

                # 変換後のデータをjsonl形式で出力
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print("すべてのファイルが変換され、Outputフォルダに保存されました。")
