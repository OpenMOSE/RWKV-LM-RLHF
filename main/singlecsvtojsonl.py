import csv
import json
import re

endtoken = '\n\n\x17'

def process_text(text):
    # '\n\n'を'\n'に置換
    text = text.replace('\n\n', '\n')
    # 連続した'\n'を1つの'\n'に置換
    text = re.sub(r'\n+', '\n', text)
    return text

def csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, 'r', encoding='utf-8') as csv_file, \
         open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            prompt = row['prompt']
            chosen = row['chosen']

            prompt = process_text(prompt)
            chosen = process_text(chosen)
            
            text = f"User: {prompt}{endtoken}Assistant: {chosen}{endtoken}"
            
            jsonl_entry = {
                'text': text
            }
            
            json.dump(jsonl_entry, jsonl_file, ensure_ascii=False)
            jsonl_file.write('\n')

# 使用例
input_csv_file = 'myfolder/datasets/tweets01.csv'
output_jsonl_file = 'myfolder/datasets/tweets01.csv.jsonl'

csv_to_jsonl(input_csv_file, output_jsonl_file)
print(f"変換が完了しました。結果は {output_jsonl_file} に保存されています。")