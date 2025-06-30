import csv
import json
import os
import re
from argparse import ArgumentParser
import csv
endtoken = '\n\n\x17'

parser = ArgumentParser()

parser.add_argument("--input_folder", default="myfolder/nsha.csv", type=str)
parser.add_argument("--output_folder", default="myfolder/.csv", type=str)
args2 = parser.parse_args()

def process_text(text):
    # '\n\n'を'\n'に置換
    text = text.replace('\n\n', '\n')
    # 連続した'\n'を1つの'\n'に置換
    text = re.sub(r'\n+', '\n', text)
    return text

def process_csv_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csv_file, \
         open(output_file, 'w', encoding='utf-8') as jsonl_file:
        
        csv_reader = csv.DictReader(csv_file)
        conversation = []
        
        for row in csv_reader:
            processed_system = process_text(row['system'])
            processed_prompt = process_text(row['prompt'])
            processed_chosen = process_text(row['chosen'])
            conversation.append(f"System: {processed_system}{endtoken}User: {processed_prompt}{endtoken}Assistant: {processed_chosen}{endtoken}")
            
            jsonl_entry = {
                'text': ''.join(conversation)
            }
            
            json.dump(jsonl_entry, jsonl_file, ensure_ascii=False)
            jsonl_file.write('\n')

def main():
    input_folder = args2.input_folder#'OpenAI'  # CSVファイルが格納されているフォルダ
    output_folder = args2.output_folder#'OpenAIOutput'  # 出力JSONLファイルを保存するフォルダ
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_converted_jsonl.jsonl"
            output_path = os.path.join(output_folder, output_filename)
            
            process_csv_file(input_path, output_path)
            print(f"Processed {filename} -> {output_filename}")

if __name__ == "__main__":
    main()
