import json
import re
# 入力ファイルと出力ファイルのパス
input_file_path = 'may2.jsonl'
output_file_path = 'may2_cleaned.jsonl'

# 入力ファイルを開き、出力ファイルに書き込む
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # JSON文字列を辞書に変換
        data = json.loads(line)
        
        # 'text'データを抽出し、変換
        if 'text' in data:
            text = data['text'].replace('\n\n', '$ugyuugyu$')# + '\n\n'\
            text = text.replace('\n', '')
            text = text.replace('$ugyuugyu$', '\n')
            text = text.replace('\n\n', '\n')
            text = text.replace('\n\n', '\n')
            text = re.sub(r'\s+', ' ', text)

            text = '\x16' + text + '\n\n\x17' 
            print(text)



            data['text'] = text
        
        # 変換後のデータをJSONL形式で出力ファイルに書き込む
        json.dump(data, output_file,ensure_ascii=False)
        output_file.write('\n')  # JSONLの各オブジェクトは新しい行に書く

print("処理が完了しました。")