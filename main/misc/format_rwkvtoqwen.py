import os
import json

def convert_text_format(original_text: str) -> str:
    """
    入力のテキスト（例: "System: ...\n\n\x17User: ...\n\n\x17Assistant: ...\n\n\x17"）
    を、指定のフォーマットに変換した文字列を返す。
    """
    # "\n\n\x17" で分割（末尾の空行・空要素は排除）
    parts = [p.strip() for p in original_text.split("\n\n\x17")]
    parts = [p for p in parts if p]  # 空文字は除外

    new_segments = []
    for part in parts:
        # "System: " / "User: " / "Assistant: " で始まる部分を解析
        if part.startswith("System: "):
            role = "system"
            content = part[len("System: "):]
        elif part.startswith("User: "):
            role = "user"
            content = part[len("User: "):]
        elif part.startswith("Assistant: "):
            role = "assistant"
            content = part[len("Assistant: "):]
        else:
            # もしどれにも当てはまらない場合はスキップする
            continue

        # 指定のフォーマット
        # <|im_start|>
        # {role}
        # {content}
        # <|im_end|>
        segment_str = f"<|im_start|>\n{role}\n{content}\n<|im_end|>"
        new_segments.append(segment_str)

    # 各セグメントを改行で結合して返す
    return "\n".join(new_segments)


def main():
    input_dir = "myfolder/datasets/may"
    output_dir = "myfolder/2024_dataset/may_qwen"

    # 出力先フォルダが無ければ作成する
    os.makedirs(output_dir, exist_ok=True)

    # Inputフォルダ内のjsonlファイルを処理
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as fin, \
                 open(output_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    if not line.strip():
                        continue  # 空行はスキップ
                    
                    data = json.loads(line)
                    if "text" in data:
                        # 変換を行う
                        data["text"] = convert_text_format(data["text"])
                    
                    # 変換後のデータをjsonl形式で出力
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
