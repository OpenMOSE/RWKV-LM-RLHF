import json
import requests
import multiprocessing
import math
import sys
import argparse
import time
from typing import List, Tuple

# -------------------------------------------
# OpenAI API互換のエンドポイント
# -------------------------------------------
API_URL = "http://127.0.0.1:9000/v1/chat/completions"

# -------------------------------------------
# LLMに問い合わせてreject文字列を生成する関数
# -------------------------------------------
def generate_reject(prompt: str) -> str:
    """
    LLMに問い合わせて reject 文字列を生成する。
    """
    payload = {
        "model": "RWKV x070 2B9 CJE Instruct-1",  # 必要に応じてモデル名を変更
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        # 想定されるレスポンス構造
        # {
        #   "id": ...,
        #   "object": ...,
        #   "created": ...,
        #   "choices": [
        #       {
        #           "message": {
        #               "role": "assistant",
        #               "content": "生成されたテキスト..."
        #           },
        #           ...
        #       }
        #   ],
        #   ...
        # }
        reject_text = data["choices"][0]["message"]["content"]
        return reject_text
    except Exception as e:
        print(f"[Worker] Error in generate_reject: {e}", file=sys.stderr)
        return ""

# -------------------------------------------
# 各プロセスが担当するチャンクを処理する関数
# -------------------------------------------
def process_chunk(json_lines: List[str]) -> Tuple[List[str], int]:
    """
    JSON文字列リストを受け取り、"reject" フィールドをLLMから生成して更新する。
    * 戻り値:
      - results: 更新後の各行(JSON文字列)
      - total_words: 生成した reject 部分のトータル単語数（ざっくりsplitベース）
    """
    results = []
    total_words = 0

    for line in json_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # JSONとして読み込めない行はスキップ
            continue

        # "reject"キーが無い or 空文字の場合のみ生成
        if "reject" not in data or not data["reject"]:
            prompt_text = data.get("prompt", "")
            if prompt_text:
                generated = generate_reject(prompt_text)
                data["reject"] = generated
                # 生成テキストのワード数をカウント（簡易的に空白splitで計測）
                total_words += len(generated.split())
            else:
                data["reject"] = ""

        # 更新後データをシリアライズ
        results.append(json.dumps(data, ensure_ascii=False))

    return results, total_words

# -------------------------------------------
# リストを複数のチャンクに分割する関数
# -------------------------------------------
def chunkify(data_list: List[str], num_chunks: int) -> List[List[str]]:
    """
    data_list を num_chunks 個のチャンクに分割して返す。
    """
    if num_chunks <= 1:
        return [data_list]

    chunk_size = math.ceil(len(data_list) / num_chunks)
    return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

# -------------------------------------------
# メイン処理
# -------------------------------------------
def main(input_file: str,
         output_file: str,
         num_processes: int,
         chunk_size: int) -> None:
    """
    ストリーミング方式で input_file を chunk_size 行ずつ読み込みつつ、
    'reject' フィールドを生成して output_file に書き出す。
    """

    # マルチプロセスプールを用意
    pool = multiprocessing.Pool(processes=num_processes)

    # 全体進捗を測るための変数
    global_start_time = time.time()
    global_word_count = 0  # 全チャンクで生成した単語数の合計
    global_chunk_count = 0
    total_processed_lines = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        print(f"[Main] Start processing.")
        print(f" - Input file: {input_file}")
        print(f" - Output file: {output_file}")
        print(f" - Processes: {num_processes}")
        print(f" - Chunk size: {chunk_size}")
        print("------------------------------------------------")

        while True:
            # chunk_size 行だけ読み込む
            lines = []
            for _ in range(chunk_size):
                line = fin.readline()
                if not line:
                    break
                lines.append(line)

            # ファイル末尾に達した場合は終了
            if not lines:
                print("[Main] Reached end of file.")
                break

            global_chunk_count += 1
            chunk_id = global_chunk_count

            # どれくらいの行を今回処理するか
            n_lines = len(lines)
            total_processed_lines += n_lines

            print(f"[Main] Processing chunk #{chunk_id} (lines={n_lines}) ...")

            # チャンクを並列実行用に分割
            splitted = chunkify(lines, num_processes)
            chunk_start_time = time.time()

            # 並列処理実行
            # process_chunk の戻り値は List[Tuple[List[str], int]]
            results_list = pool.map(process_chunk, splitted)

            chunk_end_time = time.time()
            chunk_elapsed = chunk_end_time - chunk_start_time

            # 結果をまとめて output_file に書き込み
            chunk_word_count = 0
            write_start_time = time.time()
            for results, sub_word_count in results_list:
                chunk_word_count += sub_word_count
                for line_result in results:
                    fout.write(line_result + "\n")
            write_end_time = time.time()
            write_elapsed = write_end_time - write_start_time

            global_word_count += chunk_word_count

            # ログ出力
            print(f"[Main]  => Chunk #{chunk_id} done.")
            print(f"     - Processing time: {chunk_elapsed:.2f} sec")
            print(f"     - Writing time   : {write_elapsed:.2f} sec")
            print(f"     - Word count     : {chunk_word_count}")
            print(f"     - Lines processed in chunk: {n_lines}")

            # 現在までの累計で Words/Sec を出す
            overall_time = time.time() - global_start_time
            overall_wps = global_word_count / overall_time if overall_time > 0 else 0

            print(f"[Main]  => Overall WPS: {overall_wps:.2f} words/sec "
                  f"(Total words={global_word_count}, Elapsed={overall_time:.2f} sec)")
            print("------------------------------------------------")

    pool.close()
    pool.join()

    total_time = time.time() - global_start_time
    print("[Main] All done.")
    print(f" - Total lines processed: {total_processed_lines}")
    print(f" - Total words generated: {global_word_count}")
    print(f" - Elapsed time         : {total_time:.2f} sec")
    if total_time > 0:
        print(f" - Average WPS         : {global_word_count / total_time:.2f}")
    print("------------------------------------------------")



def parse_args():
    parser = argparse.ArgumentParser(description="JSONL の 'reject' フィールドを LLM で生成するスクリプト")
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        default="myfolder/RLHF/bancho.jsonl",
        help="入力 JSONL ファイルパス (デフォルト: input.jsonl)"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="myfolder/RLHF/bancho_reject.jsonl",
        help="出力 JSONL ファイルパス (デフォルト: output.jsonl)"
    )
    parser.add_argument(
        "-n", "--num_processes",
        type=int,
        default=32,
        help="並列処理数 (デフォルト: 4)"
    )
    parser.add_argument(
        "-c", "--chunk_size",
        type=int,
        default=96,
        help="一度に読み込む行数 (デフォルト: 1000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size
    )
