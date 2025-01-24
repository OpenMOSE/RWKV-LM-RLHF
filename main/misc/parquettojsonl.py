# -*- coding: utf-8 -*-
import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="Input Parquet file path")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    # Parquetを読み込み
    df = pd.read_parquet(args.input_parquet)

    # 1行ずつ読み込み、JSONLに追記 (UTF-8 で書き込み)
    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "prompt": row["instruction"],
                "chosen": row["response"],
                "reject": ""
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
