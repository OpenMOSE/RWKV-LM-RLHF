import requests
import json
import numpy as np
import base64
import pickle

import requests
import base64
import pickle
import torch

def base64_to_tensor(b64_str):
    pickle_bytes = base64.b64decode(b64_str.encode('utf-8'))
    return pickle.loads(pickle_bytes)

# APIエンドポイントURL
BASE_URL = "http://localhost:10000"  # 変更されたポート番号
LOAD_MODEL_URL = f"{BASE_URL}/LoadModel"
PROCESS_LOGITS_URL = f"{BASE_URL}/ProcessLogits_ex"

def test_load_model(model_name="/home/client/Projects/llm/Qwen3-8B", use_4bit=False, use_cuda=True):
    """LoadModelエンドポイントをテスト"""
    payload = {
        "modelname": model_name,
        "use_4bit": use_4bit,
        "use_cuda": use_cuda
    }
    response = requests.post(LOAD_MODEL_URL, json=payload)
    print(f"LoadModelレスポンス (ステータス: {response.status_code}):")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_process_logits():
    """Pickle形式で返ってくる /ProcessLogits エンドポイントのテスト"""
    payload = {
        "input_ids": [
            [128000, 145692, 668, 1412, 382, 6446, 30, 575, 2411, 6446, 382, 4358, 5716, 13, 6446, 382, 1869, 23172, 13, 100020, 100019],
            [128000, 145692, 668, 1412, 382, 6446, 30, 575, 2411, 6446, 382, 4358, 5716, 13, 6446, 382, 1869, 23172, 13, 100020, 100019]
        ],
        "topk": 30000  # Top-k を大きくする場合でも Pickleなら高速
    }

    response = requests.post(PROCESS_LOGITS_URL, json=payload)
    print(f"ProcessLogitsレスポンス (ステータス: {response.status_code}):")

    if response.status_code == 200:
        result = response.json()

        # Base64からTensorに復元
        logits_tensor = base64_to_tensor(result["logits"])
        indices_tensor = base64_to_tensor(result["indices"])
        loss = result["loss"]

        print(f"Loss: {loss}")
        print(f"バッチ数: {indices_tensor.shape[0]}")
        print(f"シーケンス長: {indices_tensor.shape[1]}")
        print(f"Topkサイズ: {indices_tensor.shape[2]}")

        # ロジットの一部を表示（最初のバッチ・最初の位置・先頭3つ）
        print("\nロジット例（[0,0,:3]):")
        print(logits_tensor[0, 0, :3])

        return True
    else:
        print(response.json())
        return False

def run_tests():
    """すべてのAPIテストを実行"""
    print("=== LoadModel APIのテスト ===")
    if test_load_model():
        print("\n=== ProcessLogits APIのテスト ===")
        test_process_logits()
    else:
        print("LoadModelが失敗したため、ProcessLogitsのテストをスキップします")

if __name__ == "__main__":
    run_tests()