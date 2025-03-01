import requests
import json
import numpy as np

# APIエンドポイントURL
BASE_URL = "http://localhost:10000"  # 変更されたポート番号
LOAD_MODEL_URL = f"{BASE_URL}/LoadModel"
PROCESS_LOGITS_URL = f"{BASE_URL}/ProcessLogits"

def test_load_model(model_name="myfolder/Phi-4-mini-instruct", use_4bit=True, use_cuda=True):
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
    """ProcessLogitsエンドポイントをテスト"""
    # 2つのバッチのトークンを持つ入力例
    payload = {
        #"input_ids": [[200021, 145692, 668, 1412, 382, 6446, 30, 200020, 200019], [200021, 145692, 668, 1412, 382, 6446, 30, 200020, 200019]],
        "input_ids": [[200021, 145692, 668, 1412, 382, 6446, 30, 575, 2411, 6446, 382, 4358, 5716, 13, 6446, 382, 1869, 23172, 13, 200020, 200019], [200021, 145692, 668, 1412, 382, 6446, 30, 200020, 200019]],
        
        "topk": 2000  # テスト用に小さい値
    }
    
    response = requests.post(PROCESS_LOGITS_URL, json=payload)
    print(f"ProcessLogitsレスポンス (ステータス: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(f"Loss: {result['loss']}")
        print(f"バッチ数: {len(result['indices'])}")
        print(f"シーケンス長: {[len(batch) for batch in result['indices']]}")
        print(f"Topkサイズ: {len(result['indices'][0][0])}")
        
        # ロジットの一部を表示
        print("\nロジット例（最初のバッチ、最初のトークン位置の先頭3つ）:")
        if len(result['logits']) > 0 and len(result['logits'][0]) > 0:
            print(result['logits'][0][0][:3])
    else:
        print(response.json())
    
    return response.status_code == 200

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