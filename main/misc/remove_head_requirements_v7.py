import argparse
import torch
import argparse
import torch
import os
def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove specific keys from a PyTorch model's state_dict.")
    parser.add_argument('--inputmodel', type=str, default='myfolder/models/rwkv-x070-1b5-world-v3-80%trained-20250120-ctx4k.pth', help='Path to the input PTH file.')
    parser.add_argument('--outputmodel', type=str, default='myfolder/models/rwkv-x070-1b5-world-v3-NonHead2.pth', help='Path to save the output PTH file.')
    return parser.parse_args()


 
def remove_keys(state_dict, keys_to_remove):
    """
    Remove keys that contain any of the specified substrings.

    Args:
        state_dict (dict): The state dictionary of the model.
        keys_to_remove (list of str): Substrings to identify keys to remove.

    Returns:
        dict: The filtered state dictionary.
        list: List of removed keys.
    """
    removed_keys = [k for k in state_dict.keys() if any(sub in k for sub in keys_to_remove)]
    filtered_dict = {k: v for k, v in state_dict.items() if not any(sub in k for sub in keys_to_remove)}
    return filtered_dict, removed_keys

def main():
    args = parse_arguments()
    input_path = args.inputmodel
    output_path = args.outputmodel
    keys_to_remove = ['r_k']

    if not os.path.isfile(input_path):
        print(f"入力ファイルが存在しません: {input_path}")
        return

    with torch.no_grad():
        # Load the checkpoint from the input model
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # Determine if the checkpoint contains 'state_dict'
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("チェックポイントに 'state_dict' が含まれています。")
                state_dict = checkpoint['state_dict']
            else:
                print("チェックポイントは直接 state_dict 形式です。")
                state_dict = checkpoint
        else:
            print("チェックポイントは辞書形式ではありません。処理を中止します。")
            return

        # デバッグ: 全てのキーを表示（最初の10個）
        print(f"state_dict の総キー数: {len(state_dict)}")
        print("一部のキーを表示します:")
        for idx, key in enumerate(state_dict.keys()):
            if idx < 10:
                print(f"  {key}")
            else:
                break

        # 削除対象のキーを削除
        filtered_state_dict, removed_keys = remove_keys(state_dict, keys_to_remove)
        
        # デバッグ: 削除されたキーを表示
        if removed_keys:
            print(f"削除されたキー数: {len(removed_keys)}")
            print("削除されたキー一覧:")
            for rk in removed_keys:
                print(f"  - {rk}")
        else:
            print("削除対象のキーは見つかりませんでした。")

        # フィルタリングされた state_dict をチェックポイントに戻す
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = filtered_state_dict
        else:
            checkpoint = filtered_state_dict

        # 保存
        torch.save(checkpoint, output_path)
        print(f"フィルタリングされたモデルを保存しました: {output_path}")

        # ファイルサイズの比較
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        print(f"入力ファイルサイズ: {input_size} bytes")
        print(f"出力ファイルサイズ: {output_size} bytes")

if __name__ == "__main__":
    main()
