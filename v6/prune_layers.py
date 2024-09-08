import torch
import re
import os
from typing import Dict

def count_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(p.numel() for p in state_dict.values())

def print_model_keys(state_dict: Dict[str, torch.Tensor], max_keys: int = None):
    keys = list(state_dict.keys())
    if max_keys is not None:
        print(f"First {max_keys} keys of the model:")
        keys = keys[:max_keys]
    else:
        print("All keys of the model:")
    
    for key in keys:
        print(f"  {key}")
    
    if max_keys is not None and len(state_dict) > max_keys:
        print(f"  ... (and {len(state_dict) - max_keys} more)")

def prune_layers(input_path: str, output_path: str, layers_to_remove: list):
    # モデルの読み込み
    state_dict: Dict[str, torch.Tensor] = torch.load(input_path, map_location='cpu')
    
    print(f"Original model size: {count_parameters(state_dict)} parameters")

    # 新しいstate_dictを作成
    new_state_dict: Dict[str, torch.Tensor] = {}
    block_pattern = re.compile(r'blocks\.(\d+)\.')
    current_block = -1
    new_block_index = -1
    removed_count = 0

    for key, value in state_dict.items():
        match = block_pattern.match(key)
        if match:
            block_num = int(match.group(1))
            if block_num in layers_to_remove:
                removed_count += 1
                print(f"Removing layer: {key}")
                continue  # 指定されたレイヤーをスキップ

            if block_num != current_block:
                current_block = block_num
                new_block_index += 1

            new_key = block_pattern.sub(f'blocks.{new_block_index}.', key)
            new_state_dict[new_key] = value.clone()  # 値のコピーを作成
        else:
            # ブロック以外のレイヤーはそのまま保持
            new_state_dict[key] = value.clone()  # 値のコピーを作成

    print(f"Removed {removed_count} layers")
    print(f"New model size: {count_parameters(new_state_dict)} parameters")

    # 更新されたモデルを保存
    torch.save(new_state_dict, output_path)

    # パラメータ数を計算（Billion単位）
    num_params = count_parameters(new_state_dict)
    num_params_billion = num_params / 1e9

    print(f"Pruned model saved to {output_path}")
    print(f"Number of parameters: {num_params_billion:.2f}B")
    
    # ファイルサイズの比較
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print(f"Input file size: {input_size:,} bytes")
    print(f"Output file size: {output_size:,} bytes")
    print(f"Size difference: {input_size - output_size:,} bytes")

    # モデルのキー一覧を表示
    print("\nOutput model structure:")
    print_model_keys(new_state_dict, max_keys=50)  # 最初の50個のキーを表示

    # 保存されたモデルを読み込んで確認
    print("\nVerifying saved model:")
    saved_state_dict = torch.load(output_path, map_location='cpu')
    print(f"Saved model size: {count_parameters(saved_state_dict)} parameters")
    print_model_keys(saved_state_dict, max_keys=10000)  # 最初の10個のキーを表示

# 使用例
input_path = "models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth"
output_path = "Output.pth"
layers_to_remove = [3, 7, 11, 15, 19, 23, 27]

prune_layers(input_path, output_path, layers_to_remove)