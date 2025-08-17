import os
import torch
import json
from pathlib import Path
from typing import Dict, Union
import gc

def get_tensor_shapes(file_path: Union[str, Path]) -> Dict[str, tuple]:
    """
    PyTorchモデル(.pth)またはSafeTensorsファイル/ディレクトリから
    テンソル名とShapeを取得する関数
    
    Args:
        file_path: .pthファイル、.safetensorsファイル、またはsafetensorsを含むディレクトリのパス
        
    Returns:
        Dict[str, tuple]: テンソル名をキー、Shapeをタプルで値とする辞書
    """
    file_path = Path(file_path)
    shapes_dict = {}
    
    if file_path.is_file():
        if file_path.suffix == '.pth' or file_path.suffix == '.pt':
            # PyTorchモデルの処理
            shapes_dict = _extract_pytorch_shapes(file_path)
        elif file_path.suffix == '.safetensors':
            # 単一のSafeTensorsファイルの処理
            shapes_dict = _extract_safetensors_shapes(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    elif file_path.is_dir():
        # ディレクトリの場合、SafeTensorsファイルを探す
        safetensor_files = list(file_path.glob("*.safetensors"))
        if not safetensor_files:
            raise ValueError(f"No .safetensors files found in directory: {file_path}")
        
        for st_file in safetensor_files:
            file_shapes = _extract_safetensors_shapes(st_file)
            # ファイル名をプレフィックスとして追加（オプション）
            # shapes_dict.update({f"{st_file.stem}/{k}": v for k, v in file_shapes.items()})
            shapes_dict.update(file_shapes)
    else:
        raise ValueError(f"Path does not exist: {file_path}")
    
    return shapes_dict


def _extract_pytorch_shapes(file_path: Path) -> Dict[str, tuple]:
    """
    PyTorchモデルからテンソルShapeを抽出（メモリ効率的）
    """
    shapes_dict = {}
    
    try:
        # mmap=Trueでメモリマップを使用してロード
        checkpoint = torch.load(file_path, map_location='cpu', mmap=True)
        
        # state_dictが含まれている場合
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            # 直接state_dictの場合、またはmodel等のキーがある場合
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            # checkpoint自体がstate_dictの場合
            state_dict = checkpoint
        
        # 各テンソルのShapeを取得
        for key, tensor in state_dict.items():
            if torch.is_tensor(tensor):
                shapes_dict[key] = tuple(tensor.shape)
                # テンソルの参照を削除
                del tensor
        
        # checkpointとstate_dictを削除
        del state_dict
        del checkpoint
        
        # ガベージコレクションを強制実行してメモリを解放
        gc.collect()
        
        # CUDAメモリも解放（GPUを使用している場合）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        raise RuntimeError(f"Error loading PyTorch model: {e}")
    
    return shapes_dict


def _extract_safetensors_shapes(file_path: Path) -> Dict[str, tuple]:
    """
    SafeTensorsファイルからテンソルShapeを抽出（メモリ効率的）
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors library is not installed. Install it with: pip install safetensors")
    
    shapes_dict = {}
    
    try:
        # SafeTensorsファイルを開く（メモリ効率的な方法）
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # メタデータからテンソル名を取得
            for key in f.keys():
                # get_tensorを使わずにshapeだけを取得
                # SafeTensorsは内部でメタデータにshape情報を持っている
                tensor_slice = f.get_slice(key)
                # shapeを取得
                shapes_dict[key] = tuple(tensor_slice.get_shape())
                
    except Exception as e:
        raise RuntimeError(f"Error loading SafeTensors file: {e}")
    
    return shapes_dict


def print_tensor_info(shapes_dict: Dict[str, tuple], max_display: int = None):
    """
    テンソル情報を見やすく表示するヘルパー関数
    
    Args:
        shapes_dict: テンソル名とShapeの辞書
        max_display: 表示する最大数（Noneの場合は全て表示）
    """
    print(f"Total tensors: {len(shapes_dict)}")
    print("-" * 60)
    
    items = list(shapes_dict.items())
    if max_display and len(items) > max_display:
        items = items[:max_display]
        print(f"Showing first {max_display} tensors...")
    
    for key, shape in items:
        # テンソルサイズの計算（要素数）
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        # サイズをMB単位で表示（float32と仮定）
        size_mb = (num_elements * 4) / (1024 * 1024)
        
        print(f"{key:40s} | Shape: {str(shape):20s} | ~{size_mb:.2f} MB")
    
    if max_display and len(shapes_dict) > max_display:
        print(f"... and {len(shapes_dict) - max_display} more tensors")


def GetAutoModelConfig(args,inputpath):
    shapes = get_tensor_shapes(inputpath)
    rwkvarch = "x060"
    args.HFMode = False
    for name, shape in shapes.items():
        if "r_k" in name:
            rwkvarch = "x070"
            break
    for name, shape in shapes.items():
        if "k_first" in name:
            rwkvarch = "hxa079"
            break
    headsize=64
    num_attention_heads=0
    num_kv_heads = 0
    hiddensize=0
    num_lora_w = 0
    num_lora_v = 0
    num_lora_g = 0
    num_lora_a = 0

    rwkvlayers = []
    gqalayers = []

    num_totallayers = 0

    for i in range(100):
        for name, shape in shapes.items():
            if f"blocks.{i}." in name:
                num_totallayers += 1
                break

    for i in range(num_totallayers):
        for name, shape in shapes.items():
            if f"blocks.{i}.q_proj" in name:
                rwkvlayers.append(i)
                break
            else:
                rwkvlayers.append(i)
                break

    if rwkvarch == "x070":
        for name, shape in shapes.items():
            if "r_k" in name:
                headsize = shape[1] #r_k[h,n]
            if "w1" in name:
                num_lora_w = shape[1]
            if "v1" in name:
                num_lora_v = shape[1]
            if "g1" in name:
                num_lora_g = shape[1]
            if "a1" in name:
                num_lora_a = shape[1]

        for name, shape in shapes.items():
            if "att.receptance.weight" in name:
                hiddensize=shape[1]
                num_attention_heads = shape[0] // headsize
            if "att.key.weight" in name:
                num_kv_heads = shape[0] // headsize

        for name, shape in shapes.items():
            if "emb" in name or "embedding" in name:
                vocabsize = shape[0]
                break

    args.my_testing = rwkvarch
    args.head_size_a = headsize
    args.num_attention_heads = num_attention_heads
    args.num_kv_heads = num_kv_heads
    args.n_layer = num_totallayers
    args.n_embd = hiddensize
    args.num_lora_w = num_lora_w
    args.num_lora_v = num_lora_v
    args.num_lora_g = num_lora_g
    args.num_lora_a = num_lora_a
    args.vocab_size = vocabsize
    print(f"auto detect result-----------------------------------")
    print(f"rwkvarch = {rwkvarch}")
    print(f"vocabsize = {vocabsize}")
    print(f"head_size_a = {headsize}")
    print(f"gqa_attention_heads = {num_attention_heads}")
    print(f"gqa_kv_heads = {num_kv_heads}")
    print(f"n_layer = {num_totallayers}")
    print(f"n_embd = {hiddensize}")
    print(f"num_lora_w = {num_lora_w}")
    print(f"num_lora_v = {num_lora_v}")
    print(f"num_lora_g = {num_lora_g}")
    print(f"num_lora_a = {num_lora_a}")
    print(f"-----------------------------------------------------")
    return args

# 使用例
if __name__ == "__main__":
    # PyTorchモデルの場合
    shapes = get_tensor_shapes("/home/client/Projects/llm/rwkv7-g0-7.2b-20250722-ctx4096.pth")
    
    # SafeTensorsファイルの場合
    # shapes = get_tensor_shapes("model.safetensors")
    
    # SafeTensorsディレクトリの場合
    # shapes = get_tensor_shapes("./model_directory/")
    
    # 結果を表示
    # print_tensor_info(shapes, max_display=20)
    
   # 辞書として直接アクセス
    for name, shape in shapes.items():
        print(f"{name}: {shape}")

    print("All tensor shapes extracted successfully.")

    pass