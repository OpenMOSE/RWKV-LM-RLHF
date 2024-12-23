import torch
import gc
import os
from typing import Dict
import logging
from tqdm import tqdm
import psutil
import time

class BFloat16ModelMerger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def log_memory_usage(self):
        """メモリ使用状況をログ出力"""
        process = psutil.Process(os.getpid())
        self.logger.info(f"メモリ使用量: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    def clear_memory(self, *tensors):
        """テンソルのメモリ解放"""
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        gc.collect()

    def merge_pth_models(self,
                        jpn_path: str,
                        v3_path: str,
                        base_v2_path: str,
                        output_path: str,
                        jpn_ratio: float = 0.3,
                        v3_ratio: float = 0.7,
                        chunk_size: int = 100) -> None:
        """
        メモリ効率を重視したモデルマージ
        chunk_size: 一度に処理するパラメータの数
        """
        try:
            self.logger.info("Base modelを読み込み中...")
            base_keys = torch.load(base_v2_path, map_location='cpu', mmap=True).keys()
            total_keys = len(base_keys)
            base_keys = list(base_keys)

            # 出力ディレクトリの作成
            output_dir = f"{output_path}_chunks"
            os.makedirs(output_dir, exist_ok=True)

            # チャンク単位で処理
            for chunk_start in tqdm(range(0, total_keys, chunk_size), desc="Processing chunks"):
                chunk_end = min(chunk_start + chunk_size, total_keys)
                chunk_keys = base_keys[chunk_start:chunk_end]
                
                # 各モデルから必要なパラメータのみ読み込み
                self.logger.info(f"Chunk {chunk_start}-{chunk_end}の処理中...")
                
                base_chunk = {}
                jpn_chunk = {}
                v3_chunk = {}
                merged_chunk = {}

                # Base modelのチャンク読み込み
                base_state = torch.load(base_v2_path, map_location='cpu', mmap=True)
                for key in chunk_keys:
                    if key in base_state:
                        base_chunk[key] = base_state[key]
                self.clear_memory(base_state)

                # JPN modelのチャンク読み込み
                jpn_state = torch.load(jpn_path, map_location='cpu', mmap=True)
                for key in chunk_keys:
                    if key in jpn_state:
                        jpn_chunk[key] = jpn_state[key]
                self.clear_memory(jpn_state)

                # V3 modelのチャンク読み込み
                v3_state = torch.load(v3_path, map_location='cpu', mmap=True)
                for key in chunk_keys:
                    if key in v3_state:
                        v3_chunk[key] = v3_state[key]
                self.clear_memory(v3_state)

                # チャンク内の各パラメータをマージ
                for key in chunk_keys:
                    if key in base_chunk and key in jpn_chunk and key in v3_chunk:
                        # Float32に変換して計算
                        base_float32 = base_chunk[key].to(torch.float32).detach()
                        jpn_float32 = jpn_chunk[key].to(torch.float32).detach()
                        v3_float32 = v3_chunk[key].to(torch.float32).detach()

                        # 差分計算
                        jpn_diff = jpn_float32 - base_float32
                        v3_diff = v3_float32 - base_float32

                        # マージ計算
                        merged_tensor = (
                            base_float32 +
                            (jpn_diff * jpn_ratio) +
                            (v3_diff * v3_ratio)
                        )

                        # BFLOAT16に変換して保存
                        merged_chunk[key] = merged_tensor.to(torch.bfloat16)

                        # 中間テンソルのメモリ解放
                        self.clear_memory(
                            base_float32, jpn_float32, v3_float32,
                            jpn_diff, v3_diff, merged_tensor
                        )

                # チャンクの保存
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_start}.pt")
                torch.save(merged_chunk, chunk_file)

                # チャンクデータのメモリ解放
                self.clear_memory(base_chunk, jpn_chunk, v3_chunk, merged_chunk)
                self.log_memory_usage()

            # 全チャンクの結合
            self.logger.info("チャンクを結合中...")
            final_state = {}
            chunk_files = sorted(os.listdir(output_dir), 
                               key=lambda x: int(x.split('_')[1].split('.')[0]))

            for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
                chunk_path = os.path.join(output_dir, chunk_file)
                chunk_data = torch.load(chunk_path)
                final_state.update(chunk_data)
                self.clear_memory(chunk_data)
                os.remove(chunk_path)  # チャンクファイルの削除

            # 最終モデルの保存
            self.logger.info(f"最終モデルを保存中: {output_path}")
            torch.save(final_state, output_path)
            self.clear_memory(final_state)

            # 一時ディレクトリの削除
            os.rmdir(output_dir)
            
        except Exception as e:
            self.logger.error(f"マージ処理中にエラー発生: {e}")
            # クリーンアップ
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
            raise
        finally:
            gc.collect()

    def verify_merge_result(self, 
                          merged_path: str,
                          sample_size: int = 5,
                          cleanup: bool = True):
        """
        マージ結果の検証（メモリ効率重視）
        """
        try:
            self.logger.info("マージ結果を検証中...")
            state = torch.load(merged_path, map_location='cpu')
            
            # サンプリングしたキーの検証
            keys = list(state.keys())[:sample_size]
            
            for key in keys:
                tensor = state[key]
                self.logger.info(f"Layer {key}:")
                self.logger.info(f"  データ型: {tensor.dtype}")
                self.logger.info(f"  シェイプ: {tensor.shape}")
                
                # テンソルのメモリ解放
                if cleanup:
                    self.clear_memory(tensor)
            
            if cleanup:
                self.clear_memory(state)
                
        except Exception as e:
            self.logger.error(f"検証中にエラー発生: {e}")
            raise

def main():
    merger = BFloat16ModelMerger()
    
    base_path = "myfolder/models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth"
    jpn_path = "myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth"
    v3_path = "myfolder/models/rwkv-x060-7b-world-v3-80_trained-20241025-ctx4k.pth"
    output_path = "myfolder/models/rwkv-x060-chimera.pth"
    
    # チャンクサイズはメモリ容量に応じて調整
    merger.merge_pth_models(
        jpn_path=jpn_path,
        v3_path=v3_path,
        base_v2_path=base_path,
        output_path=output_path,
        chunk_size=100  # メモリ使用量に応じて調整
    )
    
    # 結果の検証（cleanup=Trueで検証後にメモリ解放）
    merger.verify_merge_result(output_path, cleanup=True)

if __name__ == "__main__":
    main()