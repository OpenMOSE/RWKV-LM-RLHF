import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def analyze_model_differences_complete(model_path_a: str, model_path_b: str, output_dir: str = "weight_diff", y_limit: float = 0.002):
    """
    モデルの各コンポーネントごとに層別の差分を分析し、個別と統合の両方の画像を保存します
    
    Args:
        model_path_a: 1つ目のモデルのパス
        model_path_b: 2つ目のモデルのパス
        output_dir: 出力ディレクトリ
        y_limit: Y軸の絶対値の上限（デフォルト: 0.01）
    """
    # モデルの読み込み
    state_dict_a = torch.load(model_path_a, map_location='cpu')
    state_dict_b = torch.load(model_path_b, map_location='cpu')
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    plots_path = output_path / 'component_plots'
    plots_path.mkdir(exist_ok=True, parents=True)
    
    # 共通のキーを抽出
    common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    
    # コンポーネントの分類定義
    components = {
        'att_receptance': r'\.att\.receptance',
        'att_key': r'\.att\.key',
        'att_value': r'\.att\.value',
        'att_output': r'\.att\.output',
        'att_gate': r'\.att\.gate',
        'ffn_key': r'\.ffn\.key',
        'ffn_receptance': r'\.ffn\.receptance',
        'ffn_value': r'\.ffn\.value'
    }
    
    # 差分データを格納するリスト
    diff_data = []
    
    # ブロックパターンの定義
    block_pattern = re.compile(r'blocks\.(\d+)\.')
    
    for key in common_keys:
        block_match = block_pattern.search(key)
        if block_match:
            block_num = int(block_match.group(1))
            
            # コンポーネントの特定
            component = 'other'
            for comp_name, pattern in components.items():
                if re.search(pattern, key):
                    component = comp_name
                    break
            
            # 差分の計算
            weight_a = state_dict_a[key].detach().float().numpy()
            weight_b = state_dict_b[key].detach().float().numpy()
            diff = weight_a - weight_b
            
            # 統計量の計算
            diff_data.append({
                'block': block_num,
                'component': component,
                'key': key,
                'mean_diff': np.mean(diff),
                'std_diff': np.std(diff),
                'max_abs_diff': np.max(np.abs(diff)),
                'min_diff': np.min(diff),
                'max_diff': np.max(diff),
                'param_count': diff.size
            })
    
    # DataFrameを作成
    df = pd.DataFrame(diff_data)
    df = df.sort_values(['component', 'block'])
    
    # CSVとして保存
    csv_path = output_path / 'weight_differences_by_component.csv'
    df.to_csv(csv_path, index=False)
    
    def create_component_plot(plt, component_data, component, y_limit):
        """コンポーネントのプロットを作成する補助関数"""
        # メイン差分プロット
        plt.plot(component_data['block'], 
                component_data['mean_diff'],
                'b-',
                linewidth=2,
                label='Mean Difference')
        
        # 標準偏差の範囲
        plt.fill_between(component_data['block'],
                        component_data['mean_diff'] - component_data['std_diff'],
                        component_data['mean_diff'] + component_data['std_diff'],
                        alpha=0.2,
                        color='blue',
                        label='±1 Std Dev')
        
        # 最大/最小差分
        plt.plot(component_data['block'],
                component_data['max_diff'],
                'r--',
                alpha=0.7,
                label='Max Difference')
        plt.plot(component_data['block'],
                component_data['min_diff'],
                'g--',
                alpha=0.7,
                label='Min Difference')
        
        # Y軸の範囲を固定
        plt.ylim(-y_limit, y_limit)
        
        # グリッドを追加
        plt.grid(True, which='major', alpha=0.3)
        plt.grid(True, which='minor', alpha=0.1)
        
        plt.title(f'Weight Differences: {component}')
        plt.xlabel('Layer Number')
        plt.ylabel('Difference Value')
        plt.legend()
        
        # 統計情報をテキストとして追加
        stats_text = (
            f"Statistics:\n"
            f"Mean of means: {component_data['mean_diff'].mean():.6f}\n"
            f"Std of means: {component_data['mean_diff'].std():.6f}\n"
            f"Max abs diff: {component_data['max_abs_diff'].max():.6f}\n"
            f"Scale: ±{y_limit}"
        )
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Y軸のティックを設定
        major_ticks = np.linspace(-y_limit, y_limit, 11)
        minor_ticks = np.linspace(-y_limit, y_limit, 21)
        plt.gca().set_yticks(major_ticks)
        plt.gca().set_yticks(minor_ticks, minor=True)
    
    # 個別のプロットを作成
    for component in df['component'].unique():
        if component == 'other':
            continue
            
        component_data = df[df['component'] == component]
        
        plt.figure(figsize=(12, 8))
        create_component_plot(plt, component_data, component, y_limit)
        plt.tight_layout()
        plt.savefig(plots_path / f'{component}_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 統合プロットの作成
    components_list = sorted([c for c in df['component'].unique() if c != 'other'])
    n_components = len(components_list)
    n_cols = 2
    n_rows = (n_components + 1) // 2
    
    plt.figure(figsize=(20, 8 * n_rows))
    
    for i, component in enumerate(components_list, 1):
        component_data = df[df['component'] == component]
        plt.subplot(n_rows, n_cols, i)
        create_component_plot(plt, component_data, component, y_limit)
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_components_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df
# 使用例
if __name__ == "__main__":
    diffs = analyze_model_differences_complete("myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth", "myfolder/Outputs/7b-sft-code2/rwkv-2-merged.pth")
    print("分析が完了しました。'weight_diff'ディレクトリに結果が保存されています。")
 