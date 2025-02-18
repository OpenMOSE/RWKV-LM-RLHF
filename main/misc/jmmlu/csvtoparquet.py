import os
import pandas as pd
from pathlib import Path

def convert_csvs_to_parquet():
    # testフォルダのパスを設定
    test_dir = Path('test')
    
    # 結果を格納するリスト
    all_data = []
    
    # testフォルダ内のすべてのCSVファイルを処理
    for csv_file in test_dir.glob('*.csv'):
        try:
            # ファイル名からsubjectを抽出（ディレクトリと拡張子を除く）
            subject = csv_file.stem
            
            # CSVファイルを読み込む
            df = pd.read_csv(csv_file)
            
            # 必要なカラムが存在することを確認
            required_columns = ['question', 'A', 'B', 'C', 'D', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: {csv_file} is missing columns: {missing_columns}")
                continue
            
            # subjectカラムを追加
            df['subject'] = subject
            
            # カラムの順序を指定
            df = df[['subject', 'question', 'A', 'B', 'C', 'D', 'answer']]
            
            # データフレームをリストに追加
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    if not all_data:
        print("No valid CSV files found")
        return
    
    # すべてのデータフレームを結合
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Parquetファイルとして保存
    output_path = Path('output.parquet')
    final_df.to_parquet(output_path, index=False)
    print(f"Successfully saved data to {output_path}")

if __name__ == "__main__":
    convert_csvs_to_parquet()