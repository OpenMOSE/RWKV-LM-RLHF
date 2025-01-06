import torch

with torch.no_grad():
# モデルの読み込み
    state_dict = torch.load('myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth',map_location=torch.device('cpu'))

# 保持したいキーパターンのリスト
    keep_patterns = [
        'emb',
        'head',
        'ln',
        'att.receptance',
        'att.key',
        'att.value',
        'att.output',
        'ffn.key',
        'ffn.value'
    ]
    
    # 保持するキーを格納する新しいstate_dict
    new_state_dict = {}
    
    # state_dictの各キーをチェック
    for key in state_dict.keys():
        print(key)
        # いずれかのパターンがキーに含まれているかチェック
        if any(pattern in key for pattern in keep_patterns):
            new_state_dict[key] = state_dict[key].clone()
    
    # 抽出したキーの確認（オプション）
    print("保持されるキー:")
    for key in new_state_dict.keys():
        print(key)
# 新しいモデルとして保存
torch.save(new_state_dict, 'myfolder/converted/x060-deleted.pth')