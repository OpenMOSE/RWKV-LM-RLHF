from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import numpy as np
import gc

# グローバル変数の定義
Model = None
Tokenizer = None
Config = None

app = FastAPI(title="Transformer KL Distillation API")

# リクエストとレスポンスのモデル定義
class LoadModelRequest(BaseModel):
    modelname: str
    use_4bit: Optional[bool] = True
    use_cuda: Optional[bool] = True

class ProcessLogitsRequest(BaseModel):
    input_ids: List[List[int]]
    topk: Optional[int] = 2000

class ProcessLogitsResponse(BaseModel):
    indices: List[List[List[int]]]
    logits: List[List[List[float]]]
    loss: float

@app.post("/LoadModel")
async def load_model(request: LoadModelRequest):
    """
    Huggingface TransformerからモデルをロードするエンドポイントCUDA+4bit量子化対応
    """
    global Model, Tokenizer, Config, Device

    Model = None
    Tokenizer = None
    Config = None

    gc.collect
    
    try:
        # 使用するデバイスを設定（CUDAまたはCPU）
        if request.use_cuda and torch.cuda.is_available():
            Device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            Device = torch.device("cpu")
            print("Using CPU")
            
        #request.use_4bit = False
        # 4bit量子化の設定
        if request.use_4bit:
            print("Loading model with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Huggingfaceからモデルとトークナイザーをロード（4bit量子化）
            Model = AutoModelForCausalLM.from_pretrained(
                request.modelname,
                device_map="auto",  # デバイスマッピングを自動的に決定
                quantization_config=quantization_config,
                trust_remote_code=True
            )

            Model = torch.compile(Model)
        else:
            print("Loading model without quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8"
            )
            # 通常のロード（量子化なし）
            Model = AutoModelForCausalLM.from_pretrained(
                request.modelname,
                device_map="auto" if request.use_cuda and torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if request.use_cuda and torch.cuda.is_available() else None,
                quantization_config=quantization_config,
                trust_remote_code=True
            )

            Model = torch.compile(Model)
        
        # トークナイザーをロード
        Tokenizer = AutoTokenizer.from_pretrained(request.modelname)
        Config = Model.config
        
        # モデルを評価モードに設定
        Model.eval()

        # KVCacheを無効化する設定
        if hasattr(Model, "config"):
            if hasattr(Model.config, "use_cache"):
                Model.config.use_cache = False
                print("KV cache disabled")
        
        return {
            "status": "success", 
            "message": f"Model {request.modelname} loaded successfully",
            "device": str(Device),
            "quantized": request.use_4bit,
            "model_type": Config.model_type,
            "vocab_size": Config.vocab_size,
            "hidden_size": Config.hidden_size if hasattr(Config, "hidden_size") else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/ProcessLogits", response_model=ProcessLogitsResponse)
async def process_logits(request: ProcessLogitsRequest):
    """
    入力トークンを処理し、ロジット、インデックス、損失を返すエンドポイント
    """
    global Model, Config, Device
    
    if Model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /LoadModel endpoint first.")
    
    try:
        with torch.no_grad():
            # リクエストから入力IDを取得
            input_ids_list = request.input_ids
            topk = request.topk
            
            # 最大シーケンス長を見つける
            max_length = max(len(seq) for seq in input_ids_list)
            
            # モデル設定からpad_token_idを取得
            pad_token_id = Config.pad_token_id
            if pad_token_id is None:
                # pad_token_idが定義されていない場合、フォールバックとしてeos_token_idを使用
                pad_token_id = Config.eos_token_id
            
            # パディングされた入力テンソルとアテンションマスクを作成
            padded_input_ids = []
            attention_masks = []
            
            for seq in input_ids_list:
                # シーケンスをmax_lengthにパディング
                padding_length = max_length - len(seq)
                padded_seq = seq + [pad_token_id] * padding_length
                padded_input_ids.append(padded_seq)
                
                # アテンションマスクを作成（トークンは1、パディングは0）
                mask = [1] * len(seq) + [0] * padding_length
                attention_masks.append(mask)
            
            # テンソルに変換
            input_ids_tensor = torch.tensor(padded_input_ids)
            attention_mask_tensor = torch.tensor(attention_masks)
            
            # ラベルを作成（言語モデリングのため、入力の1つ右にシフト）
            # [t₁, t₂, t₃, t₄] -> ラベル: [t₂, t₃, t₄, pad]
            labels = torch.full((input_ids_tensor.shape[0], input_ids_tensor.shape[1]), -100)  # デフォルトですべてマスク
            
            for i in range(len(input_ids_list)):
                seq_len = len(input_ids_list[i])
                # 実際のシーケンス長から1つ少ない位置までをコピー
                if seq_len > 1:  # シーケンスが少なくとも2つのトークンを持っている場合
                    # 入力の2番目から最後までのトークンをラベルの最初からコピー
                    labels[i, :seq_len-1] = input_ids_tensor[i, 1:seq_len]
            
            # モデルを通じてフォワードパス
            
            outputs = Model(
                    input_ids=input_ids_tensor.to(device='cuda'), 
                    attention_mask=attention_mask_tensor.to(device='cuda'), 
                    labels=labels.to(device='cuda'),
                    use_cache=False  # KVキャッシュを明示的に無効化
                )
            
            # ロジットと損失を取得
            logits = outputs.logits  # 形状: [batch_size, seq_length, vocab_size]
            loss = outputs.loss.item()
            
            # トップkのインデックスとそれに対応するロジットを取得
            topk_logits, topk_indices = torch.topk(logits, k=min(topk, logits.size(-1)), dim=-1)
            
            # CPUに移動してからJSONシリアライズのためにリストに変換
            indices_list = topk_indices.cpu().tolist()
            logits_list = topk_logits.cpu().tolist()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return ProcessLogitsResponse(
                indices=indices_list,
                logits=logits_list,
                loss=loss
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing logits: {str(e)}")
    
# サーバー起動用コード
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")