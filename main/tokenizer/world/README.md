---
license: apache-2.0
language:
- en
- zh
- ja
- ko
- fr
- ar
- es
- pt
metrics:
- accuracy
base_model:
- BlinkDL/rwkv-7-world
pipeline_tag: text-generation
---

# rwkv7-191M-world

<!-- Provide a quick summary of what the model is/does. -->

This is RWKV-7 model under flash-linear attention format.

## Model Details


### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Bo Peng, Yu Zhang, Songlin Yang, Ruichong Zhang
- **Funded by:** RWKV Project (Under LF AI & Data Foundation)
- **Model type:** RWKV7
- **Language(s) (NLP):** English
- **License:** Apache-2.0
- **Parameter count:** 191M
- **Tokenizer:** RWKV World tokenizer
- **Vocabulary size:** 65,536

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/fla-org/flash-linear-attention ; https://github.com/BlinkDL/RWKV-LM
- **Paper:** With in Progress

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
Install `flash-linear-attention` and the latest version of `transformers` before using this model:

```bash
pip install git+https://github.com/fla-org/flash-linear-attention
pip install 'transformers>=4.48.0'
```

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
You can use this model just as any other HuggingFace models:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('fla-hub/rwkv7-191M-world', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-191M-world', trust_remote_code=True)
```

## Training Details

### Training Data

This model is trained on the World v2.8 with a total of 1.0 trillion tokens.

#### Training Hyperparameters

- **Training regime:** bfloat16, lr 8e-4 to 1e-5 "delayed" cosine decay, wd 0.1, bsz 8x30x4096 (Please contact Peng Bo for details)

## Evaluation

#### Metrics

`lambada_openai`: ppl 12.4 acc 49.1%

## FAQ
Q: safetensors metadata is none.

A: upgrade transformers to >=4.48.0: `pip install 'transformers>=4.48.0'`