---
language:
- en
- fr
- de
- es
- it
- pt
- zh
- ja
- ru
- ko
license: apache-2.0
library_name: vllm
inference: false
base_model:
- mistralai/Mistral-Small-24B-Base-2501
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
tags:
- transformers
---

# Model Card for Mistral-Small-24B-Instruct-2501

Mistral Small 3 ( 2501 ) sets a new benchmark in the "small" Large Language Models category below 70B, boasting 24B parameters and achieving state-of-the-art capabilities comparable to larger models!  
This model is an instruction-fine-tuned version of the base model: [Mistral-Small-24B-Base-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501).

Mistral Small can be deployed locally and is exceptionally "knowledge-dense", fitting in a single RTX 4090 or a 32GB RAM MacBook once quantized.  
Perfect for:
- Fast response conversational agents.
- Low latency function calling.
- Subject matter experts via fine-tuning.
- Local inference for hobbyists and organizations handling sensitive data.

For enterprises that need specialized capabilities (increased context, particular modalities, domain specific knowledge, etc.), we will be releasing commercial models beyond what Mistral AI contributes to the community.

This release demonstrates our commitment to open source, serving as a strong base model. 

Learn more about Mistral Small in our [blog post](https://mistral.ai/news/mistral-small-3/).

Model developper: Mistral AI Team

## Key Features
- **Multilingual:** Supports dozens of languages, including English, French, German, Spanish, Italian, Chinese, Japanese, Korean, Portuguese, Dutch, and Polish.
- **Agent-Centric:** Offers best-in-class agentic capabilities with native function calling and JSON outputting.
- **Advanced Reasoning:** State-of-the-art conversational and reasoning capabilities.
- **Apache 2.0 License:** Open license allowing usage and modification for both commercial and non-commercial purposes.
- **Context Window:** A 32k context window.
- **System Prompt:** Maintains strong adherence and support for system prompts.
- **Tokenizer:** Utilizes a Tekken tokenizer with a 131k vocabulary size.

## Benchmark results


### Human evaluated benchmarks

| Category | Gemma-2-27B | Qwen-2.5-32B | Llama-3.3-70B | Gpt4o-mini |
|----------|-------------|--------------|---------------|------------|
| Mistral is better | 0.536 | 0.496 | 0.192 | 0.200 |
| Mistral is slightly better | 0.196 | 0.184 | 0.164 | 0.204 |
| Ties | 0.052 | 0.060 | 0.236 | 0.160 |
| Other is slightly better | 0.060 | 0.088 | 0.112 | 0.124 |
| Other is better | 0.156 | 0.172 | 0.296 | 0.312 |

**Note**:

- We conducted side by side evaluations with an external third-party vendor, on a set of over 1k proprietary coding and generalist prompts.
- Evaluators were tasked with selecting their preferred model response from anonymized generations produced by Mistral Small 3 vs another model.
- We are aware that in some cases the benchmarks on human judgement starkly differ from publicly available benchmarks, but have taken extra caution in verifying a fair evaluation. We are confident that the above benchmarks are valid.

### Publicly accesible benchmarks

**Reasoning & Knowledge**

| Evaluation | mistral-small-24B-instruct-2501 | gemma-2b-27b | llama-3.3-70b | qwen2.5-32b | gpt-4o-mini-2024-07-18 |
|------------|---------------|--------------|---------------|---------------|-------------|
| mmlu_pro_5shot_cot_instruct | 0.663 | 0.536 | 0.666 | 0.683 | 0.617 |
| gpqa_main_cot_5shot_instruct | 0.453 | 0.344 | 0.531 | 0.404 | 0.377 |

**Math & Coding**

| Evaluation | mistral-small-24B-instruct-2501 | gemma-2b-27b | llama-3.3-70b | qwen2.5-32b | gpt-4o-mini-2024-07-18 |
|------------|---------------|--------------|---------------|---------------|-------------|
| humaneval_instruct_pass@1 | 0.848 | 0.732 | 0.854 | 0.909 | 0.890 |
| math_instruct | 0.706 | 0.535 | 0.743 | 0.819 | 0.761 |

**Instruction following**

| Evaluation | mistral-small-24B-instruct-2501 | gemma-2b-27b | llama-3.3-70b | qwen2.5-32b | gpt-4o-mini-2024-07-18 |
|------------|---------------|--------------|---------------|---------------|-------------|
| mtbench_dev | 8.35 | 7.86 | 7.96 | 8.26 | 8.33 |
| wildbench | 52.27 | 48.21 | 50.04 | 52.73 | 56.13 |
| arena_hard | 0.873 | 0.788 | 0.840 | 0.860 | 0.897 |
| ifeval | 0.829 | 0.8065 | 0.8835 | 0.8401 | 0.8499 |

**Note**:

- Performance accuracy on all benchmarks were obtained through the same internal evaluation pipeline - as such, numbers may vary slightly from previously reported performance
([Qwen2.5-32B-Instruct](https://qwenlm.github.io/blog/qwen2.5/), [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [Gemma-2-27B-IT](https://huggingface.co/google/gemma-2-27b-it)). 
- Judge based evals such as Wildbench, Arena hard and MTBench were based on gpt-4o-2024-05-13.

### Basic Instruct Template (V7-Tekken)

```
<s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
```
*`<system_prompt>`, `<user message>` and `<assistant response>` are placeholders.*

***Please make sure to use [mistral-common](https://github.com/mistralai/mistral-common) as the source of truth***

## Usage

The model can be used with the following frameworks;
- [`vllm`](https://github.com/vllm-project/vllm): See [here](#vllm)
- [`transformers`](https://github.com/huggingface/transformers): See [here](#transformers)

### vLLM

We recommend using this model with the [vLLM library](https://github.com/vllm-project/vllm)
to implement production-ready inference pipelines.

**Note 1**: We recommond using a relatively low temperature, such as `temperature=0.15`.

**Note 2**: Make sure to add a system prompt to the model to best tailer it for your needs. If you want to use the model as a general assistant, we recommend the following 
system prompt:

```
system_prompt = """You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
Your knowledge base was last updated on 2023-10-01. The current date is 2025-01-30.
When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel from?\")"""
```

**_Installation_**

Make sure you install [`vLLM >= 0.6.4`](https://github.com/vllm-project/vllm/releases/tag/v0.6.4):

```
pip install --upgrade vllm
```

Also make sure you have [`mistral_common >= 1.5.2`](https://github.com/mistralai/mistral-common/releases/tag/v1.5.2) installed:

```
pip install --upgrade mistral_common
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/latest/images/sha256-de9032a92ffea7b5c007dad80b38fd44aac11eddc31c435f8e52f3b7404bbf39).

#### Server

We recommand that you use Mistral-Small-24B-Instruct-2501 in a server/client setting. 

1. Spin up a server:

```
vllm serve mistralai/Mistral-Small-24B-Instruct-2501 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice
```

**Note:** Running Mistral-Small-24B-Instruct-2501 on GPU requires ~55 GB of GPU RAM in bf16 or fp16. 


2. To ping the client you can use a simple Python snippet.

```py
import requests
import json
from datetime import datetime, timedelta

url = "http://<your-server>:8000/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

model = "mistralai/Mistral-Small-24B-Instruct-2501"

messages = [
    {
        "role": "system",
        "content": "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."
    },
    {
        "role": "user",
        "content": "Give me 5 non-formal ways to say 'See you later' in French."
    },
]

data = {"model": model, "messages": messages}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json()["choices"][0]["message"]["content"])

# Sure, here are five non-formal ways to say "See you later" in French:
#
# 1. À plus tard
# 2. À plus
# 3. Salut
# 4. À toute
# 5. Bisous
#
# ```
#  /\_/\
# ( o.o )
#  > ^ <
# ```
```

### Function calling

Mistral-Small-24-Instruct-2501 is excellent at function / tool calling tasks via vLLM. *E.g.:*

<details>
  <summary>Example</summary>

```py
import requests
import json
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta

url = "http://<your-url>:8000/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

model = "mistralai/Mistral-Small-24B-Instruct-2501"


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "The state abbreviation, e.g. 'CA' for California",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit for temperature",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite",
            "description": "Rewrite a given text for improved clarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The input text to rewrite",
                    }
                },
            },
        },
    },
]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Could you please make the below article more concise?\n\nOpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership.",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "bbc5b7ede",
                "type": "function",
                "function": {
                    "name": "rewrite",
                    "arguments": '{"text": "OpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership."}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"action":"rewrite","outcome":"OpenAI is a FOR-profit company."}',
        "tool_call_id": "bbc5b7ede",
        "name": "rewrite",
    },
    {
        "role": "assistant",
        "content": "---\n\nOpenAI is a FOR-profit company.",
    },
    {
        "role": "user",
        "content": "Can you tell me what the temperature will be in Dallas, in Fahrenheit?",
    },
]

data = {"model": model, "messages": messages, "tools": tools}

response = requests.post(url, headers=headers, data=json.dumps(data))
import ipdb; ipdb.set_trace()
print(response.json()["choices"][0]["message"]["tool_calls"])
# [{'id': '8PdihwL6d', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': '{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}'}}]
```

</details>

#### Offline

```py
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datetime import datetime, timedelta

SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."

user_prompt = "Give me 5 non-formal ways to say 'See you later' in French."

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": user_prompt
    },
]

# note that running this model on GPU requires over 60 GB of GPU RAM
llm = LLM(model=model_name, tokenizer_mode="mistral", tensor_parallel_size=8)

sampling_params = SamplingParams(max_tokens=512, temperature=0.15)
outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
# Sure, here are five non-formal ways to say "See you later" in French:
#
# 1. À plus tard
# 2. À plus
# 3. Salut
# 4. À toute
# 5. Bisous
#
# ```
#  /\_/\
# ( o.o )
#  > ^ <
# ```
```

### Transformers

If you want to use Hugging Face transformers to generate text, you can do something like this.

```py
from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Give me 5 non-formal ways to say 'See you later' in French."},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-Small-24B-Instruct-2501", max_new_tokens=256, torch_dtype=torch.bfloat16)
chatbot(messages)
```


### Ollama

[Ollama](https://github.com/ollama/ollama) can run this model locally on MacOS, Windows and Linux. 

```
ollama run mistral-small
```

4-bit quantization (aliased to default): 
```
ollama run mistral-small:24b-instruct-2501-q4_K_M
```

8-bit quantization:
```
ollama run mistral-small:24b-instruct-2501-q8_0
```

FP16:
```
ollama run mistral-small:24b-instruct-2501-fp16
```