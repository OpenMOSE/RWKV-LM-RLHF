---
license: apache-2.0
---
# Reka Flash 3

Reka Flash 3 is a 21B general-purpose reasoning model that was trained from scratch. It was trained in synthetic and public datasets for supervised finetuning, followed by RLOO with model-based and rule-based rewards. It performs competitively with proprietary models such as OpenAI o1-mini, making it a good foundation to build applications that require low latency or on-device deployment. It is currently the best open model in its size category.

Try it out at [Reka Space](https://space.reka.ai).

Reka Flash 3 powers Nexus, our new platform for organizations to create and manage AI workers. Nexus workers have native deep research capabilities and can browse the web, execute code, and analyse internal files including documents, images, videos and audio. Learn more about Nexus at [getnexus.reka.ai](https://getnexus.reka.ai).

![Performance](./eval-new.png)

## Quickstart

For ease of deployment, the model is released in a Llama-compatible format. You may use any library compatible with Llama to run the model.

### Via Hugging Face

```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("RekaAI/reka-flash-3")
model = transformers.AutoModelForCausalLM.from_pretrained("RekaAI/reka-flash-3", torch_dtype='auto', device_map='auto')

prompt = {"role": "human", "content": "Write a poem about large language model."}
text = tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**model_inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Via vLLM

```bash
docker run --rm -it --network=host --gpus '"device=0"' -v  --shm-size=10.24gb vllm/vllm-openai:latest serve RekaAI/reka-flash-3 --dtype auto -tp 1
```

## Model Details

### Prompt Format

Reka Flash 3 uses cl100k_base tokenizer and adds no additional special tokens. Its prompt format is as follows:

```
human: this is round 1 prompt <sep> assistant: this is round 1 response <sep> ...
```

Generation should stop on seeing the string `<sep>` or seeing the special token `<|endoftext|>`.

System prompt can be added by prepending to the first user round.

```
human: You are a friendly assistant blah ... this is round 1 user prompt <sep> assistant: this is round 1 response <sep> ...
```

For multi-round conversations, it is recommended to drop the reasoning traces in the previous assistant round to save tokens for the model to think.

If you are using HF or vLLM, the built-in chat_template shall handle prompt formatting automatically.

### Budget Forcing

Reka Flash thinks before it produces an output. We use <reasoning> </reasoning> tags to indicate the beginning and the end of its thinking process. For some problems, the model might think for a long time. You can make the model to stop its thinking process by forcing it to output </reasoning> after a certain number of steps. We observe such a budget forcing mechanism will still produce a reasonable output. We show performance on AIME-2024 (cons@16) for various budgets below.

![AIME'24](./aime.png)


### Language Support

This model is primarily built for the English language, and you should consider this an English only model. However, the model is able to converse and understand other languages to some degree.


### Release Notes

- As a smaller model, it is not the best model for knowledge-intensive tasks. We recommend coupling Reka Flash 3 with web search for knowledge-related tasks.
- The model often thinks in English when prompted with questions in non-English languages. We observe that this sometimes affects the output quality in non-English languages.
- The model has not undergone extensive alignment or persona training.