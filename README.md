# tiktoken-chatml

Adding support for ChatML chat template to tiktoken tokenizers:

- Remap or remove OpenAI special tokens to support only ChatML special tokens: `<|im_start|>`, `<|im_end|>`;
- Always maintain the original vocuabulary size if possible;
- Add `apply_chat_template` method known from HF tokenizers;
- Maintain full functionality of tiktoken tokenizer.

Use for training models from scratch. For your model safety - recheck all changes before using.

### Installation
```sh
pip install tiktoken-chatml
```

### Quickstart

```python
import tiktoken_chatml

enc = tiktoken_chatml.get_encoding("cl100k_base-chatml")

output = enc.apply_chat_template(
    [
        {"role": "system", "content": "This is a system message."},
        {"role": "user", "content": "Hello!"},
    ],
    tokenize=False,
)
print(output)
```

Output:
```
<|im_start|>system
This is a system message
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
```

Setting `tokenize=True` invokes tiktoken `encoding.encode()`.

You can use this encoding as a drag and drop replacement for tiktoken. 

Supported encodings:
```python
SUPPORTED_ENCODINGS = ["o200k_base-chatml", "cl100k_base-chatml", "gpt2-chatml"]
```

The `eot_token` is now `<|im_end|>`:
```python
>> enc.eot_token == enc.encode("<|im_end|>", allowed_special="all")[0]
True
```