from functools import partial

import tiktoken
from tiktoken import Encoding

SUPPORTED_ENCODINGS = ["o200k_base-chatml", "cl100k_base-chatml", "gpt2-chatml"]

# Chat ML template
# More information: https://huggingface.co/docs/transformers/v4.35.2/en/chat_templating
DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def get_encoding(encoding_name: str) -> Encoding:
    """
    Special tokens remapping rules:
    - map "<|endoftext|>" to "<|im_end|>" to preseve enc.eot_token property
    - keep same vocabulary size if possible
    """

    if encoding_name not in SUPPORTED_ENCODINGS:
        print(
            f"Encoding: {encoding_name} not in chatml {SUPPORTED_ENCODINGS=}. Will default to tiktoken."
        )
        return tiktoken.get_encoding(encoding_name)

    if encoding_name == "o200k_base-chatml":
        o200k_base = tiktoken.get_encoding("o200k_base")

        # o200k_base._special_tokens
        # '<|endoftext|>': 199999,
        # '<|endofprompt|>': 200018

        enc = tiktoken.Encoding(
            name="chatml",
            pat_str=o200k_base._pat_str,
            mergeable_ranks=o200k_base._mergeable_ranks,
            special_tokens={
                "<|im_start|>": 200018,
                "<|im_end|>": 199999,
                "<|endoftext|>": 199999,
            },
        )

    elif encoding_name == "cl100k_base-chatml":
        cl100k_base = tiktoken.get_encoding("cl100k_base")

        # cl100k_base._special_tokens
        # '<|endoftext|>': 100257,
        # '<|fim_prefix|>': 100258,
        # '<|fim_middle|>': 100259,
        # '<|fim_suffix|>': 100260,
        # '<|endofprompt|>': 100276

        enc = tiktoken.Encoding(
            name="chatml",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                "<|im_start|>": 100276,
                "<|im_end|>": 100257,
                "<|endoftext|>": 100257,
            },
        )
    elif encoding_name == "gpt2-chatml":
        gpt2_base = tiktoken.get_encoding("gpt2")

        # gpt2_base._special_tokens
        # '<|endoftext|>': 50256

        enc = tiktoken.Encoding(
            name="chatml",
            pat_str=gpt2_base._pat_str,
            mergeable_ranks=gpt2_base._mergeable_ranks,
            special_tokens={
                "<|im_start|>": 100276,
                "<|im_end|>": 100257,
                "<|endoftext|>": 100257,
            },
        )

    enc.apply_chat_template = partial(apply_chat_template, enc)
    enc.default_chat_template = DEFAULT_CHAT_TEMPLATE
    return enc


def apply_chat_template(encoding, msgs, add_generation_prompt=False, tokenize=True):
    prompt = ""
    for msg in msgs:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}\n<|im_end|>\n"
    prompt = prompt.strip()
    if add_generation_prompt:
        prompt += "\n<|im_start|>assistant\n"
    if tokenize:
        return encoding.encode(prompt, allowed_special="all")
    return prompt
