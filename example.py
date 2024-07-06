import tiktoken_chatml

enc = tiktoken_chatml.get_encoding("cl100k_base-chatml")

out = enc.apply_chat_template(
    [
        {"role": "system", "content": "This is a system message."},
        {"role": "user", "content": "Hello!"},
    ],
    tokenize=False,
)

print(out)
print(enc.eot_token)
print(enc.eot_token == enc.encode("<|im_end|>", allowed_special="all")[0])
