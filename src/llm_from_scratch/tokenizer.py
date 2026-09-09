import tiktoken


def get_tokenizer(encoding_name: str = "gpt2"):
    return tiktoken.get_encoding(encoding_name)


def encode(text: str, encoding_name: str = "gpt2") -> list[int]:
    return get_tokenizer(encoding_name).encode(text)
