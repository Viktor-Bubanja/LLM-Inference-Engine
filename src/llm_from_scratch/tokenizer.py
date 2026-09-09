import tiktoken


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)
