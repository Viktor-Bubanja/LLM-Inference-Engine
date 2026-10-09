import tiktoken
import torch


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Encode ``text`` as a batch of one sequence.
    str -> (1, num_tokens)
    """
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Decode a batch of one sequence back to text.
    (1, num_tokens) -> str
    """
    return tokenizer.decode(token_ids.squeeze(0).tolist())
