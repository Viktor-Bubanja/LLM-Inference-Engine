from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import tiktoken
import tokenizers
import torch


class Tokenizer(Protocol):
    """What model-agnostic code needs from a tokenizer."""

    @property
    def eos_id(self) -> int: ...

    @property
    def vocab_size(self) -> int:
        """Tokens the tokenizer can produce. A model's embedding table may be
        larger, padded for efficiency."""
        ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class TiktokenTokenizer:
    """GPT-2's byte-pair encoding, through tiktoken."""

    def __init__(self, encoding_name: str = "gpt2"):
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def eos_id(self) -> int:
        return self._encoding.eot_token

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._encoding.decode(list(token_ids))


class HFTokenizer:
    """A HuggingFace ``tokenizer.json``, through the ``tokenizers`` library.

    Encoding adds no special tokens: base models such as Qwen3 expect raw text.
    """

    def __init__(self, tokenizer_json: Path, eos_id: int):
        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_json))
        self._eos_id = eos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def text_to_token_ids(
    text: str, tokenizer: Tokenizer | tiktoken.Encoding
) -> torch.Tensor:
    """Encode ``text`` as a batch of one sequence.
    str -> (1, num_tokens)
    """
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(
    token_ids: torch.Tensor, tokenizer: Tokenizer | tiktoken.Encoding
) -> str:
    """Decode a batch of one sequence back to text.
    (1, num_tokens) -> str
    """
    return tokenizer.decode(token_ids.squeeze(0).tolist())
