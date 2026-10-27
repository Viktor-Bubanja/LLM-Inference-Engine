from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import tokenizers


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

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids, and each token's ``(start, end)`` character span in
        ``text``.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"{token!r} is not in the vocabulary")
        return token_id
