import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.tokenizer import HFTokenizer


@pytest.fixture(scope="session")
def byte_chat_format(tmp_path_factory) -> ChatFormat:
    """A ``ChatFormat`` over a byte-level tokenizer with no merges and Qwen3's
    chat special tokens, so chat code can be tested without downloads.
    """
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    tokenizer = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(sorted(alphabet))}, merges=[])
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tokenizer.add_tokens(["<think>", "</think>"])
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return ChatFormat(HFTokenizer(path, eos_id=tokenizer.token_to_id("<|im_end|>")))
