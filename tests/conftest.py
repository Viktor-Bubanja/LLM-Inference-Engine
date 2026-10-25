import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model
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


@pytest.fixture
def tiny_chat_model(byte_chat_format) -> Qwen3Model:
    """A tiny model whose replies depend on the context. With tied, unit-scale
    embeddings it would repeat one token whatever came before, and every
    equivalence check would pass trivially.
    """
    torch.manual_seed(123)
    config = Qwen3Config(
        vocab_size=byte_chat_format.tokenizer.vocab_size,
        context_length=512,
        emb_dim=32,
        hidden_dim=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        tie_embeddings=False,
    )
    model = Qwen3Model(config).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(0.1)
    return model
