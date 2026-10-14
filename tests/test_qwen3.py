"""Our Qwen3 against the transformers implementation, on a tiny random model so
the test needs no download.
"""

import pytest
import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM

from llm_from_scratch.hf_loading import load_hf_state_dict
from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model

TINY_HF_CONFIG = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
)


def build_hf_model(tie_embeddings: bool = True) -> Qwen3ForCausalLM:
    torch.manual_seed(123)
    config = HFQwen3Config(**TINY_HF_CONFIG, tie_word_embeddings=tie_embeddings)
    model = Qwen3ForCausalLM(config)
    # Norm gains start at 1, which would hide a mixed-up norm mapping.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return model.eval()


def build_from_hf(hf_model: Qwen3ForCausalLM) -> Qwen3Model:
    model = Qwen3Model(Qwen3Config.from_hf_config(hf_model.config.to_dict()))
    load_hf_state_dict(model, hf_model.state_dict())
    return model.eval()


def build_model() -> Qwen3Model:
    return build_from_hf(build_hf_model())


def token_ids(batch_size: int = 2, num_tokens: int = 10) -> torch.Tensor:
    torch.manual_seed(321)
    return torch.randint(TINY_HF_CONFIG["vocab_size"], (batch_size, num_tokens))


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_logits_match_huggingface(tie_embeddings):
    hf_model = build_hf_model(tie_embeddings)
    model = build_from_hf(hf_model)
    ids = token_ids()

    with torch.no_grad():
        expected = hf_model(ids).logits

    torch.testing.assert_close(model(ids), expected, atol=1e-5, rtol=1e-5)


def test_loading_rejects_an_incomplete_checkpoint():
    hf_model = build_hf_model()
    state = hf_model.state_dict()
    del state["model.layers.1.self_attn.k_norm.weight"]
    model = Qwen3Model(Qwen3Config.from_hf_config(hf_model.config.to_dict()))

    with pytest.raises(ValueError, match="trf_blocks.1.att.k_norm.weight"):
        load_hf_state_dict(model, state)


def test_without_hooks_the_output_is_bit_identical_to_the_plain_forward_pass():
    model = build_model()
    ids = token_ids()

    positions = torch.arange(ids.shape[1]).expand_as(ids)
    cos, sin = model.rope(positions)
    x = model.tok_emb(ids)
    for block in model.trf_blocks:
        x = block(x, cos, sin)
    expected = model.out_head(model.final_norm(x))

    assert torch.equal(model(ids), expected)


def test_hooks_see_every_layer_with_its_positions():
    model = build_model()
    ids = token_ids()
    positions = torch.arange(5, 5 + ids.shape[1]).expand_as(ids)
    seen = []

    def record(layer: int, x: torch.Tensor, hook_positions: torch.Tensor):
        seen.append((layer, x.shape, hook_positions))
        return x

    model.residual_hooks.append(record)
    model(ids, positions=positions)

    assert [layer for layer, _, _ in seen] == [0, 1]
    for _, shape, hook_positions in seen:
        assert shape == (*ids.shape, TINY_HF_CONFIG["hidden_size"])
        assert torch.equal(hook_positions, positions)
