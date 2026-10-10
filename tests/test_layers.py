import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from llm_from_scratch.layers import FeedForward, LayerNorm, RMSNorm, SwiGLU

EMB_DIM = 8


def test_layer_norm_standardizes_the_last_dimension():
    torch.manual_seed(123)
    norm = LayerNorm(EMB_DIM)
    x = torch.randn(4, EMB_DIM) * 10 + 3

    out = norm(x)

    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        out.var(dim=-1, unbiased=False), torch.ones(4), atol=1e-4, rtol=0
    )


def test_layer_norm_applies_scale_and_shift():
    torch.manual_seed(123)
    norm = LayerNorm(EMB_DIM)
    with torch.no_grad():
        norm.scale.fill_(2.0)
        norm.shift.fill_(1.0)
    x = torch.randn(4, EMB_DIM)

    out = norm(x)

    torch.testing.assert_close(out.mean(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        out.std(dim=-1, unbiased=False), torch.full((4,), 2.0), atol=1e-4, rtol=0
    )


def test_feed_forward_accepts_explicit_hidden_dim():
    ff = FeedForward(EMB_DIM, hidden_dim=5)

    assert ff.layers[0].out_features == 5


def test_feed_forward_preserves_shape():
    torch.manual_seed(123)
    ff = FeedForward(EMB_DIM)
    x = torch.rand(2, 3, EMB_DIM)

    assert ff(x).shape == x.shape


def test_rms_norm_gives_unit_root_mean_square():
    torch.manual_seed(123)
    x = torch.randn(4, EMB_DIM) * 10 + 3

    out = RMSNorm(EMB_DIM)(x)

    torch.testing.assert_close(out.pow(2).mean(dim=-1), torch.ones(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_the_huggingface_qwen3_reference(dtype):
    torch.manual_seed(123)
    norm = RMSNorm(EMB_DIM, eps=1e-6)
    reference = Qwen3RMSNorm(EMB_DIM, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_()
        reference.weight.copy_(norm.weight)
    x = torch.randn(2, 3, EMB_DIM, dtype=dtype)

    torch.testing.assert_close(
        norm.to(dtype)(x), reference.to(dtype)(x), atol=0, rtol=0
    )


def test_swiglu_gates_the_up_projection_with_silu_of_the_gate():
    torch.manual_seed(123)
    ff = SwiGLU(EMB_DIM, hidden_dim=5)
    x = torch.randn(2, 3, EMB_DIM)

    gate = torch.nn.functional.silu(ff.gate_proj(x))
    expected = ff.down_proj(gate * ff.up_proj(x))

    assert ff(x).shape == x.shape
    torch.testing.assert_close(ff(x), expected)
