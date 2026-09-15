import torch

from llm_from_scratch.layers import FeedForward, LayerNorm

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
