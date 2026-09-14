import pytest
import torch

from llm_from_scratch.attention import MultiHeadAttention

CONTEXT_LENGTH = 6
D_IN = 3
D_OUT = 4
NUM_HEADS = 2


def build_attention(dropout: float = 0.0) -> MultiHeadAttention:
    torch.manual_seed(123)
    attention = MultiHeadAttention(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LENGTH,
        dropout=dropout,
        num_heads=NUM_HEADS,
    )
    return attention.eval()


def test_single_token_attends_only_to_itself():
    attention = build_attention()
    x = torch.rand(2, 1, D_IN)

    expected = attention.out_proj(attention.W_value(x))

    torch.testing.assert_close(attention(x), expected)


@pytest.mark.parametrize("num_tokens", [1, 3, CONTEXT_LENGTH - 1])
def test_output_for_prefix_is_independent_of_later_tokens(num_tokens):
    attention = build_attention()
    x = torch.rand(1, CONTEXT_LENGTH, D_IN)

    full = attention(x)
    prefix_only = attention(x[:, :num_tokens])

    assert prefix_only.shape == (1, num_tokens, D_OUT)
    torch.testing.assert_close(full[:, :num_tokens], prefix_only)


def test_batch_elements_do_not_influence_each_other():
    attention = build_attention()
    x = torch.rand(3, CONTEXT_LENGTH, D_IN)

    batched = attention(x)
    single = attention(x[1:2])

    torch.testing.assert_close(batched[1:2], single)


def test_dropout_is_applied_only_in_training_mode():
    attention = build_attention(dropout=0.5)
    x = torch.rand(1, CONTEXT_LENGTH, D_IN)

    torch.testing.assert_close(attention(x), attention(x))

    attention.train()
    assert not torch.allclose(attention(x), attention(x))


def test_rejects_more_tokens_than_context_length():
    attention = build_attention()

    with pytest.raises(ValueError, match="more tokens than context_length"):
        attention(torch.rand(1, CONTEXT_LENGTH + 1, D_IN))
