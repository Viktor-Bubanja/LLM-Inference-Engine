import torch

from llm_from_scratch.attention import GroupedQueryAttention
from llm_from_scratch.rope import RotaryEmbedding

D_IN = 32
NUM_HEADS = 4
NUM_KV_HEADS = 2
# NUM_HEADS * HEAD_DIM != D_IN, as in Qwen3-0.6B (16 * 128 vs 1024).
HEAD_DIM = 16
NUM_TOKENS = 6


def build_attention(num_kv_heads: int = NUM_KV_HEADS) -> GroupedQueryAttention:
    torch.manual_seed(123)
    attention = GroupedQueryAttention(
        d_in=D_IN, num_heads=NUM_HEADS, num_kv_heads=num_kv_heads, head_dim=HEAD_DIM
    )
    assert attention.q_norm is not None and attention.k_norm is not None
    # Norm gains start at 1; perturb them so the tests can tell q from k.
    with torch.no_grad():
        attention.q_norm.weight.normal_(1.0, 0.2)
        attention.k_norm.weight.normal_(1.0, 0.2)
    return attention.eval()


def run(attention: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    batch_size, num_tokens, _ = x.shape
    positions = torch.arange(num_tokens).expand(batch_size, num_tokens)
    return attention(x, *RotaryEmbedding(HEAD_DIM, 10_000.0)(positions))


def test_output_for_prefix_is_independent_of_later_tokens():
    attention = build_attention()
    x = torch.randn(1, NUM_TOKENS, D_IN)

    torch.testing.assert_close(run(attention, x)[:, :3], run(attention, x[:, :3]))


def test_grouped_heads_equal_multi_head_attention_with_repeated_key_value_heads():
    gqa = build_attention(num_kv_heads=NUM_KV_HEADS)
    mha = build_attention(num_kv_heads=NUM_HEADS)
    group_size = NUM_HEADS // NUM_KV_HEADS

    def repeat_heads(weight: torch.Tensor) -> torch.Tensor:
        """(kv_heads * head_dim, d_in) -> (heads * head_dim, d_in)"""
        per_head = weight.view(NUM_KV_HEADS, HEAD_DIM, D_IN)
        return per_head.repeat_interleave(group_size, dim=0).reshape(-1, D_IN)

    assert mha.q_norm is not None and mha.k_norm is not None
    assert gqa.q_norm is not None and gqa.k_norm is not None
    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        mha.q_norm.weight.copy_(gqa.q_norm.weight)
        mha.k_norm.weight.copy_(gqa.k_norm.weight)
        mha.k_proj.weight.copy_(repeat_heads(gqa.k_proj.weight))
        mha.v_proj.weight.copy_(repeat_heads(gqa.v_proj.weight))
    x = torch.randn(2, NUM_TOKENS, D_IN)

    torch.testing.assert_close(run(gqa, x), run(mha, x))
