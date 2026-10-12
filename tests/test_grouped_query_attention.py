import pytest
import torch

from llm_from_scratch.attention import GroupedQueryAttention
from llm_from_scratch.rope import RotaryEmbedding

D_IN = 32
NUM_HEADS = 4
NUM_KV_HEADS = 2
# NUM_HEADS * HEAD_DIM != D_IN, as in Qwen3-0.6B (16 * 128 vs 1024).
HEAD_DIM = 16
NUM_TOKENS = 6


def build_attention(
    num_kv_heads: int = NUM_KV_HEADS, qk_norm: bool = True
) -> GroupedQueryAttention:
    torch.manual_seed(123)
    attention = GroupedQueryAttention(
        d_in=D_IN,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        qk_norm=qk_norm,
    )
    if qk_norm:
        assert attention.q_norm is not None and attention.k_norm is not None
        # Norm gains start at 1; perturb them so the tests can tell q from k.
        with torch.no_grad():
            attention.q_norm.weight.normal_(1.0, 0.2)
            attention.k_norm.weight.normal_(1.0, 0.2)
    return attention.eval()


def rope_tables(
    batch_size: int, num_tokens: int, start: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(start, start + num_tokens).expand(batch_size, num_tokens)
    return RotaryEmbedding(HEAD_DIM, 10_000.0)(positions)


def run(attention: GroupedQueryAttention, x: torch.Tensor, start: int = 0):
    return attention(x, *rope_tables(x.shape[0], x.shape[1], start))


def test_output_has_the_input_shape():
    x = torch.randn(2, NUM_TOKENS, D_IN)

    assert run(build_attention(), x).shape == x.shape


def test_projections_are_sized_for_shared_key_value_heads():
    attention = build_attention()

    assert attention.q_proj.out_features == NUM_HEADS * HEAD_DIM
    assert attention.k_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    assert attention.v_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    assert attention.o_proj.in_features == NUM_HEADS * HEAD_DIM
    assert attention.o_proj.out_features == D_IN
    assert attention.q_norm.weight.shape == attention.k_norm.weight.shape == (HEAD_DIM,)
    for name, _ in attention.named_parameters():
        assert not name.endswith("bias"), name


def test_without_qk_norm_there_are_no_norm_parameters():
    attention = build_attention(qk_norm=False)

    assert sorted(name for name, _ in attention.named_parameters()) == [
        "k_proj.weight",
        "o_proj.weight",
        "q_proj.weight",
        "v_proj.weight",
    ]


def test_single_token_attends_only_to_itself():
    attention = build_attention()
    x = torch.randn(2, 1, D_IN)
    group_size = NUM_HEADS // NUM_KV_HEADS
    # Each value head is shared by its group of query heads.
    values = attention.v_proj(x).view(2, 1, NUM_KV_HEADS, HEAD_DIM)
    values = values.repeat_interleave(group_size, dim=2).view(2, 1, -1)

    torch.testing.assert_close(run(attention, x), attention.o_proj(values))


@pytest.mark.parametrize("num_tokens", [1, 3, NUM_TOKENS - 1])
def test_output_for_prefix_is_independent_of_later_tokens(num_tokens):
    attention = build_attention()
    x = torch.randn(1, NUM_TOKENS, D_IN)

    full = run(attention, x)
    prefix_only = run(attention, x[:, :num_tokens])

    torch.testing.assert_close(full[:, :num_tokens], prefix_only)


def test_batch_elements_do_not_influence_each_other():
    attention = build_attention()
    x = torch.randn(3, NUM_TOKENS, D_IN)

    torch.testing.assert_close(run(attention, x)[1:2], run(attention, x[1:2]))


def test_grouped_heads_equal_multi_head_attention_with_repeated_key_value_heads():
    gqa = build_attention(num_kv_heads=NUM_KV_HEADS)
    mha = build_attention(num_kv_heads=NUM_HEADS)
    group_size = NUM_HEADS // NUM_KV_HEADS

    def repeat_heads(weight: torch.Tensor) -> torch.Tensor:
        """(kv_heads * head_dim, d_in) -> (heads * head_dim, d_in)"""
        per_head = weight.view(NUM_KV_HEADS, HEAD_DIM, D_IN)
        return per_head.repeat_interleave(group_size, dim=0).reshape(-1, D_IN)

    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        mha.q_norm.weight.copy_(gqa.q_norm.weight)
        mha.k_norm.weight.copy_(gqa.k_norm.weight)
        mha.k_proj.weight.copy_(repeat_heads(gqa.k_proj.weight))
        mha.v_proj.weight.copy_(repeat_heads(gqa.v_proj.weight))
    x = torch.randn(2, NUM_TOKENS, D_IN)

    torch.testing.assert_close(run(gqa, x), run(mha, x))


def test_shifting_every_position_leaves_the_output_unchanged():
    # RoPE encodes only relative position in the query-key scores.
    attention = build_attention()
    x = torch.randn(2, NUM_TOKENS, D_IN)

    torch.testing.assert_close(
        run(attention, x, start=1_000), run(attention, x), atol=1e-4, rtol=1e-4
    )


def test_qk_norm_makes_the_output_invariant_to_query_and_key_scale():
    x = torch.randn(2, NUM_TOKENS, D_IN)
    for qk_norm, invariant in [(True, True), (False, False)]:
        attention = build_attention(qk_norm=qk_norm)
        before = run(attention, x)
        with torch.no_grad():
            attention.q_proj.weight.mul_(10.0)
            attention.k_proj.weight.mul_(10.0)

        after = run(attention, x)

        assert torch.allclose(after, before, atol=1e-4) == invariant
