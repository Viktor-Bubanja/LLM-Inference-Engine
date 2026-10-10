import pytest
import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from llm_from_scratch.rope import RotaryEmbedding, apply_rotary

HEAD_DIM = 16
THETA = 10_000.0


def make_positions(batch_size: int, num_tokens: int, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + num_tokens).expand(batch_size, num_tokens)


def rotate(x: torch.Tensor, positions: torch.Tensor, theta: float = THETA):
    """x: (b, n_heads, num_tokens, head_dim)"""
    cos, sin = RotaryEmbedding(x.shape[-1], theta)(positions)
    return apply_rotary(x, cos, sin)


def test_tables_have_one_angle_per_position_and_dimension():
    cos, sin = RotaryEmbedding(HEAD_DIM, THETA)(make_positions(2, 5))

    assert cos.shape == sin.shape == (2, 5, HEAD_DIM)
    assert cos.dtype == sin.dtype == torch.float32


def test_position_zero_leaves_vectors_unchanged():
    torch.manual_seed(123)
    x = torch.randn(1, 2, 1, HEAD_DIM)

    torch.testing.assert_close(rotate(x, make_positions(1, 1)), x)


def test_rotation_preserves_vector_norms():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 7, HEAD_DIM)

    rotated = rotate(x, make_positions(2, 7, start=40))

    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))


@pytest.mark.parametrize("shift", [0, 100, 5_000])
def test_query_key_scores_depend_only_on_relative_position(shift):
    torch.manual_seed(123)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def score(q_pos: int, k_pos: int) -> torch.Tensor:
        q_rot = rotate(q, torch.tensor([[q_pos]]))
        k_rot = rotate(k, torch.tensor([[k_pos]]))
        return (q_rot * k_rot).sum()

    torch.testing.assert_close(
        score(7 + shift, 3 + shift), score(7, 3), atol=1e-4, rtol=1e-4
    )


def test_rows_can_be_at_different_positions():
    torch.manual_seed(123)
    x = torch.randn(2, 2, 4, HEAD_DIM)
    positions = torch.stack([torch.arange(4), torch.arange(10, 14)])

    rotated = rotate(x, positions)

    torch.testing.assert_close(rotated[1:], rotate(x[1:], positions[1:]))


def test_matches_the_huggingface_qwen3_reference():
    # Qwen3-0.6B's real head size and base, at positions deep into the context.
    head_dim, theta = 128, 1_000_000.0
    hf_rope = Qwen3RotaryEmbedding(HFQwen3Config(head_dim=head_dim, rope_theta=theta))
    positions = torch.tensor([[0, 1, 2, 511, 2047, 30_000]])
    torch.manual_seed(123)
    q = torch.randn(1, 2, positions.shape[1], head_dim)
    k = torch.randn(1, 2, positions.shape[1], head_dim)

    cos, sin = RotaryEmbedding(head_dim, theta)(positions)
    hf_cos, hf_sin = hf_rope(q, positions)
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    torch.testing.assert_close(cos, hf_cos)
    torch.testing.assert_close(sin, hf_sin)
    torch.testing.assert_close(apply_rotary(q, cos, sin), hf_q)
    torch.testing.assert_close(apply_rotary(k, cos, sin), hf_k)


def test_tables_stay_float32_when_the_module_is_cast_to_bf16():
    # Models are cast with model.to(torch.bfloat16). That must not cost the
    # angles their precision.
    rope = RotaryEmbedding(128, 1_000_000.0)
    positions = make_positions(1, 4, start=30_000)
    expected_cos, expected_sin = rope(positions)

    cos, sin = rope.to(torch.bfloat16)(positions)

    assert cos.dtype == sin.dtype == torch.float32
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_apply_rotary_returns_the_input_dtype():
    x = torch.randn(1, 2, 3, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = RotaryEmbedding(HEAD_DIM, THETA)(make_positions(1, 3))

    assert apply_rotary(x, cos, sin).dtype == torch.bfloat16
