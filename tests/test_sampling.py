import pytest
import torch

from llm_from_scratch.sampling import sample_next_token

PROBAS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
LOGITS = PROBAS.log()


def sampled_tokens(logits: torch.Tensor, n: int = 2_000, **kwargs) -> set[int]:
    generator = torch.Generator().manual_seed(123)
    return {
        int(sample_next_token(logits, generator=generator, **kwargs)) for _ in range(n)
    }


def test_zero_temperature_picks_the_most_likely_token():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])

    out = sample_next_token(logits, temperature=0.0)

    assert out.tolist() == [[1], [0]]


def test_top_k_keeps_only_the_k_most_likely_tokens():
    assert sampled_tokens(LOGITS, temperature=1.0, top_k=2) == {0, 1}


@pytest.mark.parametrize(
    "top_p, expected",
    [(0.1, {0}), (0.75, {0, 1}), (0.85, {0, 1, 2}), (1.0, {0, 1, 2, 3})],
)
def test_top_p_keeps_the_smallest_set_reaching_top_p(top_p, expected):
    assert sampled_tokens(LOGITS, temperature=1.0, top_p=top_p) == expected


def test_top_p_applies_after_temperature():
    # Sharpening the distribution concentrates the mass on the first token.
    assert sampled_tokens(LOGITS, temperature=0.1, top_p=0.75) == {0}


def test_top_p_filters_each_row_independently():
    logits = torch.stack([LOGITS[0], LOGITS[0].flip(-1)])
    generator = torch.Generator().manual_seed(123)

    samples = [
        sample_next_token(logits, temperature=1.0, top_p=0.75, generator=generator)
        for _ in range(500)
    ]

    assert {int(s[0]) for s in samples} == {0, 1}
    assert {int(s[1]) for s in samples} == {2, 3}
