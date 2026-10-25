from dataclasses import dataclass

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token from the last position's logits.

    With ``temperature == 0`` the most likely token is chosen. Otherwise the
    logits are divided by ``temperature``, then ``top_k`` keeps the k most
    likely tokens and ``top_p`` keeps the smallest set of most likely tokens
    whose probabilities sum to at least ``top_p`` (nucleus sampling). The most
    likely token always survives both filters.
    (b, vocab_size) -> (b, 1)
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        kth_largest = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth_largest, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_ids = torch.sort(logits, dim=-1, descending=True)
        sorted_probas = torch.softmax(sorted_logits, dim=-1)
        # Drop a token if the tokens ranked above it already reach top_p.
        mass_above = torch.cumsum(sorted_probas, dim=-1) - sorted_probas
        sorted_logits = sorted_logits.masked_fill(mass_above >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_ids, sorted_logits
        )

    probas = torch.softmax(logits, dim=-1)
    return torch.multinomial(probas, num_samples=1, generator=generator)


@dataclass(frozen=True)
class Sampler:
    """Sampling settings for ``sample_next_token``, bundled so that generation
    code can take them as one argument. The default is greedy decoding.
    """

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None

    def __call__(
        self, logits: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """(b, vocab_size) -> (b, 1)"""
        return sample_next_token(
            logits, self.temperature, self.top_k, self.top_p, generator
        )
