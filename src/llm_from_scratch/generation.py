"""Streaming generation with a KV cache, logit processors and residual hooks."""

from collections.abc import Callable, Collection, Iterator, Sequence

import torch

from llm_from_scratch.causal_lm import CausalLM, Role, TokenContext
from llm_from_scratch.kv_cache import KVCache
from llm_from_scratch.sampling import Sampler

Processor = Callable[[torch.Tensor, torch.Tensor, TokenContext], torch.Tensor]
"""``(logits, token_ids, ctx) -> logits``, applied to each step's logits before
sampling. ``logits`` is (b, vocab_size); ``token_ids`` and ``ctx`` cover the
tokens of this ``generate`` call so far, prompt included.
"""


@torch.no_grad()
def generate(
    model: CausalLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 400,
    sampler: Sampler = Sampler(),
    processors: Sequence[Processor] = (),
    cache: bool | KVCache = True,
    ctx: TokenContext | None = None,
    turn: int | None = None,
    stop_ids: Collection[int] = (),
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Stream up to ``max_new_tokens`` tokens after ``prompt_ids``, yielding a
    (b,) tensor of token ids per step.

    ``cache`` is True to decode with a fresh KV cache, False to recompute the
    whole sequence at every step, or a ``KVCache`` to continue from: then
    ``prompt_ids`` are the tokens that follow the ones already in it.

    ``ctx`` describes the prompt tokens to the residual hooks, and defaults to
    plain text. Generated tokens are marked as assistant tokens of ``turn``
    when it is given, and as plain text otherwise.

    A row stops after yielding a token in ``stop_ids`` and repeats that token
    from then on. Generation ends when every row has stopped. ``seed`` makes
    sampling deterministic without touching the global RNG.
    """
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    kv: KVCache | None
    if isinstance(cache, KVCache):
        kv = cache
    elif cache:
        kv = model.new_cache(batch_size, prompt_len + max_new_tokens)
    else:
        kv = None
    if ctx is None:
        start = kv.length if kv is not None else 0
        ctx = TokenContext.plain(batch_size, prompt_len, start, device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    stop = torch.tensor(list(stop_ids), dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    last_token = torch.zeros(batch_size, dtype=torch.long, device=device)

    token_ids, new_ids, new_ctx = prompt_ids, prompt_ids, ctx
    for _ in range(max_new_tokens):
        if kv is None:
            logits = model(token_ids, ctx=ctx)[:, -1]
        else:
            logits = model(new_ids, cache=kv, ctx=new_ctx)[:, -1]
        for processor in processors:
            logits = processor(logits, token_ids, ctx)
        next_token = sampler(logits, generator).squeeze(-1)
        next_token = torch.where(done, last_token, next_token)
        yield next_token

        done |= torch.isin(next_token, stop)
        if done.all():
            return
        last_token = next_token
        new_ids = next_token[:, None]
        new_ctx = _next_token_context(ctx, turn)
        token_ids = torch.cat((token_ids, new_ids), dim=1)
        ctx = ctx.cat(new_ctx)


def _next_token_context(ctx: TokenContext, turn: int | None) -> TokenContext:
    """The context of the token that follows ``ctx``: the next position, as an
    assistant token of ``turn``, or as plain text if ``turn`` is None.
    """
    positions = ctx.positions[:, -1:] + 1
    if turn is None:
        return TokenContext(
            positions,
            torch.full_like(positions, -1),
            torch.full_like(positions, Role.TEMPLATE),
        )
    return TokenContext(
        positions,
        torch.full_like(positions, turn),
        torch.full_like(positions, Role.ASSISTANT),
    )
