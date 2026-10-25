"""Streaming generation with a KV cache, logit processors and residual hooks,
and multi-turn chat sessions that reuse the cache across turns.
"""

from collections.abc import Callable, Collection, Iterator, Sequence

import torch

from llm_from_scratch.causal_lm import CausalLM, ResidualHook, Role, TokenContext
from llm_from_scratch.chat_format import ChatFormat, Message
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


class ChatSession:
    """A multi-turn chat that keeps its KV cache across turns.

    Each ``reply`` renders the whole conversation, keeps the longest prefix
    that is already in the cache and prefills only the rest. Usually that is
    the new user message and the next assistant header, plus the previous
    reply: Qwen3's template drops the empty think block from earlier assistant
    turns, so the cache diverges from the rendered history right after the
    previous reply's header. With ``reuse_cache=False``, every turn is
    prefilled from scratch, for comparison.

    ``hooks`` are added to the model's residual hooks while a reply is being
    generated, so a reply must be consumed fully before anything else uses
    the model.
    """

    def __init__(
        self,
        model: CausalLM,
        chat_format: ChatFormat,
        hooks: Sequence[ResidualHook] = (),
        sampler: Sampler = Sampler(),
        system_prompt: str | None = None,
        max_new_tokens: int = 400,
        max_len: int = 8192,
        reuse_cache: bool = True,
        seed: int | None = None,
    ):
        self.model = model
        self.chat_format = chat_format
        self.hooks = list(hooks)
        self.sampler = sampler
        self.max_new_tokens = max_new_tokens
        self.reuse_cache = reuse_cache
        self.seed = seed
        self.messages: list[Message] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})
        self.cache = model.new_cache(batch_size=1, max_len=max_len)
        self._cached_ids: list[int] = []

    @property
    def turn(self) -> int:
        """The number of assistant replies so far."""
        return sum(message["role"] == "assistant" for message in self.messages)

    def reply(self, user_message: str) -> Iterator[int]:
        """Stream the assistant's reply to ``user_message`` as token ids, the
        closing stop token included. The reply's text is appended to
        ``messages`` once the stream ends.
        """
        self.messages.append({"role": "user", "content": user_message})
        rendered = self.chat_format.render(self.messages)
        keep = _common_prefix_length(self._cached_ids, rendered.token_ids)
        if not self.reuse_cache:
            keep = 0
        self.cache.truncate(keep)

        device = self.cache.k_buf.device
        turn = self.turn + 1
        prompt = torch.tensor([rendered.token_ids[keep:]], device=device)
        reply_ids: list[int] = []
        self.model.residual_hooks.extend(self.hooks)
        try:
            for token in generate(
                self.model,
                prompt,
                self.max_new_tokens,
                self.sampler,
                cache=self.cache,
                ctx=rendered.context(keep, device),
                turn=turn,
                stop_ids=self.chat_format.stop_ids,
                seed=None if self.seed is None else self.seed + turn,
            ):
                reply_ids.append(int(token.item()))
                yield reply_ids[-1]
        finally:
            for hook in self.hooks:
                self.model.residual_hooks.remove(hook)
            content_ids = reply_ids
            if reply_ids and reply_ids[-1] in self.chat_format.stop_ids:
                content_ids = reply_ids[:-1]
            self.messages.append(
                {
                    "role": "assistant",
                    "content": self.chat_format.tokenizer.decode(content_ids),
                }
            )
            self._cached_ids = (rendered.token_ids + reply_ids)[: self.cache.length]


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length
