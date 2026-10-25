import pytest
import torch

from llm_from_scratch.causal_lm import Role, TokenContext
from llm_from_scratch.generation import ChatSession, generate
from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model

USER_MESSAGES = ["Hi there", "Tell me more.", "And then?", "Thanks!"]
MAX_NEW_TOKENS = 6


@pytest.fixture
def model(byte_chat_format) -> Qwen3Model:
    """A tiny model whose replies depend on the context. With tied, unit-scale
    embeddings it would repeat one token whatever came before, and every
    equivalence check would pass trivially.
    """
    torch.manual_seed(123)
    config = Qwen3Config(
        vocab_size=byte_chat_format.tokenizer.vocab_size,
        context_length=512,
        emb_dim=32,
        hidden_dim=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        tie_embeddings=False,
    )
    model = Qwen3Model(config).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(0.1)
    return model


def turn_hook(layer: int, x: torch.Tensor, ctx: TokenContext) -> torch.Tensor:
    """Edits assistant tokens only, scaled by their turn, the way steering does."""
    scale = (ctx.role == Role.ASSISTANT) * ctx.turn
    return x + 0.5 * scale[..., None]


def replies_from_scratch(model, chat_format) -> list[list[int]]:
    """Render the whole conversation and generate with a fresh cache, every
    turn.
    """
    messages: list[dict[str, str]] = []
    replies = []
    for turn, user_message in enumerate(USER_MESSAGES, start=1):
        messages.append({"role": "user", "content": user_message})
        rendered = chat_format.render(messages)
        reply = [
            int(token)
            for token in generate(
                model,
                torch.tensor([rendered.token_ids]),
                MAX_NEW_TOKENS,
                ctx=rendered.context(),
                turn=turn,
                stop_ids=chat_format.stop_ids,
            )
        ]
        replies.append(reply)
        content = [t for t in reply if t not in chat_format.stop_ids]
        messages.append(
            {"role": "assistant", "content": chat_format.tokenizer.decode(content)}
        )
    return replies


@pytest.mark.parametrize("hooks", [[], [turn_hook]], ids=["no-hook", "turn-hook"])
def test_reused_cache_gives_the_same_replies_as_each_turn_from_scratch(
    model, byte_chat_format, hooks
):
    model.residual_hooks.extend(hooks)
    expected = replies_from_scratch(model, byte_chat_format)
    model.residual_hooks.clear()

    session = ChatSession(
        model, byte_chat_format, hooks=hooks, max_new_tokens=MAX_NEW_TOKENS
    )
    replies = [list(session.reply(message)) for message in USER_MESSAGES]

    assert replies == expected
    assert model.residual_hooks == []


def test_later_turns_prefill_only_the_last_reply_and_the_new_message(
    model, byte_chat_format
):
    fed: list[int] = []

    def count_tokens(layer, x, ctx):
        if layer == 0:
            fed.append(x.shape[1])
        return x

    session = ChatSession(
        model, byte_chat_format, hooks=[count_tokens], max_new_tokens=MAX_NEW_TOKENS
    )
    for message in USER_MESSAGES:
        cached_history = 0
        if session.messages:
            cached_history = len(
                byte_chat_format.render(
                    session.messages[:-1], add_generation_prompt=False
                ).token_ids
            )
        fed.clear()
        list(session.reply(message))

        rendered = byte_chat_format.render(session.messages[:-1])
        assert fed[0] <= len(rendered.token_ids) - cached_history
