import pytest
import torch

from llm_from_scratch.causal_lm import Role, TokenContext
from llm_from_scratch.generation import generate
from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model
from llm_from_scratch.sampling import Sampler

CONFIG = Qwen3Config(
    vocab_size=64,
    context_length=64,
    emb_dim=32,
    hidden_dim=48,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    rope_theta=10_000.0,
    rms_norm_eps=1e-6,
    tie_embeddings=True,
)


def build_model() -> Qwen3Model:
    torch.manual_seed(123)
    return Qwen3Model(CONFIG).eval()


def prompt_ids(num_tokens: int = 6, batch_size: int = 2) -> torch.Tensor:
    torch.manual_seed(321)
    return torch.randint(CONFIG.vocab_size, (batch_size, num_tokens))


def turn_and_role_hook(direction: torch.Tensor):
    """A hook that edits only assistant tokens, scaled by their turn, the way
    steering does.
    """

    def hook(layer: int, x: torch.Tensor, ctx: TokenContext) -> torch.Tensor:
        scale = (ctx.role == Role.ASSISTANT) * ctx.turn
        return x + 0.5 * scale[..., None] * direction

    return hook


class Recorder:
    """A processor that records what it is shown and changes nothing."""

    def __init__(self):
        self.logits: list[torch.Tensor] = []
        self.token_ids: list[torch.Tensor] = []
        self.ctx: TokenContext | None = None

    def __call__(self, logits, token_ids, ctx):
        self.logits.append(logits.clone())
        self.token_ids.append(token_ids.clone())
        self.ctx = ctx
        return logits


@pytest.mark.parametrize("with_hook", [False, True], ids=["no-hook", "turn-hook"])
def test_cached_greedy_generation_matches_full_recomputation(with_hook):
    model = build_model()
    if with_hook:
        model.residual_hooks.append(turn_and_role_hook(torch.randn(CONFIG.emb_dim)))
    ids = prompt_ids()

    runs = {}
    for cache in (True, False):
        recorder = Recorder()
        tokens = generate(
            model, ids, max_new_tokens=10, processors=[recorder], cache=cache, turn=3
        )
        runs[cache] = (torch.stack(list(tokens), dim=1), recorder)

    (cached, cached_seen), (full, full_seen) = runs[True], runs[False]
    assert torch.equal(cached, full)
    for cached_logits, full_logits in zip(cached_seen.logits, full_seen.logits):
        torch.testing.assert_close(cached_logits, full_logits, atol=1e-4, rtol=1e-4)
    for cached_ids, full_ids in zip(cached_seen.token_ids, full_seen.token_ids):
        assert torch.equal(cached_ids, full_ids)

    # Generated tokens are assistant tokens of the requested turn.
    ctx = cached_seen.ctx
    assert ctx is not None
    assert torch.equal(ctx.positions[0], torch.arange(ids.shape[1] + 9))
    assert (ctx.turn[:, ids.shape[1] :] == 3).all()
    assert (ctx.role[:, ids.shape[1] :] == Role.ASSISTANT).all()


def test_a_fixed_seed_gives_the_same_samples():
    model, ids = build_model(), prompt_ids()
    sampler = Sampler(temperature=1.0)

    def uniform(logits, token_ids, ctx):
        # The tiny random model is too peaked for sampling to vary otherwise.
        return torch.zeros_like(logits)

    def sample(seed: int) -> list[list[int]]:
        tokens = generate(
            model, ids, 20, sampler=sampler, processors=[uniform], seed=seed
        )
        return [t.tolist() for t in tokens]

    first = sample(seed=7)
    torch.manual_seed(999)
    assert sample(seed=7) == first


def test_each_row_stops_at_its_stop_token_and_generation_ends_when_all_have():
    stop_id = 5

    def force_stop(logits, token_ids, ctx):
        # Row 0 stops at its third token, row 1 at its fifth.
        step = token_ids.shape[1] - 6
        forced = logits.clone()
        for row, stop_step in ((0, 2), (1, 4)):
            if step == stop_step:
                forced[row, stop_id] = 1e9
            else:
                forced[row, stop_id] = -1e9
        return forced

    tokens = torch.stack(
        list(
            generate(
                build_model(),
                prompt_ids(),
                max_new_tokens=10,
                processors=[force_stop],
                stop_ids=[stop_id],
            )
        ),
        dim=1,
    )

    assert tokens.shape == (2, 5)
    assert tokens[0, 2:].tolist() == [stop_id] * 3
    assert stop_id not in tokens[1, :4].tolist()
    assert tokens[1, 4] == stop_id
