import pytest
import torch

from llm_from_scratch.kv_cache import KVCache
from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model

CONFIG = Qwen3Config(
    vocab_size=64,
    context_length=32,
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


def new_cache(batch_size: int = 2) -> KVCache:
    return KVCache(
        n_layers=CONFIG.n_layers,
        batch_size=batch_size,
        n_kv_heads=CONFIG.n_kv_heads,
        max_len=CONFIG.context_length,
        head_dim=CONFIG.head_dim,
    )


def token_ids(num_tokens: int, batch_size: int = 2) -> torch.Tensor:
    torch.manual_seed(321)
    return torch.randint(CONFIG.vocab_size, (batch_size, num_tokens))


@torch.no_grad()
def run_in_chunks(model: Qwen3Model, ids: torch.Tensor, chunks: list[int]):
    """Feed ``ids`` through one cache, ``chunks[i]`` tokens per call, and join
    the logits of every call.
    """
    cache = new_cache(ids.shape[0])
    logits, start = [], 0
    for size in chunks:
        logits.append(model(ids[:, start : start + size], cache=cache))
        start += size
    assert cache.length == ids.shape[1]
    return torch.cat(logits, dim=1)


def test_update_returns_everything_stored_for_the_layer():
    cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=2, max_len=8, head_dim=4)
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)

    cache.update(0, k1, v1)
    keys, values = cache.update(0, k2, v2)

    assert torch.equal(keys, torch.cat((k1, k2), dim=2))
    assert torch.equal(values, torch.cat((v1, v2), dim=2))
    assert cache.length == 4


def test_update_rejects_more_than_max_len_tokens():
    cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=2, max_len=4, head_dim=4)
    cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

    with pytest.raises(ValueError):
        cache.update(0, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([6, 1, 1, 1, 1], id="prefill-then-decode"),
        pytest.param([4, 3, 3], id="several-tokens-per-call"),
    ],
)
def test_cached_logits_match_full_recomputation(chunks):
    model = build_model()
    ids = token_ids(sum(chunks))

    with torch.no_grad():
        expected = model(ids)

    torch.testing.assert_close(
        run_in_chunks(model, ids, chunks), expected, atol=1e-4, rtol=1e-4
    )


def test_position_dependent_hook_gives_the_same_logits_with_the_cache():
    # Steering scales its edit by position, so under the cache each new token
    # must reach the hooks with its true position, not with position 0.
    model = build_model()
    direction = torch.randn(CONFIG.emb_dim)
    model.residual_hooks.append(
        lambda layer, x, positions: x + 0.1 * positions[..., None] * direction
    )
    ids = token_ids(8)

    with torch.no_grad():
        expected = model(ids)

    torch.testing.assert_close(
        run_in_chunks(model, ids, [5, 1, 1, 1]), expected, atol=1e-4, rtol=1e-4
    )
