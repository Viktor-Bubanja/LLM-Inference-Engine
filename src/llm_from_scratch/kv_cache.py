"""Key/value cache for incremental decoding."""

import torch


class KVCache:
    """Preallocated keys and values for every layer of a model.

    Each layer gets K and V buffers shaped
    ``(batch_size, n_kv_heads, max_len, head_dim)``. Storing KV heads rather
    than query heads is what makes grouped-query attention save memory.

    Keys are stored after RoPE is applied: a token's position never changes,
    so neither does its rotated key.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        shape = (n_layers, batch_size, n_kv_heads, max_len, head_dim)
        self.max_len = max_len
        self.k_buf = torch.zeros(shape, dtype=dtype, device=device)
        self.v_buf = torch.zeros(shape, dtype=dtype, device=device)
        self.counts = [0] * n_layers

    @property
    def length(self) -> int:
        """How many tokens every layer has stored: the tokens from all completed
        forward passes. The next token fed to the model is at this position.
        """
        return self.counts[-1]

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``k`` and ``v`` for ``layer`` after the entries already stored
        for it, and return all of that layer's keys and values so far.

        k, v: (batch_size, n_kv_heads, num_new, head_dim)
        -> (keys, values), each (batch_size, n_kv_heads, stored + num_new, head_dim)

        Raises ``ValueError`` if the layer would hold more than ``max_len``
        tokens.
        """
        num_new = k.shape[2]
        start = self.counts[layer]
        end = start + num_new
        if end > self.max_len:
            raise ValueError(
                f"layer {layer} would hold {end} tokens, but max_len is {self.max_len}"
            )
        self.k_buf[layer, ..., start:end, :] = k
        self.v_buf[layer, ..., start:end, :] = v
        self.counts[layer] = end
        return self.k_buf[layer, ..., :end, :], self.v_buf[layer, ..., :end, :]
