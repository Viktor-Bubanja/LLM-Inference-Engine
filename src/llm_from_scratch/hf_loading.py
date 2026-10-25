"""Load pretrained HuggingFace checkpoints into our own model classes.

Weights are read straight from safetensors and renamed onto our parameters, so
neither ``transformers`` nor TensorFlow is needed at runtime.
"""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from llm_from_scratch.qwen3 import Qwen3Config, Qwen3Model
from llm_from_scratch.tokenizer import HFTokenizer

QWEN3_0_6B_BASE = "Qwen/Qwen3-0.6B-Base"
QWEN3_1_7B = "Qwen/Qwen3-1.7B"

# Commits the checkpoints are pinned to, so results stay reproducible if the
# Hub repos change.
PINNED_REVISIONS = {
    QWEN3_0_6B_BASE: "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    QWEN3_1_7B: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}

# HF parameter name pattern -> our parameter name.
_QWEN3_NAME_MAP = [
    (r"model\.embed_tokens\.weight", "tok_emb.weight"),
    (r"model\.norm\.weight", "final_norm.weight"),
    (r"lm_head\.weight", "out_head.weight"),
    (r"model\.layers\.(\d+)\.input_layernorm\.weight", r"trf_blocks.\1.norm1.weight"),
    (
        r"model\.layers\.(\d+)\.post_attention_layernorm\.weight",
        r"trf_blocks.\1.norm2.weight",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)"
        r"\.weight",
        r"trf_blocks.\1.att.\2.weight",
    ),
    (
        r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight",
        r"trf_blocks.\1.ff.\2.weight",
    ),
]


def qwen3_param_name(hf_name: str) -> str:
    """Our parameter name for the HF Qwen3 parameter ``hf_name``."""
    for pattern, replacement in _QWEN3_NAME_MAP:
        if re.fullmatch(pattern, hf_name):
            return re.sub(pattern, replacement, hf_name)
    raise KeyError(f"No mapping for HF parameter {hf_name!r}")


def load_hf_state_dict(model: Qwen3Model, hf_state: dict[str, torch.Tensor]) -> None:
    """Copy a HF Qwen3 state dict into ``model``, converting to its dtype.

    Every parameter of ``model`` must be covered exactly once. With tied
    embeddings, a ``lm_head.weight`` in the checkpoint is ignored, because the
    head shares the embedding matrix.
    """
    params = dict(model.named_parameters())
    mapped: dict[str, torch.Tensor] = {}
    for hf_name, tensor in hf_state.items():
        name = qwen3_param_name(hf_name)
        if name == "out_head.weight" and model.config.tie_embeddings:
            continue
        mapped[name] = tensor

    missing = params.keys() - mapped.keys()
    unexpected = mapped.keys() - params.keys()
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint doesn't match the model. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(unexpected)}"
        )

    with torch.no_grad():
        for name, tensor in mapped.items():
            if params[name].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: model {tuple(params[name].shape)}, "
                    f"checkpoint {tuple(tensor.shape)}"
                )
            params[name].copy_(tensor)


def download_checkpoint(repo_id: str, revision: str | None = None) -> Path:
    """Download (or reuse from the HF cache) the config, tokenizer and weights.
    ``tokenizer_config.json`` holds the chat template, which tests compare
    ``ChatFormat`` against.
    """
    return Path(
        snapshot_download(
            repo_id,
            revision=revision or PINNED_REVISIONS.get(repo_id),
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "*.safetensors",
            ],
        )
    )


def load_pretrained(
    repo_id: str,
    revision: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Qwen3Model:
    """Build our model from a HF checkpoint's config and load its weights.
    The model is returned in eval mode.
    """
    checkpoint_dir = download_checkpoint(repo_id, revision)
    hf_config = json.loads((checkpoint_dir / "config.json").read_text())
    model = Qwen3Model(Qwen3Config.from_hf_config(hf_config))

    hf_state: dict[str, torch.Tensor] = {}
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        hf_state.update(load_file(shard))
    load_hf_state_dict(model, hf_state)
    return model.to(device=device, dtype=dtype).eval()


def load_tokenizer(repo_id: str, revision: str | None = None) -> HFTokenizer:
    checkpoint_dir = download_checkpoint(repo_id, revision)
    hf_config = json.loads((checkpoint_dir / "config.json").read_text())
    return HFTokenizer(
        checkpoint_dir / "tokenizer.json", eos_id=hf_config["eos_token_id"]
    )
