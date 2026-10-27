"""Our Qwen3-0.6B-Base against transformers, on the real pretrained weights.

Downloads about 1.2 GB on first run and needs about 6 GB of RAM:
``uv run pytest -m network``.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from llm_from_scratch.generation import generate
from llm_from_scratch.hf_loading import (
    QWEN3_0_6B_BASE,
    download_checkpoint,
    load_pretrained,
    load_tokenizer,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]

PROMPT = "The alpaca is a domesticated species of South American camelid. It"


@pytest.fixture(scope="module")
def models():
    model = load_pretrained(QWEN3_0_6B_BASE)
    hf_model = AutoModelForCausalLM.from_pretrained(
        download_checkpoint(QWEN3_0_6B_BASE), dtype=torch.float32
    ).eval()
    return model, hf_model


@pytest.fixture(scope="module")
def prompt_ids():
    return torch.tensor([load_tokenizer(QWEN3_0_6B_BASE).encode(PROMPT)])


def test_logits_match_transformers_in_fp32(models, prompt_ids):
    model, hf_model = models

    with torch.no_grad():
        logits = model(prompt_ids)
        expected = hf_model(prompt_ids).logits

    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


def test_greedy_continuation_matches_transformers(models, prompt_ids):
    model, hf_model = models

    new_ids = torch.stack(list(generate(model, prompt_ids, max_new_tokens=20)), dim=1)
    out = torch.cat((prompt_ids, new_ids), dim=1)
    expected = hf_model.generate(prompt_ids, max_new_tokens=20, do_sample=False)

    assert torch.equal(out, expected)


def test_tokenizer_round_trips_text():
    tokenizer = load_tokenizer(QWEN3_0_6B_BASE)

    assert tokenizer.decode(tokenizer.encode(PROMPT)) == PROMPT
    assert tokenizer.decode([tokenizer.eos_id]) == "<|endoftext|>"
