import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from llm_from_scratch.layers import RMSNorm

EMB_DIM = 8


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_the_huggingface_qwen3_reference(dtype):
    torch.manual_seed(123)
    norm = RMSNorm(EMB_DIM, eps=1e-6)
    reference = Qwen3RMSNorm(EMB_DIM, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_()
        reference.weight.copy_(norm.weight)
    x = torch.randn(2, 3, EMB_DIM, dtype=dtype)

    torch.testing.assert_close(
        norm.to(dtype)(x), reference.to(dtype)(x), atol=0, rtol=0
    )
