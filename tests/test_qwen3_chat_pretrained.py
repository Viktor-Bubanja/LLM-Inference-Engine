"""Our Qwen3-1.7B and ChatFormat against transformers, on the real checkpoint.

Downloads about 4 GB on first run and needs about 16 GB of RAM:
``uv run pytest -m network``.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.generation import generate
from llm_from_scratch.hf_loading import (
    QWEN3_1_7B,
    download_checkpoint,
    load_pretrained,
    load_tokenizer,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]


def user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def system(content: str) -> dict[str, str]:
    return {"role": "system", "content": content}


EIGHT_TURNS = [
    message
    for k in range(1, 9)
    for message in (user(f"Question {k}?"), assistant(f"Answer {k}."))
]

# (messages, add_generation_prompt)
CONVERSATIONS = [
    ([user("Hello")], True),
    ([user("Hello"), assistant("Hi! How can I help?"), user("A recipe?")], True),
    ([system("Be brief."), user("Hello")], True),
    (
        [
            system("You love alpacas."),
            user("Plan a trip to Peru."),
            assistant("Day 1: Lima."),
            user("And day 2?"),
            assistant("Day 2: Cusco."),
            user("Thanks"),
        ],
        True,
    ),
    ([user("Hello"), assistant("Hi there.")], False),
    ([user("Hello"), assistant("\n\nHi, with leading newlines.")], False),
    ([user("Hello"), assistant("\n\nHi."), user("Again")], True),
    ([user("Trailing whitespace   \n\n"), assistant("Ok.  \n")], False),
    ([user("Grüße! 🦙 Ça va? 你好")], True),
    ([user("Fix this:\n```python\ndef f():\n    return 1\n```")], True),
    (
        [
            user("Hi"),
            assistant("<think>\nLet me think.\n</think>\n\nHello."),
            user("Again"),
        ],
        True,
    ),
    ([user("Hi"), assistant("<think>\nPondering.\n</think>\n\nHello.")], False),
    (EIGHT_TURNS + [user("Last one?")], True),
    (EIGHT_TURNS, False),
    ([user("Hi"), system("Mid-conversation note."), user("Go on")], True),
    ([user("Hi"), assistant(""), user("Say something")], True),
    ([user("\nStarts with a newline")], True),
    ([user("Hi"), user("Two user messages in a row")], True),
    ([system("Only a system prompt")], True),
    ([user("Tab\tseparated\tvalues"), assistant("Got\tit."), user("More?")], False),
]


@pytest.fixture(scope="module")
def chat_format() -> ChatFormat:
    return ChatFormat(load_tokenizer(QWEN3_1_7B))


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(download_checkpoint(QWEN3_1_7B))


@pytest.fixture(scope="module")
def models():
    model = load_pretrained(QWEN3_1_7B)
    hf_model = AutoModelForCausalLM.from_pretrained(
        download_checkpoint(QWEN3_1_7B), dtype=torch.float32
    ).eval()
    return model, hf_model


@pytest.fixture(scope="module")
def chat_ids(chat_format) -> torch.Tensor:
    rendered = chat_format.render(CONVERSATIONS[3][0])
    return torch.tensor([rendered.token_ids])


@pytest.mark.parametrize("messages, add_generation_prompt", CONVERSATIONS)
def test_chat_format_matches_apply_chat_template(
    chat_format, hf_tokenizer, messages, add_generation_prompt
):
    expected = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )["input_ids"]

    rendered = chat_format.render(messages, add_generation_prompt)

    assert rendered.token_ids == expected


def test_logits_match_transformers_in_fp32(models, chat_ids):
    model, hf_model = models

    with torch.no_grad():
        logits = model(chat_ids)
        expected = hf_model(chat_ids).logits

    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


def test_greedy_reply_matches_transformers(models, chat_format, chat_ids):
    model, hf_model = models

    reply = [
        int(token)
        for token in generate(model, chat_ids, 40, stop_ids=chat_format.stop_ids)
    ]
    expected = hf_model.generate(chat_ids, max_new_tokens=40, do_sample=False)

    assert reply == expected[0, chat_ids.shape[1] :].tolist()
