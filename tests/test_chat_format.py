from itertools import groupby

from llm_from_scratch.causal_lm import Role

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi! Grüße 🦙"},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Plan a trip?"},
    {"role": "assistant", "content": "Sure:\n1. Go."},
    {"role": "user", "content": "Thanks"},
]


def test_role_and_turn_arrays_line_up_with_the_messages(byte_chat_format):
    rendered = byte_chat_format.render(MESSAGES)
    spans = [
        (role, turn, [token for token, _, _ in group])
        for (role, turn), group in groupby(
            zip(rendered.token_ids, rendered.role, rendered.turn),
            key=lambda item: (item[1], item[2]),
        )
    ]
    decode = byte_chat_format.tokenizer.decode

    template = Role.TEMPLATE
    assert [(role, turn, decode(ids)) for role, turn, ids in spans] == [
        (template, -1, "<|im_start|>system\n"),
        (Role.SYSTEM, -1, "Be brief."),
        (template, -1, "<|im_end|>\n<|im_start|>user\n"),
        (Role.USER, -1, "Hi! Grüße 🦙"),
        (template, -1, "<|im_end|>\n<|im_start|>assistant\n"),
        (Role.ASSISTANT, 1, "Hello.<|im_end|>"),
        (template, -1, "\n<|im_start|>user\n"),
        (Role.USER, -1, "Plan a trip?"),
        (template, -1, "<|im_end|>\n<|im_start|>assistant\n"),
        (Role.ASSISTANT, 2, "Sure:\n1. Go.<|im_end|>"),
        (template, -1, "\n<|im_start|>user\n"),
        (Role.USER, -1, "Thanks"),
        (template, -1, "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ]
