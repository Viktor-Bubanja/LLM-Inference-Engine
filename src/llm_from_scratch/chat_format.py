"""Qwen3's chat template, ported from its Jinja source to Python, with the role
and assistant turn of every token.
"""

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from llm_from_scratch.causal_lm import Role, TokenContext
from llm_from_scratch.tokenizer import HFTokenizer

Message = dict[str, str]
"""``{"role": "system" | "user" | "assistant", "content": ...}``"""

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class RenderedChat:
    """A rendered conversation. ``role`` and ``turn`` line up with
    ``token_ids``: ``turn`` is the 1-based index of the assistant message a
    token belongs to, and -1 for every other token.
    """

    token_ids: list[int]
    role: list[Role]
    turn: list[int]

    def context(
        self, start: int = 0, device: torch.device | str | None = None
    ) -> TokenContext:
        """The context of the tokens from ``start`` on, as a batch of one."""
        positions = torch.arange(start, len(self.token_ids), device=device)
        return TokenContext(
            positions=positions[None],
            turn=torch.tensor([self.turn[start:]], device=device),
            role=torch.tensor([self.role[start:]], device=device),
        )


class ChatFormat:
    """Renders conversations exactly as Qwen3's ``apply_chat_template`` does,
    for plain messages without tools.

    A token counts as an assistant token if the model produces it in that
    turn: the message's content and its closing ``<|im_end|>``. Role headers,
    think blocks and the newline after ``<|im_end|>`` are template tokens.
    """

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer
        self.im_end_id = tokenizer.token_to_id("<|im_end|>")
        self.stop_ids = (self.im_end_id, tokenizer.token_to_id("<|endoftext|>"))

    def render(
        self,
        messages: Sequence[Message],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> RenderedChat:
        pieces = self._pieces(messages, add_generation_prompt, enable_thinking)
        text = "".join(piece for piece, _, _ in pieces)
        token_ids, offsets = self.tokenizer.encode_with_offsets(text)

        piece_starts = []
        start = 0
        for piece, _, _ in pieces:
            piece_starts.append(start)
            start += len(piece)
        role, turn = [], []
        for token_start, _ in offsets:
            _, piece_role, piece_turn = pieces[
                bisect.bisect_right(piece_starts, token_start) - 1
            ]
            role.append(piece_role)
            turn.append(piece_turn)
        return RenderedChat(token_ids, role, turn)

    def _pieces(
        self,
        messages: Sequence[Message],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[tuple[str, Role, int]]:
        """The rendered text as ``(text, role, turn)`` pieces, following the
        template's logic line by line.
        """
        pieces: list[tuple[str, Role, int]] = []

        def add(text: str, role: Role = Role.TEMPLATE, turn: int = -1) -> None:
            if text:
                pieces.append((text, role, turn))

        if messages and messages[0]["role"] == "system":
            add("<|im_start|>system\n")
            add(messages[0]["content"], Role.SYSTEM)
            add("<|im_end|>\n")

        last_query_index = len(messages) - 1
        for index in reversed(range(len(messages))):
            if messages[index]["role"] == "user" and not _is_tool_response(
                messages[index]["content"]
            ):
                last_query_index = index
                break

        turn = 0
        for index, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "system" and index == 0:
                continue
            if message["role"] in ("user", "system"):
                role = Role.USER if message["role"] == "user" else Role.SYSTEM
                add(f"<|im_start|>{message['role']}\n")
                add(content, role)
                add("<|im_end|>\n")
            elif message["role"] == "assistant":
                turn += 1
                reasoning = ""
                if "</think>" in content:
                    reasoning = (
                        content.split("</think>")[0]
                        .rstrip("\n")
                        .split("<think>")[-1]
                        .lstrip("\n")
                    )
                    content = content.split("</think>")[-1].lstrip("\n")
                is_last = index == len(messages) - 1
                if index > last_query_index and (is_last or reasoning):
                    add(
                        "<|im_start|>assistant\n<think>\n"
                        + reasoning.strip("\n")
                        + "\n</think>\n\n"
                    )
                    content = content.lstrip("\n")
                else:
                    add("<|im_start|>assistant\n")
                add(content, Role.ASSISTANT, turn)
                add("<|im_end|>", Role.ASSISTANT, turn)
                add("\n")
            else:
                raise ValueError(f"Unsupported role {message['role']!r}")

        if add_generation_prompt:
            add("<|im_start|>assistant\n")
            if not enable_thinking:
                add(EMPTY_THINK_BLOCK)
        return pieces


def _is_tool_response(content: str) -> bool:
    return content.startswith("<tool_response>") and content.endswith(
        "</tool_response>"
    )
