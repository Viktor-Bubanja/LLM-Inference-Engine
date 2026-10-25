"""Chat with Qwen3-1.7B in the terminal, streaming each reply:
``uv run python -m llm_from_scratch.chat``.
"""

import argparse
from collections.abc import Iterable, Iterator

import torch

from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.device import get_device
from llm_from_scratch.generation import ChatSession
from llm_from_scratch.hf_loading import QWEN3_1_7B, load_pretrained, load_tokenizer
from llm_from_scratch.sampling import Sampler
from llm_from_scratch.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Qwen3 in the terminal.")
    parser.add_argument("--model", default=QWEN3_1_7B)
    parser.add_argument("--device", default=str(get_device()))
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def stream_text(tokenizer: Tokenizer, token_ids: Iterable[int]) -> Iterator[str]:
    """Decode a stream of token ids into pieces of text. One character can
    span several byte-level tokens, so text is held back while it ends in an
    incomplete character.
    """
    ids: list[int] = []
    shown = 0
    for token_id in token_ids:
        ids.append(token_id)
        text = tokenizer.decode(ids)
        if not text.endswith("\ufffd"):
            yield text[shown:]
            shown = len(text)


def main() -> None:
    args = parse_args()
    model = load_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device=args.device
    )
    chat_format = ChatFormat(load_tokenizer(args.model))
    session = ChatSession(
        model,
        chat_format,
        sampler=Sampler(temperature=args.temperature, top_p=args.top_p),
        system_prompt=args.system,
        seed=args.seed,
    )
    print("Ctrl-D to quit.")
    while True:
        try:
            message = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("\nassistant> ", end="", flush=True)
        reply = (t for t in session.reply(message) if t not in chat_format.stop_ids)
        for piece in stream_text(chat_format.tokenizer, reply):
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
