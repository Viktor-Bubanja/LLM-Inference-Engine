"""Benchmark generation speed with and without the KV cache.

Reports prefill and decode tokens per second, and memory, for one sequence,
then the per-turn latency of an 8-turn chat with and without reusing the
cache across turns: ``uv run python -m llm_from_scratch.bench_generate``.
"""

import argparse
import resource
import sys
import time

import torch

from llm_from_scratch.causal_lm import CausalLM
from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.device import get_device
from llm_from_scratch.generation import ChatSession, generate
from llm_from_scratch.hf_loading import QWEN3_1_7B, load_pretrained, load_tokenizer

CHAT = [
    "Can you suggest a vegetarian dinner I can cook in 30 minutes?",
    "I don't have chickpeas. What can I use instead?",
    "How long will the leftovers keep in the fridge?",
    "Could I freeze it instead?",
    "What side dish would go with it?",
    "Is there a dessert that uses similar spices?",
    "How would I make the whole meal for eight people?",
    "Write me a shopping list for that.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark generation speed with and without the KV cache."
    )
    parser.add_argument("--model", default=QWEN3_1_7B)
    parser.add_argument("--device", default=str(get_device()))
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16", "float16"]
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument(
        "--reply-tokens", type=int, default=64, help="Tokens per chat reply."
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_gb(device: torch.device) -> float:
    """Peak allocated memory on CUDA. MPS has no peak counter, so it reports
    the memory the driver currently holds. On CPU it reports the process's
    peak resident set size, which never goes down between runs.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / 2**30 if sys.platform == "darwin" else max_rss / 2**20


def time_generation(
    model: CausalLM, prompt: torch.Tensor, new_tokens: int, cache: bool
) -> tuple[float, float]:
    """Prefill and decode speed in tokens per second. Prefill is the time to
    the first token; decode covers every token after it.
    """
    device = prompt.device
    synchronize(device)
    start = time.perf_counter()
    stamps = []
    for token in generate(model, prompt, new_tokens, cache=cache):
        token.tolist()
        stamps.append(time.perf_counter())
    prefill = stamps[0] - start
    decode = stamps[-1] - stamps[0]
    return prompt.shape[1] / prefill, (len(stamps) - 1) / decode


def time_chat(
    model: CausalLM, chat_format: ChatFormat, reply_tokens: int, reuse_cache: bool
) -> list[tuple[int, float]]:
    """For each turn of ``CHAT``: the rendered prompt length and the seconds
    taken to generate the whole reply.
    """
    session = ChatSession(
        model, chat_format, max_new_tokens=reply_tokens, reuse_cache=reuse_cache
    )
    turns = []
    for message in CHAT:
        start = time.perf_counter()
        list(session.reply(message))
        seconds = time.perf_counter() - start
        prompt_len = len(chat_format.render(session.messages[:-1]).token_ids)
        turns.append((prompt_len, seconds))
    return turns


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = load_pretrained(args.model, dtype=dtype, device=device)
    torch.manual_seed(0)
    prompt = torch.randint(
        model.config.vocab_size, (1, args.prompt_tokens), device=device
    )
    time_generation(model, prompt[:, :16], 4, cache=True)

    print(
        f"{args.model}, {args.prompt_tokens} prompt tokens, "
        f"{args.new_tokens} new tokens\n"
    )
    print("| device | dtype | KV cache | prefill tok/s | decode tok/s | memory GB |")
    print("|---|---|---|---|---|---|")
    for cache in (True, False):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prefill, decode = time_generation(model, prompt, args.new_tokens, cache)
        print(
            f"| {device.type} | {args.dtype} | {'yes' if cache else 'no'} "
            f"| {prefill:,.0f} | {decode:,.1f} | {memory_gb(device):.2f} |"
        )

    chat_format = ChatFormat(load_tokenizer(args.model))
    print(f"\n{len(CHAT)}-turn chat, up to {args.reply_tokens} tokens per reply\n")
    print("| turn | prompt tokens | cache reused (s) | from scratch (s) |")
    print("|---|---|---|---|")
    reused = time_chat(model, chat_format, args.reply_tokens, reuse_cache=True)
    scratch = time_chat(model, chat_format, args.reply_tokens, reuse_cache=False)
    for turn, ((prompt_len, reused_s), (_, scratch_s)) in enumerate(
        zip(reused, scratch), start=1
    ):
        print(f"| {turn} | {prompt_len} | {reused_s:.2f} | {scratch_s:.2f} |")


if __name__ == "__main__":
    main()
