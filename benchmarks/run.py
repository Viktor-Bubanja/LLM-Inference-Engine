"""Run one benchmark and write its results to ``benchmarks/results/<name>.json``:
``uv run python benchmarks/run.py {decode,chat,kv-memory,hf,parity}``.

``plot.py`` turns the results into the README's plots and tables.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

from llm_from_scratch.bench_generate import CHAT, synchronize, time_generation
from llm_from_scratch.causal_lm import CausalLM
from llm_from_scratch.chat_format import ChatFormat
from llm_from_scratch.generation import ChatSession, generate
from llm_from_scratch.hf_loading import (
    PINNED_REVISIONS,
    QWEN3_0_6B_BASE,
    QWEN3_1_7B,
    download_checkpoint,
    load_pretrained,
    load_tokenizer,
)
from llm_from_scratch.kv_cache import KVCache
from llm_from_scratch.qwen3 import Qwen3Config

RESULTS = Path(__file__).parent / "results"
MODELS = [QWEN3_0_6B_BASE, QWEN3_1_7B]
PARITY_PROMPT = (
    "A key/value cache stores the attention keys and values of earlier tokens, so"
)


def machine() -> dict[str, str]:
    chip = platform.processor()
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {"chip": chip, "os": platform.platform(), "torch": torch.__version__}


def random_prompt(vocab_size: int, num_tokens: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(vocab_size, (1, num_tokens), generator=generator).to(device)


def median_of(runs: int, measure: Callable[[], float]) -> float:
    return statistics.median(measure() for _ in range(runs))


def bench_decode(args: argparse.Namespace) -> dict:
    """Milliseconds per decoded token as the context grows, with and without
    the KV cache.
    """
    model = load_pretrained(args.model, dtype=args.dtype, device=args.device)
    time_generation(model, random_prompt(1000, 16, args.device), 4, cache=True)
    rows = []
    for context in args.contexts:
        prompt = random_prompt(model.config.vocab_size, context, args.device)
        row = {"context": context}
        for cache in (True, False):

            def measure() -> float:
                _, decode = time_generation(model, prompt, args.new_tokens, cache)
                return 1000 / decode

            row["cache_ms" if cache else "no_cache_ms"] = median_of(
                args.repeats, measure
            )
        print(row)
        rows.append(row)
    return {"new_tokens": args.new_tokens, "repeats": args.repeats, "rows": rows}


def time_chat_turns(
    model: CausalLM, chat_format: ChatFormat, reply_tokens: int, reuse_cache: bool
) -> list[dict]:
    """For each turn of ``CHAT``: the rendered prompt length, the time to the
    first token of the reply, and the time to the whole reply.
    """
    session = ChatSession(
        model, chat_format, max_new_tokens=reply_tokens, reuse_cache=reuse_cache
    )
    turns = []
    for message in CHAT:
        start = time.perf_counter()
        stamps = [time.perf_counter() for _ in session.reply(message)]
        prompt_len = len(chat_format.render(session.messages[:-1]).token_ids)
        turns.append(
            {
                "prompt_tokens": prompt_len,
                "first_token_s": stamps[0] - start,
                "reply_s": stamps[-1] - start,
            }
        )
    return turns


def bench_chat(args: argparse.Namespace) -> dict:
    """Time to first token and to the whole reply, per turn of an 8-turn chat,
    reusing the KV cache across turns or prefilling the whole conversation
    again every turn.
    """
    model = load_pretrained(args.model, dtype=args.dtype, device=args.device)
    chat_format = ChatFormat(load_tokenizer(args.model))
    time_chat_turns(model, chat_format, 4, reuse_cache=True)
    reused = time_chat_turns(model, chat_format, args.reply_tokens, reuse_cache=True)
    scratch = time_chat_turns(model, chat_format, args.reply_tokens, reuse_cache=False)
    rows = []
    for turn, (r, s) in enumerate(zip(reused, scratch), start=1):
        row = {
            "turn": turn,
            "prompt_tokens": r["prompt_tokens"],
            "reused_first_token_s": r["first_token_s"],
            "scratch_first_token_s": s["first_token_s"],
            "reused_reply_s": r["reply_s"],
            "scratch_reply_s": s["reply_s"],
        }
        print(row)
        rows.append(row)
    return {"reply_tokens": args.reply_tokens, "rows": rows}


def bench_kv_memory(args: argparse.Namespace) -> dict:
    """KV cache size per token with grouped-query attention, against the same
    model with one KV head per query head. Measured from the cache's buffers.
    """
    rows = []
    for model in MODELS:
        hf_config = json.loads((download_checkpoint(model) / "config.json").read_text())
        config = Qwen3Config.from_hf_config(hf_config)
        row = {"model": model, "n_layers": config.n_layers}
        for name, n_kv_heads in (("gqa", config.n_kv_heads), ("mha", config.n_heads)):
            cache = KVCache(
                config.n_layers, 1, n_kv_heads, args.tokens, config.head_dim, args.dtype
            )
            nbytes = cache.k_buf.nbytes + cache.v_buf.nbytes
            row[f"{name}_kv_heads"] = n_kv_heads
            row[f"{name}_bytes_per_token"] = nbytes // args.tokens
        print(row)
        rows.append(row)
    return {"rows": rows}


class TimestampStreamer(BaseStreamer):
    """Records when each generated token reaches the host. ``generate`` puts
    the prompt first, then one token per step.
    """

    def __init__(self) -> None:
        self.stamps: list[float] = []
        self.seen_prompt = False

    def put(self, value: torch.Tensor) -> None:
        value.tolist()
        if self.seen_prompt:
            self.stamps.append(time.perf_counter())
        self.seen_prompt = True

    def end(self) -> None:
        pass


def time_hf_generation(
    hf_model: torch.nn.Module, prompt: torch.Tensor, new_tokens: int
) -> tuple[float, float]:
    """Prefill and decode speed of ``transformers`` generation, measured as in
    ``time_generation``.
    """
    streamer = TimestampStreamer()
    synchronize(prompt.device)
    start = time.perf_counter()
    hf_model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        streamer=streamer,
    )
    stamps = streamer.stamps
    return prompt.shape[1] / (stamps[0] - start), (len(stamps) - 1) / (
        stamps[-1] - stamps[0]
    )


def bench_hf(args: argparse.Namespace) -> dict:
    """Our generation against ``transformers``: same weights, dtype, device,
    prompt, greedy decoding and KV cache.
    """
    rows = []
    for model_name in MODELS:
        ours = load_pretrained(model_name, dtype=args.dtype, device=args.device)
        prompt = random_prompt(ours.config.vocab_size, args.prompt_tokens, args.device)
        time_generation(ours, prompt[:, :16], 4, cache=True)
        ours_prefill, ours_decode = (
            statistics.median(x)
            for x in zip(
                *(
                    time_generation(ours, prompt, args.new_tokens, cache=True)
                    for _ in range(args.repeats)
                )
            )
        )
        del ours

        hf_model = AutoModelForCausalLM.from_pretrained(
            download_checkpoint(model_name), dtype=args.dtype
        )
        hf_model = hf_model.to(args.device).eval()
        time_hf_generation(hf_model, prompt[:, :16], 4)
        hf_prefill, hf_decode = (
            statistics.median(x)
            for x in zip(
                *(
                    time_hf_generation(hf_model, prompt, args.new_tokens)
                    for _ in range(args.repeats)
                )
            )
        )
        del hf_model

        row = {
            "model": model_name,
            "ours_prefill_tok_s": ours_prefill,
            "hf_prefill_tok_s": hf_prefill,
            "ours_decode_tok_s": ours_decode,
            "hf_decode_tok_s": hf_decode,
        }
        print(row)
        rows.append(row)
    return {
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "rows": rows,
    }


def bench_parity(args: argparse.Namespace) -> dict:
    """Largest logit difference from ``transformers`` in fp32 on CPU, and
    whether greedy continuations agree token for token.
    """
    rows = []
    for model_name in MODELS:
        ours = load_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            download_checkpoint(model_name), dtype=torch.float32
        ).eval()
        prompt = torch.tensor([load_tokenizer(model_name).encode(PARITY_PROMPT)])
        with torch.no_grad():
            logit_diff = (ours(prompt) - hf_model(prompt).logits).abs().max().item()
        new_ids = torch.stack(list(generate(ours, prompt, args.new_tokens)), dim=1)
        expected = hf_model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=args.new_tokens,
            min_new_tokens=args.new_tokens,
            do_sample=False,
        )[:, prompt.shape[1] :]
        mismatches = (new_ids != expected).nonzero()
        row = {
            "model": model_name,
            "prompt_tokens": prompt.shape[1],
            "max_abs_logit_diff": logit_diff,
            "greedy_tokens": args.new_tokens,
            "first_mismatch": int(mismatches[0, 1]) if len(mismatches) else None,
        }
        print(row)
        rows.append(row)
        del ours, hf_model
    return {"device": "cpu", "dtype": "float32", "rows": rows}


BENCHMARKS = {
    "decode": bench_decode,
    "chat": bench_chat,
    "kv-memory": bench_kv_memory,
    "hf": bench_hf,
    "parity": bench_parity,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark.")
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=None)
    parser.add_argument("--reply-tokens", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    if args.model is None:
        args.model = QWEN3_1_7B if args.benchmark == "chat" else QWEN3_0_6B_BASE
    if args.new_tokens is None:
        args.new_tokens = {"decode": 32, "hf": 128, "parity": 64}.get(args.benchmark, 0)
    return args


def main() -> None:
    args = parse_args()
    result = BENCHMARKS[args.benchmark](args)
    settings = {
        "benchmark": args.benchmark,
        "device": args.device if args.benchmark != "parity" else "cpu",
        "dtype": str(args.dtype).removeprefix("torch."),
        "revisions": PINNED_REVISIONS,
        "machine": machine(),
    }
    if args.benchmark in ("decode", "chat"):
        settings["model"] = args.model
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.benchmark}.json"
    path.write_text(json.dumps(settings | result, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
