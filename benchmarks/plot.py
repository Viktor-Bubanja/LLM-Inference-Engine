"""Draw the README's plots from ``benchmarks/results`` into ``docs/img`` (a
light and a dark version of each), and print the README's tables:
``uv run python benchmarks/plot.py``.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).parent / "results"
IMG = Path(__file__).parent.parent / "docs" / "img"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "cached": "#2a78d6",
        "uncached": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "cached": "#3987e5",
        "uncached": "#d95926",
    },
}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def short_name(model: str) -> str:
    return model.removeprefix("Qwen/")


def line_chart(
    theme: dict[str, str],
    x: list[float],
    series: list[tuple[str, str, str, list[float]]],
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    log_x: bool = False,
) -> None:
    """``series`` holds ``(label, color key, marker, values)``. Each line is
    labelled at its last point and in the legend.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for label, color, marker, values in series:
        ax.plot(
            x,
            values,
            color=theme[color],
            linewidth=2,
            marker=marker,
            markersize=7,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            label=label,
        )
        ax.annotate(
            f"{values[-1]:,.0f}" if values[-1] >= 10 else f"{values[-1]:.2f}",
            (x[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=theme["text"],
            fontsize=10,
        )
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{v:,}" for v in x])
        ax.minorticks_off()
    else:
        ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=10)
    ax.set_xlabel(xlabel, color=theme["muted"], fontsize=10)
    ax.set_ylabel(ylabel, color=theme["muted"], fontsize=10)
    fig.text(
        0.01, 0.97, title, color=theme["text"], fontsize=13, weight="bold", va="top"
    )
    fig.text(0.01, 0.905, subtitle, color=theme["muted"], fontsize=10, va="top")
    legend = ax.legend(
        frameon=False, loc="upper left", fontsize=10, labelcolor=theme["text"]
    )
    legend.set_zorder(5)
    fig.tight_layout(rect=(0, 0, 0.97, 0.86))
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def plot_decode(theme_name: str, theme: dict[str, str]) -> None:
    data = load("decode")
    rows = data["rows"]
    line_chart(
        theme,
        [r["context"] for r in rows],
        [
            ("Recompute everything", "uncached", "s", [r["no_cache_ms"] for r in rows]),
            ("KV cache", "cached", "o", [r["cache_ms"] for r in rows]),
        ],
        "Without a cache, every new token costs a full pass over the context",
        f"{short_name(data['model'])}, {data['dtype']}, "
        f"{data['machine']['chip']} ({data['device'].upper()}), "
        f"median of {data['repeats']} runs",
        "Tokens already in the context",
        "Milliseconds per new token",
        IMG / f"decode-{theme_name}.png",
        log_x=True,
    )


def plot_chat(theme_name: str, theme: dict[str, str]) -> None:
    data = load("chat")
    rows = data["rows"]
    line_chart(
        theme,
        [r["turn"] for r in rows],
        [
            (
                "Re-prefill the whole chat each turn",
                "uncached",
                "s",
                [r["scratch_first_token_s"] for r in rows],
            ),
            (
                "Reuse the cache across turns",
                "cached",
                "o",
                [r["reused_first_token_s"] for r in rows],
            ),
        ],
        "Reusing the cache keeps the wait for the first token flat",
        f"{short_name(data['model'])}, {data['dtype']}, "
        f"{data['machine']['chip']} ({data['device'].upper()}), "
        f"up to {data['reply_tokens']} tokens per reply",
        "Turn",
        "Seconds to the first token of the reply",
        IMG / f"chat-{theme_name}.png",
    )


def print_tables() -> None:
    decode = load("decode")
    print("Decode latency (ms per new token)\n")
    print("| context | KV cache | recompute | speed-up |")
    print("|---:|---:|---:|---:|")
    for r in decode["rows"]:
        print(
            f"| {r['context']:,} | {r['cache_ms']:.1f} | {r['no_cache_ms']:,.1f} "
            f"| {r['no_cache_ms'] / r['cache_ms']:.1f}× |"
        )

    chat = load("chat")
    print("\nChat turns (seconds)\n")
    print(
        "| turn | prompt tokens | first token, cache reused "
        "| first token, re-prefilled | whole reply, reused | whole reply, re-prefilled |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for r in chat["rows"]:
        print(
            f"| {r['turn']} | {r['prompt_tokens']:,} "
            f"| {r['reused_first_token_s']:.2f} | {r['scratch_first_token_s']:.2f} "
            f"| {r['reused_reply_s']:.1f} | {r['scratch_reply_s']:.1f} |"
        )

    kv = load("kv-memory")
    print("\nKV cache size (bf16)\n")
    print(
        "| model | KV heads (GQA) | per token | at 32k tokens "
        "| with one KV head per query head |"
    )
    print("|---|---:|---:|---:|---:|")
    for r in kv["rows"]:
        gqa, mha = r["gqa_bytes_per_token"], r["mha_bytes_per_token"]
        print(
            f"| {short_name(r['model'])} | {r['gqa_kv_heads']} of {r['mha_kv_heads']} "
            f"| {gqa / 2**10:.0f} KiB | {gqa * 2**15 / 2**30:.1f} GiB "
            f"| {mha / 2**10:.0f} KiB, {mha * 2**15 / 2**30:.1f} GiB |"
        )

    hf = load("hf")
    print(
        f"\nAgainst transformers ({hf['prompt_tokens']}-token prompt, "
        f"{hf['new_tokens']} new tokens, tokens/s)\n"
    )
    print("| model | prefill (ours) | prefill (HF) | decode (ours) | decode (HF) |")
    print("|---|---:|---:|---:|---:|")
    for r in hf["rows"]:
        print(
            f"| {short_name(r['model'])} | {r['ours_prefill_tok_s']:,.0f} "
            f"| {r['hf_prefill_tok_s']:,.0f} | {r['ours_decode_tok_s']:.1f} "
            f"| {r['hf_decode_tok_s']:.1f} |"
        )

    parity = load("parity")
    print("\nParity with transformers (fp32, CPU)\n")
    print("| model | max \\|Δ logit\\| | greedy tokens identical |")
    print("|---|---:|---|")
    for r in parity["rows"]:
        same = (
            f"{r['greedy_tokens']} of {r['greedy_tokens']}"
            if r["first_mismatch"] is None
            else f"first {r['first_mismatch']} of {r['greedy_tokens']}"
        )
        diff = r["max_abs_logit_diff"]
        diff_text = "0 (bit-identical)" if diff == 0 else f"{diff:.1e}"
        print(f"| {short_name(r['model'])} | {diff_text} | {same} |")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for theme_name, theme in THEMES.items():
        plot_decode(theme_name, theme)
        plot_chat(theme_name, theme)
    print_tables()


if __name__ == "__main__":
    main()
