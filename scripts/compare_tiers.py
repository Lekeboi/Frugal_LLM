"""Put the study's GPU tiers side by side: the comparison claim 2 rests on.

Analysis only, like analyse_study.py (whose loading and analysis it reuses): it reads saved
result folders and never contacts a server.

    uv run --with matplotlib python scripts/compare_tiers.py \\
        results/t1-rtx4090 results/t2-l4 results/ref-a100 --out results/analysis/tiers

Each folder is one tier, named after its TIER label. Cost per 1M output tokens is the hourly
rate divided by measured output throughput: the GPU kept 100% busy at that concurrency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyse_study as one

LEVELS = (1, 16, 64)  # single user, the quality-run setting, the highest measured


def load_tier(folder: Path) -> dict | None:
    tier = folder.name
    bench = one.find_bench(folder, tier)
    if not bench:
        return None
    meta = next(iter(bench.values()))["environment"].get("meta") or {}
    _, latency = one.latency_section(bench)
    accuracy = {}
    for task in one.TASKS:
        for name, doc in one.find_quality(folder, tier, task).items():
            if name == "bf16-b":  # the noise-floor rerun; run a is the baseline
                continue
            fmt, items = name.removesuffix("-a"), doc["items"]
            accuracy.setdefault(task, {})[fmt] = sum(1 for i in items if i["correct"]) / len(items)
    return {
        "tier": tier,
        "gpu": meta.get("gpu", tier),
        "rate": float(meta["rate_usd_per_hour"]),
        "rate_date": meta.get("rate_date", ""),
        "provider": meta.get("provider", ""),
        "latency": latency,
        "accuracy": accuracy,
    }


def at(tier: dict, fmt: str, concurrency: int, key: str) -> float | None:
    for point in tier["latency"].get(fmt, []):
        if point["concurrency"] == concurrency:
            return point[key]
    return None


def cost_per_mtok(tier: dict, fmt: str, concurrency: int) -> float | None:
    tok_s = at(tier, fmt, concurrency, "output_tokens_per_sec")
    return tier["rate"] / (tok_s * 3600) * 1e6 if tok_s else None


def _short(gpu: str) -> str:
    return gpu.replace("NVIDIA ", "").replace("GeForce ", "")


def _cell(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def _grid(tiers: list[dict], title: str, value, spec: str) -> list[str]:
    lines = [title, "", "| GPU | " + " | ".join(one.FORMAT_NAMES[f] for f in one.FORMATS) + " |"]
    lines.append("|---|" + "---:|" * len(one.FORMATS))
    for t in tiers:
        cells = [_cell(value(t, f), spec) for f in one.FORMATS]
        lines.append(f"| {_short(t['gpu'])} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def summary(tiers: list[dict]) -> list[str]:
    lines = [
        "# Results across GPU tiers",
        "",
        "## GPUs",
        "",
        "| GPU | Tier | USD/hour | Provider | Rate date |",
        "|---|---|---:|---|---|",
    ]
    for t in tiers:
        lines.append(
            f"| {_short(t['gpu'])} | {t['tier']} | {t['rate']:.2f} | "
            f"{t['provider']} | {t['rate_date']} |"
        )
    lines += ["", "## Cost per 1M output tokens", ""]
    for c in (64, 16):
        lines += _grid(
            tiers, f"At concurrency {c}, USD:", lambda t, f, c=c: cost_per_mtok(t, f, c), ".3f"
        )
    lines += ["![Cost per 1M output tokens](cost_per_mtok.png)", ""]

    # Findings computed from the numbers, so the prose cannot drift from the tables.
    lines += ["Cheapest GPU per format, at concurrency 64:", ""]
    for fmt in one.FORMATS:
        costs = [(cost_per_mtok(t, fmt, 64), t) for t in tiers if cost_per_mtok(t, fmt, 64)]
        if len(costs) > 1:
            costs.sort(key=lambda x: x[0])
            ranked = ", ".join(f"{_short(t['gpu'])} {c:.3f}" for c, t in costs)
            lines.append(f"- {one.FORMAT_NAMES[fmt]}: {ranked} (USD per 1M output tokens)")
    cheapest_hour = min(tiers, key=lambda t: t["rate"])
    lines += [
        "",
        f"- Cheapest per hour: {_short(cheapest_hour['gpu'])} "
        f"(USD {cheapest_hour['rate']:.2f}). Hourly rate does not rank cost per token: "
        "throughput decides it.",
        "",
    ]

    lines += ["## Throughput", ""]
    for c in (1, 64):
        what = "one request at a time" if c == 1 else f"{c} concurrent requests"
        lines += _grid(
            tiers,
            f"Output tokens per second, {what}:",
            lambda t, f, c=c: at(t, f, c, "output_tokens_per_sec"),
            ".1f",
        )
    lines += _grid(
        tiers,
        "Time to first token, p95 at concurrency 16, ms:",
        lambda t, f: at(t, f, 16, "ttft_p95_ms"),
        ".0f",
    )
    lines += ["![Throughput](throughput.png)", ""]

    lines += [
        "## Quality across tiers",
        "",
        "The same weights should score the same on every GPU; this is the cross-check. "
        "BF16 is the baseline run (run a; its rerun is the noise floor).",
        "",
    ]
    for task in one.TASKS:
        rated = [t for t in tiers if task in t["accuracy"]]
        if not rated:
            continue
        lines += _grid(
            rated,
            f"{task.upper()} accuracy, %:",
            lambda t, f, task=task: _pct(t["accuracy"][task].get(f)),
            ".1f",
        )
        bf16 = [t["accuracy"][task]["bf16"] for t in rated if "bf16" in t["accuracy"][task]]
        if len(bf16) > 1:
            spread = (max(bf16) - min(bf16)) * 100
            lines += [f"BF16 {task.upper()} spread across GPUs: {spread:.1f} points.", ""]
    lines.append("Tiers without quality runs (the reference) are left out of these tables.")
    return [*lines, ""]


def _pct(x: float | None) -> float | None:
    return None if x is None else x * 100


def _bars(plt, tiers: list[dict], panels: list[tuple[str, object, str]], path: Path) -> None:
    """Grouped bars, one group per GPU, one bar per format in its fixed colour."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(5.6, 3.6 * len(panels) + 0.4), 3.6), dpi=160, squeeze=False
    )
    fig.patch.set_facecolor(one.SURFACE)
    width = 0.8 / len(one.FORMATS)
    for ax, (title, value, label_spec) in zip(axes[0], panels, strict=True):
        ax.set_facecolor(one.SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(one.AXIS)
        ax.grid(axis="y", color=one.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=one.MUTED, labelsize=8, length=0)
        ax.set_title(title, loc="left", color=one.INK, fontsize=10, pad=10)
        for i, fmt in enumerate(one.FORMATS):
            for g, tier in enumerate(tiers):
                v = value(tier, fmt)
                if v is None:
                    continue
                x = g - 0.4 + width * (i + 0.5)
                # A surface-coloured edge keeps a visible gap between adjacent bars.
                ax.bar(
                    x,
                    v,
                    width,
                    color=one.COLOURS[fmt],
                    edgecolor=one.SURFACE,
                    linewidth=1.2,
                    label=one.FORMAT_NAMES[fmt] if g == 0 else None,
                )
        # Direct label only on each group's lowest bar (the cheapest / slowest option).
        for g, tier in enumerate(tiers):
            vals = [(value(tier, f), i) for i, f in enumerate(one.FORMATS) if value(tier, f)]
            if vals and label_spec:
                v, i = min(vals)
                x = g - 0.4 + width * (i + 0.5)
                ax.annotate(
                    format(v, label_spec),
                    (x, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=one.INK_2,
                )
        ticks = [f"{_short(t['gpu'])}\n${t['rate']:.2f}/h" for t in tiers]
        ax.set_xticks(range(len(tiers)), ticks, fontsize=8)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=one.INK_2,
        loc="lower center",
        ncol=len(one.FORMATS),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, facecolor=one.SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+", help="one result folder per tier")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [t for t in (load_tier(f) for f in args.folders) if t]
    if len(tiers) < 2:
        raise SystemExit("need at least two tiers with bench results")
    tiers.sort(key=lambda t: t["rate"])  # cheapest per hour first
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text("\n".join(summary(tiers)), encoding="utf-8", newline="\n")
    data = {
        t["tier"]: {
            **t,
            "cost_per_mtok": {f: {c: cost_per_mtok(t, f, c) for c in LEVELS} for f in one.FORMATS},
        }
        for t in tiers
    }
    (args.out / "chart-data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    _bars(
        plt,
        tiers,
        [("USD per 1M output tokens, concurrency 64", lambda t, f: cost_per_mtok(t, f, 64), ".2f")],
        args.out / "cost_per_mtok.png",
    )
    _bars(
        plt,
        tiers,
        [
            ("Output tok/s, 1 request", lambda t, f: at(t, f, 1, "output_tokens_per_sec"), ""),
            ("Output tok/s, 64 concurrent", lambda t, f: at(t, f, 64, "output_tokens_per_sec"), ""),
        ],
        args.out / "throughput.png",
    )
    print(f"wrote {args.out}/summary.md, chart-data.json and charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
