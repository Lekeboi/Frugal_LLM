"""Turn one tier's study results into the tables and charts for the write-up.

Analysis only: it reads saved result files and never contacts a server, so a presentation
change never needs a (paid) re-run. Every number comes from frugal-llm's own analysis code
(`report`, `compare`, `cost`), so the write-up cannot drift from what the tool prints.

    uv run --with matplotlib python scripts/analyse_study.py results/t1-rtx4090 \\
        --tier t1-rtx4090 --out docs/results/t1-rtx4090 \\
        --api "name=together-qwen2.5-7b-turbo,in=0.30,out=0.30,source=https://www.together.ai/pricing,date=2026-10-15"

Charts follow one fixed format-to-colour mapping (validated for colour-blind separation),
carry direct labels as well as a legend, and every chart's numbers are also in the tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from matplotlib.ticker import FuncFormatter, MaxNLocator

from frugal_llm import compare as cmp
from frugal_llm import cost_report
from frugal_llm.cost import ApiPrice
from frugal_llm.metrics import summarise
from frugal_llm.report import fmt_ms, fmt_rate, fmt_spread, load_document, runs_from_document

FORMATS = ("bf16", "awq", "gptq4", "gptq8")
FORMAT_NAMES = {"bf16": "BF16", "awq": "AWQ 4-bit", "gptq4": "GPTQ 4-bit", "gptq8": "GPTQ 8-bit"}
# Reference categorical slots 1-4, in fixed order: a format keeps its colour on every chart.
COLOURS = {"bf16": "#2a78d6", "awq": "#eb6834", "gptq4": "#1baf7a", "gptq8": "#eda100"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)  # fmt: skip
TASKS = ("gsm8k", "mmlu")


# --- loading -----------------------------------------------------------------------


def find_bench(folder: Path, tier: str) -> dict[str, dict]:
    found = {}
    for fmt in FORMATS:
        paths = [p for p in folder.glob(f"*{tier}-{fmt}.json") if "-quality-" not in p.name]
        if paths:
            found[fmt] = load_document(sorted(paths)[-1])
    return found


def find_quality(folder: Path, tier: str, task: str) -> dict[str, dict]:
    found = {}
    for path in folder.glob(f"*-quality-{task}-*{tier}-*.json"):
        match = re.search(rf"{re.escape(tier)}-(\w+?)(-a|-b)?\.json$", path.name)
        if match and match.group(1) in FORMATS:
            found[match.group(1) + (match.group(2) or "")] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return found


# --- tables ------------------------------------------------------------------------


def environment_section(bench: dict[str, dict], folder: Path) -> list[str]:
    doc = next(iter(bench.values()))
    meta = doc["environment"].get("meta") or {}
    server = doc["environment"].get("server") or {}
    lines = ["## Environment", ""]
    for key, label in (("gpu", "GPU"), ("driver", "Driver"), ("vllm", "vLLM"),
                       ("provider", "Provider"), ("rate_usd_per_hour", "Rate, USD/hour"),
                       ("rate_date", "Rate date")):  # fmt: skip
        if meta.get(key):
            lines.append(f"- {label}: {meta[key]}")
    if server.get("system_fingerprint"):
        lines.append(f"- Server build: `{server['system_fingerprint']}`")
    lines.append(f"- frugal-llm: {doc['environment']['frugal_llm_version']}")
    host = folder / "host-info.txt"
    if host.exists():
        text = host.read_text(encoding="utf-8", errors="replace")
        cpu = re.search(r"Model name:\s*(.+)", text)
        mem = re.search(r"Mem:\s+(\d+)", text)
        if cpu:
            lines.append(f"- Host CPU: {cpu.group(1).strip()}")
        if mem:
            lines.append(f"- Host memory: {mem.group(1)} GB")
    lines.append("- Weights (pinned revisions):")
    for fmt, d in bench.items():
        lines.append(
            f"  - {FORMAT_NAMES[fmt]}: `{(d['environment'].get('meta') or {}).get('weights')}`"
        )
    return [*lines, ""]


def latency_section(bench: dict[str, dict]) -> tuple[list[str], dict]:
    lines = [
        "## Latency and throughput",
        "",
        "| Format | Conc | TTFT p50 | TTFT p95 | ITL p50 | Output tok/s, median (range) | Err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    data: dict = {}
    for fmt, doc in bench.items():
        for s in summarise(runs_from_document(doc), None):
            lines.append(
                f"| {FORMAT_NAMES[fmt]} | {s.concurrency} | {fmt_ms(s.ttft_ms[50])} | "
                f"{fmt_ms(s.ttft_ms[95])} | {fmt_ms(s.itl_ms[50])} | "
                f"{fmt_spread(s.output_tokens_per_sec, 1)} | {fmt_rate(s.error_rate)} |"
            )
            tps = s.output_tokens_per_sec.median if s.output_tokens_per_sec else None
            data.setdefault(fmt, []).append(
                {
                    "concurrency": s.concurrency,
                    "ttft_p95_ms": s.ttft_ms[95],
                    "output_tokens_per_sec": tps,
                }
            )
    return [*lines, ""], data


def quality_section(folder: Path, tier: str) -> tuple[list[str], dict]:
    lines, data = ["## Quality against BF16", ""], {}
    for task in TASKS:
        runs = find_quality(folder, tier, task)
        if "bf16-a" not in runs:
            continue
        candidates = [runs[f] for f in FORMATS[1:] if f in runs]
        floor = runs.get("bf16-b")
        results, floor_cmp, warnings = cmp.compare_runs(runs["bf16-a"], candidates, floor)
        extra = [*([floor] if floor else []), *candidates]
        lines += [cmp.render_markdown(runs["bf16-a"], results, floor_cmp, warnings, extra), ""]
        data[task] = [
            {
                "format": f,
                "difference": r.difference,
                "ci": list(r.difference_ci),
                "answer_agreement": r.answer_agreement,
            }
            for f, r in zip([f for f in FORMATS[1:] if f in runs], results, strict=True)
        ]
        if floor_cmp:
            data[task + "_noise_floor_agreement"] = floor_cmp.answer_agreement
    if not data:  # e.g. the reference run, which measures latency only
        return ["## Quality against BF16", "", "No quality runs in this tier.", ""], data
    return [*lines, "![Quality](quality.png)", ""], data


def cost_section(bench: dict[str, dict], prices: list[ApiPrice]) -> tuple[list[str], dict]:
    lines, data = ["## Cost", ""], {}
    for fmt, doc in bench.items():
        meta = doc["environment"].get("meta") or {}
        rate = cost_report.HourlyRate(
            float(meta.get("rate_usd_per_hour") or doc["config"]["cost_per_hour"]),
            f"{meta.get('provider', 'unknown provider')}, {meta.get('gpu', '')}",
            meta.get("rate_date", doc["started_at"][:10]),
        )
        levels = cost_report.analyse(doc, rate, prices)
        report = cost_report.render_markdown(doc, levels, rate, "USD")
        lines += [f"### {FORMAT_NAMES[fmt]}", "", *report.splitlines()[2:], ""]
        best = max((lv for lv in levels if lv.capacity_rps), key=lambda lv: lv.capacity_rps)
        data[fmt] = {"rate": rate.amount, "concurrency": best.concurrency,
                     "curve": cost_report.curve(best, rate)}  # fmt: skip
    return lines, data


# --- charts ------------------------------------------------------------------------


def _axes(plt, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    return fig, ax


def _end_labels(ax, ends: list[tuple[float, float, str]]) -> None:
    """Direct labels at line ends, nudged apart so close values never overlap."""
    low, high = ax.get_ylim()
    gap = (high - low) * 0.05
    placed: list[float] = []
    for y, x, fmt in sorted(ends):
        y_label = max([y, *(p + gap for p in placed)])
        placed.append(y_label)
        # x * 1.08 sits just right of the last point on the log-2 axis.
        ax.text(x * 1.08, y_label, FORMAT_NAMES[fmt], va="center", fontsize=8, color=INK_2)


def _by_concurrency_chart(plt, data: dict, key: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = _axes(plt, title, "Concurrent requests", ylabel)
    ends = []
    for fmt in FORMATS:
        points = [p for p in data.get(fmt, []) if p[key] is not None]
        if not points:
            continue
        xs, ys = [p["concurrency"] for p in points], [p[key] for p in points]
        ax.plot(xs, ys, color=COLOURS[fmt], linewidth=2, marker="o", markersize=5,
                label=FORMAT_NAMES[fmt])  # fmt: skip
        ends.append((ys[-1], xs[-1], fmt))
    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    _end_labels(ax, ends)
    xs = sorted({p["concurrency"] for v in data.values() for p in v})
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _quality_chart(plt, data: dict, path: Path) -> None:
    tasks = [t for t in TASKS if t in data]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.2, 3.2), dpi=160, sharey=True,
                             squeeze=False)  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes[0], tasks, strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.axvline(0, color=AXIS, linewidth=1)
        rows = data[task]
        for i, row in enumerate(rows):
            lo, hi = (v * 100 for v in row["ci"])
            ax.plot([lo, hi], [i, i], color=COLOURS[row["format"]], linewidth=2)
            ax.plot(row["difference"] * 100, i, "o", color=COLOURS[row["format"]], markersize=7)
        ax.set_yticks(range(len(rows)), [FORMAT_NAMES[r["format"]] for r in rows])
        ax.set_title(task.upper(), loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Accuracy vs BF16, points (95% CI)", color=INK_2, fontsize=8)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _thousands(value: float, _pos: int) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    return f"{value / 1e3:g}k" if value else "0"


def _breakeven_chart(plt, data: dict, path: Path) -> None:
    if not data:
        return
    fig, ax = _axes(plt, "Daily cost: self-hosted vs API", "Requests per day", "USD per day")
    api_done = set()
    for fmt in FORMATS:
        curve = data.get(fmt, {}).get("curve")
        if not curve:
            continue
        xs = [p["requests_per_day"] for p in curve]
        ax.step(xs, [p["self_hosted"] for p in curve], where="post", color=COLOURS[fmt],
                linewidth=2, label=f"{FORMAT_NAMES[fmt]} (self-hosted)")  # fmt: skip
        for name in [k for k in curve[0] if k not in ("requests_per_day", "self_hosted")]:
            if name in api_done:
                continue
            api_done.add(name)
            ys = [p[name] for p in curve]
            ax.plot(xs, ys, color=MUTED, linewidth=2)
            # Label at the line's own end, inside the plot, so it reads as that line's name.
            ax.annotate(f"API: {name}", (xs[-1], ys[-1]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8, color=INK_2)  # fmt: skip
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --- main ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="one tier's result folder")
    parser.add_argument("--tier", required=True, help="the TIER label used on the pod")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api", action="append", default=[], help="price spec, as for `cost`")
    parser.add_argument("--note", action="append", default=[], help="a line for the summary")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = find_bench(args.folder, args.tier)
    if not bench:
        raise SystemExit(f"no bench results for tier {args.tier!r} in {args.folder}")
    prices = [ApiPrice.parse(spec) for spec in args.api]
    args.out.mkdir(parents=True, exist_ok=True)

    latency_md, latency = latency_section(bench)
    quality_md, quality = quality_section(args.folder, args.tier)
    cost_md, cost = cost_section(bench, prices)
    summary = [
        f"# Results: {args.tier}",
        "",
        *(f"> {n}" for n in args.note),
        *([""] if args.note else []),
        *environment_section(bench, args.folder),
        *latency_md,
        "![Throughput](throughput.png)",
        "",
        "![TTFT p95](ttft_p95.png)",
        "",
        *quality_md,
        *cost_md,
        "![Break-even](breakeven.png)",
        "",
    ]
    (args.out / "summary.md").write_text("\n".join(summary), encoding="utf-8", newline="\n")
    (args.out / "chart-data.json").write_text(
        json.dumps({"latency": latency, "quality": quality, "cost": cost}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _by_concurrency_chart(plt, latency, "output_tokens_per_sec", "Output throughput",
                          "Output tokens per second", args.out / "throughput.png")  # fmt: skip
    _by_concurrency_chart(plt, latency, "ttft_p95_ms", "Time to first token, p95",
                          "Milliseconds", args.out / "ttft_p95.png")  # fmt: skip
    _quality_chart(plt, quality, args.out / "quality.png")
    _breakeven_chart(plt, cost, args.out / "breakeven.png")
    print(f"wrote {args.out}/summary.md, chart-data.json and charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
