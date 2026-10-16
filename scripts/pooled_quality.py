"""Exploratory: is a quantised format's accuracy difference from BF16 consistent across GPUs?

Not part of the pre-registered analysis (docs/study-protocol.md, change 5): added after the
per-GPU results showed both 4-bit formats about one point lower on MMLU on every GPU.

The GPUs answered the same items with the same weights, so their results are not independent
samples and must not be pooled as if they were (that would multiply the evidence by the
number of GPUs). The unit here is the item: for each item, the format-minus-BF16 correctness
difference is averaged over the GPUs (and, for "both 4-bit", over the two 4-bit formats),
and the mean over items is tested against zero by an item-level sign-flip permutation test,
with a bootstrap 95% interval. The BF16 rerun against BF16 is the control. Every format and
both tasks are tested, so the 4-bit MMLU result is not singled out; p-values are Holm-corrected
across all of them.

    uv run --with numpy python scripts/pooled_quality.py study/2026-10-qwen2.5-7b \\
        t1-rtx3090 t1-rtx4090 t2-l4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("gsm8k", "mmlu")
GROUPS = {
    "BF16 rerun (control)": ("bf16-b",),
    "AWQ 4-bit": ("awq",),
    "GPTQ 4-bit": ("gptq4",),
    "GPTQ 8-bit": ("gptq8",),
    "Both 4-bit formats": ("awq", "gptq4"),
}
DRAWS = 200_000
SEED = 20261016


def correct(folder: Path, tier: str, task: str, run: str) -> dict[str, float]:
    (path,) = folder.glob(f"{tier}/*-quality-{task}-{tier}-{run}.json")
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return {i["id"]: float(bool(i["correct"])) for i in items}


def item_differences(folder: Path, tiers: list[str], task: str, runs: tuple[str, ...]):
    per_item = []
    for tier in tiers:
        base = correct(folder, tier, task, "bf16-a")
        for run in runs:
            other = correct(folder, tier, task, run)
            assert other.keys() == base.keys(), f"{tier} {run}: different items"
            per_item.append([other[k] - base[k] for k in sorted(base)])
    return np.mean(per_item, axis=0)  # one value per item


def test(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    observed = d.mean()
    signs = rng.choice((-1.0, 1.0), size=(DRAWS, d.size))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (DRAWS + 1)
    boots = rng.choice(d, size=(DRAWS // 20, d.size)).mean(axis=1)
    low, high = np.percentile(boots, (2.5, 97.5))
    return observed, low, high, p


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("tiers", nargs="+")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for task in TASKS:
        for name, runs in GROUPS.items():
            d = item_differences(args.folder, args.tiers, task, runs)
            rows.append((task, name, d.size, *test(d, rng)))
    tested = [i for i, r in enumerate(rows) if not r[1].startswith("BF16")]
    adjusted = dict(zip(tested, holm([rows[i][6] for i in tested]), strict=True))
    print(f"Item-level test across {', '.join(args.tiers)} (exploratory; see docstring)\n")
    print("| Task | Format | Items | Mean difference, points (95% CI) | p | p (Holm) |")
    print("|---|---|---:|---:|---:|---:|")
    for i, (task, name, n, mean, low, high, p) in enumerate(rows):
        ci = f"{mean * 100:+.2f} ({low * 100:+.2f} to {high * 100:+.2f})"
        holm_p = f"{adjusted[i]:.3f}" if i in adjusted else "control"
        print(f"| {task.upper()} | {name} | {n} | {ci} | {p:.3f} | {holm_p} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
