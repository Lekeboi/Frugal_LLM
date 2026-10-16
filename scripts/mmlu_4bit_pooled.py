"""Exploratory: is the ~1-point 4-bit deficit on MMLU real, judged across all three GPUs?

Not part of the pre-registered analysis (docs/study-protocol.md, change 7): added after the
per-GPU results showed AWQ and GPTQ-Int4 about one point below BF16 on MMLU on every GPU.

1. Determinism check: per-question answer agreement between BF16 run a and run b on each
   GPU, and for each format across the three GPUs. If the GPUs mostly give the same answer to
   the same question, the question (not question x GPU) is the unit of analysis.
2. For each MMLU question, correctness is averaged across the GPUs for each format, and the
   paired difference (4-bit minus BF16 run a) is taken.
3. A paired bootstrap 95% interval (10,000 resamples of questions) and a sign-flip
   permutation test (100,000 draws) on the mean of that difference, separately for AWQ and
   GPTQ-Int4, with Holm-adjusted p-values across the two.

    uv run --with numpy python scripts/mmlu_4bit_pooled.py study/2026-10-qwen2.5-7b \\
        t1-rtx3090 t1-rtx4090 t2-l4
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

TASK = "mmlu"
FORMATS = {"AWQ 4-bit": "awq", "GPTQ-Int4": "gptq4"}
BOOTSTRAP = 10_000
PERMUTATIONS = 100_000
SEED = 20261016


def load(folder: Path, tier: str, run: str) -> dict[str, dict]:
    (path,) = folder.glob(f"{tier}/*-quality-{TASK}-{tier}-{run}.json")
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return {i["id"]: i for i in items}


def agreement(a: dict[str, dict], b: dict[str, dict]) -> float:
    assert a.keys() == b.keys()
    return sum(a[k]["extracted"] == b[k]["extracted"] for k in a) / len(a)


def all_agree(runs: list[dict[str, dict]]) -> float:
    keys = runs[0].keys()
    return sum(len({r[k]["extracted"] for r in runs}) == 1 for k in keys) / len(keys)


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
    runs = {
        tier: {r: load(args.folder, tier, r) for r in ("bf16-a", "bf16-b", *FORMATS.values())}
        for tier in args.tiers
    }

    print("## 1. Determinism check (same extracted answer, per question)\n")
    print("| Comparison | " + " | ".join(args.tiers) + " |")
    print("|---|" + "---:|" * len(args.tiers))
    cells = [f"{100 * agreement(runs[t]['bf16-a'], runs[t]['bf16-b']):.1f}%" for t in args.tiers]
    print("| BF16 run a vs run b, same GPU | " + " | ".join(cells) + " |")
    print(
        "\n| Format | Same answer on all GPUs | "
        + " | ".join(f"{a} vs {b}" for a, b in itertools.combinations(args.tiers, 2))
        + " |"
    )
    print("|---|---:|" + "---:|" * len(list(itertools.combinations(args.tiers, 2))))
    for name, run in {"BF16 run a": "bf16-a", **FORMATS}.items():
        per_gpu = [runs[t][run] for t in args.tiers]
        pairs = [
            f"{100 * agreement(runs[a][run], runs[b][run]):.1f}%"
            for a, b in itertools.combinations(args.tiers, 2)
        ]
        print(f"| {name} | {100 * all_agree(per_gpu):.1f}% | " + " | ".join(pairs) + " |")

    print("\n## 2-3. Item-level difference from BF16, averaged across the GPUs\n")
    rng = np.random.default_rng(SEED)
    keys = sorted(runs[args.tiers[0]]["bf16-a"])
    results = []
    for name, run in FORMATS.items():
        diffs = np.mean(
            [
                [
                    float(runs[t][run][k]["correct"]) - float(runs[t]["bf16-a"][k]["correct"])
                    for k in keys
                ]
                for t in args.tiers
            ],
            axis=0,
        )
        observed = diffs.mean()
        boots = rng.choice(diffs, size=(BOOTSTRAP, diffs.size)).mean(axis=1)
        low, high = np.percentile(boots, (2.5, 97.5))
        signs = rng.choice((-1.0, 1.0), size=(PERMUTATIONS, diffs.size))
        null = (signs * diffs).mean(axis=1)
        p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (PERMUTATIONS + 1)
        results.append((name, observed, low, high, p))
    adjusted = holm([r[4] for r in results])
    print(
        f"{len(keys)} questions; {BOOTSTRAP:,} bootstrap resamples; "
        f"{PERMUTATIONS:,} sign-flip draws; Holm across the {len(results)} formats.\n"
    )
    print("| Format | Mean difference, points | Bootstrap 95% CI | p (sign-flip) | p (Holm) |")
    print("|---|---:|---:|---:|---:|")
    for (name, observed, low, high, p), p_holm in zip(results, adjusted, strict=True):
        print(
            f"| {name} | {100 * observed:+.2f} | {100 * low:+.2f} to {100 * high:+.2f} "
            f"| {p:.3f} | {p_holm:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
