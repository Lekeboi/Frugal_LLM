"""Self-hosting against commercial APIs, from saved benchmark results.

Analysis only: it reads a results file and never contacts a server, so prices and rates
can be changed and the comparison redone without re-running a (paid) measurement.
Method: docs/cost-method.md.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from frugal_llm.cost import (
    SECONDS_PER_DAY,
    ApiPrice,
    BreakEven,
    api_cost_per_request,
    break_even,
    cost_per_million_tokens,
    daily_cost_api,
    daily_cost_self_hosted,
    self_hosted_cost_per_request,
)
from frugal_llm.metrics import is_stalled, summarise
from frugal_llm.report import runs_from_document


@dataclass(frozen=True)
class HourlyRate:
    amount: float
    source: str
    date: str


@dataclass(frozen=True)
class ApiComparison:
    price: ApiPrice
    api_per_request: float | None
    break_even: BreakEven | None


@dataclass(frozen=True)
class LevelCost:
    concurrency: int
    capacity_rps: float | None
    ttft_p95_ms: float | None
    prompt_tokens: float | None
    output_tokens: float | None
    self_per_request: float | None
    self_per_million_output: float | None
    comparisons: list[ApiComparison]


def _mean(values: list[float]) -> float | None:
    # A mean, not a quantile: cost is additive, so the average request is what is billed.
    return statistics.fmean(values) if values else None


def analyse(doc: dict, rate: HourlyRate, prices: list[ApiPrice]) -> list[LevelCost]:
    runs = runs_from_document(doc)
    levels = []
    for summary in summarise(runs, rate.amount):
        valid = [r for r in runs if r.concurrency == summary.concurrency and not is_stalled(r)]
        ok = [q for run in valid for q in run.requests if q.ok]
        prompt = _mean([q.prompt_tokens for q in ok if q.prompt_tokens is not None])
        output = _mean([q.output_tokens for q in ok])
        capacity = summary.requests_per_sec.median if summary.requests_per_sec else None
        tps = summary.output_tokens_per_sec.median if summary.output_tokens_per_sec else 0.0
        self_cost = self_hosted_cost_per_request(rate.amount, capacity) if capacity else None
        comparisons = []
        for price in prices:
            # Without the server's prompt-token count the input side can't be priced, and
            # pricing output alone would flatter self-hosting. So no comparison is made.
            api = (
                api_cost_per_request(price, prompt, output)
                if prompt is not None and output is not None
                else None
            )
            be = break_even(rate.amount, capacity, api) if api and capacity else None
            comparisons.append(ApiComparison(price, api, be))
        levels.append(
            LevelCost(
                concurrency=summary.concurrency,
                capacity_rps=capacity,
                ttft_p95_ms=summary.ttft_ms[95],
                prompt_tokens=prompt,
                output_tokens=output,
                self_per_request=self_cost,
                self_per_million_output=cost_per_million_tokens(rate.amount, tps),
                comparisons=comparisons,
            )
        )
    return levels


def curve(level: LevelCost, rate: HourlyRate, points: int = 60) -> list[dict]:
    """Daily cost of each option across traffic from zero to three instances' capacity:
    the data behind a break-even chart. Self-hosting steps up as instances are added."""
    if not level.capacity_rps:
        return []
    top = 3 * level.capacity_rps * SECONDS_PER_DAY
    rows = []
    for i in range(points + 1):
        traffic = top * i / points
        self_hosted = daily_cost_self_hosted(traffic, rate.amount, level.capacity_rps)
        row = {"requests_per_day": traffic, "self_hosted": self_hosted}
        for c in level.comparisons:
            if c.api_per_request is not None:
                row[c.price.name] = daily_cost_api(traffic, c.api_per_request)
        rows.append(row)
    return rows


# --- rendering ----------------------------------------------------------------------


def _money(x: float | None, currency: str, digits: int = 2) -> str:
    return "n/a" if x is None else f"{currency} {x:,.{digits}f}"


def _count(x: float) -> str:
    return f"{x:,.0f}"


def render_markdown(doc: dict, levels: list[LevelCost], rate: HourlyRate, currency: str) -> str:
    cfg = doc["config"]
    lines = [
        f"## frugal-llm cost — {cfg['model']} @ {cfg['url']}",
        "",
        f"Hardware: {currency} {rate.amount:,.2f}/hour ({rate.source}, {rate.date}). "
        f"Benchmark: {doc.get('label') or doc['started_at']}, max_tokens={cfg['max_tokens']}.",
        "",
        "Self-hosted, kept 100% busy at the capacity measured at each concurrency level:",
        "",
        "| Conc | Capacity req/s | TTFT p95 | Tokens in / out per request | Cost per 1k requests | Cost per 1M output tokens |",  # noqa: E501
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in levels:
        tokens = (
            "n/a"
            if lv.output_tokens is None
            else f"{'?' if lv.prompt_tokens is None else _count(lv.prompt_tokens)} / "
            f"{_count(lv.output_tokens)}"
        )
        ttft = "n/a" if lv.ttft_p95_ms is None else f"{lv.ttft_p95_ms:.0f}ms"
        cap = "n/a" if lv.capacity_rps is None else f"{lv.capacity_rps:.3g}"
        per_k = None if lv.self_per_request is None else lv.self_per_request * 1000
        lines.append(
            f"| {lv.concurrency} | {cap} | {ttft} | {tokens} | {_money(per_k, currency, 4)} "
            f"| {_money(lv.self_per_million_output, currency)} |"
        )

    for i, price in enumerate(levels[0].comparisons if levels else []):
        lines += [
            "",
            f"Against **{price.price.name}** ({currency} {price.price.input_per_million} in / "
            f"{price.price.output_per_million} out per 1M tokens; {price.price.source}, "
            f"{price.price.date}):",
            "",
            "| Conc | API cost per 1k requests | Break-even traffic, requests/day | Break-even utilisation | |",  # noqa: E501
            "|---:|---:|---:|---:|---|",
        ]
        for lv in levels:
            c = lv.comparisons[i]
            api_k = None if c.api_per_request is None else c.api_per_request * 1000
            if c.break_even is None:
                verdict = "no comparison: the server did not report prompt tokens"
                be_day = be_util = "n/a"
            else:
                # Unreachable on this hardware: a traffic figure would read as achievable.
                be_day = _count(c.break_even.requests_per_day) if c.break_even.reachable else "—"
                be_util = f"{c.break_even.utilisation:.0%}"
                verdict = (
                    "self-hosting cheaper above this traffic"
                    if c.break_even.reachable
                    else "API cheaper even at 100% use"
                )
            api_cell = _money(api_k, currency, 4)
            lines.append(f"| {lv.concurrency} | {api_cell} | {be_day} | {be_util} | {verdict} |")

    lines += [
        "",
        "- Utilisation is the assumption every figure rests on. Self-hosted costs above are at "
        "100% use; at utilisation u they are 1/u times higher. Break-even utilisation is the "
        "share of measured capacity that must be used, around the clock, for one instance to "
        "cost no more than the API.",
        "- Requests are priced at the shape that was measured (mean tokens in and out per "
        "request). Self-hosting has no separate input and output price: one GPU-hour does "
        "both, so whole requests are compared instead.",
        "- Capacity is what was measured at each concurrency level. Higher concurrency might "
        "raise it, at a cost in latency (see TTFT p95).",
        "- Not included: idle time between workloads, engineering time, storage, egress, and "
        "the quality difference between the models compared.",
        f"- All prices are taken as entered, in {currency}; nothing is converted between "
        "currencies.",
        "",
    ]
    return "\n".join(lines)
