"""Percentiles and aggregation. Pure functions over plain data.

Rules: latency is only ever reported as quantiles, never a mean;
samples are never pooled across concurrency levels; TTFT and ITL stay separate.
Within one level, latency samples from all repeats form one distribution, while
throughput is computed per run and its spread across repeats is reported.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frugal_llm.cost import cost_per_million_tokens

if TYPE_CHECKING:
    from frugal_llm.bench import RunResult


# A run during which the client's event loop froze for longer than this is invalid:
# the machine slept, was suspended, or was too overloaded to time anything. Such runs
# are excluded from every aggregate and named in the report; their samples stay in
# the JSON.
STALL_MS = 1000.0


def is_stalled(run: RunResult) -> bool:
    return run.client_lag_max_ms is not None and run.client_lag_max_ms > STALL_MS


def percentiles(samples: Iterable[float], ps: Sequence[int]) -> dict[int, float | None]:
    """Percentiles via ``statistics.quantiles(method="inclusive")``.

    Inclusive interpolation stays within the observed range and matches the usual
    hand calculation: p = x[k] + f * (x[k+1] - x[k]) at rank (n - 1) * p / 100.
    """
    data = sorted(samples)
    if not data:
        return dict.fromkeys(ps)
    if len(data) == 1:  # quantiles() needs two points before Python 3.13
        return dict.fromkeys(ps, data[0])
    cuts = statistics.quantiles(data, n=100, method="inclusive")
    return {p: cuts[p - 1] for p in ps}


@dataclass(frozen=True)
class Spread:
    """A per-run figure across repeats: the median and the full range."""

    median: float
    low: float
    high: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Spread | None:
        if not values:
            return None
        return cls(statistics.median(values), min(values), max(values))


@dataclass(frozen=True)
class LevelSummary:
    concurrency: int
    runs: int
    excluded_runs: int
    attempted: int
    completed: int
    requests_per_run: int
    ttft_ms: dict[int, float | None]
    itl_ms: dict[int, float | None]
    e2e_ms: dict[int, float | None]
    output_tokens_per_sec: Spread | None
    requests_per_sec: Spread | None
    cost_per_million: Spread | None
    errors: dict[str, int]
    token_count_sources: tuple[str, ...]
    client_lag_p99_ms: float | None
    output_tokens_median: float | None = None
    prompt_tokens_median: float | None = None
    cached_prompt_share: float | None = None
    """Fraction of prompt tokens the server reported serving from cache (None if unreported)."""

    @property
    def error_rate(self) -> float:
        return (self.attempted - self.completed) / self.attempted if self.attempted else 0.0


def run_output_tokens_per_sec(run: RunResult) -> float:
    """Output tokens of completed requests / wall-clock measurement window."""
    if run.window_seconds <= 0:
        return 0.0
    return sum(r.output_tokens for r in run.requests if r.ok) / run.window_seconds


def run_requests_per_sec(run: RunResult) -> float:
    if run.window_seconds <= 0:
        return 0.0
    return sum(1 for r in run.requests if r.ok) / run.window_seconds


def summarise_level(runs: Sequence[RunResult], cost_per_hour: float | None) -> LevelSummary:
    """Aggregate the repeats of one concurrency level, leaving out stalled runs."""
    levels = {run.concurrency for run in runs}
    if len(levels) != 1:
        raise ValueError(f"refusing to pool samples across concurrency levels {sorted(levels)}")

    concurrency = runs[0].concurrency
    requests_per_run = max(len(run.requests) for run in runs)
    excluded = sum(1 for run in runs if is_stalled(run))
    runs = [run for run in runs if not is_stalled(run)]
    requests = [r for run in runs for r in run.requests]
    ok = [r for r in requests if r.ok]
    tps = [run_output_tokens_per_sec(run) for run in runs]
    costs = []
    if cost_per_hour is not None:
        costs = [c for t in tps if (c := cost_per_million_tokens(cost_per_hour, t)) is not None]
    lags = [run.client_lag_p99_ms for run in runs if run.client_lag_p99_ms is not None]

    return LevelSummary(
        concurrency=concurrency,
        runs=len(runs),
        excluded_runs=excluded,
        attempted=len(requests),
        completed=len(ok),
        requests_per_run=requests_per_run,
        ttft_ms=percentiles((r.ttft_ms for r in ok), (50, 95, 99)),
        itl_ms=percentiles((r.itl_ms for r in ok if r.itl_ms is not None), (50, 95)),
        e2e_ms=percentiles((r.e2e_ms for r in ok), (50, 95, 99)),
        output_tokens_per_sec=Spread.of(tps),
        requests_per_sec=Spread.of([run_requests_per_sec(run) for run in runs]),
        cost_per_million=Spread.of(costs),
        errors=dict(Counter(r.error for r in requests if not r.ok).most_common()),
        token_count_sources=tuple(sorted({r.token_count_source for r in ok})),
        client_lag_p99_ms=max(lags, default=None),
        output_tokens_median=_median([r.output_tokens for r in ok]),
        prompt_tokens_median=_median([r.prompt_tokens for r in ok if r.prompt_tokens is not None]),
        cached_prompt_share=_cached_share(ok),
    )


def _cached_share(requests: Sequence) -> float | None:
    reported = [r for r in requests if r.cached_prompt_tokens is not None and r.prompt_tokens]
    if not reported:
        return None
    return sum(r.cached_prompt_tokens for r in reported) / sum(r.prompt_tokens for r in reported)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def summarise(runs: Sequence[RunResult], cost_per_hour: float | None) -> list[LevelSummary]:
    """One summary per concurrency level, in the order the levels were run."""
    by_level: dict[int, list[RunResult]] = {}
    for run in runs:
        by_level.setdefault(run.concurrency, []).append(run)
    return [summarise_level(level_runs, cost_per_hour) for level_runs in by_level.values()]
