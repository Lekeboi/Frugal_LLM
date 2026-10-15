"""JSON and Markdown writers, plus the live terminal table.

The JSON keeps every individual sample; nothing is pre-aggregated. Markdown is always
rendered *from* that document, so a report can be regenerated without re-running.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from frugal_llm.bench import HTTP_VERSION, BenchConfig, RequestResult, RunResult
from frugal_llm.environment import client_info, summary_line
from frugal_llm.metrics import STALL_MS, LevelSummary, Spread, summarise

SCHEMA_VERSION = "0.1"
# Above this p99 event-loop lag the client, not the server, may be shaping latencies.
CLIENT_LAG_WARN_MS = 50.0
CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥", "INR": "₹"}
_REQUEST_FIELDS = (
    "ttft_ms",
    "itl_ms",
    "e2e_ms",
    "output_tokens",
    "prompt_tokens",
    "token_count_source",
    "error",
    "cached_prompt_tokens",
)
# A little cache use is normal: the chat template's fixed preamble is shared by every
# prompt. Above this share, cached prompts are flattering TTFT.
CACHED_PROMPT_WARN = 0.25


# --- JSON ---------------------------------------------------------------------------


def _round(value: Any) -> Any:
    return round(value, 3) if isinstance(value, float) else value


def build_document(
    config: BenchConfig,
    runs: list[RunResult],
    started_at: datetime,
    prompt_seed: int | None = None,
    server: dict | None = None,
    meta: dict | None = None,
    complete: bool = True,
) -> dict:
    """The full results document. The API key is never written."""
    return {
        "schema_version": SCHEMA_VERSION,
        "label": config.label,
        "started_at": started_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "url": config.url,
            "model": config.model,
            "concurrency": config.concurrency,
            "requests": config.requests,
            "max_tokens": config.max_tokens,
            "prompt_tokens": config.prompt_tokens,
            "prompts": str(config.prompts_path) if config.prompts_path else None,
            "prompt_seed": prompt_seed,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "timeout": config.timeout,
            "cost_per_hour": config.cost_per_hour,
            "currency": config.currency,
        },
        "complete": complete,
        "environment": {
            **environment_info(server, meta),
            "token_count_source": _overall_token_source(runs),
        },
        "runs": [
            {
                "concurrency": run.concurrency,
                "repeat": run.repeat,
                "window_seconds": _round(run.window_seconds),
                "client_lag_p99_ms": _round(run.client_lag_p99_ms),
                "client_lag_max_ms": _round(run.client_lag_max_ms),
                "requests": [
                    {field: _round(getattr(r, field)) for field in _REQUEST_FIELDS}
                    for r in run.requests
                ],
            }
            for run in runs
        ],
    }


def environment_info(server: dict | None = None, meta: dict | None = None) -> dict:
    """What every result file records about where it came from (see environment.py)."""
    return {
        **client_info(),
        "http_version": HTTP_VERSION,
        "server": server,
        "meta": meta or {},
    }


def _overall_token_source(runs: list[RunResult]) -> str | None:
    sources = {r.token_count_source for run in runs for r in run.requests if r.ok}
    if not sources:
        return None
    return sources.pop() if len(sources) == 1 else "mixed"


def runs_from_document(doc: dict) -> list[RunResult]:
    """Rebuild runs from a results document so they can be re-aggregated."""
    return [
        RunResult(
            concurrency=run["concurrency"],
            repeat=run["repeat"],
            window_seconds=run["window_seconds"],
            client_lag_p99_ms=run.get("client_lag_p99_ms"),
            client_lag_max_ms=run.get("client_lag_max_ms"),
            requests=[
                RequestResult(
                    sent_at_s=0.0, finished_at_s=0.0, **{f: r.get(f) for f in _REQUEST_FIELDS}
                )
                for r in run["requests"]
            ],
        )
        for run in doc["runs"]
    ]


def load_document(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {doc.get('schema_version')!r}")
    return doc


# --- formatting ---------------------------------------------------------------------


def currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(currency.upper(), f"{currency} ")


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


def fmt_spread(spread: Spread | None, digits: int) -> str:
    if spread is None:
        return "n/a"
    text = f"{spread.median:.{digits}f}"
    if spread.low != spread.high:
        text += f" ({spread.low:.{digits}f}-{spread.high:.{digits}f})"
    return text


def fmt_rate(rate: float) -> str:
    # Never round a non-zero error rate down to a reassuring "0%".
    if rate == 0:
        return "0%"
    if rate < 0.01:
        return f"{rate * 100:.1f}%"
    return f"{rate * 100:.0f}%"


def _columns(has_cost: bool, currency: str) -> list[str]:
    cols = ["Conc", "TTFT p50", "TTFT p95", "ITL p50", "Thrpt tok/s"]
    if has_cost:
        cols.append(f"{currency_symbol(currency).strip()}/1M tok")
    cols.append("Err")
    return cols


def _row(s: LevelSummary, has_cost: bool) -> list[str]:
    row = [
        str(s.concurrency),
        fmt_ms(s.ttft_ms[50]),
        fmt_ms(s.ttft_ms[95]),
        fmt_ms(s.itl_ms[50]),
        fmt_spread(s.output_tokens_per_sec, 1),
    ]
    if has_cost:
        row.append(fmt_spread(s.cost_per_million, 2))
    row.append(fmt_rate(s.error_rate))
    return row


def notes(summaries: list[LevelSummary], doc: dict) -> list[str]:
    """Everything a reader needs to judge how each number was obtained."""
    cfg = doc["config"]
    out = []
    if cfg["repeats"] > 1:
        out.append(
            f"Latency percentiles pool the {cfg['repeats']} repeats at each level (never "
            "across levels). Throughput and cost show the median run (min-max)."
        )
    chunk_levels = [s.concurrency for s in summaries if "chunks" in s.token_count_sources]
    if chunk_levels:
        out.append(
            f"Output tokens at c={_levels(chunk_levels)} were counted from stream chunks "
            "because the server did not report `usage`. Servers that send several tokens "
            "per chunk are undercounted, which understates throughput."
        )
    for run in doc["runs"]:
        lag = run.get("client_lag_max_ms")
        if lag is not None and lag > STALL_MS:
            out.append(
                f"c={run['concurrency']} repeat {run['repeat']}: EXCLUDED. The client was frozen "
                f"for {lag / 1000:.1f}s during this run (system sleep or suspension?), so its "
                "timings are meaningless. Its samples remain in the JSON; re-run to replace it."
            )
    max_tokens = cfg["max_tokens"]
    short = [
        s
        for s in summaries
        if s.output_tokens_median is not None and s.output_tokens_median < 0.9 * max_tokens
    ]
    if short:
        detail = ", ".join(f"c={s.concurrency}: {s.output_tokens_median:.0f}" for s in short)
        out.append(
            f"Median output length was below --max-tokens {max_tokens} ({detail}): the model "
            "often stopped early. Throughput and cost reflect these shorter generations."
        )
    prompt_medians = sorted(
        {f"{s.prompt_tokens_median:.0f}" for s in summaries if s.prompt_tokens_median},
        key=float,
    )
    if prompt_medians and not cfg.get("prompts"):
        low, high = prompt_medians[0], prompt_medians[-1]
        span = low if low == high else f"{low}-{high}"
        out.append(
            f"Prompt length as counted by the server: median {span} tokens "
            f"(requested ≈{cfg['prompt_tokens']}, including the chat template)."
        )
    for s in summaries:
        if s.cached_prompt_share is not None and s.cached_prompt_share > CACHED_PROMPT_WARN:
            out.append(
                f"c={s.concurrency}: WARNING the server reported serving "
                f"{s.cached_prompt_share:.0%} of prompt tokens from its prompt cache, so TTFT "
                "is lower than new prompts would see"
                + (" (--prompts repeats its lines)." if cfg.get("prompts") else ".")
            )
        if s.runs == 0:
            out.append(f"c={s.concurrency}: no valid runs; every run was excluded.")
        if s.errors:
            detail = ", ".join(f"{n} {kind}" for kind, n in s.errors.items())
            out.append(
                f"c={s.concurrency}: {s.attempted - s.completed} of {s.attempted} "
                f"requests failed ({detail})."
            )
        if s.requests_per_run < s.concurrency:
            out.append(
                f"c={s.concurrency}: only {s.requests_per_run} requests per run, so "
                f"{s.concurrency} were never in flight at once. Use -n of several times "
                "the concurrency."
            )
        if s.client_lag_p99_ms is not None and s.client_lag_p99_ms > CLIENT_LAG_WARN_MS:
            out.append(
                f"c={s.concurrency}: WARNING the benchmarking client's event loop lagged "
                f"{s.client_lag_p99_ms:.0f}ms (p99). Latencies at this level include client "
                "overhead; run the client on a less loaded machine."
            )
    out.append(
        f"{doc['environment']['http_version']}, one connection per concurrent request. "
        "Measures speed and cost only; output quality is not measured."
    )
    return out


def _levels(levels: list[int]) -> str:
    return ",".join(str(c) for c in levels)


# --- Markdown -----------------------------------------------------------------------


def render_markdown(doc: dict) -> str:
    cfg = doc["config"]
    summaries = summarise(runs_from_document(doc), cfg["cost_per_hour"])
    has_cost = cfg["cost_per_hour"] is not None

    config_line = (
        f"Config: max_tokens={cfg['max_tokens']}, "
        + (
            f"prompts={cfg['prompts']}"
            if cfg.get("prompts")
            else f"prompt_tokens≈{cfg['prompt_tokens']}"
        )
        + f", requests={cfg['requests']}, warmup={cfg['warmup']}, repeats={cfg['repeats']}"
    )
    if has_cost:
        config_line += f", {cfg['currency']} {cfg['cost_per_hour']}/hour"
    if doc.get("label"):
        config_line += f", label={doc['label']}"

    header = _columns(has_cost, cfg["currency"])
    rows = [_row(s, has_cost) for s in summaries]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.rjust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    environment = summary_line(doc["environment"])
    lines = [
        f"## frugal-llm — {cfg['model']} @ {cfg['url']}",
        "",
        *(
            []
            if doc.get("complete", True)
            else [
                f"> **Incomplete run.** Stopped after {len(doc['runs'])} of "
                f"{len(cfg['concurrency']) * cfg['repeats']} runs; the table covers only "
                "those.",
                "",
            ]
        ),
        config_line,
        *([f"Environment: {environment}"] if environment else []),
        "",
        line(header),
        "|" + "|".join("-" * (w + 1) + ":" for w in widths) + "|",
        *(line(r) for r in rows),
        "",
        *(f"- {n}" for n in notes(summaries, doc)),
        "",
        f"Generated by frugal-llm {doc['environment']['frugal_llm_version']} "
        f"on {doc['started_at']}",
        "",
    ]
    return "\n".join(lines)


# --- terminal -----------------------------------------------------------------------


def summary_table(
    summaries: list[LevelSummary], cost_per_hour: float | None, currency: str
) -> Table:
    has_cost = cost_per_hour is not None
    table = Table(title="frugal-llm results", title_justify="left")
    for col in _columns(has_cost, currency):
        table.add_column(col, justify="right")
    for s in summaries:
        table.add_row(*_row(s, has_cost))
    return table


# --- files --------------------------------------------------------------------------


def output_stem(doc: dict) -> str:
    stamp = doc["started_at"].replace("-", "").replace(":", "")
    label = doc.get("label")
    if not label:
        return stamp
    return f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-')}"


def write_json(doc: dict, path: Path) -> None:
    """Write then rename, so a crash mid-write never leaves a half-written results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_reports(doc: dict, out_dir: Path, fmt: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(doc)
    written = []
    if fmt in ("json", "both"):
        path = out_dir / f"{stem}.json"
        # Explicit "\n": identical bytes on every OS, and diffable against stdout.
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    if fmt in ("md", "both"):
        path = out_dir / f"{stem}.md"
        path.write_text(render_markdown(doc), encoding="utf-8", newline="\n")
        written.append(path)
    return written
