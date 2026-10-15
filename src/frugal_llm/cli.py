"""Command-line entry point: argument parsing and orchestration."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console, Group
from rich.text import Text

from frugal_llm import __version__

if TYPE_CHECKING:
    from frugal_llm.bench import BenchConfig, RequestResult, RunResult

# Heavy modules (httpx, the measurement code) are imported inside commands so that
# --help and --version start instantly.

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark latency, throughput and cost of an OpenAI-compatible endpoint.",
)
console = Console()


class OutputFormat(StrEnum):
    md = "md"
    json = "json"
    both = "both"


def parse_concurrency(value: str) -> list[int]:
    """Parse a comma-separated list such as ``1,4,16`` into positive ints."""
    try:
        levels = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"expected comma-separated integers, got {value!r}") from None
    if not levels or any(level < 1 for level in levels):
        raise typer.BadParameter("concurrency levels must be positive integers")
    return levels


def parse_requests(value: str, levels: list[int]) -> int | list[int]:
    """``50`` for every level, or ``50,100,200,400`` with one count per concurrency level."""
    try:
        counts = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"expected integers, got {value!r}", param_hint="-n") from None
    if not counts or any(n < 1 for n in counts):
        raise typer.BadParameter("request counts must be positive integers", param_hint="-n")
    if len(counts) == 1:
        return counts[0]
    if len(counts) != len(levels):
        raise typer.BadParameter(
            f"{len(counts)} request counts for {len(levels)} concurrency levels; give one count, "
            "or one per level",
            param_hint="-n",
        )
    return counts


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"frugal-llm {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Benchmark latency, throughput and cost of an OpenAI-compatible endpoint."""


@app.command()
def bench(
    url: Annotated[str, typer.Option(help="Base URL, e.g. http://localhost:11434/v1")],
    model: Annotated[str, typer.Option(help="Model identifier as the server knows it")],
    api_key: Annotated[
        str | None, typer.Option(envvar="FRUGAL_LLM_API_KEY", help="Bearer token")
    ] = None,
    concurrency: Annotated[
        str,
        typer.Option("-c", "--concurrency", help="Comma-separated levels to sweep, e.g. 1,4,16"),
    ] = "1",
    requests: Annotated[
        str,
        typer.Option(
            "-n",
            "--requests",
            help="Measured requests per run: one count, or one per level, e.g. 50,100,200,400",
        ),
    ] = "50",
    max_tokens: Annotated[int, typer.Option(min=1, help="Hard cap on output length")] = 128,
    prompt_tokens: Annotated[
        int, typer.Option(min=1, help="Approx. generated prompt length")
    ] = 200,
    prompts: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help='JSONL of {"prompt": "..."}'),
    ] = None,
    warmup: Annotated[
        int, typer.Option(min=0, help="Min warm-up requests per level (at least one per worker)")
    ] = 5,
    repeats: Annotated[int, typer.Option(min=1, help="Independent runs per configuration")] = 3,
    cost_per_hour: Annotated[
        float | None, typer.Option(min=0, help="Hourly cost; enables the cost model")
    ] = None,
    currency: Annotated[str, typer.Option(help="Currency label")] = "GBP",
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Seconds; per streamed chunk and per request")
    ] = 120.0,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("./results"),
    format: Annotated[OutputFormat, typer.Option(help="Report format")] = OutputFormat.both,
    label: Annotated[str | None, typer.Option(help="Free-text tag stored with results")] = None,
    meta: Annotated[
        list[str] | None,
        typer.Option(
            "--meta",
            help="key=value the API can't report, e.g. gpu=RTX3060 provider=runpod; repeatable",
        ),
    ] = None,
) -> None:
    """Run a benchmark sweep against an endpoint."""
    import asyncio
    from datetime import UTC, datetime

    from rich.live import Live

    from frugal_llm import report
    from frugal_llm.bench import BenchAborted, BenchConfig, make_client, run_sweep
    from frugal_llm.environment import server_info
    from frugal_llm.metrics import summarise
    from frugal_llm.prompts import FilePrompts, GeneratedPrompts

    levels = parse_concurrency(concurrency)
    config = BenchConfig(
        url=url.rstrip("/"),
        model=model,
        api_key=api_key,
        concurrency=levels,
        requests=parse_requests(requests, levels),
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        prompts_path=prompts,
        warmup=warmup,
        repeats=repeats,
        cost_per_hour=cost_per_hour,
        currency=currency,
        timeout=timeout,
        out=out,
        format=format.value,
        label=label,
    )
    try:
        prompt_source = FilePrompts.load(prompts) if prompts else GeneratedPrompts(prompt_tokens)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--prompts") from None

    for level in config.concurrency:
        if (n := config.requests_for(level)) < level:
            console.print(
                f"[yellow]Note:[/] -n {n} < concurrency {level}: that level "
                "will never have all its requests in flight at once."
            )

    meta_values = _meta_or_exit(meta)
    view = LiveView(config)
    started_at = datetime.now(UTC)
    seed = getattr(prompt_source, "seed", None)
    server: dict = {}
    partial: list[Path] = []

    def save_partial(run) -> None:
        # Each finished run is saved at once: a crash or dropped connection hours into a
        # sweep costs the run in progress, not everything before it.
        view.on_run(run)
        doc = report.build_document(config, view.runs, started_at, seed, server, meta_values,
                                    complete=False)  # fmt: skip
        path = config.out / f"{report.output_stem(doc)}.partial.json"
        report.write_json(doc, path)
        partial[:] = [path]

    async def probe_then_sweep():
        async with make_client(config) as client:
            server.update(await server_info(client, config.model))
        return await run_sweep(config, prompt_source, on_request=view.on_request,
                               on_run=save_partial)  # fmt: skip

    try:
        with keep_awake(), Live(view, console=console, refresh_per_second=2):
            runs = asyncio.run(probe_then_sweep())
        if not console.is_terminal:
            console.line()  # Live leaves its last frame unterminated when not on a TTY
    except (BenchAborted, KeyboardInterrupt) as exc:
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
        if partial:
            console.print(f"[bold red]Aborted:[/] {reason}. Finished runs are saved in "
                          f"{partial[0]} (marked incomplete).")  # fmt: skip
        else:
            console.print(f"[bold red]Aborted:[/] {reason}. No results were written.")
        if "unreachable" in reason:
            console.print("Is the server running, and does --url include the /v1 prefix?")
        raise typer.Exit(2) from None

    # Save first: nothing that happens while displaying may cost the user their data.
    doc = report.build_document(config, runs, started_at, seed, server, meta_values)
    paths = report.write_reports(doc, config.out, config.format)
    for path in partial:
        path.unlink(missing_ok=True)
    for note in report.notes(summarise(runs, config.cost_per_hour), doc):
        console.print(f"- {note}", highlight=False)
    for path in paths:
        console.print(f"Wrote [bold]{path}[/]")


@app.command("report")
def report_command(
    results: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A results .json file")
    ],
    write: Annotated[bool, typer.Option(help="Also (re)write the .md next to it")] = False,
) -> None:
    """Re-aggregate a saved results file into the Markdown table, without re-running."""
    from frugal_llm import report

    try:
        markdown = report.render_markdown(report.load_document(results))
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(f"not a frugal-llm results file: {exc}") from None
    # Markdown is UTF-8 (≈, £, —). Windows pipes default to cp1252, which would crash
    # `frugal-llm report x.json > x.md`, so write bytes explicitly.
    sys.stdout.buffer.write(markdown.encode("utf-8"))
    sys.stdout.flush()
    if write:
        results.with_suffix(".md").write_text(markdown, encoding="utf-8", newline="\n")


@app.command("cost")
def cost_command(
    results: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A bench results .json file")
    ],
    hourly_rate: Annotated[float, typer.Option(min=0, help="What the hardware costs per hour")],
    rate_source: Annotated[
        str, typer.Option(help="Where that rate came from, e.g. 'runpod RTX 3090'")
    ],
    rate_date: Annotated[str, typer.Option(help="When the rate was quoted, YYYY-MM-DD")],
    api: Annotated[
        list[str] | None,
        typer.Option(
            "--api",
            help="A commercial price to compare against, repeatable: "
            "'name=...,in=<per 1M>,out=<per 1M>,source=...,date=YYYY-MM-DD'",
        ),
    ] = None,
    currency: Annotated[
        str | None, typer.Option(help="Currency of every price given (default: the results')")
    ] = None,
    write: Annotated[
        Path | None, typer.Option(help="Also write the report to this .md file")
    ] = None,
    curve_json: Annotated[
        Path | None, typer.Option("--curve-json", help="Write break-even curve data for charting")
    ] = None,
) -> None:
    """Compare self-hosting against commercial API prices, from saved benchmark results."""
    import json
    from datetime import date

    from frugal_llm import cost_report
    from frugal_llm.cost import ApiPrice
    from frugal_llm.report import load_document

    try:
        date.fromisoformat(rate_date)
    except ValueError:
        raise typer.BadParameter("expected YYYY-MM-DD", param_hint="--rate-date") from None
    try:
        prices = [ApiPrice.parse(spec) for spec in api or []]
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--api") from None
    try:
        doc = load_document(results)
    except (ValueError, KeyError):
        raise typer.BadParameter(
            f"{results} is not a bench result; cost needs measured throughput"
        ) from None

    rate = cost_report.HourlyRate(hourly_rate, rate_source, rate_date)
    levels = cost_report.analyse(doc, rate, prices)
    markdown = cost_report.render_markdown(doc, levels, rate, currency or doc["config"]["currency"])
    sys.stdout.buffer.write(markdown.encode("utf-8"))
    sys.stdout.flush()
    if write:
        write.write_text(markdown, encoding="utf-8", newline="\n")
    if curve_json:
        data = {
            "results": str(results),
            "hourly_rate": rate.__dict__,
            "currency": currency or doc["config"]["currency"],
            "levels": {str(lv.concurrency): cost_report.curve(lv, rate) for lv in levels},
        }
        curve_json.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


class TaskName(StrEnum):
    gsm8k = "gsm8k"
    mmlu = "mmlu"


@app.command()
def quality(
    url: Annotated[str, typer.Option(help="Base URL, e.g. http://localhost:8080/v1")],
    model: Annotated[str, typer.Option(help="Model identifier as the server knows it")],
    task: Annotated[TaskName, typer.Option(help="Task set to score")],
    api_key: Annotated[
        str | None, typer.Option(envvar="FRUGAL_LLM_API_KEY", help="Bearer token")
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            min=1, help="Score only the first N items: a development subset, never a result"
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            "-c",
            "--concurrency",
            min=1,
            help="Requests in flight; keep it the same across runs you compare",
        ),
    ] = 1,
    max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Output cap (default 512; see docs/quality-method.md)")
    ] = None,
    max_retries: Annotated[
        int, typer.Option(min=0, help="Retries per item after HTTP 429/503, honouring Retry-After")
    ] = 8,
    timeout: Annotated[float, typer.Option(min=0.001, help="Seconds per request phase")] = 300.0,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("./results"),
    label: Annotated[str | None, typer.Option(help="Name for this run, e.g. fp16-run1")] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Continue an interrupted run from its .partial.jsonl"
        ),
    ] = None,
    meta: Annotated[
        list[str] | None,
        typer.Option(
            "--meta",
            help="key=value the API can't report, e.g. gpu=RTX3060 provider=runpod; repeatable",
        ),
    ] = None,
) -> None:
    """Score a model on a shipped task set with greedy decoding (see docs/quality-method.md)."""
    import asyncio
    from datetime import UTC, datetime

    from rich.live import Live

    from frugal_llm import quality as q
    from frugal_llm.report import output_stem

    cfg = q.QualityConfig(
        url=url.rstrip("/"),
        model=model,
        api_key=api_key,
        task=task.value,
        limit=limit,
        concurrency=concurrency,
        max_tokens=max_tokens or q.DEFAULT_MAX_TOKENS[task.value],
        max_retries=max_retries,
        timeout=timeout,
        label=label,
    )
    if resume:
        partial = resume
    else:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        stem = output_stem({"started_at": stamp, "label": f"quality-{task.value}-{label or model}"})
        partial = out / f"{stem}.partial.jsonl"
    meta_values = _meta_or_exit(meta)
    server = None if resume else asyncio.run(_probe(cfg.url, cfg.model, cfg.api_key))
    try:
        run = q.prepare_run(
            cfg, partial, resume=resume is not None, server=server, meta=meta_values
        )
    except q.QualityAborted as exc:
        console.print(f"[bold red]Can't resume:[/] {exc}")
        raise typer.Exit(2) from None

    progress = QualityProgress(len(run.items), already=len(run.done))
    try:
        with keep_awake(), Live(progress, console=console, refresh_per_second=2):
            doc = asyncio.run(q.run_quality(cfg, run, on_item=progress.on_item))
        if not console.is_terminal:
            console.line()  # Live leaves its last frame unterminated when not on a TTY
    except q.QualityAborted as exc:
        console.print(f"[bold red]Aborted:[/] {exc}. Is the server running, and is --model right?")
        if not resume:  # a fresh run that failed on its first request has nothing worth keeping
            partial.unlink(missing_ok=True)
        raise typer.Exit(2) from None
    except KeyboardInterrupt:
        console.print(
            f"Interrupted. Finished items are saved; continue with:\n  --resume {partial}"
        )
        raise typer.Exit(130) from None

    final = partial.with_name(partial.name.removesuffix(".partial.jsonl") + ".json")
    q.write_result(doc, final)
    partial.unlink(missing_ok=True)
    s = q.summarise_run(doc)
    if s["scored"]:
        console.print(
            f"{s['correct']}/{s['scored']} correct ({s['correct'] / s['scored']:.1%}); "
            f"{s['no_answer']} with no extractable answer, {s['truncated']} truncated, "
            f"{s['errors']} errors, {s['retried']} retried after rate limiting."
        )
    if cfg.limit:
        console.print(
            f"[yellow]Development subset ({cfg.limit} items): not a publishable result.[/]"
        )
    console.print(f"Wrote [bold]{final}[/]")


class QualityProgress:
    def __init__(self, total: int, already: int = 0) -> None:
        self.total, self.done, self.correct, self.errors = total, already, 0, 0

    def on_item(self, result) -> None:
        self.done += 1
        self.correct += result.correct
        self.errors += result.error is not None

    def __rich__(self) -> Text:
        return Text(
            f"Scored {self.done}/{self.total} items, {self.correct} correct this session"
            + (f", {self.errors} errors" if self.errors else ""),
            style="cyan",
        )


@app.command("compare")
def compare_command(
    baseline: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="FP16 baseline run (.json)")
    ],
    candidates: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, help="Runs to compare against it")
    ],
    noise_floor: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="A second baseline run, to measure run-to-run noise"
        ),
    ] = None,
    write: Annotated[
        Path | None, typer.Option(help="Also write the report to this .md file")
    ] = None,
) -> None:
    """Compare saved quality runs against an FP16 baseline, without re-running anything."""
    import json

    from frugal_llm import compare as cmp

    def load(path: Path) -> dict:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("kind") != "quality":
            raise typer.BadParameter(f"{path} is not a frugal-llm quality result")
        return doc

    base = load(baseline)
    others = [load(p) for p in candidates]
    floor_doc = load(noise_floor) if noise_floor else None
    try:
        results, floor, warnings = cmp.compare_runs(base, others, floor_doc)
    except cmp.NotComparable as exc:
        console.print(f"[bold red]Not comparable:[/] {exc}")
        raise typer.Exit(2) from None
    extra = [*([floor_doc] if floor_doc else []), *others]
    markdown = cmp.render_markdown(base, results, floor, warnings, extra)
    sys.stdout.buffer.write(markdown.encode("utf-8"))
    sys.stdout.flush()
    if write:
        write.write_text(markdown, encoding="utf-8", newline="\n")


def _meta_or_exit(pairs: list[str] | None) -> dict[str, str]:
    from frugal_llm.environment import parse_meta

    try:
        return parse_meta(pairs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--meta") from None


async def _probe(url: str, model: str, api_key: str | None) -> dict:
    import httpx

    from frugal_llm.bench import request_timeout, ssl_context
    from frugal_llm.environment import server_info

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        base_url=url, headers=headers, timeout=request_timeout(30), verify=ssl_context()
    ) as client:
        return await server_info(client, model)


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Best effort: stop the machine idling into sleep mid-benchmark.

    A laptop that sleeps during a run freezes the client mid-request and wrecks that
    run (the report detects and excludes it, but the run is still lost). Closing the
    lid or a battery policy can still force sleep.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        try:
            yield
        finally:
            kernel32.SetThreadExecutionState(es_continuous)
    elif sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        try:
            yield
        finally:
            proc.terminate()
    else:
        yield


class LiveView:
    """What the terminal shows while a sweep runs: results so far plus progress."""

    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.total_runs = len(config.concurrency) * config.repeats
        self.runs: list[RunResult] = []
        self.done_in_run = 0
        self.errors_in_run = 0
        self.table = None

    def on_request(self, result: RequestResult) -> None:
        self.done_in_run += 1
        self.errors_in_run += not result.ok

    def on_run(self, run: RunResult) -> None:
        from frugal_llm.metrics import summarise
        from frugal_llm.report import summary_table

        self.runs.append(run)
        self.done_in_run = self.errors_in_run = 0
        summaries = summarise(self.runs, self.config.cost_per_hour)
        self.table = summary_table(summaries, self.config.cost_per_hour, self.config.currency)

    def __rich__(self) -> Group:
        if len(self.runs) == self.total_runs:
            status = Text(f"Done: {self.total_runs} runs.", style="green")
        else:
            index = len(self.runs)
            level = self.config.concurrency[index // self.config.repeats]
            status = Text(
                f"Run {index + 1}/{self.total_runs}: c={level}, "
                f"repeat {index % self.config.repeats + 1}/{self.config.repeats}, "
                f"{self.done_in_run}/{self.config.requests_for(level)} requests"
                + (f", {self.errors_in_run} failed" if self.errors_in_run else ""),
                style="cyan",
            )
        return Group(*([self.table] if self.table else []), status)


def main() -> None:
    # A terminal or pipe that cannot encode a character (cp1252 on Windows has no ≈)
    # must degrade the display, never crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    app()


if __name__ == "__main__":
    main()
