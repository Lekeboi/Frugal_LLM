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
        int, typer.Option("-n", "--requests", min=1, help="Measured requests per level")
    ] = 50,
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
) -> None:
    """Run a benchmark sweep against an endpoint."""
    import asyncio
    from datetime import UTC, datetime

    from rich.live import Live

    from frugal_llm import report
    from frugal_llm.bench import BenchAborted, BenchConfig, run_sweep
    from frugal_llm.metrics import summarise
    from frugal_llm.prompts import FilePrompts, GeneratedPrompts

    config = BenchConfig(
        url=url.rstrip("/"),
        model=model,
        api_key=api_key,
        concurrency=parse_concurrency(concurrency),
        requests=requests,
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
        if config.requests < level:
            console.print(
                f"[yellow]Note:[/] -n {config.requests} < concurrency {level}: that level "
                "will never have all its requests in flight at once."
            )

    view = LiveView(config)
    started_at = datetime.now(UTC)
    try:
        with keep_awake(), Live(view, console=console, refresh_per_second=2):
            runs = asyncio.run(
                run_sweep(config, prompt_source, on_request=view.on_request, on_run=view.on_run)
            )
        if not console.is_terminal:
            console.line()  # Live leaves its last frame unterminated when not on a TTY
    except BenchAborted as exc:
        console.print(f"[bold red]Aborted:[/] {exc}. No results were written.")
        if "unreachable" in str(exc):
            console.print("Is the server running, and does --url include the /v1 prefix?")
        raise typer.Exit(2) from None

    doc = report.build_document(config, runs, started_at)
    for note in report.notes(summarise(runs, config.cost_per_hour), doc):
        console.print(f"- {note}", highlight=False)
    for path in report.write_reports(doc, config.out, config.format):
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
                f"{self.done_in_run}/{self.config.requests} requests"
                + (f", {self.errors_in_run} failed" if self.errors_in_run else ""),
                style="cyan",
            )
        return Group(*([self.table] if self.table else []), status)


def main() -> None:
    app()
