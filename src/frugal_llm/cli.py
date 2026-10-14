"""Command-line entry point: argument parsing and orchestration."""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from frugal_llm import __version__
from frugal_llm.bench import BenchConfig

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
    warmup: Annotated[int, typer.Option(min=0, help="Warm-up requests per level")] = 5,
    repeats: Annotated[int, typer.Option(min=1, help="Independent runs per configuration")] = 3,
    cost_per_hour: Annotated[
        float | None, typer.Option(min=0, help="Hourly cost; enables the cost model")
    ] = None,
    currency: Annotated[str, typer.Option(help="Currency label")] = "GBP",
    timeout: Annotated[float, typer.Option(min=0, help="Per-request timeout (s)")] = 120.0,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("./results"),
    format: Annotated[OutputFormat, typer.Option(help="Report format")] = OutputFormat.both,
    label: Annotated[str | None, typer.Option(help="Free-text tag stored with results")] = None,
) -> None:
    """Run a benchmark sweep against an endpoint."""
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

    # Scaffold stage: show the parsed configuration. Orchestration lands in a later step.
    table = Table(title="frugal-llm bench: configuration", show_header=False)
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if field.name == "api_key" and value:
            value = "***"
        table.add_row(field.name, str(value))
    console.print(table)


def main() -> None:
    app()
