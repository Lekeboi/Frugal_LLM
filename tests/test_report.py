from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mock_server import MockBehaviour
from typer.testing import CliRunner

from frugal_llm.bench import BenchConfig, RequestResult, RunResult
from frugal_llm.cli import app
from frugal_llm.metrics import summarise
from frugal_llm.report import (
    build_document,
    fmt_rate,
    load_document,
    notes,
    output_stem,
    render_markdown,
    runs_from_document,
    write_reports,
)

STARTED = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)


def _config(**overrides) -> BenchConfig:
    values = dict(
        url="http://localhost:8000/v1",
        model="mistral-7b-awq",
        api_key="sk-secret-key",
        concurrency=[1],
        requests=3,
        max_tokens=128,
        prompt_tokens=200,
        prompts_path=None,
        warmup=5,
        repeats=1,
        cost_per_hour=0.52,
        currency="GBP",
        timeout=120.0,
        out=Path("results"),
        format="both",
        label="vllm-awq-L4",
    )
    values.update(overrides)
    return BenchConfig(**values)


def req(ttft, itl=20.0, e2e=300.0, tokens=10, source="usage", error=None):
    if error:
        return RequestResult(0.0, 1.0, error=error)
    return RequestResult(0.0, 1.0, ttft, itl, e2e, tokens, 203, source)


def _known_run(**kwargs) -> RunResult:
    # 3 requests x 10 tokens over 1 s -> 30 tok/s; GBP 0.52/h -> 4.81 per 1M tokens
    defaults = dict(concurrency=1, repeat=0, window_seconds=1.0, client_lag_p99_ms=5.0)
    defaults.update(kwargs)
    return RunResult(requests=[req(100), req(120), req(140)], **defaults)


def test_json_keeps_every_sample_and_never_the_api_key():
    doc = build_document(_config(), [_known_run()], STARTED)
    text = json.dumps(doc)
    assert "sk-secret-key" not in text
    assert doc["started_at"] == "2026-10-14T12:00:00Z"
    assert doc["environment"]["http_version"] == "HTTP/1.1"
    assert doc["environment"]["token_count_source"] == "usage"
    assert [r["ttft_ms"] for r in doc["runs"][0]["requests"]] == [100, 120, 140]


def test_reaggregation_from_json_matches_live_summary():
    runs = [_known_run(), _known_run(repeat=1, window_seconds=2.0)]
    doc = json.loads(json.dumps(build_document(_config(repeats=2), runs, STARTED)))
    assert summarise(runs_from_document(doc), 0.52) == summarise(runs, 0.52)


def test_markdown_matches_spec_layout():
    md = render_markdown(build_document(_config(), [_known_run()], STARTED))
    lines = md.splitlines()
    assert lines[0] == "## frugal-llm — mistral-7b-awq @ http://localhost:8000/v1"
    assert lines[2].startswith("Config: max_tokens=128, prompt_tokens≈200")
    assert "GBP 0.52/hour" in lines[2]
    assert lines[4] == "| Conc | TTFT p50 | TTFT p95 | ITL p50 | Thrpt tok/s | £/1M tok | Err |"
    assert lines[5] == "|-----:|---------:|---------:|--------:|------------:|---------:|----:|"
    # p50 of 100,120,140 = 120; p95 = 120 + 0.9 * 20 = 138
    assert lines[6] == "|    1 |    120ms |    138ms |    20ms |        30.0 |     4.81 |  0% |"
    assert "output quality is not measured" in md
    assert md.rstrip().endswith("on 2026-10-14T12:00:00Z")


def test_no_cost_column_without_hourly_rate():
    md = render_markdown(build_document(_config(cost_per_hour=None), [_known_run()], STARTED))
    assert "/1M tok" not in md and "/hour" not in md


def test_notes_say_how_numbers_were_obtained():
    run = RunResult(
        concurrency=8,
        repeat=0,
        window_seconds=1.0,
        client_lag_p99_ms=120.0,
        requests=[
            req(100, source="chunks"),
            req(110, source="chunks"),
            req(0, error="read_timeout"),
        ],
    )
    doc = build_document(_config(concurrency=[8]), [run], STARTED)
    text = "\n".join(notes(summarise([run], None), doc))
    assert "counted from stream chunks" in text
    assert "1 of 3 requests failed (1 read_timeout)" in text
    assert "only 3 requests per run, so 8 were never in flight" in text
    assert "WARNING the benchmarking client's event loop lagged 120ms" in text


@pytest.mark.parametrize(("rate", "text"), [(0, "0%"), (0.004, "0.4%"), (0.25, "25%")])
def test_small_error_rates_are_not_rounded_away(rate, text):
    assert fmt_rate(rate) == text


def test_output_names_and_formats(tmp_path):
    doc = build_document(_config(label="vllm awq/L4"), [_known_run()], STARTED)
    assert output_stem(doc) == "20261014T120000Z-vllm-awq-L4"
    assert [p.name for p in write_reports(doc, tmp_path, "md")] == [f"{output_stem(doc)}.md"]
    paths = write_reports(doc, tmp_path / "both", "both")
    assert sorted(p.suffix for p in paths) == [".json", ".md"]
    assert b"\r\n" not in paths[1].read_bytes()


def test_rejects_unknown_schema(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"schema_version": "9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_document(path)


# --- end to end through the CLI -----------------------------------------------------

runner = CliRunner()


def test_cli_bench_then_report(mock_endpoint, tmp_path):
    with mock_endpoint(MockBehaviour(first_chunk_delay=0.02)) as (_, url):
        result = runner.invoke(
            app,
            [
                "bench", "--url", url, "--model", "mock", "-c", "1,2", "-n", "4",
                "--warmup", "1", "--repeats", "2", "--max-tokens", "4",
                "--cost-per-hour", "1.5", "--label", "e2e", "--out", str(tmp_path),
            ],
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    (json_path,) = tmp_path.glob("*-e2e.json")
    doc = load_document(json_path)
    assert [(r["concurrency"], r["repeat"]) for r in doc["runs"]] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert all(len(r["requests"]) == 4 for r in doc["runs"])

    regenerated = runner.invoke(app, ["report", str(json_path)])
    assert regenerated.exit_code == 0
    assert regenerated.stdout_bytes.decode() == json_path.with_suffix(".md").read_text("utf-8")


def test_cli_unreachable_endpoint_fails_loudly_and_writes_nothing(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
        result = runner.invoke(app, ["bench", "--url", url, "--model", "m", "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert "unreachable" in result.output
    assert not list(tmp_path.iterdir())


def test_cli_rejects_bad_concurrency():
    result = runner.invoke(app, ["bench", "--url", "http://x/v1", "--model", "m", "-c", "4,0"])
    assert result.exit_code != 0


def test_stalled_run_is_named_and_excluded():
    slept = _known_run(repeat=1, window_seconds=1429.0, client_lag_max_ms=1_217_591.5)
    doc = build_document(_config(repeats=2), [_known_run(), slept], STARTED)
    md = render_markdown(doc)
    assert "c=1 repeat 1: EXCLUDED. The client was frozen for 1217.6s" in md
    # The table only reflects the valid run: 30.0 tok/s, no range from the stalled one.
    assert "|        30.0 |" in md


def test_keep_awake_is_safe_to_enter_and_exit():
    from frugal_llm.cli import keep_awake

    with keep_awake():
        pass


def test_notes_report_actual_output_and_prompt_lengths():
    run = RunResult(1, 0, 1.0, [req(100, tokens=60), req(110, tokens=80)])
    text = "\n".join(notes(summarise([run], None), build_document(_config(), [run], STARTED)))
    assert "Median output length was below --max-tokens 128 (c=1: 70)" in text
    assert "Prompt length as counted by the server: median 203 tokens (requested ≈200" in text
