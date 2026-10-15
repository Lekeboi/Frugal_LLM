from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frugal_llm.bench import BenchConfig, RequestResult, RunResult
from frugal_llm.cli import app
from frugal_llm.cost import ApiPrice
from frugal_llm.cost_report import HourlyRate, analyse, curve, render_markdown
from frugal_llm.report import build_document

RATE = HourlyRate(0.90, "example GPU host", "2026-10-15")
PRICE = ApiPrice("example-api", 0.50, 1.50, "https://example.com/pricing", "2026-10-15")


def _req(prompt_tokens: int | None = 200, output_tokens: int = 100) -> RequestResult:
    return RequestResult(0.0, 1.0, 100.0, 20.0, 2000.0, output_tokens, prompt_tokens, "usage")


def _doc(runs: list[RunResult], concurrency: list[int]) -> dict:
    config = BenchConfig(
        url="http://localhost:8000/v1", model="m", api_key=None, concurrency=concurrency,
        requests=40, max_tokens=128, prompt_tokens=200, prompts_path=None, warmup=5, repeats=1,
        cost_per_hour=None, currency="USD", timeout=120.0, out=Path("."), format="json",
        label="cost-test",
    )  # fmt: skip
    return json.loads(json.dumps(build_document(config, runs, datetime(2026, 10, 15, tzinfo=UTC))))


# 40 requests of 200 in / 100 out over 10 s at c=4: capacity 4 requests/s.
FAST = RunResult(4, 0, 10.0, [_req() for _ in range(40)], 5.0, 9.0)


def test_hand_computed_break_even():
    (level,) = analyse(_doc([FAST], [4]), RATE, [PRICE])
    assert level.capacity_rps == pytest.approx(4.0)
    assert (level.prompt_tokens, level.output_tokens) == (200, 100)
    # $0.90/h / 3600 / 4 req/s = $0.0000625 per request at 100% use.
    assert level.self_per_request == pytest.approx(0.0000625)
    (c,) = level.comparisons
    assert c.api_per_request == pytest.approx(0.00025)  # 200*0.5/1M + 100*1.5/1M
    assert c.break_even.requests_per_day == pytest.approx(86_400)  # 1 request/s
    assert c.break_even.utilisation == pytest.approx(0.25)


def test_report_states_the_verdict_and_its_assumptions():
    doc = _doc([FAST], [4])
    md = render_markdown(doc, analyse(doc, RATE, [PRICE]), RATE, "USD")
    assert "| 4 | 4 | 100ms | 200 / 100 | USD 0.0625 | USD 0.62 |" in md
    assert "| 4 | USD 0.2500 | 86,400 | 25% | self-hosting cheaper above this traffic |" in md
    assert "Utilisation is the assumption every figure rests on" in md
    assert "example GPU host, 2026-10-15" in md and "https://example.com/pricing" in md


def test_slow_hardware_never_breaks_even():
    slow = RunResult(1, 0, 10.0, [_req() for _ in range(2)], 5.0, 9.0)  # 0.2 requests/s
    doc = _doc([slow], [1])
    md = render_markdown(doc, analyse(doc, RATE, [PRICE]), RATE, "USD")
    assert "| — | 500% | API cheaper even at 100% use |" in md


def test_no_comparison_without_server_reported_prompt_tokens():
    unknown = RunResult(4, 0, 10.0, [_req(prompt_tokens=None) for _ in range(40)], 5.0, 9.0)
    (level,) = analyse(_doc([unknown], [4]), RATE, [PRICE])
    assert level.comparisons[0].break_even is None  # pricing output alone would flatter


def test_stalled_runs_are_left_out():
    slept = RunResult(4, 1, 1200.0, [_req(prompt_tokens=9999) for _ in range(40)], 5.0, 1.2e6)
    (level,) = analyse(_doc([FAST, slept], [4]), RATE, [PRICE])
    assert level.capacity_rps == pytest.approx(4.0) and level.prompt_tokens == 200


def test_curve_steps_with_instances_and_crosses_at_break_even():
    (level,) = analyse(_doc([FAST], [4]), RATE, [PRICE])
    rows = curve(level, RATE, points=6)  # up to 3 instances' capacity: 1,036,800/day
    assert [r["self_hosted"] for r in rows] == pytest.approx([21.6] * 3 + [43.2] * 2 + [64.8] * 2)
    assert rows[0]["example-api"] == 0


# --- CLI ----------------------------------------------------------------------------

runner = CliRunner()
API = "name=example-api,in=0.50,out=1.50,source=https://example.com/pricing,date=2026-10-15"


def test_cli_cost(tmp_path):
    results = tmp_path / "r.json"
    results.write_text(json.dumps(_doc([FAST], [4])), encoding="utf-8")
    out = runner.invoke(app, [
        "cost", str(results), "--hourly-rate", "0.90", "--rate-source", "example GPU host",
        "--rate-date", "2026-10-15", "--api", API, "--curve-json", str(tmp_path / "c.json"),
    ])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "86,400" in out.stdout_bytes.decode()
    data = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert data["hourly_rate"]["source"] == "example GPU host" and "4" in data["levels"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--rate-date", "yesterday"], "YYYY-MM-DD"),
        (["--rate-date", "2026-10-15", "--api", "name=x,in=1,out=2"], "missing"),
    ],
)
def test_cli_cost_rejects_undated_or_unsourced_prices(tmp_path, args, message):
    results = tmp_path / "r.json"
    results.write_text(json.dumps(_doc([FAST], [4])), encoding="utf-8")
    out = runner.invoke(
        app, ["cost", str(results), "--hourly-rate", "1", "--rate-source", "s", *args]
    )
    assert out.exit_code != 0 and message in out.output


def test_cli_cost_refuses_a_quality_result(tmp_path):
    results = tmp_path / "q.json"
    results.write_text(json.dumps({"schema_version": "quality-0.1", "kind": "quality"}),
                       encoding="utf-8")  # fmt: skip
    out = runner.invoke(app, ["cost", str(results), "--hourly-rate", "1", "--rate-source", "s",
                              "--rate-date", "2026-10-15"])  # fmt: skip
    assert out.exit_code != 0 and "not a bench result" in out.output
