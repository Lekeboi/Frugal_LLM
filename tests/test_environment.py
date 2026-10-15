from __future__ import annotations

import json
import socket

import httpx
import pytest
from mock_server import MockBehaviour
from typer.testing import CliRunner

from frugal_llm.cli import app
from frugal_llm.environment import parse_meta, server_info, summary_line

runner = CliRunner()


def test_meta_pairs():
    assert parse_meta(["gpu=RTX 3060", "provider = runpod", "note=a=b"]) == {
        "gpu": "RTX 3060",
        "provider": "runpod",
        "note": "a=b",
    }
    assert parse_meta(None) == {}
    with pytest.raises(ValueError, match="key=value"):
        parse_meta(["gpu"])


async def test_server_reports_itself_verbatim(mock_endpoint):
    with mock_endpoint(MockBehaviour(reply="hi")) as (_, url):
        async with httpx.AsyncClient(base_url=url) as client:
            info = await server_info(client, "mock")
    assert info["server_header"] == "frugal-mock"
    assert info["system_fingerprint"] == "mock-build-1"
    assert info["models_listed"] == ["mock"]
    assert info["model_entry"]["meta"]["ftype"] == "F16"


async def test_unknown_model_and_dead_server_never_fail_the_probe(mock_endpoint):
    with mock_endpoint(MockBehaviour(reply="hi")) as (_, url):
        async with httpx.AsyncClient(base_url=url) as client:
            info = await server_info(client, "not-listed")
    assert info["model_entry"] is None and info["models_listed"] == ["mock"]

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{s.getsockname()[1]}/v1") as c:
            info = await server_info(c, "m")
    assert info["models_error"] == "ConnectError" and info["system_fingerprint"] is None


def test_summary_line():
    env = {
        "server": {"server_header": "llama.cpp", "system_fingerprint": "b11604-45451c366",
                   "model_entry": {"meta": {"ftype": "F16"}}},
        "meta": {"gpu": "none"},
    }  # fmt: skip
    assert summary_line(env) == (
        "server llama.cpp, build b11604-45451c366, weights F16 (as reported), gpu=none"
    )
    assert summary_line({}) is None


def test_bench_records_environment_and_meta(mock_endpoint, tmp_path):
    with mock_endpoint(MockBehaviour(reply="hi")) as (_, url):
        result = runner.invoke(app, [
            "bench", "--url", url, "--model", "mock", "-n", "2", "--warmup", "1",
            "--repeats", "1", "--max-tokens", "2", "--out", str(tmp_path),
            "--meta", "gpu=none", "--meta", "provider=laptop",
        ])  # fmt: skip
    assert result.exit_code == 0, result.output
    (path,) = tmp_path.glob("*.json")
    env = json.loads(path.read_text(encoding="utf-8"))["environment"]
    assert env["meta"] == {"gpu": "none", "provider": "laptop"}
    assert env["server"]["system_fingerprint"] == "mock-build-1"
    md = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Environment: server frugal-mock, build mock-build-1, weights F16" in md
    assert not list(tmp_path.glob("*.partial.json"))  # cleaned up after a complete sweep


def test_finished_runs_survive_a_server_that_dies_mid_sweep(mock_endpoint, tmp_path):
    # Requests: probe 1, preflight 1, then c=1 (1 warm-up + 3 measured) = 6. Everything after
    # fails, so the c=2 level aborts. The finished c=1 run must already be on disk.
    with mock_endpoint(MockBehaviour(fail_after=6)) as (_, url):
        result = runner.invoke(app, [
            "bench", "--url", url, "--model", "mock", "-c", "1,2", "-n", "3",
            "--warmup", "1", "--repeats", "1", "--max-tokens", "2", "--out", str(tmp_path),
        ])  # fmt: skip
    assert result.exit_code == 2
    assert "Finished runs are saved" in result.output
    (partial,) = tmp_path.glob("*.partial.json")
    doc = json.loads(partial.read_text(encoding="utf-8"))
    assert doc["complete"] is False and [r["concurrency"] for r in doc["runs"]] == [1]

    rendered = runner.invoke(app, ["report", str(partial)])
    assert "Incomplete run.** Stopped after 1 of 2 runs" in rendered.stdout_bytes.decode()


def test_compare_warns_when_the_baseline_is_not_full_precision(mock_endpoint, tmp_path):
    def run(label: str, ftype: str) -> None:
        with mock_endpoint(MockBehaviour(reply="Answer: 18", reported_ftype=ftype)) as (_, url):
            result = runner.invoke(app, [
                "quality", "--url", url, "--model", "mock", "--task", "gsm8k", "--limit", "2",
                "--label", label, "--out", str(tmp_path), "--meta", f"quant={ftype}",
            ])  # fmt: skip
        assert result.exit_code == 0, result.output

    run("base", "Q4_K_M")  # the "baseline" is actually quantised
    run("cand", "Q2_K")
    paths = {p.stem.split("-quality-gsm8k-")[1]: p for p in tmp_path.glob("*.json")}
    result = runner.invoke(app, ["compare", str(paths["base"]), str(paths["cand"])])
    md = result.stdout_bytes.decode()
    assert "WARNING: the baseline's server reported its weights as Q4_K_M" in md
    assert (
        "- cand: server frugal-mock, build mock-build-1, weights Q2_K (as reported), quant=Q2_K"
        in md
    )
