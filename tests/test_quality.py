from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from mock_server import MockBehaviour
from typer.testing import CliRunner

from frugal_llm import quality as q
from frugal_llm.cli import app

# --- task sets ----------------------------------------------------------------------


def test_shipped_task_sets_load_and_match_their_manifest():
    gsm, gsm_hash = q.load_task("gsm8k")
    mmlu, _ = q.load_task("mmlu")
    assert (len(gsm), len(mmlu)) == (1319, 1000)
    assert gsm[0].id == "gsm8k-test-0000" and gsm[0].answer == "18"
    assert gsm[0].prompt.endswith("End with a final line of the form `Answer: <number>`.")
    assert "\nA. " in mmlu[0].prompt and mmlu[0].answer in "ABCD"
    assert len(gsm_hash) == 64


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        q.load_task("humaneval")


# --- answer extraction (rules fixed in docs/quality-method.md) ---------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("16 - 3 - 4 = 9\n9 * 2 = 18\nAnswer: 18", ("18", "answer_line")),
        ("Answer: 5\nWait, recheck.\nAnswer: 6", ("6", "answer_line")),  # the last one
        ("Answer: **$1,234.50**", ("1234.5", "answer_line")),
        ("answer：42", ("42", "answer_line")),  # noqa: RUF001 -- fullwidth colon
        ("The total is 70,000 dollars.", ("70000", "last_number")),
        ("Answer: -3", ("-3", "answer_line")),
        ("so the total is\n" r"\[ \boxed{18} \]", ("18", "boxed")),
        (r"Answer: 7 is wrong; \boxed{9}", ("7", "answer_line")),  # explicit line wins
        ("I don't know.", (None, "none")),
    ],
)
def test_gsm8k_extraction(output, expected):
    assert q.extract_gsm8k(output) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("C", ("C", "leading_letter")),
        ("C. 24", ("C", "leading_letter")),
        ("(B)", ("B", "leading_letter")),
        ("**D**", ("D", "leading_letter")),
        ("A good answer is C", ("C", "answer_phrase")),  # not A
        ("The answer is B.", ("B", "answer_phrase")),
        ("I think B because it fits", ("B", "short_reply_letter")),
        ("None of these.", (None, "none")),
        # Real Qwen2.5-0.5B outputs: the stated answer is at the end, after working that
        # contains other letters. The old first-letter rule scored these as A.
        (
            "To determine the maximum possible order... A cycle of length 7 and 3 gives 21.\n"
            "Therefore, the correct answer is:\n"
            r"\["
            "\n"
            r"\boxed{D}"
            "\n"
            r"\]",
            ("D", "boxed"),
        ),
        (
            "Statement 2 is false.\n\n### Final Answer\nThe correct option is C. True, False.",
            ("C", "answer_phrase"),
        ),
        (r"The correct answer is \(\boxed{\text{B}}\).", ("B", "boxed")),
        # Long working with no stated answer: no guess from a stray letter.
        (" ".join(["A group of order 6 is cyclic"] * 6), (None, "none")),
    ],
)
def test_mmlu_extraction(output, expected):
    assert q.extract_mmlu(output) == expected


def test_development_subset_spans_the_whole_task():
    items, _ = q.load_task("mmlu")
    subset = q.development_subset(items, 100)
    assert len(subset) == 100 and subset[0] is items[0]
    subjects = {i.prompt.split("question about ")[1].split(".\n")[0] for i in subset}
    assert len(subjects) == 57  # the first 100 items would cover only 6


def test_numeric_answers_compare_as_exact_decimals():
    assert q.is_correct("gsm8k", "18.0", "18")
    assert q.is_correct("gsm8k", "70000", "70000")
    assert not q.is_correct("gsm8k", "18.5", "18")
    assert not q.is_correct("gsm8k", None, "18")
    assert q.is_correct("mmlu", "C", "C") and not q.is_correct("mmlu", "A", "C")


# --- rate limits ---------------------------------------------------------------------


def _response(headers: dict) -> httpx.Response:
    return httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x"))


def test_retry_after_honours_the_server():
    assert q._retry_after(_response({"retry-after": "7"}), attempt=1) == 7.0
    soon = (datetime.now(UTC) + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert 25 <= q._retry_after(_response({"retry-after": soon}), attempt=1) <= 30
    assert q._retry_after(_response({}), attempt=3) == 8.0  # exponential backoff
    assert q._retry_after(_response({"retry-after": "9999"}), attempt=1) == q.MAX_BACKOFF_S


def _cfg(url: str, **overrides) -> q.QualityConfig:
    values = dict(url=url, model="mock", api_key=None, task="gsm8k", limit=3, concurrency=1,
                  max_tokens=64, max_retries=3, timeout=30.0, label="t")  # fmt: skip
    values.update(overrides)
    return q.QualityConfig(**values)


async def test_rate_limited_items_are_retried_not_scored_wrong(mock_endpoint, tmp_path):
    with mock_endpoint(MockBehaviour(reply="Answer: 18", rate_limit_first=2)) as (server, url):
        run = q.prepare_run(_cfg(url, limit=1), tmp_path / "r.partial.jsonl")
        doc = await q.run_quality(_cfg(url, limit=1), run)
        assert server.stats.requests_received == 3
    (item,) = doc["items"]
    assert (item["attempts"], item["correct"], item["error"]) == (3, True, None)


# --- whole runs ----------------------------------------------------------------------


async def test_run_scores_items_with_greedy_decoding(mock_endpoint, tmp_path):
    # Item 0's answer is 18; items 1 and 2 have other answers.
    with mock_endpoint(MockBehaviour(reply="Working...\nAnswer: 18")) as (server, url):
        run = q.prepare_run(_cfg(url), tmp_path / "r.partial.jsonl")
        doc = await q.run_quality(_cfg(url), run)
        body = server.stats.last_body
    assert [i["correct"] for i in doc["items"]] == [True, False, False]
    assert (body["temperature"], body["top_p"], body["seed"], body["stream"]) == (0, 1, 0, False)
    assert doc["config"]["development_subset"] is True
    assert q.summarise_run(doc)["correct"] == 1


async def test_cut_off_output_is_never_graded(mock_endpoint, tmp_path):
    # With no fixed reply the mock emits max_tokens filler words and finish_reason=length;
    # its text even contains digits, which the last-number rule would otherwise grab.
    with mock_endpoint() as (_, url):
        doc = await q.run_quality(
            _cfg(url, limit=1), q.prepare_run(_cfg(url, limit=1), tmp_path / "p")
        )
    (item,) = doc["items"]
    assert (item["finish_reason"], item["extracted"], item["rule"]) == ("length", None, "truncated")


async def test_interrupted_run_resumes_without_redoing_items(mock_endpoint, tmp_path):
    partial = tmp_path / "r.partial.jsonl"
    with mock_endpoint(MockBehaviour(reply="Answer: 1")) as (server, url):
        await q.run_quality(_cfg(url, limit=4), q.prepare_run(_cfg(url, limit=4), partial))
        lines = partial.read_text(encoding="utf-8").splitlines()
        partial.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # header + 2 items
        first_ids = [json.loads(line)["id"] for line in lines[1:3]]

    with mock_endpoint(MockBehaviour(reply="Answer: 1")) as (server, url):
        run = q.prepare_run(_cfg(url, limit=4), partial, resume=True)
        assert sorted(run.done) == sorted(first_ids)
        doc = await q.run_quality(_cfg(url, limit=4), run)
        assert server.stats.requests_received == 2
    assert len(doc["items"]) == 4


def test_resume_refuses_different_settings(tmp_path):
    partial = tmp_path / "r.partial.jsonl"
    q.prepare_run(_cfg("http://x/v1"), partial)
    with pytest.raises(q.QualityAborted, match="different settings"):
        q.prepare_run(_cfg("http://x/v1", max_tokens=99), partial, resume=True)


async def test_unusable_endpoint_aborts_on_the_first_item(mock_endpoint, tmp_path):
    with mock_endpoint(MockBehaviour(status_code=404)) as (_, url):
        run = q.prepare_run(_cfg(url), tmp_path / "r.partial.jsonl")
        with pytest.raises(q.QualityAborted, match="http_404"):
            await q.run_quality(_cfg(url), run)


# --- end to end through the CLI -----------------------------------------------------

runner = CliRunner()


def test_cli_quality_then_compare(mock_endpoint, tmp_path):
    def run(label: str, reply: str) -> None:
        with mock_endpoint(MockBehaviour(reply=reply)) as (_, url):
            result = runner.invoke(app, [
                "quality", "--url", url, "--model", "m", "--task", "gsm8k", "--limit", "3",
                "--label", label, "--out", str(tmp_path),
            ])  # fmt: skip
        assert result.exit_code == 0, result.output

    run("fp16-a", "Answer: 18")
    run("fp16-b", "Answer: 18")
    run("q4", "Answer: 17")
    paths = {p.stem.split("-quality-gsm8k-")[1]: p for p in tmp_path.glob("*.json")}
    assert not list(tmp_path.glob("*.partial.jsonl"))  # cleaned up after completion

    result = runner.invoke(app, [
        "compare", str(paths["fp16-a"]), str(paths["q4"]),
        "--noise-floor", str(paths["fp16-b"]), "--write", str(tmp_path / "cmp.md"),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    md = (tmp_path / "cmp.md").read_text(encoding="utf-8")
    assert "Development subset" in md
    assert "| q4 | 0.0%" in md
    assert "Noise floor: two baseline runs agreed on 100.0% of final answers" in md
    assert "no difference detected" in md  # 1 of 3 items: far below any resolution
