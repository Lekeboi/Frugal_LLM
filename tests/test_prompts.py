import pytest

from frugal_llm.prompts import FilePrompts, GeneratedPrompts


def test_generated_prompts_are_reproducible_from_their_seed():
    assert GeneratedPrompts(200, seed=7).get(5) == GeneratedPrompts(200, seed=7).get(5)


def test_requests_in_a_run_differ_right_after_the_id():
    p = GeneratedPrompts(200)
    body = lambda i: p.get(i).split(": ", 1)[1]  # noqa: E731
    assert body(1)[:12] != body(2)[:12]


def test_every_run_gets_new_prompts():
    """Regression: prompts used to depend only on the request index, so a second run
    re-sent the first run's prompts and hit the server's prompt cache (TTFT 28 ms
    against ~1000 ms on llama.cpp)."""
    a, b = GeneratedPrompts(200), GeneratedPrompts(200)
    assert a.seed != b.seed
    assert a.get(0)[:8] != b.get(0)[:8]


@pytest.mark.parametrize("tokens", [50, 200, 1000])
def test_generated_prompt_length_tracks_target(tokens):
    words = len(GeneratedPrompts(tokens).get(0).split())
    assert abs(words - tokens) <= 20


def test_file_prompts_rotate(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    p = FilePrompts.load(path)
    assert [p.get(i) for i in range(5)] == ["a", "b", "a", "b", "a"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"prompt": "a"}\nnot json\n', r"p.jsonl:2: invalid JSON"),
        ('{"text": "a"}\n', r"p.jsonl:1: expected"),
        ('{"prompt": ""}\n', r"p.jsonl:1: expected"),
        ("\n", "no prompts found"),
    ],
)
def test_file_prompt_errors_name_the_line(tmp_path, content, message):
    path = tmp_path / "p.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        FilePrompts.load(path)
