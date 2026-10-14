import pytest

from frugal_llm.prompts import FilePrompts, GeneratedPrompts


def test_generated_prompts_are_deterministic_and_distinct_from_the_start():
    p = GeneratedPrompts(200)
    assert p.get(5) == p.get(5)
    # Different from the first characters, so no shared prefix for a cache to reuse.
    assert p.get(1)[:8] != p.get(2)[:8]


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
