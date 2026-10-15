import pytest

from frugal_llm.compare import (
    NotComparable,
    compare_pair,
    compare_runs,
    first_divergence,
    render_markdown,
    word_edit_distance,
)


def test_word_edit_distance_hand_computed():
    assert word_edit_distance("a b c", "a b c") == 0
    assert word_edit_distance("a b c", "a x c") == 1  # substitution
    assert word_edit_distance("a b c", "a c") == 1  # deletion
    assert word_edit_distance("", "a b") == 2
    assert word_edit_distance("kitten sat", "sitting sat") == 1  # words, not characters


def test_first_divergence():
    assert first_divergence("the cat sat", "the cat sat") is None
    assert first_divergence("the cat sat", "the dog sat") == 1
    assert first_divergence("the cat", "the cat sat") == 2


def _item(i: int, correct: bool, output: str = "x", extracted: str | None = "1") -> dict:
    return {"id": f"i{i}", "output": output, "extracted": extracted, "rule": "answer_line",
            "correct": correct, "finish_reason": "stop", "prompt_tokens": 5,
            "completion_tokens": 1, "attempts": 1, "error": None}  # fmt: skip


def _doc(label: str, correct: list[bool], **config) -> dict:
    cfg = {"task": "gsm8k", "task_version": "v1", "task_sha256": "abc", "items": len(correct),
           "development_subset": False, "max_tokens": 512, "temperature": 0, "top_p": 1,
           "seed": 0, "concurrency": 1, "model": label, "url": "http://x/v1"}  # fmt: skip
    cfg.update(config)
    return {"kind": "quality", "label": label, "config": cfg,
            "items": [_item(i, c) for i, c in enumerate(correct)]}  # fmt: skip


def test_paired_counts():
    base = _doc("fp16", [True, True, False, False])
    cand = _doc("q4", [True, False, True, False])
    r = compare_pair(base, cand)
    assert (r.both_right, r.baseline_only, r.candidate_only, r.both_wrong) == (1, 1, 1, 1)
    assert r.difference == 0 and r.accuracy == 0.5


def test_runs_with_different_task_files_are_refused():
    with pytest.raises(NotComparable, match="task_sha256"):
        compare_runs(_doc("fp16", [True]), [_doc("q4", [True], task_sha256="zzz")])


def test_different_concurrency_is_warned_about():
    _, _, warnings = compare_runs(_doc("fp16", [True]), [_doc("q4", [True], concurrency=8)])
    assert "batching can change greedy outputs" in warnings[0]


def test_a_large_drop_is_reported_as_significant():
    base = _doc("fp16", [True] * 400 + [False] * 100)
    cand = _doc("q2", [True] * 300 + [False] * 200)  # 100 items lost, none gained
    results, floor, warnings = compare_runs(base, [cand])
    md = render_markdown(base, results, floor, warnings)
    assert "**q2**: accuracy 20.0 points lower than the baseline" in md
    assert "No noise floor supplied" in md


def test_errored_items_are_excluded_from_both_sides():
    base = _doc("fp16", [True, True, True])
    cand = _doc("q4", [True, True, True])
    cand["items"][1]["error"] = "read_timeout"
    r = compare_pair(base, cand)
    assert (r.n, r.excluded) == (2, 1)
