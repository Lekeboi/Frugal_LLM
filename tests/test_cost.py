import pytest

from frugal_llm.cost import cost_per_million_tokens


def test_hand_computed():
    # 3.60/hour = 0.001/s; at 1000 tok/s a million tokens take 1000 s -> 1.00
    assert cost_per_million_tokens(3.60, 1000) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("tokens_per_sec", "expected"), [(47.2, 3.06), (162.8, 0.89), (398.1, 0.36), (512.4, 0.28)]
)
def test_matches_spec_example_table(tokens_per_sec, expected):
    """The SPEC section 6 example: GBP 0.52/hour."""
    assert round(cost_per_million_tokens(0.52, tokens_per_sec), 2) == expected


@pytest.mark.parametrize("tokens_per_sec", [0, -1])
def test_no_output_has_no_cost(tokens_per_sec):
    assert cost_per_million_tokens(0.52, tokens_per_sec) is None
