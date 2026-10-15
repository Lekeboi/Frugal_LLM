import pytest

from frugal_llm.stats import (
    holm,
    mcnemar_exact,
    minimum_detectable_difference,
    paired_difference_ci,
    wilson,
)


def test_wilson_matches_published_values():
    # Standard textbook example: 8 of 10 -> 0.490 to 0.943.
    lo, hi = wilson(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_hand_computed():
    # b=0, c=5: two-sided binomial tail = 2 * (1/2)^5 = 0.0625
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    # b=1, c=9: 2 * (C(10,0) + C(10,1)) / 2^10 = 22/1024
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_paired_ci_contains_estimate_and_respects_bounds():
    a, b, c, d = 600, 60, 40, 300
    lo, hi = paired_difference_ci(a, b, c, d)
    estimate = (c - b) / (a + b + c + d)
    assert lo < estimate < hi
    assert -1 <= lo and hi <= 1


def test_paired_ci_is_antisymmetric_when_roles_swap():
    lo, hi = paired_difference_ci(600, 60, 40, 300)
    lo2, hi2 = paired_difference_ci(600, 40, 60, 300)
    assert (lo, hi) == pytest.approx((-hi2, -lo2))


def test_paired_ci_approaches_wald_for_large_samples():
    # n = 100,000: score and Wald intervals should agree closely.
    a, b, c, d = 60_000, 6_000, 4_000, 30_000
    n = a + b + c + d
    diff = (c - b) / n
    se = ((b + c) / n - diff**2) ** 0.5 / n**0.5
    lo, hi = paired_difference_ci(a, b, c, d)
    assert lo == pytest.approx(diff - 1.96 * se, abs=5e-4)
    assert hi == pytest.approx(diff + 1.96 * se, abs=5e-4)


def test_identical_models_give_zero_centred_interval():
    lo, hi = paired_difference_ci(700, 0, 0, 300)
    assert lo <= 0 <= hi


def test_holm_hand_computed():
    # sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.9, 0.9]) == [1.0, 1.0]


def test_minimum_detectable_difference_inverts_the_sample_size_table():
    # From the method note's table: 870 items detect 3 points at 10% disagreement.
    assert minimum_detectable_difference(870, 0.10) == pytest.approx(0.03, abs=0.0005)
    assert minimum_detectable_difference(1319, 0.10) < 0.03
    assert minimum_detectable_difference(1319, 0.0) is None
