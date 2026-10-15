"""Statistics for paired quality comparisons. Pure functions, standard library only.

See docs/quality-method.md section 5 for why each of these is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.975)


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The bounds are exactly 0 and 1 at the extremes; don't report floating-point dust.
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return (lower, upper)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: a binomial test on the b + c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference_ci(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float]:
    """95% CI for p_candidate - p_baseline from a paired 2x2 table (Newcombe 1998, method 10).

    a: both right, b: only the baseline right, c: only the candidate right, d: both wrong.
    """
    n = a + b + c + d
    if n == 0:
        return (-1.0, 1.0)
    p1, p2 = (a + c) / n, (a + b) / n  # candidate, baseline
    l1, u1 = wilson(a + c, n, z)
    l2, u2 = wilson(a + b, n, z)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if denom == 0 else (a * d - b * c) / math.sqrt(denom)
    theta = p1 - p2
    lower = theta - math.sqrt(
        max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    )
    upper = theta + math.sqrt(
        max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    )
    return (max(-1.0, lower), min(1.0, upper))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def minimum_detectable_difference(
    n: int, discordant_share: float, alpha: float = 0.05, power: float = 0.80
) -> float | None:
    """Smallest accuracy difference a paired test of n items can detect.

    Inverts the McNemar sample-size formula n = (za*sqrt(p) + zb*sqrt(p - d^2))^2 / d^2 by
    bisection. None when nothing is detectable (no disagreement observed or n too small).
    """
    if n <= 0 or discordant_share <= 0:
        return None
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    p = discordant_share

    def needed(diff: float) -> float:
        return (za * math.sqrt(p) + zb * math.sqrt(max(p - diff * diff, 0.0))) ** 2 / diff**2

    lo, hi = 1e-6, min(p, 1.0)
    if needed(hi) > n:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if needed(mid) > n else (lo, mid)
    return hi
