"""Cost model. Pure functions."""

from __future__ import annotations

SECONDS_PER_HOUR = 3600
PER_MILLION = 1_000_000


def cost_per_million_tokens(cost_per_hour: float, output_tokens_per_sec: float) -> float | None:
    """``(cost_per_hour / 3600) / output_tokens_per_sec * 1_000_000``.

    ``None`` when nothing was produced: the cost of a token that never arrived is not
    a number, and reporting infinity or zero would both mislead.
    """
    if output_tokens_per_sec <= 0:
        return None
    return cost_per_hour / SECONDS_PER_HOUR / output_tokens_per_sec * PER_MILLION
