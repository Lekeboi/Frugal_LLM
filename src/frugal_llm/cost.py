"""Cost model. Pure functions.

Self-hosted cost per token depends entirely on utilisation: a rented GPU costs the same
per hour whether it serves a million tokens or none. So the comparison with a commercial
API is not a ratio but a break-even point: the sustained traffic above which self-hosting
is cheaper. See docs/cost-method.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86_400
PER_MILLION = 1_000_000


def cost_per_million_tokens(cost_per_hour: float, output_tokens_per_sec: float) -> float | None:
    """``(cost_per_hour / 3600) / output_tokens_per_sec * 1_000_000``.

    ``None`` when nothing was produced: the cost of a token that never arrived is not
    a number, and reporting infinity or zero would both mislead.
    """
    if output_tokens_per_sec <= 0:
        return None
    return cost_per_hour / SECONDS_PER_HOUR / output_tokens_per_sec * PER_MILLION


# --- comparison against a commercial API --------------------------------------------


@dataclass(frozen=True)
class ApiPrice:
    """A commercial per-token price, entered by the user with its source and date.

    Prices are never built into the tool: they change, and a stale price presented as
    current is exactly the kind of number this project exists not to publish.
    """

    name: str
    input_per_million: float
    output_per_million: float
    source: str
    date: str

    @classmethod
    def parse(cls, spec: str) -> ApiPrice:
        """``"name=gpt-x,in=0.15,out=0.60,source=https://...,date=2026-10-15"``."""
        fields = dict(_pairs(spec))
        missing = {"name", "in", "out", "source", "date"} - fields.keys()
        if missing:
            raise ValueError(f"API price {spec!r} is missing {', '.join(sorted(missing))}")
        try:
            date.fromisoformat(fields["date"])
            prices = float(fields["in"]), float(fields["out"])
        except ValueError:
            raise ValueError(
                f"API price {spec!r}: in/out must be numbers and date YYYY-MM-DD"
            ) from None
        if min(prices) < 0:
            raise ValueError(f"API price {spec!r}: prices cannot be negative")
        return cls(fields["name"], prices[0], prices[1], fields["source"], fields["date"])


def _pairs(spec: str):
    # Split on commas that start a new key=, so a source URL may itself contain commas.
    for part in re.split(r",(?=\s*[A-Za-z_]+\s*=)", spec):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"expected key=value in {spec!r}, got {part!r}")
        yield key.strip(), value.strip()


def api_cost_per_request(price: ApiPrice, prompt_tokens: float, output_tokens: float) -> float:
    """What the API charges for one request of this shape."""
    return (
        prompt_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / PER_MILLION


def self_hosted_cost_per_request(hourly_rate: float, capacity_rps: float) -> float | None:
    """Cost of one request when the hardware is kept 100% busy at its measured capacity."""
    if capacity_rps <= 0:
        return None
    return hourly_rate / SECONDS_PER_HOUR / capacity_rps


@dataclass(frozen=True)
class BreakEven:
    requests_per_sec: float
    """Sustained traffic at which one instance costs the same as the API."""
    utilisation: float
    """That traffic as a share of the instance's measured capacity."""

    @property
    def requests_per_day(self) -> float:
        return self.requests_per_sec * SECONDS_PER_DAY

    @property
    def reachable(self) -> bool:
        """False when even 100% utilisation costs more than the API."""
        return self.utilisation <= 1.0


def break_even(hourly_rate: float, capacity_rps: float, api_per_request: float) -> BreakEven | None:
    """Traffic at which hourly self-hosting matches per-request API pricing.

    One instance costs ``hourly_rate`` per hour regardless of traffic r, while the API
    costs ``r * 3600 * api_per_request``. They are equal at r* = hourly_rate / (3600 * A).
    ``None`` when the API is free (no finite break-even exists).
    """
    if api_per_request <= 0 or capacity_rps <= 0:
        return None
    rps = hourly_rate / (SECONDS_PER_HOUR * api_per_request)
    return BreakEven(rps, rps / capacity_rps)


def daily_cost_self_hosted(
    requests_per_day: float, hourly_rate: float, capacity_rps: float
) -> float:
    """Instances needed to carry the traffic (at least one, kept running), for a day."""
    instances = max(1, math.ceil(requests_per_day / (capacity_rps * SECONDS_PER_DAY)))
    return instances * hourly_rate * 24


def daily_cost_api(requests_per_day: float, api_per_request: float) -> float:
    return requests_per_day * api_per_request
