import pytest

from frugal_llm.cost import (
    ApiPrice,
    api_cost_per_request,
    break_even,
    cost_per_million_tokens,
    daily_cost_api,
    daily_cost_self_hosted,
    self_hosted_cost_per_request,
)


def test_hand_computed():
    # 3.60/hour = 0.001/s; at 1000 tok/s a million tokens take 1000 s -> 1.00
    assert cost_per_million_tokens(3.60, 1000) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("tokens_per_sec", "expected"), [(47.2, 3.06), (162.8, 0.89), (398.1, 0.36), (512.4, 0.28)]
)
def test_matches_spec_example_table(tokens_per_sec, expected):
    """A worked example: GBP 0.52/hour across a concurrency sweep."""
    assert round(cost_per_million_tokens(0.52, tokens_per_sec), 2) == expected


@pytest.mark.parametrize("tokens_per_sec", [0, -1])
def test_no_output_has_no_cost(tokens_per_sec):
    assert cost_per_million_tokens(0.52, tokens_per_sec) is None


# --- comparison against a commercial API --------------------------------------------


PRICE = ApiPrice("example", 0.50, 1.50, "https://example.com/pricing", "2026-10-15")


def test_api_price_parsing():
    spec = "name=example,in=0.50,out=1.50,source=https://example.com/a,b/pricing,date=2026-10-15"
    p = ApiPrice.parse(spec)
    assert (p.name, p.input_per_million, p.output_per_million) == ("example", 0.5, 1.5)
    assert p.source == "https://example.com/a,b/pricing"  # commas in the URL survive


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("name=x,in=1,out=2,source=s", "missing date"),
        ("name=x,in=cheap,out=2,source=s,date=2026-10-15", "must be numbers"),
        ("name=x,in=1,out=2,source=s,date=15/10/2026", "YYYY-MM-DD"),
        ("name=x,in=-1,out=2,source=s,date=2026-10-15", "cannot be negative"),
    ],
)
def test_api_price_rejects_incomplete_or_bad_input(spec, message):
    with pytest.raises(ValueError, match=message):
        ApiPrice.parse(spec)


def test_api_cost_of_a_request_hand_computed():
    # 200 in x $0.50/M + 100 out x $1.50/M = 0.0001 + 0.00015 = $0.00025
    assert api_cost_per_request(PRICE, 200, 100) == pytest.approx(0.00025)


def test_break_even_hand_computed():
    # $0.90/hour, API $0.00025/request -> break-even at 0.9 / (3600 * 0.00025) = 1 request/s.
    # Capacity 4 requests/s -> 25% utilisation; 86,400 requests/day.
    be = break_even(0.90, capacity_rps=4.0, api_per_request=0.00025)
    assert be.requests_per_sec == pytest.approx(1.0)
    assert be.utilisation == pytest.approx(0.25)
    assert be.requests_per_day == pytest.approx(86_400)
    assert be.reachable
    # At full use the hardware costs 0.9/3600/4 = $0.0000625/request: cheaper than the API.
    assert self_hosted_cost_per_request(0.90, 4.0) == pytest.approx(0.0000625)


def test_hardware_too_slow_never_breaks_even():
    # Capacity 0.5 requests/s but break-even needs 1/s: costlier than the API even when full.
    be = break_even(0.90, capacity_rps=0.5, api_per_request=0.00025)
    assert be.utilisation == pytest.approx(2.0) and not be.reachable


def test_daily_costs_step_with_instances():
    # One instance carries 4 req/s = 345,600/day. 400,000/day needs two instances.
    assert daily_cost_self_hosted(0, 0.90, 4.0) == pytest.approx(21.6)  # idle still costs
    assert daily_cost_self_hosted(345_600, 0.90, 4.0) == pytest.approx(21.6)
    assert daily_cost_self_hosted(400_000, 0.90, 4.0) == pytest.approx(43.2)
    assert daily_cost_api(86_400, 0.00025) == pytest.approx(21.6)  # equal at break-even


def test_no_break_even_against_a_free_api():
    assert break_even(0.90, 4.0, 0.0) is None
