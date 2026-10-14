import pytest

from frugal_llm.bench import RequestResult, RunResult
from frugal_llm.metrics import Spread, percentiles, summarise, summarise_level


def req(ttft=100.0, itl=10.0, e2e=500.0, tokens=10, source="usage", error=None):
    if error:
        return RequestResult(0.0, 0.0, error=error)
    return RequestResult(0.0, 0.0, ttft, itl, e2e, tokens, None, source)


def run(requests, concurrency=1, repeat=0, window=1.0):
    return RunResult(concurrency, repeat, window, requests)


def test_percentiles_hand_computed():
    # Inclusive method: rank (n-1)*p/100 into the sorted data, linear in between.
    # 1..10: p50 rank 4.5 -> 5.5; p95 rank 8.55 -> 9.55; p99 rank 8.91 -> 9.91
    assert percentiles(range(1, 11), (50, 95, 99)) == pytest.approx({50: 5.5, 95: 9.55, 99: 9.91})
    # [10,20,30,40]: p50 rank 1.5 -> 25; p95 rank 2.85 -> 38.5
    assert percentiles([40, 10, 30, 20], (50, 95)) == pytest.approx({50: 25.0, 95: 38.5})


def test_percentiles_edge_cases():
    assert percentiles([], (50, 95)) == {50: None, 95: None}
    assert percentiles([7.0], (50, 99)) == {50: 7.0, 99: 7.0}


def test_tail_is_not_hidden_by_averaging():
    samples = [100] * 95 + [5000] * 5
    p = percentiles(samples, (50, 99))
    assert p[50] == 100 and p[99] == 5000


def test_throughput_error_rate_and_breakdown():
    r = run([req(tokens=100), req(tokens=100), req(error="read_timeout")], window=2.0)
    (s,) = summarise([r], None)
    assert s.output_tokens_per_sec == Spread(100.0, 100.0, 100.0)
    assert s.requests_per_sec.median == 1.0
    assert (s.attempted, s.completed) == (3, 2)
    assert s.error_rate == pytest.approx(1 / 3)
    assert s.errors == {"read_timeout": 1}


def test_spread_across_repeats():
    runs = [run([req(tokens=t)], repeat=i) for i, t in enumerate([90, 120, 100])]
    s = summarise_level(runs, cost_per_hour=3.6)
    assert s.output_tokens_per_sec == Spread(100.0, 90.0, 120.0)
    assert s.cost_per_million.median == pytest.approx(10.0)  # 0.001/s / 100 tok/s * 1e6


def test_latency_pools_repeats_within_a_level():
    runs = [run([req(ttft=100)], repeat=0), run([req(ttft=300)], repeat=1)]
    assert summarise_level(runs, None).ttft_ms[50] == 200


def test_never_pools_across_levels():
    with pytest.raises(ValueError, match="across concurrency levels"):
        summarise_level([run([req()], concurrency=1), run([req()], concurrency=4)], None)
    summaries = summarise(
        [run([req(ttft=100)], concurrency=4), run([req(ttft=900)], concurrency=1)], None
    )
    assert [(s.concurrency, s.ttft_ms[50]) for s in summaries] == [(4, 100), (1, 900)]


def test_single_chunk_requests_have_no_itl():
    s = summarise_level([run([req(itl=None), req(itl=30)])], None)
    assert s.itl_ms[50] == 30


def test_token_sources_and_no_successes():
    s = summarise_level([run([req(source="chunks"), req(source="usage")])], None)
    assert s.token_count_sources == ("chunks", "usage")
    s = summarise_level([run([req(error="http_500")], window=0.0)], 1.0)
    assert s.ttft_ms[50] is None and s.cost_per_million is None and s.error_rate == 1.0


def test_stalled_runs_are_excluded_from_every_aggregate():
    good = RunResult(1, 0, 1.0, [req(ttft=100, tokens=50)], 10.0, 20.0)
    # The machine slept mid-run: one 20-minute lag sample, as seen on a real laptop.
    slept = RunResult(1, 1, 1200.0, [req(ttft=100, tokens=50)], 11.0, 1_217_591.5)
    s = summarise_level([good, slept], None)
    assert (s.runs, s.excluded_runs) == (1, 1)
    assert s.output_tokens_per_sec == Spread(50.0, 50.0, 50.0)
    assert s.attempted == 1


def test_level_with_only_stalled_runs_reports_nothing():
    slept = RunResult(1, 0, 1200.0, [req()], 11.0, 5000.0)
    s = summarise_level([slept], 1.0)
    assert s.runs == 0 and s.ttft_ms[50] is None and s.output_tokens_per_sec is None
