from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import httpx
import pytest
from conftest import assert_result_invariants
from mock_server import MockBehaviour

from frugal_llm.bench import (
    BenchConfig,
    classify_error,
    make_client,
    request_timeout,
    ssl_context,
    stream_one,
)


def _config(url: str, **overrides) -> BenchConfig:
    values = dict(
        url=url,
        model="mock",
        api_key=None,
        concurrency=[1],
        requests=10,
        max_tokens=1,
        prompt_tokens=10,
        prompts_path=None,
        warmup=0,
        repeats=1,
        cost_per_hour=None,
        currency="GBP",
        timeout=30.0,
        out=Path("results"),
        format="json",
        label=None,
    )
    values.update(overrides)
    return BenchConfig(**values)


async def _hold_open(client: httpx.AsyncClient, url: str = "/chat/completions") -> None:
    body = {"model": "mock", "messages": [], "max_tokens": 1, "stream": True}
    async with client.stream("POST", url, json=body) as resp:
        resp.raise_for_status()
        async for _ in resp.aiter_bytes():
            pass


def test_timeouts_are_per_phase():
    t = request_timeout(42)
    assert (t.connect, t.read, t.write, t.pool) == (10.0, 42, 10.0, 42)


# Sweep past httpx's default pool of 100, so a change to client construction or library
# defaults cannot quietly reintroduce serialisation.
CONCURRENCY_PAST_DEFAULT_POOL = 150


async def _rounds_with_worker_clients(url: str, n: int, rounds: int) -> list[float]:
    """Wall time of each round of n concurrent requests, one worker client each."""
    async with contextlib.AsyncExitStack() as stack:
        clients = [await stack.enter_async_context(make_client(_config(url))) for _ in range(n)]
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            await asyncio.gather(*(_hold_open(c) for c in clients))
            times.append(time.perf_counter() - t0)
        return times


async def test_worker_clients_reach_concurrency_past_default_pool(mock_endpoint):
    with mock_endpoint(MockBehaviour(first_chunk_delay=0.5)) as (server, url):
        await _rounds_with_worker_clients(url, CONCURRENCY_PAST_DEFAULT_POOL, rounds=1)
        assert server.stats.max_in_flight == CONCURRENCY_PAST_DEFAULT_POOL


async def test_default_pool_would_serialise(mock_endpoint):
    """Negative control: proves the test above can actually detect pool queuing."""
    with mock_endpoint(MockBehaviour(first_chunk_delay=0.5)) as (server, url):
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            await asyncio.gather(
                *(_hold_open(client) for _ in range(CONCURRENCY_PAST_DEFAULT_POOL))
            )
        assert server.stats.max_in_flight == 100


async def test_warm_connections_add_no_client_overhead(mock_endpoint):
    """Regression: one pool shared by 150 requests spent seconds of CPU per round
    matching requests to warm connections (httpcore's scan is requests x connections),
    turning a 1 s server delay into a ~6 s round. Per-worker clients must stay close
    to the server's own delay, especially once warm."""
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (_, url):
        _cold, warm = await _rounds_with_worker_clients(
            url, CONCURRENCY_PAST_DEFAULT_POOL, rounds=2
        )
    assert warm < 2.0


def test_ssl_context_is_loaded_once():
    assert ssl_context() is ssl_context()


async def test_server_stall_is_a_read_timeout(mock_endpoint):
    # The role chunk arrives at once; the next chunk is 1 s away. Read timeout is per chunk.
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (_, url):
        async with make_client(_config(url, timeout=0.3)) as client:
            with pytest.raises(httpx.TimeoutException) as excinfo:
                await _hold_open(client)
    assert classify_error(excinfo.value) == "read_timeout"


async def test_client_capacity_exhaustion_is_a_pool_timeout(mock_endpoint):
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (_, url):
        async with httpx.AsyncClient(
            base_url=url, limits=httpx.Limits(max_connections=1), timeout=request_timeout(0.3)
        ) as client:
            results = await asyncio.gather(
                _hold_open(client), _hold_open(client), return_exceptions=True
            )
    errors = [classify_error(r) for r in results if isinstance(r, BaseException)]
    # Classification check only: frugal-llm never shares a pool, but a pool timeout must
    # still be named as one. The second request queues behind the first's 1 s stall.
    assert "pool_timeout" in errors


def test_http_status_errors_keep_their_code():
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    response = httpx.Response(503, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    assert classify_error(exc) == "http_503"


async def test_api_key_sent_as_bearer_token():
    async with make_client(_config("http://x/v1", api_key="sk-test")) as client:
        assert client.headers["Authorization"] == "Bearer sk-test"


# --- step 3: one streamed, timed request -------------------------------------------

# asyncio timers on Windows have ~15 ms granularity.
TOL_MS = 100


async def _one(url: str, *, max_tokens: int = 4, client_timeout: float = 30, deadline_s=30):
    async with make_client(_config(url, timeout=client_timeout)) as client:
        return await stream_one(
            client, model="mock", prompt="hello there", max_tokens=max_tokens, deadline_s=deadline_s
        )


async def test_ttft_is_first_content_not_total(mock_endpoint):
    """Spec row: 500 ms to first chunk, 50 ms after; TTFT ~ 500, not ~ the 950 total."""
    with mock_endpoint(MockBehaviour(first_chunk_delay=0.5, chunk_delay=0.05)) as (_, url):
        r = await _one(url, max_tokens=10)
    assert_result_invariants(r)
    assert abs(r.ttft_ms - 500) < TOL_MS
    assert abs(r.itl_ms - 50) < 25
    assert abs(r.e2e_ms - 950) < 2 * TOL_MS
    # The role-only chunk arrives at ~0 ms; counting it would have put TTFT near zero.
    assert r.ttft_ms > 400


async def test_token_count_from_usage_when_reported(mock_endpoint):
    with mock_endpoint(MockBehaviour(output_tokens=7)) as (_, url):
        r = await _one(url, max_tokens=100)
    assert_result_invariants(r)
    assert (r.output_tokens, r.token_count_source) == (7, "usage")
    assert r.prompt_tokens == 2


async def test_token_count_from_chunks_without_usage(mock_endpoint):
    with mock_endpoint(MockBehaviour(send_usage=False)) as (_, url):
        r = await _one(url, max_tokens=6)
    assert_result_invariants(r)
    assert (r.output_tokens, r.token_count_source) == (6, "chunks")
    assert r.prompt_tokens is None


async def test_http_error_is_recorded_not_raised(mock_endpoint):
    with mock_endpoint(MockBehaviour(status_code=503)) as (_, url):
        r = await _one(url)
    assert_result_invariants(r)
    assert r.error == "http_503"


async def test_stalled_chunk_is_read_timeout(mock_endpoint):
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (_, url):
        r = await _one(url, client_timeout=0.3)
    assert r.error == "read_timeout"


async def test_slow_trickle_hits_total_deadline(mock_endpoint):
    """Each 0.2 s gap is well inside the 1 s read timeout, but 20 of them are not."""
    with mock_endpoint(MockBehaviour(chunk_delay=0.2)) as (_, url):
        r = await _one(url, max_tokens=20, client_timeout=1.0, deadline_s=1.0)
    assert r.error == "total_timeout"
    assert abs((r.finished_at_s - r.sent_at_s) - 1.0) < TOL_MS / 1000


async def test_zero_content_is_an_error(mock_endpoint):
    with mock_endpoint(MockBehaviour(output_tokens=0)) as (_, url):
        r = await _one(url)
    assert r.error == "empty_response"


# --- why nothing may block the event loop during measurement -----------------------
# Promoted from scratch/loop.py. The server (another thread) delivers the first token
# at 200 ms either way; only the client's event loop differs.

BLOCK_S = 0.6


async def _ttft_while(url: str, interference) -> float:
    async with make_client(_config(url)) as client:
        task = asyncio.create_task(
            stream_one(client, model="mock", prompt="hi", max_tokens=1, deadline_s=30)
        )
        await asyncio.sleep(0.05)  # let the request go out
        await interference()
        r = await task
    assert_result_invariants(r)
    return r.ttft_ms


async def test_blocking_the_loop_inflates_measured_ttft(mock_endpoint):
    async def blocking():
        time.sleep(BLOCK_S)  # noqa: ASYNC251 -- the point of the test

    async def cooperative():
        await asyncio.sleep(BLOCK_S)

    with mock_endpoint(MockBehaviour(first_chunk_delay=0.2)) as (_, url):
        honest = await _ttft_while(url, cooperative)
        distorted = await _ttft_while(url, blocking)

    assert abs(honest - 200) < TOL_MS
    # The chunk sat in the socket buffer until the loop woke: TTFT reads ~650 ms, a
    # 3x error that no server-side change caused.
    assert distorted > (0.05 + BLOCK_S) * 1000 - TOL_MS / 2
