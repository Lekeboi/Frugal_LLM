from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from mock_server import MockBehaviour

from frugal_llm.bench import (
    BenchConfig,
    classify_error,
    make_client,
    pool_limits,
    request_timeout,
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


@pytest.mark.parametrize(("concurrency", "size"), [(1, 100), (50, 100), (64, 128), (150, 300)])
def test_pool_limits_scale_with_concurrency(concurrency, size):
    limits = pool_limits(concurrency)
    assert limits.max_connections == size
    assert limits.max_keepalive_connections == size


def test_timeouts_are_per_phase():
    t = request_timeout(42)
    assert (t.connect, t.read, t.write, t.pool) == (10.0, 42, 10.0, 42)


# Sweep past httpx's default pool of 100, so a change to pool sizing or library
# defaults cannot quietly reintroduce serialisation.
CONCURRENCY_PAST_DEFAULT_POOL = 150


async def test_client_reaches_concurrency_past_default_pool(mock_endpoint):
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (server, url):
        async with make_client(_config(url), CONCURRENCY_PAST_DEFAULT_POOL) as client:
            await asyncio.gather(
                *(_hold_open(client) for _ in range(CONCURRENCY_PAST_DEFAULT_POOL))
            )
        assert server.stats.max_in_flight == CONCURRENCY_PAST_DEFAULT_POOL


async def test_default_pool_would_serialise(mock_endpoint):
    """Negative control: proves the test above can actually detect pool queuing."""
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (server, url):
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            await asyncio.gather(
                *(_hold_open(client) for _ in range(CONCURRENCY_PAST_DEFAULT_POOL))
            )
        assert server.stats.max_in_flight == 100


async def test_server_stall_is_a_read_timeout(mock_endpoint):
    # The role chunk arrives at once; the next chunk is 1 s away. Read timeout is per chunk.
    with mock_endpoint(MockBehaviour(first_chunk_delay=1.0)) as (_, url):
        async with make_client(_config(url, timeout=0.3), 1) as client:
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
    # The first request's own read stall also exceeds 0.3 s; the second must queue at the pool.
    assert "pool_timeout" in errors


def test_http_status_errors_keep_their_code():
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    response = httpx.Response(503, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    assert classify_error(exc) == "http_503"


async def test_api_key_sent_as_bearer_token():
    async with make_client(_config("http://x/v1", api_key="sk-test"), 1) as client:
        assert client.headers["Authorization"] == "Bearer sk-test"
