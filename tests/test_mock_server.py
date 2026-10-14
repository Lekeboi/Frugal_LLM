"""Tests of the mock itself, using a bare httpx stream rather than frugal-llm's client.

If the mock's timing or in-flight counting were wrong, every harness test built
on it would be meaningless, so it gets checked independently first.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time

import httpx
from mock_server import MockBehaviour

# asyncio.sleep on Windows has ~15 ms timer granularity; keep tolerances honest but loose.
TOLERANCE_S = 0.1


async def _stream(
    client: httpx.AsyncClient, url: str, *, max_tokens: int = 4, include_usage: bool = True
):
    """Return [(seconds_since_send, parsed_event_or_DONE), ...]."""
    body = {
        "model": "mock",
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    events = []
    t0 = time.perf_counter()
    async with client.stream("POST", f"{url}/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            parsed = "[DONE]" if payload == "[DONE]" else json.loads(payload)
            events.append((time.perf_counter() - t0, parsed))
    return events


def _content(event) -> str:
    if event == "[DONE]" or not event["choices"]:
        return ""
    return event["choices"][0]["delta"].get("content") or ""


async def test_chunks_arrive_at_controlled_intervals(mock_endpoint, client):
    behaviour = MockBehaviour(first_chunk_delay=0.5, chunk_delay=0.05)
    with mock_endpoint(behaviour) as (_, url):
        events = await _stream(client, url, max_tokens=5)

    # Role-only chunk with empty content arrives immediately, before the prefill delay.
    first_t, first = events[0]
    assert _content(first) == "" and first["choices"][0]["delta"]["role"] == "assistant"
    assert first_t < 0.5 - TOLERANCE_S

    content_times = [t for t, e in events if _content(e)]
    assert len(content_times) == 5
    assert abs(content_times[0] - 0.5) < TOLERANCE_S
    gaps = [b - a for a, b in itertools.pairwise(content_times)]
    assert all(abs(g - 0.05) < TOLERANCE_S for g in gaps)
    assert abs(content_times[-1] - (0.5 + 4 * 0.05)) < TOLERANCE_S


async def test_usage_chunk_and_done_terminator(mock_endpoint, client):
    with mock_endpoint(MockBehaviour(output_tokens=7)) as (_, url):
        events = await _stream(client, url, max_tokens=100)
        no_usage = await _stream(client, url, include_usage=False)

    assert events[-1][1] == "[DONE]"
    usage_events = [e for _, e in events if e != "[DONE]" and e.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["usage"]["completion_tokens"] == 7
    assert not any(e != "[DONE]" and e.get("usage") for _, e in no_usage)


async def test_records_genuinely_overlapping_requests(mock_endpoint, client):
    with mock_endpoint(MockBehaviour(first_chunk_delay=0.3)) as (server, url):
        await asyncio.gather(*(_stream(client, url, max_tokens=1) for _ in range(16)))
        assert server.stats.requests_received == 16
        assert server.stats.max_in_flight == 16
        assert server.stats.in_flight == 0

        server.reset_stats()
        for _ in range(3):
            await _stream(client, url, max_tokens=1)
        assert server.stats.max_in_flight == 1


async def test_failure_status(mock_endpoint, client):
    with mock_endpoint(MockBehaviour(status_code=503)) as (_, url):
        resp = await client.post(
            f"{url}/chat/completions", json={"model": "m", "messages": [], "stream": True}
        )
    assert resp.status_code == 503
