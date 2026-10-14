"""Async load generator and streaming consumption."""

from __future__ import annotations

import asyncio
import functools
import itertools
import json
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

# HTTP/2 is deliberately off: multiplexing turns N concurrent requests into N streams
# on one connection, which changes what "concurrency" means. Recorded in the results.
HTTP_VERSION = "HTTP/1.1"

CONNECT_TIMEOUT_S = 10.0
WRITE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class BenchConfig:
    url: str
    model: str
    api_key: str | None
    concurrency: list[int]
    requests: int
    max_tokens: int
    prompt_tokens: int
    prompts_path: Path | None
    warmup: int
    repeats: int
    cost_per_hour: float | None
    currency: str
    timeout: float
    out: Path
    format: str
    label: str | None


@functools.cache
def ssl_context() -> ssl.SSLContext:
    """Load the CA bundle once per process.

    It takes ~0.3 s and runs synchronously, so it must never happen per client or
    inside a measurement window.
    """
    return httpx.create_ssl_context()


def request_timeout(timeout: float) -> httpx.Timeout:
    """Per-phase timeouts. ``read`` applies to each chunk, not the whole response."""
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=timeout, write=WRITE_TIMEOUT_S, pool=timeout
    )


def make_client(config: BenchConfig) -> httpx.AsyncClient:
    """Build one worker's client: a single keep-alive connection, never shared.

    Each concurrent worker owns its own client rather than all workers sharing one
    pool. httpcore matches waiting requests to connections with a scan that grows as
    requests x connections; at 150 concurrent requests on one shared pool that CPU work
    blocked the event loop for seconds and inflated every measured latency. With one
    connection per worker the scan is trivial, concurrency means exactly N open
    connections, and httpx's default 100-connection pool limit cannot serialise us.
    """
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    return httpx.AsyncClient(
        base_url=config.url,
        headers=headers,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        timeout=request_timeout(config.timeout),
        verify=ssl_context(),
        http2=False,
    )


class StreamError(Exception):
    """The server answered 200 but the stream itself was unusable."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


def classify_error(exc: BaseException) -> str:
    """Name the failure precisely.

    A pool timeout means the client ran out of capacity; a read timeout means the
    server stalled. Reporting both as "timeout" would hide the first inside the second.
    """
    match exc:
        case StreamError():
            return exc.kind
        case TimeoutError():
            # asyncio.timeout's total deadline; httpx's own timeouts are not TimeoutError.
            return "total_timeout"
        case httpx.PoolTimeout():
            return "pool_timeout"
        case httpx.ConnectTimeout():
            return "connect_timeout"
        case httpx.ReadTimeout():
            return "read_timeout"
        case httpx.WriteTimeout():
            return "write_timeout"
        case httpx.ConnectError():
            return "connect_error"
        case httpx.HTTPStatusError():
            return f"http_{exc.response.status_code}"
        case httpx.RemoteProtocolError() | httpx.ReadError():
            return "connection_lost"
        case _:
            return type(exc).__name__


TokenCountSource = Literal["usage", "chunks"]


@dataclass(frozen=True)
class RequestResult:
    """One request's measurements. Times are milliseconds unless suffixed ``_s``.

    ``sent_at_s`` and ``finished_at_s`` are raw ``time.perf_counter()`` readings, kept
    so the run can compute its measurement window; they mean nothing on their own.
    """

    sent_at_s: float
    finished_at_s: float
    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    token_count_source: TokenCountSource | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _sse_data(line: str) -> str | None:
    """Return the payload of an SSE ``data:`` line, or None for anything else."""
    if not line.startswith("data:"):
        return None
    return line[5:].removeprefix(" ")


def _delta_content(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


async def stream_one(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    deadline_s: float,
) -> RequestResult:
    """Send one streaming chat completion and time it.

    Every timestamp is taken from the monotonic ``perf_counter`` clock at the moment
    the line is handed to us, so the stream must be consumed incrementally and nothing
    may block the event loop while requests are in flight.

    - TTFT: send -> first chunk whose ``delta.content`` is non-empty. Role-only and
      empty chunks do not count.
    - ITL: mean gap between consecutive content-bearing chunks, excluding the gap
      before the first. ``None`` when there is only one content chunk.
    - E2E: send -> the last SSE event received (usually ``[DONE]``).
    - Output tokens: ``usage.completion_tokens`` if the server sends it, otherwise the
      number of content-bearing chunks. Never estimated from text length.

    ``deadline_s`` bounds the whole request (``total_timeout``); the client's own
    timeouts bound each phase and each chunk. Failures never raise: they come back as
    a result with ``error`` set, so one bad request cannot abort a run.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    content_times: list[float] = []
    usage: dict | None = None
    sent_at = time.perf_counter()
    last_event_at = sent_at

    try:
        async with asyncio.timeout(deadline_s):
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    payload = _sse_data(line)
                    if payload is None:
                        continue
                    now = time.perf_counter()
                    last_event_at = now
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        raise StreamError("malformed_stream", payload[:200]) from None
                    if "error" in event:
                        raise StreamError("stream_error", str(event["error"])[:200])
                    if _delta_content(event):
                        content_times.append(now)
                    if event.get("usage"):
                        usage = event["usage"]
    except (httpx.HTTPError, TimeoutError, StreamError) as exc:
        return RequestResult(
            sent_at_s=sent_at, finished_at_s=time.perf_counter(), error=classify_error(exc)
        )

    if not content_times:
        return RequestResult(sent_at_s=sent_at, finished_at_s=last_event_at, error="empty_response")

    gaps = [b - a for a, b in itertools.pairwise(content_times)]
    completion_tokens = (usage or {}).get("completion_tokens")
    return RequestResult(
        sent_at_s=sent_at,
        finished_at_s=last_event_at,
        ttft_ms=(content_times[0] - sent_at) * 1000,
        itl_ms=sum(gaps) / len(gaps) * 1000 if gaps else None,
        e2e_ms=(last_event_at - sent_at) * 1000,
        output_tokens=completion_tokens if completion_tokens is not None else len(content_times),
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        token_count_source="usage" if completion_tokens is not None else "chunks",
    )
