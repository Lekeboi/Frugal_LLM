"""Async load generator and streaming consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

# HTTP/2 is deliberately off: multiplexing turns N concurrent requests into N streams
# on one connection, which changes what "concurrency" means. Recorded in the results.
HTTP_VERSION = "HTTP/1.1"

CONNECT_TIMEOUT_S = 10.0
WRITE_TIMEOUT_S = 10.0
MIN_POOL_SIZE = 100


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


def pool_limits(concurrency: int) -> httpx.Limits:
    """Size the pool from the concurrency level.

    httpx defaults to 100 connections; above that, requests queue at the pool and the
    load generator silently serialises. Keep-alive matches the pool so connections
    opened during warm-up survive into the measured window.
    """
    size = max(concurrency * 2, MIN_POOL_SIZE)
    return httpx.Limits(max_connections=size, max_keepalive_connections=size)


def request_timeout(timeout: float) -> httpx.Timeout:
    """Per-phase timeouts. ``read`` applies to each chunk, not the whole response."""
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=timeout, write=WRITE_TIMEOUT_S, pool=timeout
    )


def make_client(config: BenchConfig, concurrency: int) -> httpx.AsyncClient:
    """Build the one client shared by every request at a concurrency level.

    Construction loads the CA bundle synchronously (~0.3 s on Windows), so it must
    happen outside the measurement window and never per request.
    """
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    return httpx.AsyncClient(
        base_url=config.url,
        headers=headers,
        limits=pool_limits(concurrency),
        timeout=request_timeout(config.timeout),
        http2=False,
    )


def classify_error(exc: BaseException) -> str:
    """Name the failure precisely.

    A pool timeout means the client ran out of capacity; a read timeout means the
    server stalled. Reporting both as "timeout" would hide the first inside the second.
    """
    match exc:
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
