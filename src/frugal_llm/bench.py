"""Async load generator and streaming consumption."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import json
import ssl
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from frugal_llm.metrics import percentiles

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


# --- step 4: concurrency, warm-up, repeats -------------------------------------------

# Failures that mean "nothing is listening", as opposed to a server that answered badly.
UNREACHABLE = frozenset({"connect_error", "connect_timeout"})


class BenchAborted(Exception):
    """The endpoint cannot be benchmarked; partial data must not be reported."""


@dataclass(frozen=True)
class RunResult:
    """One ``(concurrency, repeat)`` run. Never merged with another run until report time."""

    concurrency: int
    repeat: int
    window_seconds: float
    requests: list[RequestResult]
    client_lag_p99_ms: float | None = None
    client_lag_max_ms: float | None = None


class PromptSource(Protocol):
    def get(self, index: int) -> str: ...


class LoopLagMonitor:
    """Measures how late the client's own event loop wakes up during a run.

    Every timestamp this tool takes is only as good as the loop's responsiveness: a
    loop busy for 200 ms adds up to 200 ms to whatever TTFT it is timing. This samples
    the lateness of a short sleep so the report can flag runs where the client, not the
    server, was the bottleneck. (Baseline lateness is a few ms, ~15 ms on Windows.)
    """

    INTERVAL_S = 0.01

    def __init__(self) -> None:
        self.samples_ms: list[float] = []
        self._task: asyncio.Task | None = None

    async def _sample(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.INTERVAL_S)
            self.samples_ms.append((time.perf_counter() - t0 - self.INTERVAL_S) * 1000)

    async def __aenter__(self) -> LoopLagMonitor:
        self._task = asyncio.create_task(self._sample())
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


async def _drive(
    clients: list[httpx.AsyncClient],
    total: int,
    send: Callable[[httpx.AsyncClient], Awaitable[RequestResult]],
) -> list[RequestResult]:
    """Send ``total`` requests with one worker per client, each working back to back.

    Closed loop: exactly ``len(clients)`` requests are in flight until the work runs
    out. Every worker claims its first request before any can finish, so when
    ``total >= len(clients)`` every client (and its connection) is used.
    """
    results: list[RequestResult] = []
    claimed = itertools.count()

    async def worker(client: httpx.AsyncClient) -> None:
        while next(claimed) < total:
            results.append(await send(client))

    async with asyncio.TaskGroup() as tg:
        for client in clients:
            tg.create_task(worker(client))
    return results


def _abort_if_unusable(results: list[RequestResult], phase: str) -> None:
    if not results or any(r.ok for r in results):
        return
    kinds = sorted({r.error for r in results})
    if set(kinds) <= UNREACHABLE:
        raise BenchAborted(f"endpoint unreachable during {phase} ({', '.join(kinds)})")
    raise BenchAborted(f"every {phase} request failed ({', '.join(kinds)})")


async def run_level(
    config: BenchConfig,
    concurrency: int,
    prompts: PromptSource,
    prompt_ids: Iterator[int],
    *,
    on_request: Callable[[RequestResult], None] | None = None,
    on_run: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Warm up once, then run every repeat at one concurrency level.

    Warm-up sends ``max(warmup, concurrency)`` requests at full concurrency, so every
    worker's connection is open (TCP/TLS paid) and the server is warm before the first
    measured request. Its results are discarded. Repeats reuse the warm connections.
    """

    async def send(client: httpx.AsyncClient) -> RequestResult:
        return await stream_one(
            client,
            model=config.model,
            prompt=prompts.get(next(prompt_ids)),
            max_tokens=config.max_tokens,
            deadline_s=config.timeout,
        )

    async def send_measured(client: httpx.AsyncClient) -> RequestResult:
        result = await send(client)
        if on_request:
            on_request(result)
        return result

    runs: list[RunResult] = []
    async with contextlib.AsyncExitStack() as stack:
        clients = [await stack.enter_async_context(make_client(config)) for _ in range(concurrency)]
        warmup = await _drive(clients, max(config.warmup, concurrency), send)
        _abort_if_unusable(warmup, "warm-up")

        for repeat in range(config.repeats):
            async with LoopLagMonitor() as lag:
                results = await _drive(clients, config.requests, send_measured)
            _abort_if_unusable(results, f"c={concurrency} repeat {repeat}")
            # The window opens when the first measured request is sent and closes when
            # the last one completes; warm-up is outside it.
            window = max(r.finished_at_s for r in results) - min(r.sent_at_s for r in results)
            lag_p = percentiles(lag.samples_ms, (99,))
            run = RunResult(
                concurrency=concurrency,
                repeat=repeat,
                window_seconds=window,
                requests=results,
                client_lag_p99_ms=lag_p[99],
                client_lag_max_ms=max(lag.samples_ms, default=None),
            )
            runs.append(run)
            if on_run:
                on_run(run)
    return runs


async def preflight(config: BenchConfig, prompt: str) -> None:
    """One request before anything else, so a wrong URL, model or key fails in one
    round trip with a clear reason instead of after a whole warm-up of failures."""
    async with make_client(config) as client:
        result = await stream_one(
            client, model=config.model, prompt=prompt, max_tokens=1, deadline_s=config.timeout
        )
    if result.ok:
        return
    if result.error in UNREACHABLE:
        raise BenchAborted(f"endpoint unreachable ({result.error})")
    hint = {
        "http_401": "check --api-key / FRUGAL_LLM_API_KEY",
        "http_403": "check --api-key / FRUGAL_LLM_API_KEY",
        "http_404": "check that --url ends in /v1 and --model is a model the server knows",
    }.get(result.error, "")
    raise BenchAborted(f"preflight request failed ({result.error})" + (f": {hint}" if hint else ""))


async def run_sweep(
    config: BenchConfig,
    prompts: PromptSource,
    *,
    on_request: Callable[[RequestResult], None] | None = None,
    on_run: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Run every concurrency level in order, never overlapping.

    Prompt ids increase across the whole sweep so no two requests share a prompt,
    which keeps server-side prefix caches from flattering TTFT.
    """
    prompt_ids = itertools.count()
    await preflight(config, prompts.get(next(prompt_ids)))
    runs: list[RunResult] = []
    for concurrency in config.concurrency:
        runs += await run_level(
            config, concurrency, prompts, prompt_ids, on_request=on_request, on_run=on_run
        )
    return runs
