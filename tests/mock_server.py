"""A deterministic mock of the OpenAI streaming chat-completions endpoint.

This is how the test suite proves the harness measures the server rather than
its own behaviour. It emits chunks at precisely controlled intervals and records
how many requests were genuinely in flight at the same time.

It is a bare ASGI app (no framework) served by a real uvicorn server on a real
socket, because in-process transports buffer the response and would hide
streaming bugs.

The test suite runs it as a separate process (``run_mock_process``), not a thread:
sharing a process means sharing the GIL with the client under test, which skewed
client timings under load. Tests configure it and read its counters over a small
control API at ``/_mock/*``.

Run standalone for manual poking:

    uv run python tests/mock_server.py --port 8765 --first-delay 0.5 --chunk-delay 0.05
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import httpx
import uvicorn


@dataclass
class MockBehaviour:
    first_chunk_delay: float = 0.0
    """Seconds between receiving the request and the first content chunk (prefill)."""
    chunk_delay: float = 0.0
    """Seconds between consecutive content chunks (decode)."""
    output_tokens: int | None = None
    """Content chunks to emit. ``None`` means honour the request's ``max_tokens``."""
    send_usage: bool = True
    """Send a final ``usage`` chunk when the client asks via ``stream_options``."""
    status_code: int = 200
    """Non-200 makes every request fail with this status."""
    cached_prompt_tokens: int | None = None
    """When set, usage reports this many prompt tokens as served from cache."""
    fail_every: int = 0
    reply: str | None = None
    """Non-streaming replies return this text (default: 'tok0 tok1 ...' up to max_tokens)."""
    rate_limit_first: int = 0
    """The first N requests get HTTP 429 with 'Retry-After: 0', then normal service."""
    """When > 0, every k-th request received fails with HTTP 503 (partial failures)."""


@dataclass
class MockStats:
    requests_received: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    last_body: dict | None = None


class MockServer:
    """ASGI app speaking ``POST /v1/chat/completions`` with ``stream: true``."""

    def __init__(self, behaviour: MockBehaviour | None = None) -> None:
        self.behaviour = behaviour or MockBehaviour()
        self.stats = MockStats()

    def reset_stats(self) -> None:
        self.stats = MockStats()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        if scope["type"] != "http":
            return

        if scope["path"].startswith("/_mock/"):
            await self._control(scope, receive, send)
            return

        if scope["method"] != "POST" or not scope["path"].endswith("/chat/completions"):
            await _send_json(send, 404, {"error": {"message": "not found"}})
            return

        body = await _read_body(receive)
        self.stats.requests_received += 1
        self.stats.in_flight += 1
        self.stats.max_in_flight = max(self.stats.max_in_flight, self.stats.in_flight)
        handler = asyncio.ensure_future(self._handle(json.loads(body), send))
        disconnect = asyncio.ensure_future(_wait_for_disconnect(receive))
        try:
            # A client that gives up (e.g. on a timeout) must not leave the mock streaming
            # into a dead socket: that skews in-flight counts and stalls server shutdown.
            await asyncio.wait({handler, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (handler, disconnect):
                task.cancel()
            await asyncio.gather(handler, disconnect, return_exceptions=True)
            self.stats.in_flight -= 1
        if not handler.cancelled() and handler.exception() is not None:
            raise handler.exception()  # a bug in the mock must never pass silently

    async def _control(self, scope, receive, send) -> None:
        """Out-of-band control for tests: never counted as a measured request."""
        path, method = scope["path"], scope["method"]
        if path == "/_mock/config" and method == "POST":
            self.behaviour = MockBehaviour(**json.loads(await _read_body(receive)))
            self.reset_stats()
            await _send_json(send, 200, {})
        elif path == "/_mock/stats" and method == "GET":
            await _send_json(send, 200, asdict(self.stats))
        else:
            await _send_json(send, 404, {"error": {"message": "unknown control route"}})

    async def _complete(self, request: dict, send) -> None:
        """A non-streaming chat completion (quality runs)."""
        b = self.behaviour
        await asyncio.sleep(b.first_chunk_delay)
        n = request.get("max_tokens", 16)
        text = b.reply if b.reply is not None else " ".join(f"tok{i}" for i in range(n))
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in request["messages"])
        await _send_json(send, 200, {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": request.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop" if b.reply is not None else "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split()),
                      "total_tokens": prompt_tokens + len(text.split())},
        })  # fmt: skip

    async def _handle(self, request: dict, send) -> None:
        b = self.behaviour

        self.stats.last_body = request
        if self.stats.requests_received <= b.rate_limit_first:
            await _send_json(send, 429, {"error": {"message": "slow down"}},
                             headers=[(b"retry-after", b"0")])  # fmt: skip
            return
        if b.fail_every and self.stats.requests_received % b.fail_every == 0:
            await _send_json(send, 503, {"error": {"message": "mock partial failure"}})
            return
        if b.status_code != 200:
            await _send_json(send, b.status_code, {"error": {"message": "mock failure"}})
            return
        if not request.get("stream"):
            await self._complete(request, send)
            return

        n_tokens = b.output_tokens if b.output_tokens is not None else request.get("max_tokens", 16)
        want_usage = bool((request.get("stream_options") or {}).get("include_usage"))
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in request["messages"])
        created = int(time.time())

        def chunk(delta: dict, finish_reason: str | None = None) -> dict:
            return {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.get("model", "mock"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )

        # Real servers send a role-only chunk with empty content straight away.
        # TTFT must not be fooled by it: it carries no content.
        await _send_event(send, chunk({"role": "assistant", "content": ""}))

        for i in range(n_tokens):
            await asyncio.sleep(b.first_chunk_delay if i == 0 else b.chunk_delay)
            await _send_event(send, chunk({"content": f"tok{i} "}))

        await _send_event(send, chunk({}, finish_reason="length"))

        if want_usage and b.send_usage:
            usage_chunk = chunk({})
            usage_chunk["choices"] = []
            usage_chunk["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_tokens,
                "total_tokens": prompt_tokens + n_tokens,
            }
            if b.cached_prompt_tokens is not None:
                usage_chunk["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": b.cached_prompt_tokens
                }
            await _send_event(send, usage_chunk)

        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return body


async def _wait_for_disconnect(receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def _send_event(send, payload: dict) -> None:
    data = f"data: {json.dumps(payload)}\n\n".encode()
    await send({"type": "http.response.body", "body": data, "more_body": True})


async def _send_json(send, status: int, payload: dict, headers: list | None = None) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *(headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


@contextlib.contextmanager
def run_mock_server(
    behaviour: MockBehaviour | None = None, port: int = 0
) -> Iterator[tuple[MockServer, str]]:
    """Serve a MockServer on a real socket in a background thread.

    Yields ``(server, base_url)`` where ``base_url`` ends in ``/v1``.
    """
    app = MockServer(behaviour)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    actual_port = sock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on", backlog=2048)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("mock server failed to start")
        time.sleep(0.01)

    try:
        yield app, f"http://127.0.0.1:{actual_port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


class MockControl:
    """Drives a mock running in another process through its ``/_mock/*`` routes."""

    def __init__(self, url: str) -> None:
        self._http = httpx.Client(base_url=url.removesuffix("/v1"), timeout=10)

    def configure(self, behaviour: MockBehaviour) -> None:
        """Set behaviour for subsequent requests and zero the counters."""
        self._http.post("/_mock/config", json=asdict(behaviour)).raise_for_status()

    @property
    def stats(self) -> SimpleNamespace:
        """A fresh snapshot of the server's counters."""
        response = self._http.get("/_mock/stats")
        response.raise_for_status()
        return SimpleNamespace(**response.json())

    def close(self) -> None:
        self._http.close()


@contextlib.contextmanager
def run_mock_process() -> Iterator[tuple[MockControl, str]]:
    """Run the mock in its own Python process on an ephemeral port.

    Yields ``(control, base_url)`` where ``base_url`` ends in ``/v1``.
    """
    proc = subprocess.Popen(
        [sys.executable, __file__, "--port", "0", "--first-delay", "0", "--chunk-delay", "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        if not line.startswith("READY "):
            raise RuntimeError(f"mock process failed to start: {line!r}")
        url = line.split()[1]
        control = MockControl(url)
        try:
            yield control, url
        finally:
            control.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the frugal-llm mock endpoint.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--first-delay", type=float, default=0.5)
    parser.add_argument("--chunk-delay", type=float, default=0.05)
    args = parser.parse_args()

    behaviour = MockBehaviour(first_chunk_delay=args.first_delay, chunk_delay=args.chunk_delay)
    with run_mock_server(behaviour, port=args.port) as (_, url):
        print(f"READY {url}", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            threading.Event().wait()
