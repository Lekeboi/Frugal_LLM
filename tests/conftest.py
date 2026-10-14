from __future__ import annotations

import contextlib
import time

import httpx
import pytest
from mock_server import MockBehaviour, run_mock_process


@pytest.fixture(scope="session")
def _shared_mock():
    """One mock server process for the whole session.

    A separate process, not a thread: sharing the GIL with the client under test made
    client timings drift under load. Startup (~1 s) is paid once.
    """
    with run_mock_process() as (server, url):
        yield server, url


@pytest.fixture
def mock_endpoint(_shared_mock):
    """Factory: ``with mock_endpoint(MockBehaviour(...)) as (server, url): ...``

    Reconfigures the shared server and starts from clean stats. On exit it checks that
    nothing is still in flight, so no request from one test can leak into the next.
    """
    server, url = _shared_mock

    @contextlib.contextmanager
    def configure(behaviour: MockBehaviour | None = None):
        server.configure(behaviour or MockBehaviour())
        try:
            yield server, url
        finally:
            deadline = time.monotonic() + 5
            while server.stats.in_flight and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.stats.in_flight == 0, "mock requests still in flight after test"

    return configure


@pytest.fixture(scope="session")
async def client():
    """One shared client for tests of the mock itself (not of frugal-llm's client)."""
    async with httpx.AsyncClient(timeout=30) as c:
        yield c


def assert_result_invariants(result) -> None:
    """Properties every successful measurement must satisfy, whatever the server did."""
    if not result.ok:
        assert result.ttft_ms is None and result.e2e_ms is None
        return
    assert 0 < result.ttft_ms < result.e2e_ms
    assert result.itl_ms is None or result.itl_ms >= 0
    assert result.output_tokens >= 1
    assert result.token_count_source in ("usage", "chunks")
    assert result.sent_at_s < result.finished_at_s
