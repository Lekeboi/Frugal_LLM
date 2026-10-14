from __future__ import annotations

import httpx
import pytest
from mock_server import MockBehaviour, run_mock_server


@pytest.fixture
def mock_endpoint():
    """Factory: ``with mock_endpoint(MockBehaviour(...)) as (server, url): ...``"""

    def start(behaviour: MockBehaviour | None = None):
        return run_mock_server(behaviour)

    return start


@pytest.fixture(scope="session")
async def client():
    """One shared client: building an AsyncClient loads the CA bundle synchronously
    (~0.3 s on Windows), which would block the event loop if done per request."""
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=30, limits=limits) as c:
        yield c
