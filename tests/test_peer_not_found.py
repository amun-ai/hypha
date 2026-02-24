"""Test immediate error for RPC calls to disconnected peers.

Regression test for https://github.com/amun-ai/hypha/issues/868
When a client disconnects, callers holding references to its services
should get an immediate error rather than waiting for the full RPC timeout.
"""

import asyncio
import time

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL_SQLITE, SIO_PORT_SQLITE

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_immediate_error_for_disconnected_peer(fastapi_server_sqlite):
    """RPC calls to disconnected peers should fail immediately, not timeout."""
    server_url = f"http://127.0.0.1:{SIO_PORT_SQLITE}"

    api_a = await connect_to_server(
        {"server_url": server_url, "method_timeout": 30}
    )
    api_b = await connect_to_server(
        {"server_url": server_url, "method_timeout": 30}
    )

    # Register a service on client B
    await api_b.register_service(
        {
            "id": "test-svc",
            "config": {"visibility": "public", "require_context": False},
            "hello": lambda name: f"Hello {name}",
        }
    )

    svc = await api_a.get_service(f"{api_b.config.workspace}/test-svc")
    assert await svc.hello("world") == "Hello world"

    # Disconnect client B
    await api_b.disconnect()
    await asyncio.sleep(2)  # Allow server cleanup

    # Call should fail fast (< 10s), not wait for 30s timeout
    start = time.time()
    with pytest.raises(Exception) as exc_info:
        await svc.hello("world")
    elapsed = time.time() - start
    assert elapsed < 10, f"Call took {elapsed:.1f}s, expected < 10s (immediate error)"
    error_str = str(exc_info.value).lower()
    assert "not connected" in error_str or "peer" in error_str, (
        f"Unexpected error: {exc_info.value}"
    )

    await api_a.disconnect()


async def test_immediate_error_multiple_calls(fastapi_server_sqlite):
    """Multiple rapid calls to a disconnected peer should all fail fast."""
    server_url = f"http://127.0.0.1:{SIO_PORT_SQLITE}"

    api_a = await connect_to_server(
        {"server_url": server_url, "method_timeout": 30}
    )
    api_b = await connect_to_server(
        {"server_url": server_url, "method_timeout": 30}
    )

    await api_b.register_service(
        {
            "id": "multi-svc",
            "config": {"visibility": "public", "require_context": False},
            "greet": lambda: "hi",
        }
    )

    svc = await api_a.get_service(f"{api_b.config.workspace}/multi-svc")
    assert await svc.greet() == "hi"

    # Disconnect client B
    await api_b.disconnect()
    await asyncio.sleep(2)

    # Fire many concurrent calls — all should fail fast
    start = time.time()
    results = await asyncio.gather(
        *[svc.greet() for _ in range(10)],
        return_exceptions=True,
    )
    elapsed = time.time() - start

    # All calls should have failed
    assert all(isinstance(r, Exception) for r in results), (
        f"Expected all failures, got: {results}"
    )
    assert elapsed < 10, (
        f"10 concurrent calls took {elapsed:.1f}s, expected < 10s"
    )

    await api_a.disconnect()
