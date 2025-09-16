"""Test for memory leak fix: HTTP requests should reuse connections with stable client_id."""
import asyncio
import pytest
import aiohttp
from hypha.core import RedisRPCConnection

@pytest.mark.asyncio
async def test_http_stable_client_id_no_leak(fastapi_server, test_user_token):
    """Test that HTTP requests with stable client_id don't leak connections.

    This test verifies the fix for a critical memory leak where HTTP requests
    would create new RedisRPCConnection objects on every request because they
    passed None as client_id, causing a new random ID to be generated each time.

    The fix uses stable client IDs for HTTP anonymous users (e.g., 'http-anon-public')
    to enable connection reuse across requests.
    """
    server_url = fastapi_server

    # Get initial connection count
    initial_total = len(RedisRPCConnection._connections)
    initial_http_anon = len([
        k for k in RedisRPCConnection._connections.keys()
        if 'http-anon' in k  # Look for stable HTTP connections
    ])

    print(f"Initial state: {initial_total} total, {initial_http_anon} http-anon connections")

    # Make multiple HTTP requests that should reuse connections
    async with aiohttp.ClientSession() as session:
        # Test various endpoints
        test_endpoints = [
            f"{server_url}/public/services",
            f"{server_url}/public/services/hypha-login",
            f"{server_url}/public",
        ]

        # Send multiple requests
        request_count = 0
        for endpoint in test_endpoints:
            for i in range(3):  # 3 requests per endpoint = 9 total
                request_count += 1
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        assert response.status in [200, 404, 503], f"Unexpected status: {response.status}"
                except aiohttp.ClientError:
                    pass  # Some endpoints might not exist, that's OK

                await asyncio.sleep(0.1)

    # Wait for any cleanup
    await asyncio.sleep(2)

    # Check connection count after requests
    final_total = len(RedisRPCConnection._connections)
    final_http_anon = len([
        k for k in RedisRPCConnection._connections.keys()
        if 'http-anon' in k
    ])

    # List the stable connections
    stable_connections = [
        k for k in RedisRPCConnection._connections.keys()
        if 'http-anon' in k
    ]

    print(f"Final state: {final_total} total, {final_http_anon} http-anon connections")
    print(f"Stable HTTP connections: {stable_connections}")

    connection_increase = final_total - initial_total
    http_anon_increase = final_http_anon - initial_http_anon

    # With stable client_id, we should have at most 2 connections
    # (one for 'http-anon-public' and one for 'http-anon-ws-anonymous')
    assert http_anon_increase <= 2, (
        f"Too many new HTTP connections created: {http_anon_increase}. "
        f"Should reuse stable connections like 'http-anon-public'"
    )

    # Total connections shouldn't grow much (allow small buffer for other activity)
    assert connection_increase <= 3, (
        f"Connection leak detected! Connections increased by {connection_increase} "
        f"after {request_count} requests. Should reuse stable connections."
    )

    # Verify stable connection names
    for conn_key in stable_connections:
        assert 'http-anon-' in conn_key, f"Unexpected connection format: {conn_key}"
        # Should be like 'public/http-anon-public' or 'ws-anonymous/http-anon-ws-anonymous'
        parts = conn_key.split('/')
        assert len(parts) == 2, f"Invalid connection key format: {conn_key}"
        workspace, client_id = parts
        assert client_id.startswith('http-anon-'), f"Client ID should start with 'http-anon-': {client_id}"

    print(f"✅ Test passed: {request_count} requests with only {http_anon_increase} new connections")