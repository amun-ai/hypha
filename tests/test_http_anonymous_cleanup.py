"""Test that HTTP anonymous connections are properly cleaned up to prevent memory leaks."""
import asyncio
import pytest
import aiohttp
from hypha.core import RedisRPCConnection
from hypha.server import create_application


@pytest.mark.asyncio
async def test_http_anonymous_connections_cleanup(fastapi_server, test_user_token):
    """Test that HTTP requests with anonymous users don't leak RPC connections.

    This test verifies the fix for a memory leak where HTTP requests without
    authentication would create RedisRPCConnection objects via get_workspace_interface()
    that were not properly cleaned up after the request completed.
    """
    server_url = fastapi_server
    
    # Get initial connection count
    initial_total = len(RedisRPCConnection._connections)
    initial_ws_anon = len([
        k for k in RedisRPCConnection._connections.keys()
        if k.startswith('ws-anonymous/')
    ])

    # Make multiple HTTP requests without authentication to trigger anonymous connections
    async with aiohttp.ClientSession() as session:
        # Test various endpoints that use get_workspace_interface
        test_endpoints = [
            f"{server_url}/public/services",
            f"{server_url}/public/services/hypha-login",
            f"{server_url}/public",
        ]

        for endpoint in test_endpoints:
            for i in range(3):  # Multiple requests per endpoint
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        assert response.status in [200, 404, 503], f"Unexpected status: {response.status}"
                except aiohttp.ClientError:
                    # Some endpoints might not exist, that's OK for this test
                    pass

                # Small delay between requests
                await asyncio.sleep(0.1)

    # Wait for any async cleanup to complete
    await asyncio.sleep(2)

    # Force garbage collection to ensure cleanup
    import gc
    gc.collect()
    await asyncio.sleep(1)

    # Check connection count after requests
    final_total = len(RedisRPCConnection._connections)
    final_ws_anon = len([
        k for k in RedisRPCConnection._connections.keys()
        if k.startswith('ws-anonymous/')
    ])

    # Count HTTP anonymous connections (those with user_id 'anonymouz-http')
    http_anon_count = 0
    for conn_key, conn in RedisRPCConnection._connections.items():
        if hasattr(conn, '_user_info') and hasattr(conn._user_info, 'id'):
            if conn._user_info.id == 'anonymouz-http':
                http_anon_count += 1

    # Allow for some WebSocket connections, but HTTP anonymous should be cleaned up
    connection_increase = final_total - initial_total
    ws_anon_increase = final_ws_anon - initial_ws_anon

    # The fix should ensure no accumulation of HTTP anonymous connections
    assert http_anon_count <= 2, (
        f"Too many HTTP anonymous connections remaining: {http_anon_count}. "
        f"These should be cleaned up after requests complete."
    )

    # Allow small increase for active connections, but not the 9+ we sent
    assert connection_increase <= 3, (
        f"Connection leak detected! Connections increased by {connection_increase}. "
        f"Initial: {initial_total}, Final: {final_total}"
    )

    assert ws_anon_increase <= 3, (
        f"ws-anonymous connections leaked! Increased by {ws_anon_increase}. "
        f"Initial: {initial_ws_anon}, Final: {final_ws_anon}"
    )


@pytest.mark.asyncio
async def test_workspace_context_manager_cleanup(fastapi_server, test_user_token):
    """Test that WorkspaceInterfaceContextManager properly cleans up RPC connections."""
    # This test requires a running server with actual workspace setup
    # Skip if fastapi_server is not available (as in some CI environments)
    if not fastapi_server:
        pytest.skip("fastapi_server not available - test requires full server setup")

    from hypha_rpc import connect_to_server

    # Connect to the server and use its workspace system
    async with connect_to_server(
        {"server_url": f"{fastapi_server}/ws", "token": test_user_token}
    ) as api:
        # Get initial connection count
        initial_count = len(RedisRPCConnection._connections)

        # Test multiple connections that should be cleaned up
        for i in range(3):
            # Create a new connection for testing
            async with connect_to_server(
                {
                    "server_url": f"{fastapi_server}/ws",
                    "token": test_user_token,
                    "client_id": f"test-cleanup-{i}"
                }
            ) as test_api:
                # Connection should exist while in context
                assert test_api is not None
                # Do some work
                echo_result = await test_api.echo("test")
                assert echo_result == "test"

            # After context exit, give time for cleanup
            await asyncio.sleep(0.5)

        # Check that connections were cleaned up
        final_count = len(RedisRPCConnection._connections)

        # Should have roughly same number of connections (allow for some server activity)
        connection_increase = final_count - initial_count
        assert connection_increase <= 2, (
            f"Too many connections leaked! Increased by {connection_increase}. "
            f"Initial: {initial_count}, Final: {final_count}"
        )