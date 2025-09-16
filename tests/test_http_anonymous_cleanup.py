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
    from hypha.core.store import RedisStore
    import httpx

    # Get the store from the server by making a request to get the app instance
    # Note: This test focuses on the context manager behavior, not the server itself
    # We'll create our own store instance for testing
    from hypha.core import connect_to_event_bus
    event_bus = await connect_to_event_bus(None)
    store = RedisStore(event_bus, "test-manager-id")

    # Get initial connection count
    initial_count = len(RedisRPCConnection._connections)

    # Test the context manager directly
    from hypha.core import UserInfo
    test_user = UserInfo(
        id="test-cleanup-user",
        is_anonymous=True,
        email=None,
        parent=None,
        roles=["anonymous"],
        scope={"workspaces": {"ws-anonymous": "r"}},
        expires_at=None,
    )

    # Use the context manager multiple times
    for i in range(5):
        async with store.get_workspace_interface(
            test_user,
            "ws-anonymous",
            client_id=f"test-client-{i}"
        ) as workspace:
            # Simulate some work with the workspace
            assert workspace is not None
            # The connection should be active here
            conn_key = f"ws-anonymous/test-client-{i}"
            assert conn_key in RedisRPCConnection._connections

        # After exiting context, connection should be cleaned up
        await asyncio.sleep(0.1)
        # The connection should be disconnected
        conn_key = f"ws-anonymous/test-client-{i}"
        assert conn_key not in RedisRPCConnection._connections, (
            f"Connection {conn_key} was not cleaned up after context exit"
        )

    # Final check - no connections should have leaked
    final_count = len(RedisRPCConnection._connections)
    assert final_count <= initial_count + 1, (
        f"Context manager leaked connections! Initial: {initial_count}, Final: {final_count}"
    )