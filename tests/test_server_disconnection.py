"""Test the hypha server."""

import os
import subprocess
import sys
import asyncio

import pytest
import requests
from hypha_rpc import connect_to_server

from . import (
    SERVER_URL,
    SERVER_URL_REDIS_1,
    SERVER_URL_REDIS_2,
    SIO_PORT2,
    WS_SERVER_URL,
    find_item,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_server_reconnection(fastapi_server, root_user_token):
    """Test the server reconnection."""
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")

        api = await connect_to_server(
            {"server_url": WS_SERVER_URL, "client_id": "client1"}
        )
        assert api.config["client_id"] == "client1"
        await admin.kickout_client(
            api.config.workspace,
            api.config.client_id,
            1008,
            "simulated abnormal closure",
        )
        await asyncio.sleep(1)

        # It should reconnect
        assert await api.echo("hi") == "hi"
        await api.disconnect()

        api = await connect_to_server(
            {"server_url": WS_SERVER_URL, "client_id": "client1"}
        )
        assert api.config["client_id"] == "client1"
        await admin.kickout_client(
            api.config.workspace, api.config.client_id, 1000, "normal closure"
        )
        await asyncio.sleep(1)
        try:
            assert await api.echo("hi") == "hi"
        except Exception as e:
            assert "Connection is closed" in str(e)
            await api.disconnect()


async def test_cleanup_on_auth_failure(fastapi_server):
    """Test that clients are properly cleaned up when authentication fails."""
    
    # Try to connect with an invalid token - this should fail during auth
    with pytest.raises(Exception) as exc_info:
        async with connect_to_server(
            {
                "server_url": WS_SERVER_URL,
                "client_id": "test-auth-fail-client",
                "token": "invalid_token_xyz123",  # Invalid token
            }
        ) as api:
            pass
    
    # Should get authentication error
    assert "Failed to establish connection" in str(exc_info.value) or "Authentication" in str(exc_info.value)
    
    # Wait a moment for any potential cleanup
    await asyncio.sleep(2)
    
    # Now connect as a normal client and check there are no orphaned services
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "checker-client"}
    ) as api:
        # List all services - there should be no services from the failed client
        services = await api.list_services()
        
        # Look for any services from the failed authentication client
        failed_client_services = [
            s for s in services 
            if "test-auth-fail-client" in s.get("id", "")
        ]
        
        assert len(failed_client_services) == 0, (
            f"Found orphaned services from failed auth client: {failed_client_services}"
        )


async def test_cleanup_on_abrupt_disconnect(fastapi_server):
    """Test cleanup when client disconnects abruptly."""
    
    # Connect and register a service
    api = await connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "abrupt-disconnect-client"}
    )
    
    workspace = api.config["workspace"]
    
    # Register a test service
    await api.register_service({
        "id": "test-service",
        "name": "Test Service",
        "type": "test",
        "echo": lambda x: x
    })
    
    # Verify service exists
    services = await api.list_services()
    test_services = [s for s in services if "test-service" in s.get("id", "")]
    assert len(test_services) > 0, "Test service should be registered"
    
    # Force disconnect without proper cleanup (simulate crash)
    # Access internal connection and force close
    if hasattr(api, '_connection'):
        conn = api._connection
        if hasattr(conn, '_websocket') and conn._websocket:
            # Force close WebSocket with abnormal closure code
            await conn._websocket.close(code=1006)
    
    # Wait for server to detect disconnection
    await asyncio.sleep(3)
    
    # Connect as another client to check cleanup
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "checker-client-2"}
    ) as checker:
        # If we're in same workspace, check if the old service is cleaned up
        if checker.config["workspace"] == workspace:
            services = await checker.list_services()
            old_services = [
                s for s in services 
                if "abrupt-disconnect-client" in s.get("id", "")
            ]
            
            # The services should be cleaned up after disconnection
            assert len(old_services) == 0, (
                f"Found orphaned services after abrupt disconnect: {old_services}"
            )


async def test_normal_disconnect_cleanup(fastapi_server):
    """Test that normal disconnection properly cleans up services."""
    
    # Connect and register services
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "normal-disconnect-client"}
    ) as api:
        workspace = api.config["workspace"]
        
        # Register a service
        await api.register_service({
            "id": "normal-service",
            "name": "Normal Service",
            "type": "test",
            "echo": lambda x: x
        })
        
        # Service should exist
        services = await api.list_services()
        normal_services = [
            s for s in services 
            if "normal-service" in s.get("id", "")
        ]
        assert len(normal_services) > 0
        
    # After context exit (normal disconnect), services should be cleaned up
    await asyncio.sleep(2)
    
    # Verify cleanup
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "final-checker"}
    ) as checker:
        # Try to list services from the disconnected client
        # They should not exist anymore
        services = await checker.list_services()
        old_services = [
            s for s in services
            if "normal-disconnect-client" in s.get("id", "")
        ]
        
        assert len(old_services) == 0, (
            f"Services not cleaned up after normal disconnect: {old_services}"
        )
