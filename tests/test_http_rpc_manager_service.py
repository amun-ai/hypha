"""Test HTTP RPC manager service access."""

import asyncio
import pytest
from hypha_rpc import connect_to_server
from . import SERVER_URL


@pytest.mark.asyncio
async def test_http_manager_service_access(fastapi_server):
    """Test that HTTP clients can access the manager service using */manager-id:default."""
    server_url = SERVER_URL

    # Connect via HTTP transport
    server = await connect_to_server({
        "server_url": server_url,
        "method_timeout": 10,
    })

    try:
        # The server object IS the manager service wrapper
        assert server is not None
        assert hasattr(server, "get_service")
        assert hasattr(server, "list_services")
        assert hasattr(server, "register_service")

        # Verify we can call manager service methods
        services = await server.list_services()
        assert isinstance(services, list)

        # Register a test service
        await server.register_service({
            "id": "test-http-service",
            "config": {"visibility": "public"},
            "echo": lambda x: f"Echo: {x}",
        })

        # Verify the service was registered
        services_after = await server.list_services()
        service_ids = [s.get("id") for s in services_after]
        assert any("test-http-service" in sid for sid in service_ids)

        # Get and test the service
        svc = await server.get_service("test-http-service")
        result = await svc.echo("test")
        assert result == "Echo: test"

        print("✓ HTTP manager service access working correctly")

    finally:
        await server.disconnect()


@pytest.mark.asyncio
async def test_http_cross_workspace_manager_access(fastapi_server):
    """Test that HTTP clients in their own workspace can access the manager service."""
    server_url = SERVER_URL

    # Connect as anonymous user (gets assigned a user workspace automatically)
    server = await connect_to_server({
        "server_url": server_url,
        "method_timeout": 10,
    })

    try:
        # Check we're in a user workspace
        workspace = server.config.get("workspace")
        assert workspace and workspace.startswith("ws-user-"), \
            f"Expected user workspace, got {workspace}"

        # But we should still be able to access the manager service
        assert server is not None

        # Register service in our workspace
        await server.register_service({
            "id": "cross-workspace-test",
            "config": {"visibility": "public"},
            "test": lambda: "ok",
        })

        # List services via manager
        services = await server.list_services()
        assert isinstance(services, list)

        print(f"✓ Cross-workspace manager access working from {workspace}")

    finally:
        await server.disconnect()


@pytest.mark.asyncio
async def test_http_manager_service_with_reconnection(fastapi_server):
    """Test that reconnection tokens work with manager service access."""
    server_url = SERVER_URL

    # Initial connection
    server1 = await connect_to_server({
        "server_url": server_url,
        "client_id": "reconnect-test-client",
        "method_timeout": 10,
    })

    # Get reconnection token
    reconnection_token = server1.config.get("reconnection_token")
    workspace = server1.config.get("workspace")
    client_id = server1.config.get("client_id")

    assert reconnection_token, "Reconnection token not provided"

    # Disconnect
    await server1.disconnect()

    # Wait a bit
    await asyncio.sleep(1)

    # Reconnect with token
    server2 = await connect_to_server({
        "server_url": server_url,
        "workspace": workspace,
        "client_id": client_id,
        "reconnection_token": reconnection_token,
        "method_timeout": 10,
    })

    try:
        # Should still have manager service access
        services = await server2.list_services()
        assert isinstance(services, list)

        print("✓ Reconnection with manager service access working")

    finally:
        await server2.disconnect()
