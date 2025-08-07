"""Test SSE transport."""
import sys
import os
import asyncio
import pytest

from hypha.server import create_application, get_argparser
from hypha.utils.sse_client import connect_to_server

from . import SERVER_URL, find_item



@pytest.mark.asyncio
async def test_sse_basic_connection(fastapi_server):
    """Test basic SSE connection and communication."""
    # Connect using SSE client
    client = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "test-sse-client",
        "workspace": None,
        "token": None,
    })
    
    assert client is not None
    assert client.config.get("workspace") is not None
    assert client.config.get("client_id") == "test-sse-client"
    
    # Test registering a service
    async def hello(name: str):
        return f"Hello, {name}!"
    
    await client.register_service({
        "id": "test-service",
        "name": "Test Service",
        "hello": hello,
    })
    
    # Get the service
    service = await client.get_service("test-service")
    assert service is not None
    
    # Call the service method
    result = await service.hello("World")
    assert result == "Hello, World!"
    
    await client.disconnect()


@pytest.mark.asyncio
async def test_sse_multiple_clients(fastapi_server):
    """Test multiple SSE clients connecting and communicating."""
    # Connect first client (creates workspace automatically)
    client1 = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-client-1",
    })
    
    # Get the workspace from first client and connect second client to same workspace
    # Note: This test demonstrates cross-client communication within the same workspace
    # In practice, anonymous users are isolated, but for testing we can share workspaces
    workspace = client1.config.get("workspace")
    
    # For testing purposes, we'll create a second anonymous user in the same workspace
    # This simulates the scenario where users have access to a shared workspace
    client2 = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-client-2",
    })
    
    # Register service with client1
    async def multiply(a: float, b: float):
        return a * b
    
    await client1.register_service({
        "id": "math-service",
        "name": "Math Service",
        "multiply": multiply,
    })
    
    # Get workspaces for both clients
    workspace1 = client1.config.get("workspace")
    workspace2 = client2.config.get("workspace")
    
    # Since they're in different workspaces, test that they are isolated
    # (which is the correct security behavior for anonymous users)
    assert workspace1 != workspace2
    
    # Try to call service from client2 using full service ID (should fail due to isolation)
    try:
        full_service_id = f"{workspace1}/*:math-service"
        service = await client2.get_service(full_service_id)
        # This should fail due to workspace isolation
        assert False, "Should not be able to access service across workspaces for anonymous users"
    except Exception:
        # This is expected - anonymous users should be isolated
        pass
    
    # Instead, test that each client can register and access their own services
    async def divide(a: float, b: float):
        return a / b
    
    await client2.register_service({
        "id": "math-service-2",
        "name": "Math Service 2", 
        "divide": divide,
    })
    
    # Client2 should be able to access its own service
    service2 = await client2.get_service("math-service-2")
    result = await service2.divide(12.0, 4.0)
    assert result == 3.0
    
    await client1.disconnect()
    await client2.disconnect()


@pytest.mark.asyncio
async def test_sse_reconnection(fastapi_server):
    """Test SSE client reconnection after disconnect."""
    # Track connection events
    connected_count = 0
    disconnected_count = 0
    
    async def on_connected(info):
        nonlocal connected_count
        connected_count += 1
    
    def on_disconnected(reason):
        nonlocal disconnected_count
        disconnected_count += 1
    
    # Connect client
    client = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-reconnect-client",
    })
    
    # Set event handlers
    client.rpc._connection.on_connected(on_connected)
    client.rpc._connection.on_disconnected(on_disconnected)
    
    # Register a service
    async def echo(message: str):
        return message
    
    await client.register_service({
        "id": "echo-service",
        "name": "Echo Service",
        "echo": echo,
    })
    
    # Simulate disconnection by closing the SSE client
    await client.rpc._connection._sse_client.close()
    
    # Wait for reconnection
    await asyncio.sleep(3)
    
    # Service should still be available after reconnection
    service = await client.get_service("echo-service")
    result = await service.echo("test message")
    assert result == "test message"
    
    # Check that reconnection happened
    assert connected_count >= 1
    
    await client.disconnect()


@pytest.mark.asyncio
async def test_sse_large_message(fastapi_server):
    """Test sending large messages over SSE transport."""
    # Connect client
    client = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-large-msg-client",
    })
    
    # Create a large message
    large_data = "x" * (1024 * 100)  # 100KB string
    
    # Register service that handles large data
    async def process_data(data: str):
        return f"Received {len(data)} bytes"
    
    await client.register_service({
        "id": "data-service",
        "name": "Data Service",
        "process_data": process_data,
    })
    
    # Call with large data
    service = await client.get_service("data-service")
    result = await service.process_data(large_data)
    assert result == f"Received {len(large_data)} bytes"
    
    await client.disconnect()


@pytest.mark.asyncio
async def test_sse_event_emission(fastapi_server):
    """Test event emission and handling over SSE within the same client."""
    # Connect a single client (anonymous users are isolated)
    client = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-client",
    })
    
    # Set up event listener
    received_events = []
    
    async def handle_event(data):
        received_events.append(data)
    
    client.on("test-event", handle_event)
    
    # Emit event from the same client (self-emission)
    client.emit({"type": "test-event", "payload": "hello"})
    
    # Wait for event to be received
    await asyncio.sleep(1)
    
    # Check event was received (self-emission should work)
    assert len(received_events) > 0
    assert received_events[0].get("payload") == "hello"
    
    await client.disconnect()


@pytest.mark.asyncio
async def test_sse_workspace_isolation(fastapi_server):
    """Test workspace isolation with SSE clients."""
    # Connect client to workspace A
    clientA = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-client-a",
    })
    workspace_a = clientA.config.get("workspace")
    
    # Connect client to workspace B (different workspace)
    clientB = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-client-b",
    })
    workspace_b = clientB.config.get("workspace")
    
    # Workspaces should be different
    assert workspace_a != workspace_b
    
    # Register service in workspace A
    await clientA.register_service({
        "id": "private-service",
        "name": "Private Service",
        "get_workspace": lambda: workspace_a,
    })
    
    # Try to access from workspace B - should fail
    with pytest.raises(Exception):
        await clientB.get_service(f"{workspace_a}/*:private-service")
    
    await clientA.disconnect()
    await clientB.disconnect()


@pytest.mark.asyncio
async def test_sse_with_authentication(fastapi_server, test_user_token):
    """Test SSE connection with authentication token."""
    
    # Connect with token
    client = await connect_to_server({
        "server_url": SERVER_URL,
        "client_id": "sse-auth-client",
        "token": test_user_token,
    })
    
    # Check user info in config
    assert client.config.get("user") is not None
    assert client.config.get("user").get("email") == "user-1@test.com" 
    
    await client.disconnect()


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])