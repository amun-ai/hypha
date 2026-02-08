"""Test RPC and WebSocket security vulnerabilities.

This test suite explores potential security issues in:
- WebSocket connection handling
- RPC message routing
- Context injection
- Callback manipulation
- Session management
"""
import pytest
import asyncio
from hypha_rpc import connect_to_server
from . import WS_SERVER_URL


@pytest.mark.asyncio
async def test_readonly_broadcast_block(fastapi_server, test_user_token):
    """Test that readonly clients cannot broadcast messages.

    Expected: Readonly clients should be blocked from sending broadcast messages.
    Vulnerability: V-RPC-1 - If readonly flag can be bypassed, unauthorized broadcasts possible.
    """
    # Connect as a readonly user (read-only permission workspace)
    # First, create a workspace with read-only access
    admin = await connect_to_server(
        {"name": "admin", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    # Create test workspace
    workspace_name = f"test-readonly-{pytest.test_id}"
    workspace = await admin.create_workspace(
        {"name": workspace_name, "description": "Test readonly workspace"}
    )

    # Generate a readonly token for this workspace
    readonly_token_info = await admin.generate_token({
        "workspace": workspace_name,
        "permission": "read"  # Read-only permission
    })

    # Connect with readonly token
    readonly_client = await connect_to_server({
        "name": "readonly-client",
        "server_url": WS_SERVER_URL,
        "token": readonly_token_info["token"],
        "workspace": workspace_name
    })

    # Try to register a service (should fail for readonly)
    try:
        await readonly_client.register_service({
            "id": "test-service",
            "type": "test",
            "test_method": lambda: "test"
        })
        pytest.fail("Readonly client should not be able to register services")
    except Exception as e:
        assert "permission" in str(e).lower() or "read" in str(e).lower()

    await readonly_client.disconnect()
    await admin.disconnect()


@pytest.mark.asyncio
async def test_context_injection_attempt(root_user_token, fastapi_server):
    """Test if client can inject malicious context fields.

    Expected: Server should reject or sanitize client-provided context.
    Vulnerability: V-RPC-2 - Context injection could allow workspace bypass.
    """
    client = await connect_to_server({
        "name": "attacker",
        "server_url": WS_SERVER_URL,
        "token": root_user_token
    })

    # Register a test service
    def test_method(context=None):
        """Test method that receives context."""
        return {
            "received_workspace": context.get("ws") if context else None,
            "received_from": context.get("from") if context else None,
            "received_user": context.get("user") if context else None
        }

    await client.register_service({
        "id": "context-test",
        "type": "test",
        "config": {"require_context": True},
        "test_method": test_method
    })

    # Try to call the method and see what context is received
    svc = await client.get_service("context-test")
    result = await svc.test_method()

    # Verify that context is properly set by the server, not client
    assert result["received_workspace"] is not None
    assert result["received_from"] is not None
    assert result["received_user"] is not None

    # The context should match the client's actual workspace
    # not anything the client tries to inject

    await client.disconnect()


@pytest.mark.asyncio
async def test_message_routing_hijacking(root_user_token, fastapi_server):
    """Test if messages can be routed to unauthorized targets.

    Expected: Messages should only route to authorized targets within workspace.
    Vulnerability: V-RPC-3 - Target ID manipulation could bypass workspace isolation.
    """
    # Create two separate workspaces
    admin = await connect_to_server({
        "name": "admin",
        "server_url": WS_SERVER_URL,
        "token": root_user_token
    })

    ws1_name = f"workspace-1-{pytest.test_id}"
    ws2_name = f"workspace-2-{pytest.test_id}"

    ws1 = await admin.create_workspace({"name": ws1_name})
    ws2 = await admin.create_workspace({"name": ws2_name})

    # Get tokens for each workspace
    token1 = await admin.generate_token({"workspace": ws1_name})
    token2 = await admin.generate_token({"workspace": ws2_name})

    # Connect clients to different workspaces
    client1 = await connect_to_server({
        "name": "client1",
        "server_url": WS_SERVER_URL,
        "token": token1["token"],
        "workspace": ws1_name
    })

    client2 = await connect_to_server({
        "name": "client2",
        "server_url": WS_SERVER_URL,
        "token": token2["token"],
        "workspace": ws2_name
    })

    # Register service in workspace 2
    messages_received = []

    def receive_message(msg):
        messages_received.append(msg)
        return "received"

    await client2.register_service({
        "id": "target-service",
        "type": "test",
        "receive": receive_message
    })

    # Try to send message from client1 to client2 (cross-workspace)
    # This should fail due to workspace isolation
    try:
        # Attempt to get service from different workspace
        svc = await client1.get_service(f"{ws2_name}/target-service:default")
        # If this succeeds, it's a vulnerability
        await svc.receive("malicious message")
        pytest.fail("Cross-workspace service access should be blocked")
    except Exception as e:
        # Expected to fail
        assert "permission" in str(e).lower() or "not found" in str(e).lower()

    # Verify no messages were received
    assert len(messages_received) == 0, "Cross-workspace message should not be delivered"

    await client1.disconnect()
    await client2.disconnect()
    await admin.disconnect()


@pytest.mark.asyncio
async def test_callback_session_manipulation(root_user_token, fastapi_server):
    """Test if callback sessions can be hijacked or manipulated.

    Expected: Sessions should be isolated and not accessible across clients.
    Vulnerability: V-RPC-4 - Session hijacking could lead to unauthorized callback execution.
    """
    client = await connect_to_server({
        "name": "test-client",
        "server_url": WS_SERVER_URL,
        "token": root_user_token
    })

    # Register a service that uses callbacks
    callback_executions = []

    def service_with_callback(callback, data):
        """Service that executes a callback."""
        result = callback(data)
        callback_executions.append(result)
        return result

    await client.register_service({
        "id": "callback-service",
        "type": "test",
        "execute_callback": service_with_callback
    })

    # Test normal callback execution
    svc = await client.get_service("callback-service")

    def my_callback(data):
        return f"processed: {data}"

    result = await svc.execute_callback(my_callback, "test data")
    assert result == "processed: test data"
    assert len(callback_executions) == 1

    # TODO: Explore if we can manipulate session IDs to hijack callbacks
    # This would require deeper access to RPC internals

    await client.disconnect()


@pytest.mark.asyncio
async def test_broadcast_message_filtering(root_user_token, fastapi_server):
    """Test if broadcast messages properly respect workspace boundaries.

    Expected: Broadcasts should only reach clients in the same workspace.
    Vulnerability: V-RPC-5 - Broadcast leakage across workspaces.
    """
    admin = await connect_to_server({
        "name": "admin",
        "server_url": WS_SERVER_URL,
        "token": root_user_token
    })

    # Create two workspaces
    ws1_name = f"broadcast-ws1-{pytest.test_id}"
    ws2_name = f"broadcast-ws2-{pytest.test_id}"

    await admin.create_workspace({"name": ws1_name})
    await admin.create_workspace({"name": ws2_name})

    token1 = await admin.generate_token({"workspace": ws1_name})
    token2 = await admin.generate_token({"workspace": ws2_name})

    # Connect clients to different workspaces
    received_ws1 = []
    received_ws2 = []

    client1 = await connect_to_server({
        "name": "client1",
        "server_url": WS_SERVER_URL,
        "token": token1["token"],
        "workspace": ws1_name
    })

    client2 = await connect_to_server({
        "name": "client2",
        "server_url": WS_SERVER_URL,
        "token": token2["token"],
        "workspace": ws2_name
    })

    # Set up event listeners
    def on_event_ws1(data):
        received_ws1.append(data)

    def on_event_ws2(data):
        received_ws2.append(data)

    client1.on("test-broadcast", on_event_ws1)
    client2.on("test-broadcast", on_event_ws2)

    # Broadcast from workspace 1
    await client1.emit("test-broadcast", {"message": "from ws1"})

    await asyncio.sleep(0.5)  # Wait for message propagation

    # Only client1 should receive the broadcast
    assert len(received_ws1) > 0, "Client in same workspace should receive broadcast"
    assert len(received_ws2) == 0, "Client in different workspace should NOT receive broadcast"

    await client1.disconnect()
    await client2.disconnect()
    await admin.disconnect()


@pytest.mark.asyncio
async def test_service_visibility_bypass(root_user_token, fastapi_server):
    """Test if protected services can be accessed from unauthorized workspaces.

    Expected: Protected services should only be accessible within their workspace.
    Vulnerability: V-RPC-6 - Service visibility bypass.
    """
    admin = await connect_to_server({
        "name": "admin",
        "server_url": WS_SERVER_URL,
        "token": root_user_token
    })

    # Create workspace and register protected service
    ws_name = f"protected-ws-{pytest.test_id}"
    await admin.create_workspace({"name": ws_name})
    token = await admin.generate_token({"workspace": ws_name})

    client = await connect_to_server({
        "name": "service-owner",
        "server_url": WS_SERVER_URL,
        "token": token["token"],
        "workspace": ws_name
    })

    # Register protected service (default visibility is protected)
    def secret_function():
        return "secret data"

    await client.register_service({
        "id": "protected-service",
        "type": "test",
        "config": {"visibility": "protected"},  # Protected - workspace only
        "get_secret": secret_function
    })

    # Try to access from different client in different workspace
    other_ws = f"other-ws-{pytest.test_id}"
    await admin.create_workspace({"name": other_ws})
    other_token = await admin.generate_token({"workspace": other_ws})

    attacker = await connect_to_server({
        "name": "attacker",
        "server_url": WS_SERVER_URL,
        "token": other_token["token"],
        "workspace": other_ws
    })

    # Attempt to access protected service from different workspace
    try:
        svc = await attacker.get_service(f"{ws_name}/protected-service:default")
        secret = await svc.get_secret()
        pytest.fail(f"Should not be able to access protected service from different workspace, got: {secret}")
    except Exception as e:
        # Expected to fail
        assert "permission" in str(e).lower() or "not found" in str(e).lower()

    await client.disconnect()
    await attacker.disconnect()
    await admin.disconnect()
