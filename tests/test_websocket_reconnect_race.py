"""Test for WebSocket reconnection race condition.

When a client reconnects (same client_id, using reconnection_token), there is
a race between the old connection's cleanup (handle_disconnection →
remove_client) and the new connection's setup. If the old cleanup runs after
the new connection re-registers its services, the services are deleted.

The deterministic reproduction uses overlapping connections:
1. Connect A with client_id X, register a service
2. Connect B with same client_id X + reconnection_token (skips check_client)
3. B re-registers the service — both connections are alive simultaneously
4. Close A's WebSocket — A's cleanup calls remove_client, deleting B's services
5. Verify B's service is still accessible (this fails without the fix)
"""

import asyncio
import pytest

from hypha_rpc import connect_to_server

from . import WS_SERVER_URL

pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_reconnect_overlapping_cleanup_clobbers_services(
    minio_server, fastapi_server
):
    """Old connection cleanup must NOT delete new connection's services.

    This deterministically reproduces the race by keeping both connections
    alive simultaneously, then closing the old one after the new one has
    registered services.
    """
    # --- 1. Connect A and register a service ---
    api_a = await connect_to_server(
        {
            "name": "reconnect-race-A",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )

    workspace = api_a.config["workspace"]
    client_id = api_a.config["client_id"]
    reconnection_token = api_a.config.get("reconnection_token")
    assert reconnection_token, "Server did not provide a reconnection_token"

    await api_a.register_service(
        {
            "id": "overlap-svc",
            "name": "Overlap Service",
            "type": "test",
            "config": {"visibility": "public", "require_context": False},
            "ping": lambda: "from-A",
        }
    )

    # --- 2. Connect B with same client_id + reconnection_token ---
    # With reconnection_token, the server skips check_client(), so both
    # connections A and B are alive simultaneously with the same client_id.
    # Disable auto-reconnect on A so it doesn't interfere.
    conn_a = api_a.rpc._connection
    conn_a._enable_reconnect = False

    api_b = await connect_to_server(
        {
            "name": "reconnect-race-B",
            "server_url": WS_SERVER_URL,
            "workspace": workspace,
            "client_id": client_id,
            "reconnection_token": reconnection_token,
            "method_timeout": 30,
        }
    )

    assert api_b.config["client_id"] == client_id

    # --- 3. B re-registers the service ---
    await api_b.register_service(
        {
            "id": "overlap-svc",
            "name": "Overlap Service",
            "type": "test",
            "config": {"visibility": "public", "require_context": False},
            "ping": lambda: "from-B",
        }
    )

    # --- 4. Close A's WebSocket — triggers old cleanup ---
    # This is the critical moment: A's handle_disconnection will call
    # remove_client(client_id), which deletes ALL services for client_id
    # from Redis — including B's freshly registered service.
    conn_a._closed = True
    await conn_a._websocket.close(1011, "simulated network drop for A")

    # Give the server time to process A's disconnection cleanup
    await asyncio.sleep(2.0)

    # --- 5. Verify B's service survived ---
    observer = await connect_to_server(
        {
            "name": "observer",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )
    try:
        svc = await observer.get_service(f"{workspace}/{client_id}:overlap-svc")
        result = await svc.ping()
        assert result == "from-B", (
            f"Expected 'from-B' but got '{result}'. "
            "Old connection cleanup may have partially corrupted the service."
        )
    except Exception as e:
        pytest.fail(
            f"Service lookup failed after old connection cleanup: {e}. "
            "This confirms the race condition: old handle_disconnection "
            "called remove_client() and deleted the new connection's services."
        )

    # Cleanup
    await api_b.disconnect()
    await observer.disconnect()


@pytest.mark.asyncio
async def test_reconnect_sequential_service_survives(minio_server, fastapi_server):
    """Services must survive a sequential disconnect-then-reconnect cycle.

    This tests the more common scenario: client disconnects (cleanup starts),
    then reconnects. With fast local Redis, cleanup usually finishes before
    the reconnection, but the fix should handle both orderings.
    """
    api = await connect_to_server(
        {
            "name": "reconnect-sequential",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )

    workspace = api.config["workspace"]
    client_id = api.config["client_id"]
    reconnection_token = api.config.get("reconnection_token")

    await api.register_service(
        {
            "id": "sequential-svc",
            "name": "Sequential Service",
            "type": "test",
            "config": {"visibility": "public", "require_context": False},
            "ping": lambda: "pong",
        }
    )

    # Force-close the WebSocket
    conn = api.rpc._connection
    conn._enable_reconnect = False
    conn._closed = True
    await conn._websocket.close(1011, "simulated network drop")

    # Small delay, then reconnect
    await asyncio.sleep(0.1)

    api2 = await connect_to_server(
        {
            "name": "reconnect-sequential",
            "server_url": WS_SERVER_URL,
            "workspace": workspace,
            "client_id": client_id,
            "reconnection_token": reconnection_token,
            "method_timeout": 30,
        }
    )

    await api2.register_service(
        {
            "id": "sequential-svc",
            "name": "Sequential Service",
            "type": "test",
            "config": {"visibility": "public", "require_context": False},
            "ping": lambda: "pong-v2",
        }
    )

    # Wait for any lingering cleanup
    await asyncio.sleep(2.0)

    observer = await connect_to_server(
        {
            "name": "observer-sequential",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )
    svc = await observer.get_service(f"{workspace}/{client_id}:sequential-svc")
    result = await svc.ping()
    assert result == "pong-v2", (
        f"Expected 'pong-v2' but got '{result}'. "
        "Old cleanup clobbered the new connection's service."
    )

    await api2.disconnect()
    await observer.disconnect()


@pytest.mark.asyncio
async def test_reconnect_preserves_client_id(minio_server, fastapi_server):
    """Basic test: reconnection with reconnection_token preserves client_id."""
    api = await connect_to_server(
        {
            "name": "reconnect-preserve-id",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )

    workspace = api.config["workspace"]
    client_id = api.config["client_id"]
    reconnection_token = api.config.get("reconnection_token")

    # Clean disconnect
    await api.disconnect()

    # Reconnect with same client_id
    api2 = await connect_to_server(
        {
            "name": "reconnect-preserve-id",
            "server_url": WS_SERVER_URL,
            "workspace": workspace,
            "client_id": client_id,
            "reconnection_token": reconnection_token,
            "method_timeout": 30,
        }
    )

    assert api2.config["workspace"] == workspace, "Workspace should be preserved"
    assert api2.config["client_id"] == client_id, "Client ID should be preserved"

    await api2.disconnect()
