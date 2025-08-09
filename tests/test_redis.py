import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from hypha_rpc import connect_to_server
from prometheus_client import CollectorRegistry, REGISTRY

from hypha.core.store import RedisStore

from . import SIO_PORT, find_item

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry before each test to prevent metric name collisions."""
    # Clear all collectors from the default registry
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            # Collector might have already been unregistered
            pass
    yield
    # Clean up after test as well
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


@pytest_asyncio.fixture(name="redis_store")
async def redis_store():
    """Represent the redis store."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    yield store
    await store.teardown()


async def test_redis_store(redis_store):
    """Test the redis store."""
    # Test adding a workspace
    await redis_store.register_workspace(
        dict(
            name="test",
            description="test workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    wss = await redis_store.list_all_workspaces()
    assert find_item(wss, "name", "test")

    api = await redis_store.connect_to_workspace("test", client_id="test-app-99")
    # Need to wait for the client to be registered
    await asyncio.sleep(0.1)
    clients = await api.list_clients()
    assert find_item(clients, "id", "test/test-app-99")
    await api.log("hello")
    services = await api.list_services()
    assert len(services) == 1
    assert await api.generate_token()
    ws = await api.create_workspace(
        dict(
            name="test-2",
            description="test workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    assert ws["name"] == "test-2"
    ws2 = await api.get_workspace_info("test-2")
    assert ws2["name"] == "test-2"

    def echo(data):
        return data

    interface = {
        "id": "test-service",
        "name": "my service",
        "config": {},
        "setup": print,
        "echo": echo,
    }
    await api.register_service(interface)

    wm = await redis_store.connect_to_workspace("test", client_id="test-app-22")
    await asyncio.sleep(0.1)
    # Need to wait for the client to be registered
    clients = await wm.list_clients()
    assert find_item(clients, "id", "test/test-app-22")
    assert find_item(clients, "id", "test/test-app-99")
    rpc = wm.rpc
    services = await wm.list_services()
    service = await rpc.get_remote_service("test-app-99:test-service")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"


async def test_redis_event_subscription_lifecycle(redis_store):
    """Test that RedisRPCConnection properly manages targeted event subscriptions without memory leaks."""
    
    # First create the workspace
    await redis_store.register_workspace(
        dict(
            name="test-sub-lifecycle",
            description="test workspace for subscription lifecycle",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    
    # Get event bus reference
    event_bus = redis_store._event_bus
    
    # Ensure clean state by waiting for any pending operations
    await asyncio.sleep(0.1)
    
    # Get initial state (don't assume specific values, just track changes)
    initial_patterns = set(event_bus._subscribed_patterns)
    initial_clients = set(event_bus._local_clients)
    
    # Connect first client and track what gets added
    api1 = await redis_store.connect_to_workspace("test-sub-lifecycle", client_id="client-1")
    await asyncio.sleep(0.3)  # Give more time for async registration
    
    # Verify first client was added
    patterns_after_client1 = set(event_bus._subscribed_patterns)
    clients_after_client1 = set(event_bus._local_clients)
    
    new_patterns_1 = patterns_after_client1 - initial_patterns
    new_clients_1 = clients_after_client1 - initial_clients
    
    assert len(new_patterns_1) == 1, f"Expected 1 new pattern, got {len(new_patterns_1)}: {new_patterns_1}"
    assert len(new_clients_1) == 1, f"Expected 1 new client, got {len(new_clients_1)}: {new_clients_1}"
    assert "targeted:test-sub-lifecycle/client-1:*" in new_patterns_1
    assert "test-sub-lifecycle/client-1" in new_clients_1
    
    # Connect second client
    api2 = await redis_store.connect_to_workspace("test-sub-lifecycle", client_id="client-2") 
    await asyncio.sleep(0.3)  # Give more time for async registration
    
    # Verify second client was added
    patterns_after_client2 = set(event_bus._subscribed_patterns)
    clients_after_client2 = set(event_bus._local_clients)
    
    new_patterns_2 = patterns_after_client2 - patterns_after_client1  
    new_clients_2 = clients_after_client2 - clients_after_client1
    
    assert len(new_patterns_2) == 1, f"Expected 1 new pattern for client-2, got {len(new_patterns_2)}: {new_patterns_2}"
    assert len(new_clients_2) == 1, f"Expected 1 new client for client-2, got {len(new_clients_2)}: {new_clients_2}"
    assert "targeted:test-sub-lifecycle/client-2:*" in new_patterns_2
    assert "test-sub-lifecycle/client-2" in new_clients_2
    
    # Connect a manager client (should now be treated as local like any other client)
    api_manager = await redis_store.connect_to_workspace("test-sub-lifecycle", client_id="manager-123")
    await asyncio.sleep(0.3)  # Give more time for async registration
    
    # Verify manager client was added
    patterns_after_manager = set(event_bus._subscribed_patterns)
    clients_after_manager = set(event_bus._local_clients)
    
    new_patterns_mgr = patterns_after_manager - patterns_after_client2
    new_clients_mgr = clients_after_manager - clients_after_client2
    
    assert len(new_patterns_mgr) == 1, f"Expected 1 new pattern for manager, got {len(new_patterns_mgr)}: {new_patterns_mgr}"
    assert len(new_clients_mgr) == 1, f"Expected 1 new client for manager, got {len(new_clients_mgr)}: {new_clients_mgr}"
    assert "targeted:test-sub-lifecycle/manager-123:*" in new_patterns_mgr
    assert "test-sub-lifecycle/manager-123" in new_clients_mgr
    
    # Disconnect first client
    await api1.disconnect()
    await asyncio.sleep(0.2)  # Wait for cleanup
    
    # Verify first client was removed
    patterns_after_disc1 = set(event_bus._subscribed_patterns)
    clients_after_disc1 = set(event_bus._local_clients)
    
    assert "targeted:test-sub-lifecycle/client-1:*" not in patterns_after_disc1
    assert "test-sub-lifecycle/client-1" not in clients_after_disc1
    # But others should still be there
    assert "targeted:test-sub-lifecycle/client-2:*" in patterns_after_disc1
    assert "test-sub-lifecycle/client-2" in clients_after_disc1
    assert "targeted:test-sub-lifecycle/manager-123:*" in patterns_after_disc1
    assert "test-sub-lifecycle/manager-123" in clients_after_disc1
    
    # Disconnect second client
    await api2.disconnect()
    await asyncio.sleep(0.2)  # Wait for cleanup
    
    # Verify second client was removed
    patterns_after_disc2 = set(event_bus._subscribed_patterns)
    clients_after_disc2 = set(event_bus._local_clients)
    
    assert "targeted:test-sub-lifecycle/client-2:*" not in patterns_after_disc2
    assert "test-sub-lifecycle/client-2" not in clients_after_disc2
    # Manager should still be there
    assert "targeted:test-sub-lifecycle/manager-123:*" in patterns_after_disc2
    assert "test-sub-lifecycle/manager-123" in clients_after_disc2
    
    # Disconnect manager client
    await api_manager.disconnect()
    await asyncio.sleep(0.2)  # Wait for cleanup
    
    # Verify manager was removed and we're back to initial state
    final_patterns = set(event_bus._subscribed_patterns)
    final_clients = set(event_bus._local_clients)
    
    assert "targeted:test-sub-lifecycle/manager-123:*" not in final_patterns
    assert "test-sub-lifecycle/manager-123" not in final_clients
    
    # Verify we removed all our test-specific patterns and clients
    assert final_patterns == initial_patterns, (
        f"Pattern sets don't match after cleanup:\n"
        f"Initial: {initial_patterns}\n"
        f"Final: {final_patterns}\n"
        f"Added but not removed: {final_patterns - initial_patterns}\n"
        f"Removed but shouldn't be: {initial_patterns - final_patterns}"
    )
    assert final_clients == initial_clients, (
        f"Client sets don't match after cleanup:\n"
        f"Initial: {initial_clients}\n"
        f"Final: {final_clients}\n"
        f"Added but not removed: {final_clients - initial_clients}\n"
        f"Removed but shouldn't be: {initial_clients - final_clients}"
    )


async def test_redis_metrics_connections_and_patterns(redis_store):
    """Verify Prometheus-style counters reflect connect/disconnect and subscriptions without leaks."""

    # Create a dedicated workspace
    await redis_store.register_workspace(
        dict(
            name="metrics-ws",
            description="metrics workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )

    # Snapshot initial metrics
    initial = await redis_store.get_metrics()
    init_rpc_created = initial["rpc"]["rpc_connections"]["created_total"]
    init_rpc_closed = initial["rpc"]["rpc_connections"]["closed_total"]
    init_rpc_active = initial["rpc"]["rpc_connections"]["active"]
    init_patterns_active = initial["eventbus"]["eventbus"]["patterns"]["active"]
    init_local_clients = initial["eventbus"]["eventbus"]["local_clients"]

    # Connect three clients
    api1 = await redis_store.connect_to_workspace("metrics-ws", client_id="c1")
    api2 = await redis_store.connect_to_workspace("metrics-ws", client_id="c2")
    api3 = await redis_store.connect_to_workspace("metrics-ws", client_id="manager-xyz")
    await asyncio.sleep(0.3)

    mid = await redis_store.get_metrics()
    mid_rpc_created = mid["rpc"]["rpc_connections"]["created_total"]
    mid_rpc_active = mid["rpc"]["rpc_connections"]["active"]
    mid_patterns_active = mid["eventbus"]["eventbus"]["patterns"]["active"]
    mid_local_clients = mid["eventbus"]["eventbus"]["local_clients"]

    # Expect +3 created and +3 active compared to initial
    assert mid_rpc_created - init_rpc_created >= 3
    assert mid_rpc_active - init_rpc_active >= 3
    # Each client adds one targeted pattern and one local client
    assert mid_patterns_active - init_patterns_active >= 3
    assert mid_local_clients - init_local_clients >= 3

    # Publish a broadcast message and ensure processed counter increases
    before_processed = mid["eventbus"]["eventbus"]["messages_processed_total"]
    await redis_store.get_event_bus().emit("metrics-ws/*:msg", {"type": "x", "to": "*", "data": {}})
    await asyncio.sleep(0.2)
    after_emit = await redis_store.get_metrics()
    assert (
        after_emit["eventbus"]["eventbus"]["messages_processed_total"]
        >= before_processed
    )

    # Disconnect clients and verify counters go back to initial levels
    await api1.disconnect()
    await api2.disconnect()
    await api3.disconnect()
    await asyncio.sleep(0.3)

    final = await redis_store.get_metrics()
    # Closed should increase by at least 3
    assert final["rpc"]["rpc_connections"]["closed_total"] - init_rpc_closed >= 3
    # Active returns to initial
    assert final["rpc"]["rpc_connections"]["active"] == init_rpc_active
    # Patterns and local client gauges return to initial
    assert final["eventbus"]["eventbus"]["patterns"]["active"] == init_patterns_active
    assert final["eventbus"]["eventbus"]["local_clients"] == init_local_clients


async def test_redis_stress_connect_disconnect_counters(redis_store):
    """Stress: rapid connect/disconnect should not leak subscriptions or connections."""
    await redis_store.register_workspace(
        dict(
            name="stress-ws",
            description="stress workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )

    initial = await redis_store.get_metrics()
    init_rpc_active = initial["rpc"]["rpc_connections"]["active"]
    init_patterns_active = initial["eventbus"]["eventbus"]["patterns"]["active"]
    init_local_clients = initial["eventbus"]["eventbus"]["local_clients"]

    sessions = []
    # Create 20 connect/disconnect cycles
    for i in range(20):
        api = await redis_store.connect_to_workspace("stress-ws", client_id=f"s{i}")
        sessions.append(api)
        await asyncio.sleep(0.03)
        await api.disconnect()
    await asyncio.sleep(0.5)

    final = await redis_store.get_metrics()
    assert final["rpc"]["rpc_connections"]["active"] == init_rpc_active
    assert final["eventbus"]["eventbus"]["patterns"]["active"] == init_patterns_active
    assert final["eventbus"]["eventbus"]["local_clients"] == init_local_clients

async def test_websocket_server(fastapi_server, test_user_token_7):
    """Test the websocket server."""
    wm = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            client_id="test-app-1",
            token=test_user_token_7,
        )
    )
    await wm.log("hello")

    clients = await wm.list_clients()
    assert find_item(clients, "id", wm.config.workspace + "/test-app-1")
    rpc = wm.rpc

    def echo(data):
        return data

    await rpc.register_service(
        {
            "id": "test-service",
            "name": "my service",
            "config": {"visibility": "public"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x**2,
        }
    )

    # Relative service means from the same client
    assert await rpc.get_remote_service("test-service")

    await rpc.register_service(
        {
            "id": "default",
            "name": "my service",
            "config": {},
            "setup": print,
            "echo": echo,
        }
    )

    svc = await rpc.get_remote_service("test-app-1:test-service")

    assert await svc.echo("hello") == "hello"

    services = await wm.list_services()
    assert len(services) == 3

    svc = await wm.get_service("test-app-1:test-service")
    assert await svc.echo("hello") == "hello"

    # Get public service from another workspace
    wm3 = await connect_to_server({"server_url": f"ws://127.0.0.1:{SIO_PORT}/ws"})
    rpc3 = wm3.rpc
    svc7 = await rpc3.get_remote_service(
        f"{wm.config.workspace}/test-app-1:test-service"
    )
    svc7 = await wm3.get_service(f"{wm.config.workspace}/test-app-1:test-service")
    assert await svc7.square(9) == 81

    # Change the service to protected
    await rpc.register_service(
        {
            "id": "test-service",
            "name": "my service",
            "config": {"visibility": "protected"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x**2,
        },
        {"overwrite": True},
    )

    # It should fail due to permission error
    with pytest.raises(
        Exception,
        match=r".*Permission denied for getting protected service: test-service.*",
    ):
        await rpc3.get_remote_service(f"{wm.config.workspace}/test-app-1:test-service")

    # Should fail if we try to get the protected service from another workspace
    with pytest.raises(
        Exception, match=r".*Permission denied for invoking protected method.*"
    ):
        remote_echo = rpc3._generate_remote_method(
            {
                "_rtarget": f"{wm.config.workspace}/test-app-1",
                "_rmethod": "services.test-service.echo",
                "_rpromise": True,
            }
        )
        assert await remote_echo(123) == 123

    wm2 = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            workspace=wm.config.workspace,
            client_id="test-app-6",
            token=test_user_token_7,
            method_timeout=3,
        )
    )
    rpc2 = wm2.rpc

    svc2 = await rpc2.get_remote_service(
        f"{wm.config.workspace}/test-app-1:test-service"
    )
    assert await svc2.echo("hello") == "hello"
    assert len(rpc2._object_store) == 1
    svc3 = await svc2.echo(svc2)
    assert len(rpc2._object_store) == 1
    assert await svc3.echo("hello") == "hello"

    svc4 = await svc2.echo(
        {
            "add_one": lambda x: x + 1,
        }
    )
    assert len(rpc2._object_store) > 0

    # It should fail because add_one is not a service and will be destroyed after the session
    with pytest.raises(Exception, match=r".*Session not found:.*"):
        assert await svc4.add_one(99) == 100

    svc5_info = await rpc2.register_service(
        {
            "id": "default",
            "add_one": lambda x: x + 1,
            "inner": {"square": lambda y: y**2},
        }
    )
    svc5 = await rpc2.get_remote_service(svc5_info["id"])
    svc6 = await svc2.echo(svc5)
    assert await svc6.add_one(99) == 100
    assert await svc6.inner.square(10) == 100
    array = np.zeros([2048, 1000, 1])
    array2 = await svc6.add_one(array)
    np.testing.assert_array_equal(array2, array + 1)

    await wm.disconnect()
    await asyncio.sleep(0.5)

    with pytest.raises(Exception, match=r".*Service already exists: default.*"):
        await rpc2.register_service(
            {
                "id": "default",
                "add_two": lambda x: x + 2,
            }
        )

    await rpc2.unregister_service("default")
    await rpc2.register_service(
        {
            "id": "default",
            "add_two": lambda x: x + 2,
        }
    )

    await rpc2.register_service({"id": "add-two", "blocking_sleep": time.sleep})
    svc5 = await rpc2.get_remote_service(f"{wm.config.workspace}/test-app-6:add-two")
    await svc5.blocking_sleep(2)

    await rpc2.register_service(
        {
            "id": "executor-test",
            "config": {"run_in_executor": True, "visibility": "public"},
            "blocking_sleep": time.sleep,
        }
    )
    svc5 = await rpc2.get_remote_service(
        f"{wm.config.workspace}/test-app-6:executor-test"
    )
    # This should be fine because it is run in executor
    await svc5.blocking_sleep(3)
    summary = await wm2.get_summary()
    assert summary["client_count"] == 1
    assert summary["service_count"] == 4
    assert find_item(summary["services"], "name", "executor-test")
    await wm2.disconnect()
