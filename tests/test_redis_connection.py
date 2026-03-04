import pytest
import pytest_asyncio
import asyncio
import msgpack
from fakeredis.aioredis import FakeRedis
from hypha.core.store import (
    RedisRPCConnection,
    RedisEventBus,
    UserInfo,
)  # Update with the actual module path

pytestmark = pytest.mark.asyncio


# Fixtures remain the same
@pytest_asyncio.fixture
async def fake_redis():
    redis = FakeRedis()
    await redis.flushall()
    return redis


@pytest_asyncio.fixture
async def event_bus(fake_redis):
    bus = RedisEventBus(redis=fake_redis)
    await bus.init()
    return bus


@pytest_asyncio.fixture
async def user_info_admin():
    return UserInfo(
        id="admin1", is_anonymous=False, email="admin1@example.com", roles=["admin"]
    )


@pytest_asyncio.fixture
async def user_info_user():
    return UserInfo(
        id="user1", is_anonymous=False, email="user1@example.com", roles=["user"]
    )


@pytest_asyncio.fixture
async def first_rpc(event_bus, user_info_admin):
    rpc = RedisRPCConnection(event_bus, "workspace1", "client1", user_info_admin, None)
    return rpc


@pytest_asyncio.fixture
async def another_rpc_connection(event_bus, user_info_user):
    rpc = RedisRPCConnection(event_bus, "workspace2", "client2", user_info_user, None)
    return rpc


@pytest_asyncio.fixture
async def second_client_same_workspace(event_bus, user_info_user):
    rpc = RedisRPCConnection(event_bus, "workspace1", "client3", user_info_user, None)
    return rpc


@pytest_asyncio.fixture
async def dynamic_workspace_rpc(event_bus, user_info_user):
    rpc = RedisRPCConnection(event_bus, "*", "dynamic_client", user_info_user, None)
    return rpc


async def test_dynamic_workspace_interaction(first_rpc, dynamic_workspace_rpc):
    first_messages = []
    dynamic_messages = []

    def first_handler(data):
        first_messages.append(msgpack.unpackb(data))

    def dynamic_handler(data):
        dynamic_messages.append(msgpack.unpackb(data))

    first_rpc.on_message(first_handler)
    dynamic_workspace_rpc.on_message(dynamic_handler)

    await asyncio.sleep(0.1)  # Ensure handlers are registered

    # Messages from dynamic client to fixed workspace client
    dynamic_to_first_message = msgpack.packb(
        {"to": "workspace1/client1", "content": "Hello from dynamic"}
    )
    await dynamic_workspace_rpc.emit_message(dynamic_to_first_message)

    # Messages from fixed workspace client to dynamic client
    first_to_dynamic_message = msgpack.packb(
        {"to": "*/dynamic_client", "content": "Reply from workspace1"}
    )
    await first_rpc.emit_message(first_to_dynamic_message)

    await asyncio.sleep(0.1)

    assert dynamic_messages[0]["content"] == "Reply from workspace1"
    assert first_messages[0]["content"] == "Hello from dynamic"


async def test_message_emission_within_same_workspace(
    first_rpc, second_client_same_workspace
):
    first_messages = []
    second_messages = []

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    second_client_same_workspace.on_message(
        lambda data: second_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    message_to_second = msgpack.packb({"to": "client3", "content": "Message to second"})
    message_to_first = msgpack.packb({"to": "client1", "content": "Reply to first"})

    await first_rpc.emit_message(message_to_second)
    await second_client_same_workspace.emit_message(message_to_first)

    await asyncio.sleep(0.1)

    assert second_messages[0]["content"] == "Message to second"
    assert first_messages[0]["content"] == "Reply to first"


async def test_message_emission_across_workspaces(first_rpc, another_rpc_connection):
    first_messages = []
    another_messages = []

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    another_rpc_connection.on_message(
        lambda data: another_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    message_to_another = msgpack.packb(
        {"to": "workspace2/client2", "content": "Hello to workspace2"}
    )
    message_to_first = msgpack.packb(
        {"to": "workspace1/client1", "content": "Reply to workspace1"}
    )

    await first_rpc.emit_message(message_to_another)
    await another_rpc_connection.emit_message(message_to_first)

    await asyncio.sleep(0.1)

    assert another_messages[0]["content"] == "Hello to workspace2"
    assert first_messages[0]["content"] == "Reply to workspace1"


async def test_broadcast_message_within_workspace(
    first_rpc, second_client_same_workspace
):
    first_messages = []
    second_messages = []

    # Subscribe both clients to "test-broadcast" event type
    first_rpc.subscribe("test-broadcast")
    second_client_same_workspace.subscribe("test-broadcast")

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    second_client_same_workspace.on_message(
        lambda data: second_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    broadcast_message = msgpack.packb({"to": "*", "type": "test-broadcast", "content": "Broadcast message"})
    await first_rpc.emit_message(broadcast_message)

    await asyncio.sleep(0.1)

    assert len(first_messages) == 1
    assert len(second_messages) == 1
    assert first_messages[0]["content"] == "Broadcast message"
    assert second_messages[0]["content"] == "Broadcast message"


async def test_broadcast_message_does_not_cross_workspaces(
    first_rpc, another_rpc_connection, second_client_same_workspace
):
    first_messages = []
    second_messages = []
    another_messages = []

    # Subscribe all clients to "test-broadcast" event type
    first_rpc.subscribe("test-broadcast")
    second_client_same_workspace.subscribe("test-broadcast")
    another_rpc_connection.subscribe("test-broadcast")

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    second_client_same_workspace.on_message(
        lambda data: second_messages.append(msgpack.unpackb(data))
    )
    another_rpc_connection.on_message(
        lambda data: another_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    broadcast_message = msgpack.packb({"to": "*", "type": "test-broadcast", "content": "Broadcast message"})
    await first_rpc.emit_message(broadcast_message)

    await asyncio.sleep(0.1)

    assert len(first_messages) == 1
    assert len(second_messages) == 1
    assert len(another_messages) == 0
    assert first_messages[0]["content"] == "Broadcast message"
    assert second_messages[0]["content"] == "Broadcast message"


async def test_disconnect_does_not_use_gc_get_objects(event_bus, user_info_admin):
    """Test that disconnect() completes quickly without scanning all Python objects.

    The gc.get_objects() call was removed because it scans ALL Python objects
    on every disconnect, which is O(n_total_objects) and catastrophically slow
    at scale with 100K+ concurrent users.
    """
    import time
    import ast
    import inspect
    import hypha.core

    # Verify the source AST does not call gc.get_objects() or gc.collect()
    source = inspect.getsource(hypha.core)
    tree = ast.parse(source)

    gc_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if (isinstance(func.value, ast.Name) and func.value.id == "gc"
                        and func.attr in ("get_objects", "collect")):
                    gc_calls.append(f"gc.{func.attr}() at line {func.value.col_offset}")

    assert not gc_calls, (
        f"Found gc calls that hurt performance at scale: {gc_calls}. "
        "These scan ALL Python objects and must not be used in disconnect()."
    )

    # Measure disconnect time - should be fast
    rpc = RedisRPCConnection(event_bus, "perf-ws", "perf-client", user_info_admin, None)
    messages = []
    rpc.on_message(lambda data: messages.append(data))
    await asyncio.sleep(0.1)

    start = time.perf_counter()
    await rpc.disconnect("test disconnect")
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"Disconnect took too long: {elapsed:.3f}s"


async def test_client_metrics_ttl_expiry(event_bus, user_info_admin):
    """Test that orphan _client_metrics entries are evicted after TTL expiry.

    When __rlb clients crash without calling disconnect(), their entries
    accumulate in the class-level _client_metrics dict.  The fix adds
    TTL-based sweep: any entry not updated within _METRICS_TTL_SECONDS is
    removed the next time any __rlb client calls _update_load_metric().
    """
    import time

    # Ensure fresh state
    RedisRPCConnection._client_metrics = {}

    # Plant a stale orphan entry (simulates a crashed __rlb client)
    stale_key = "workspace-orphan/crashed-client__rlb"
    RedisRPCConnection._client_metrics[stale_key] = {
        "last_time": time.time() - 700,   # > TTL (600 s default)
        "last_requests": 5,
        "last_updated": time.time() - 700,
    }

    # A fresh __rlb client connects and sends a message, triggering the sweep
    rpc = RedisRPCConnection(
        event_bus, "workspace-fresh", "newclient__rlb", user_info_admin, None
    )
    rpc.on_message(lambda data: None)
    await asyncio.sleep(0.05)

    # Directly invoke the metric update (the sweep runs inside it)
    rpc._update_load_metric()

    # The stale orphan must be gone
    assert stale_key not in RedisRPCConnection._client_metrics, (
        "Orphan _client_metrics entry was NOT evicted by TTL sweep; "
        "this causes unbounded memory growth when __rlb clients crash."
    )

    # The fresh client's own entry must still exist
    fresh_key = "workspace-fresh/newclient__rlb"
    assert fresh_key in RedisRPCConnection._client_metrics, (
        "Fresh client's own metrics entry was unexpectedly removed."
    )

    await rpc.disconnect("cleanup")
