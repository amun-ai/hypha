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

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    second_client_same_workspace.on_message(
        lambda data: second_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    broadcast_message = msgpack.packb({"to": "*", "content": "Broadcast message"})
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

    first_rpc.on_message(lambda data: first_messages.append(msgpack.unpackb(data)))
    second_client_same_workspace.on_message(
        lambda data: second_messages.append(msgpack.unpackb(data))
    )
    another_rpc_connection.on_message(
        lambda data: another_messages.append(msgpack.unpackb(data))
    )

    await asyncio.sleep(0.1)

    broadcast_message = msgpack.packb({"to": "*", "content": "Broadcast message"})
    await first_rpc.emit_message(broadcast_message)

    await asyncio.sleep(0.1)

    assert len(first_messages) == 1
    assert len(second_messages) == 1
    assert len(another_messages) == 0
    assert first_messages[0]["content"] == "Broadcast message"
    assert second_messages[0]["content"] == "Broadcast message"
