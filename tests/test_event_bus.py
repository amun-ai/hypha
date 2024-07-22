import pytest
import asyncio
import json
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from hypha.core.store import RedisEventBus

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def fake_redis():
    # Setup the fake Redis server
    redis = FakeRedis()
    await redis.flushall()
    return redis


@pytest_asyncio.fixture
async def event_bus(fake_redis):
    bus = RedisEventBus(redis=fake_redis)
    await bus.init()
    return bus


async def test_combined_event_emission(event_bus):
    local_messages = []
    redis_messages = []

    def local_handler(data):
        local_messages.append(data)

    def redis_handler(data):
        redis_messages.append(data)

    event_bus.on_local("test_event", local_handler)
    event_bus.on("test_event", redis_handler)

    await event_bus.emit("test_event", {"key": "value"})
    await asyncio.sleep(0.1)

    assert local_messages == [{"key": "value"}]
    assert redis_messages == [{"key": "value"}]


async def test_local_event_handling(event_bus):
    local_messages = []

    def local_handler(data):
        local_messages.append(data)

    event_bus.on_local("local_event", local_handler)

    await event_bus.emit("local_event", {"key": "local_value"})
    await asyncio.sleep(0.1)

    assert local_messages == [{"key": "local_value"}]


async def test_redis_event_handling(event_bus, fake_redis):
    redis_messages = []

    def redis_handler(data):
        redis_messages.append(data)

    event_bus.on("redis_event", redis_handler)

    local_messages = []

    def local_handler(data):
        local_messages.append(data)

    event_bus.on_local("local_event", local_handler)
    # Simulate a Redis event
    await fake_redis.publish("event:d:redis_event", json.dumps({"key": "redis_value"}))
    await asyncio.sleep(0.1)

    assert redis_messages == [{"key": "redis_value"}]
    assert local_messages == []


async def test_wait_for_event(event_bus):
    async def emit_event():
        await asyncio.sleep(0.1)
        await event_bus.emit("wait_event", {"key": "wait_value"})

    asyncio.create_task(emit_event())
    data = await event_bus.wait_for("wait_event", match={"key": "wait_value"})
    assert data == {"key": "wait_value"}


async def test_wait_for_local_event(event_bus):
    async def emit_event():
        await asyncio.sleep(0.1)
        await event_bus.emit("wait_local_event", {"key": "wait_local_value"})

    asyncio.create_task(emit_event())
    data = await event_bus.wait_for_local(
        "wait_local_event", match={"key": "wait_local_value"}
    )
    assert data == {"key": "wait_local_value"}


async def test_combined_event_wait(event_bus, fake_redis):
    # Local event emission
    async def emit_event_local():
        await asyncio.sleep(0.1)
        await event_bus.emit("combined_event", {"key": "combined_local_value"})

    # Redis event emission
    async def emit_event_redis():
        await asyncio.sleep(0.2)
        await fake_redis.publish(
            "event:d:combined_event", json.dumps({"key": "combined_redis_value"})
        )

    asyncio.create_task(emit_event_local())
    asyncio.create_task(emit_event_redis())

    data = await event_bus.wait_for_local(
        "combined_event", match={"key": "combined_local_value"}, timeout=1
    )
    assert data == {"key": "combined_local_value"}

    # Wait for the Redis event
    data = await event_bus.wait_for(
        "combined_event", match={"key": "combined_redis_value"}, timeout=1
    )
    assert data == {"key": "combined_redis_value"}


async def test_once_handler(event_bus):
    messages = []

    def handler(data):
        messages.append(data)

    event_bus.once("once_event", handler)

    await event_bus.emit("once_event", {"key": "value1"})
    await event_bus.emit("once_event", {"key": "value2"})
    await asyncio.sleep(0.1)

    assert messages == [{"key": "value1"}]


async def test_stop(event_bus):
    event_bus.stop()
    assert event_bus._stop is True


async def test_off_local_handler(event_bus):
    local_messages = []

    def local_handler(data):
        local_messages.append(data)

    event_bus.on_local("local_event", local_handler)
    event_bus.off_local("local_event", local_handler)

    await event_bus.emit("local_event", {"key": "local_value"})
    await asyncio.sleep(0.1)

    assert local_messages == []


async def test_off_redis_handler(event_bus, fake_redis):
    redis_messages = []

    def redis_handler(data):
        redis_messages.append(data)

    event_bus.on("redis_event", redis_handler)
    event_bus.off("redis_event", redis_handler)

    await fake_redis.publish("event:d:redis_event", json.dumps({"key": "redis_value"}))
    await asyncio.sleep(0.1)

    assert redis_messages == []


async def test_off_combined_handlers(event_bus, fake_redis):
    local_messages = []
    redis_messages = []

    def local_handler(data):
        local_messages.append(data)

    def redis_handler(data):
        redis_messages.append(data)

    event_bus.on_local("combined_event", local_handler)
    event_bus.on("combined_event", redis_handler)

    event_bus.off_local("combined_event", local_handler)
    event_bus.off("combined_event", redis_handler)

    await event_bus.emit("combined_event", {"key": "combined_local_value"})
    await fake_redis.publish(
        "event:d:combined_event", json.dumps({"key": "combined_redis_value"})
    )
    await asyncio.sleep(0.1)

    assert local_messages == []
    assert redis_messages == []
