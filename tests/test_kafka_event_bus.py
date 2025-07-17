import pytest
import asyncio
import json
import pytest_asyncio
from hypha.core import KafkaEventBus

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def kafka_event_bus(kafka_server):
    """Create a Kafka event bus for testing."""
    bus = KafkaEventBus(kafka_uri=kafka_server, server_id="test-server")
    await bus.init()
    yield bus
    await bus.stop()


async def test_kafka_event_emission(kafka_event_bus):
    """Test basic event emission with Kafka."""
    local_messages = []
    kafka_messages = []

    def local_handler(data):
        local_messages.append(data)

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.on_local("test_event", local_handler)
    kafka_event_bus.on("test_event", kafka_handler)

    await kafka_event_bus.emit("test_event", {"key": "value"})
    await asyncio.sleep(0.5)  # Give more time for Kafka message processing

    assert local_messages == [{"key": "value"}]
    assert kafka_messages == [{"key": "value"}]


async def test_kafka_local_event_handling(kafka_event_bus):
    """Test local event handling with Kafka."""
    local_messages = []

    def local_handler(data):
        local_messages.append(data)

    kafka_event_bus.on_local("local_test_event", local_handler)

    await kafka_event_bus.emit_local("local_test_event", {"local": "data"})
    await asyncio.sleep(0.1)

    assert local_messages == [{"local": "data"}]


async def test_kafka_remote_event_handling(kafka_event_bus):
    """Test remote event handling with Kafka."""
    kafka_messages = []

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.on("remote_test_event", kafka_handler)

    await kafka_event_bus.emit("remote_test_event", {"remote": "data"})
    await asyncio.sleep(0.5)  # Give more time for Kafka message processing

    assert kafka_messages == [{"remote": "data"}]


async def test_kafka_string_event_handling(kafka_event_bus):
    """Test string event handling with Kafka."""
    kafka_messages = []

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.on("string_test_event", kafka_handler)

    await kafka_event_bus.emit("string_test_event", "test_string")
    await asyncio.sleep(0.5)

    assert kafka_messages == ["test_string"]


async def test_kafka_binary_event_handling(kafka_event_bus):
    """Test binary event handling with Kafka."""
    kafka_messages = []

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.on("binary_test_event", kafka_handler)

    test_data = b"test_binary_data"
    await kafka_event_bus.emit("binary_test_event", test_data)
    await asyncio.sleep(0.5)

    assert kafka_messages == [test_data]


async def test_kafka_wait_for_event(kafka_event_bus):
    """Test waiting for events with Kafka."""
    async def emit_after_delay():
        await asyncio.sleep(0.2)
        await kafka_event_bus.emit("wait_test_event", {"delayed": "data"})

    # Start emitting after delay
    asyncio.create_task(emit_after_delay())

    # Wait for the event
    result = await kafka_event_bus.wait_for("wait_test_event", timeout=1.0)
    assert result == {"delayed": "data"}


async def test_kafka_event_off(kafka_event_bus):
    """Test removing event handlers with Kafka."""
    kafka_messages = []

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.on("off_test_event", kafka_handler)

    await kafka_event_bus.emit("off_test_event", {"first": "message"})
    await asyncio.sleep(0.5)

    # Remove handler
    kafka_event_bus.off("off_test_event", kafka_handler)

    await kafka_event_bus.emit("off_test_event", {"second": "message"})
    await asyncio.sleep(0.5)

    # Only the first message should be received
    assert kafka_messages == [{"first": "message"}]


async def test_kafka_once_event_handling(kafka_event_bus):
    """Test once event handling with Kafka."""
    kafka_messages = []

    def kafka_handler(data):
        kafka_messages.append(data)

    kafka_event_bus.once("once_test_event", kafka_handler)

    await kafka_event_bus.emit("once_test_event", {"first": "message"})
    await asyncio.sleep(0.5)

    await kafka_event_bus.emit("once_test_event", {"second": "message"})
    await asyncio.sleep(0.5)

    # Only the first message should be received due to once
    assert kafka_messages == [{"first": "message"}]


async def test_kafka_multiple_servers():
    """Test multiple Kafka event buses communicating."""
    # This test will be run with the server fixtures
    pass  # The actual multi-server test will be in test_kafka_servers.py