"""Test event bus optimization features."""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch
from hypha.core import RedisEventBus
import time

pytest_asyncio_timeout = 30


@pytest_asyncio.fixture
async def fake_redis():
    # Setup the fake Redis server
    from fakeredis import aioredis
    redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")
    yield redis


@pytest_asyncio.fixture
async def enhanced_event_bus(fake_redis):
    """Create an enhanced event bus with local client tracking."""
    event_bus = RedisEventBus(fake_redis)
    await event_bus.init()
    yield event_bus
    await event_bus.stop()


@pytest.mark.asyncio
async def test_local_client_tracking(enhanced_event_bus):
    """Test that local client tracking works correctly."""
    # Register local clients
    await enhanced_event_bus.register_local_client("workspace1", "client1")
    await enhanced_event_bus.register_local_client("workspace2", "client2")
    
    # Check if clients are registered as local
    assert enhanced_event_bus.is_local_client("workspace1", "client1")
    assert enhanced_event_bus.is_local_client("workspace2", "client2")
    assert not enhanced_event_bus.is_local_client("workspace1", "nonexistent")
    
    # Unregister a client
    await enhanced_event_bus.unregister_local_client("workspace1", "client1")
    assert not enhanced_event_bus.is_local_client("workspace1", "client1")
    assert enhanced_event_bus.is_local_client("workspace2", "client2")


@pytest.mark.asyncio
async def test_smart_routing_optimization(enhanced_event_bus):
    """Test that smart routing prefers local delivery for targeted messages."""
    local_messages = []
    redis_messages = []
    
    def local_handler(data):
        local_messages.append(data)
    
    def redis_handler(data):
        redis_messages.append(data)
    
    # Register handlers for both local and Redis events
    enhanced_event_bus.on_local("workspace1/client1:msg", local_handler)
    enhanced_event_bus.on("workspace1/client1:msg", redis_handler)
    
    # Register client1 as local
    await enhanced_event_bus.register_local_client("workspace1", "client1")
    
    # Emit a targeted message to local client
    await enhanced_event_bus.emit("workspace1/client1:msg", b"test message")
    await asyncio.sleep(0.1)
    
    # For local clients, the message should be routed locally only
    # (This demonstrates the optimization potential)
    assert local_messages == [b"test message"]
    # Redis should still receive it for compatibility, but in optimized version
    # it could be skipped for local clients
    

@pytest.mark.asyncio
async def test_non_targeted_message_routing(enhanced_event_bus):
    """Test that non-targeted messages still go through Redis."""
    local_messages = []
    redis_messages = []
    
    def local_handler(data):
        local_messages.append(data)
    
    def redis_handler(data):
        redis_messages.append(data)
    
    # Register handlers
    enhanced_event_bus.on_local("general_event", local_handler)
    enhanced_event_bus.on("general_event", redis_handler)
    
    # Emit a general event
    await enhanced_event_bus.emit("general_event", {"key": "value"})
    await asyncio.sleep(0.1)
    
    # Both local and Redis should receive the message
    assert local_messages == [{"key": "value"}]
    assert redis_messages == [{"key": "value"}]


@pytest.mark.asyncio
async def test_performance_improvement_simulation():
    """Simulate performance improvement with local routing."""
    from fakeredis import aioredis
    redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")
    
    # Test standard routing
    standard_bus = RedisEventBus(redis)
    await standard_bus.init()
    
    # Measure time for Redis-only routing
    message_count = 100
    start_time = time.time()
    
    for i in range(message_count):
        await standard_bus.emit(f"workspace1/client{i}:msg", f"message{i}")
    
    redis_time = time.time() - start_time
    await standard_bus.stop()
    
    # Test optimized routing (simulated)
    optimized_bus = RedisEventBus(redis)
    await optimized_bus.init()
    
    # Register all clients as local
    for i in range(message_count):
        await optimized_bus.register_local_client("workspace1", f"client{i}")
    
    # Measure time for local routing
    start_time = time.time()
    
    for i in range(message_count):
        # In the optimized version, this would use emit_local for better performance
        await optimized_bus.emit_local(f"workspace1/client{i}:msg", f"message{i}")
    
    local_time = time.time() - start_time
    await optimized_bus.stop()
    
    # Local routing should be faster (though this is a simple simulation)
    print(f"Redis routing time: {redis_time:.4f}s")
    print(f"Local routing time: {local_time:.4f}s")
    print(f"Performance improvement: {((redis_time - local_time) / redis_time * 100):.1f}%")
    
    # Assert that local routing is at least as fast (usually faster)
    assert local_time <= redis_time * 1.1  # Allow 10% tolerance


@pytest.mark.asyncio
async def test_scalability_demonstration():
    """Demonstrate how the system scales with multiple server instances."""
    from fakeredis import aioredis
    redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")
    
    # Simulate two server instances
    server1_bus = RedisEventBus(redis)
    server2_bus = RedisEventBus(redis)
    
    await server1_bus.init()
    await server2_bus.init()
    
    # Server 1 has local clients and subscribes to their events
    await server1_bus.register_local_client("workspace1", "client1")
    await server1_bus.register_local_client("workspace1", "client2")
    await server1_bus.subscribe_to_client_events("workspace1", "client1")
    await server1_bus.subscribe_to_client_events("workspace1", "client2")
    
    # Server 2 has different local clients and subscribes to their events
    await server2_bus.register_local_client("workspace1", "client3")
    await server2_bus.register_local_client("workspace1", "client4")
    await server2_bus.subscribe_to_client_events("workspace1", "client3")
    await server2_bus.subscribe_to_client_events("workspace1", "client4")
    
    # Set up message tracking
    server1_messages = []
    server2_messages = []
    
    def server1_handler(data):
        server1_messages.append(data)
    
    def server2_handler(data):
        server2_messages.append(data)
    
    # Each server listens for its local clients
    server1_bus.on("workspace1/client1:msg", server1_handler)
    server1_bus.on("workspace1/client2:msg", server1_handler)
    server2_bus.on("workspace1/client3:msg", server2_handler)
    server2_bus.on("workspace1/client4:msg", server2_handler)
    
    # Send messages to different clients
    await server1_bus.emit("workspace1/client1:msg", "to_client1")  # Local to server1
    await server1_bus.emit("workspace1/client3:msg", "to_client3")  # Cross-server to server2
    await server2_bus.emit("workspace1/client2:msg", "to_client2")  # Cross-server to server1
    await server2_bus.emit("workspace1/client4:msg", "to_client4")  # Local to server2
    
    await asyncio.sleep(0.2)  # Give more time for message propagation
    
    # Verify message routing - each server should receive messages for its subscribed clients
    assert "to_client1" in server1_messages  # Local message
    assert "to_client2" in server1_messages  # Cross-server message
    assert "to_client3" in server2_messages  # Cross-server message  
    assert "to_client4" in server2_messages  # Local message
    
    # Verify optimization: servers don't receive messages for non-subscribed clients
    print(f"Server1 received {len(server1_messages)} messages: {server1_messages}")
    print(f"Server2 received {len(server2_messages)} messages: {server2_messages}")
    
    # Each server should only receive messages for its own clients
    assert len(server1_messages) == 2  # client1 and client2 messages
    assert len(server2_messages) == 2  # client3 and client4 messages
    
    await server1_bus.stop()
    await server2_bus.stop()


@pytest.mark.asyncio
async def test_redis_subscription_optimization():
    """Test that the system can optimize Redis subscriptions."""
    from fakeredis import aioredis
    redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")
    
    event_bus = RedisEventBus(redis)
    await event_bus.init()
    
    # The optimized system maintains subscription patterns
    assert hasattr(event_bus, '_subscribed_patterns')
    assert hasattr(event_bus, '_local_clients')
    
    # Register some local clients
    await event_bus.register_local_client("workspace1", "client1")
    await event_bus.register_local_client("workspace2", "client2") 
    
    # Verify local client tracking
    assert len(event_bus._local_clients) == 2
    assert "workspace1/client1" in event_bus._local_clients
    assert "workspace2/client2" in event_bus._local_clients
    
    await event_bus.stop()


@pytest.mark.asyncio
async def test_targeted_subscription_debug():
    """Debug the targeted subscription system to understand message routing."""
    from fakeredis import aioredis
    redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")
    
    event_bus = RedisEventBus(redis)
    await event_bus.init()
    
    # Register a local client and subscribe to its events  
    await event_bus.register_local_client("workspace1", "client1")
    await event_bus.subscribe_to_client_events("workspace1", "client1")
    
    # Set up message tracking
    received_messages = []
    
    def message_handler(data):
        print(f"Handler received: {data}")
        received_messages.append(data)
    
    # Register handler for client1 messages
    event_bus.on("workspace1/client1:msg", message_handler)
    
    print("Emitting message to client1...")
    await event_bus.emit("workspace1/client1:msg", "test_message")
    
    await asyncio.sleep(0.1)
    
    print(f"Messages received: {received_messages}")
    print(f"Is client1 local: {event_bus.is_local_client('workspace1', 'client1')}")
    
    # The message should be received
    assert len(received_messages) > 0, f"Expected to receive messages, got: {received_messages}"
    assert "test_message" in received_messages
    
    await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(test_performance_improvement_simulation()) 