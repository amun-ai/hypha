#!/usr/bin/env python3
"""Simple test to verify Redis partitioning functionality."""

import asyncio
import json
from hypha.core import RedisEventBus, RedisRPCConnection, UserInfo
from hypha.core.auth import create_scope
from hypha.utils import EventBus
from fakeredis import aioredis


async def test_redis_partitioning():
    """Test basic Redis partitioning functionality."""
    print("Testing Redis partitioning functionality...")
    
    # Create fake Redis instances
    redis1 = aioredis.FakeRedis()
    redis2 = aioredis.FakeRedis()
    
    # Create event buses with partitioning enabled
    event_bus1 = RedisEventBus(redis1, server_id="server-1", enable_partitioning=True)
    event_bus2 = RedisEventBus(redis2, server_id="server-2", enable_partitioning=True)
    
    # Initialize event buses
    await event_bus1.init()
    await event_bus2.init()
    
    # Create user info
    user_info = UserInfo(
        id="test-user",
        is_anonymous=False,
        email="test@example.com",
        parent=None,
        roles=[],
        scope=create_scope("test-workspace#r"),
        expires_at=None,
    )
    
    # Create RPC connections
    rpc1 = RedisRPCConnection(event_bus1, "test-workspace", "client-1", user_info, "manager-1")
    rpc2 = RedisRPCConnection(event_bus2, "test-workspace", "client-2", user_info, "manager-2")
    
    # Test that clients are registered in their respective event buses
    print(f"Server 1 local clients: {event_bus1._local_clients}")
    print(f"Server 2 local clients: {event_bus2._local_clients}")
    
    assert "test-workspace/client-1" in event_bus1._local_clients
    assert "test-workspace/client-2" in event_bus2._local_clients
    assert "test-workspace/client-1" not in event_bus2._local_clients
    assert "test-workspace/client-2" not in event_bus1._local_clients
    
    # Test subscription patterns
    pattern1 = event_bus1.get_subscription_pattern()
    pattern2 = event_bus2.get_subscription_pattern()
    
    print(f"Server 1 subscription pattern: {pattern1}")
    print(f"Server 2 subscription pattern: {pattern2}")
    
    assert pattern1 == "event:*:server-1"
    assert pattern2 == "event:*:server-2"
    
    # Test message routing
    messages_received = []
    
    async def message_handler(data):
        messages_received.append(data)
    
    # Set up message handlers
    rpc1.on_message(message_handler)
    rpc2.on_message(message_handler)
    
    # Test that disconnection removes clients
    print("Before disconnect - Server 1 local clients:", event_bus1._local_clients)
    print("Before disconnect - Server 2 local clients:", event_bus2._local_clients)
    
    # Manually remove clients to test the remove_local_client method
    event_bus1.remove_local_client("test-workspace/client-1")
    event_bus2.remove_local_client("test-workspace/client-2")
    
    print("After manual remove - Server 1 local clients:", event_bus1._local_clients)
    print("After manual remove - Server 2 local clients:", event_bus2._local_clients)
    
    # Check that the removal worked
    assert "test-workspace/client-1" not in event_bus1._local_clients
    assert "test-workspace/client-2" not in event_bus2._local_clients
    
    # Stop event buses
    await event_bus1.stop()
    await event_bus2.stop()
    
    print("✓ Redis partitioning test passed!")


async def test_legacy_mode():
    """Test that legacy mode (no partitioning) still works."""
    print("Testing legacy mode (no partitioning)...")
    
    # Create fake Redis instance
    redis = aioredis.FakeRedis()
    
    # Create event bus without partitioning
    event_bus = RedisEventBus(redis, server_id="server-1", enable_partitioning=False)
    
    # Initialize event bus
    await event_bus.init()
    
    # Test subscription pattern
    pattern = event_bus.get_subscription_pattern()
    print(f"Legacy subscription pattern: {pattern}")
    
    assert pattern == "event:*"
    
    # Test that local clients are not tracked
    user_info = UserInfo(
        id="test-user",
        is_anonymous=False,
        email="test@example.com",
        parent=None,
        roles=[],
        scope=create_scope("test-workspace#r"),
        expires_at=None,
    )
    
    rpc = RedisRPCConnection(event_bus, "test-workspace", "client-1", user_info, "manager-1")
    
    # In legacy mode, local clients should not be tracked
    print(f"Legacy mode local clients: {event_bus._local_clients}")
    assert len(event_bus._local_clients) == 0
    
    await event_bus.stop()
    
    print("✓ Legacy mode test passed!")


if __name__ == "__main__":
    asyncio.run(test_redis_partitioning())
    asyncio.run(test_legacy_mode())
    print("All tests passed!")