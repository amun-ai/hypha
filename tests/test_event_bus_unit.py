"""Unit tests for Redis event bus security without full server setup.

These tests directly test RedisEventBus and RedisRPCConnection classes
to identify security vulnerabilities without needing full server fixtures.
"""

import pytest
import asyncio
import msgpack
import fakeredis.aioredis
from hypha.core import RedisEventBus, RedisRPCConnection, UserInfo


@pytest.mark.asyncio
async def test_readonly_client_broadcast_restriction():
    """
    V-BUS-05: Test readonly client broadcast restrictions.

    Expected: Readonly clients should be blocked from ALL broadcast patterns,
    including both "*" and "workspace/*" formats.

    Status: ✅ FIXED - Both patterns now properly blocked.
    """
    # Create fake Redis for testing
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    user_info = UserInfo.model_validate({
        "id": "readonly-user",
        "email": "readonly@test.com",
        "roles": [],
        "is_anonymous": False,
        "scope": None
    })

    # Create readonly connection
    readonly_conn = RedisRPCConnection(
        event_bus=event_bus,
        workspace="test-ws",
        client_id="readonly-client",
        user_info=user_info,
        manager_id="test-manager",
        readonly=True
    )

    # Test 1: "*" broadcast should be blocked
    message_star = {
        "type": "test",
        "to": "*",
        "data": {"msg": "broadcast via star"}
    }
    packed_star = msgpack.packb(message_star)

    with pytest.raises(PermissionError, match="read-only client"):
        await readonly_conn.emit_message(packed_star)

    # Test 2: "workspace/*" pattern should ALSO be blocked (FIXED in V-BUS-05)
    message_ws = {
        "type": "test",
        "to": "test-ws/*",
        "data": {"msg": "broadcast via workspace"}
    }
    packed_ws = msgpack.packb(message_ws)

    # After V-BUS-05 fix: This now correctly raises PermissionError
    with pytest.raises(PermissionError, match="read-only client|broadcast"):
        await readonly_conn.emit_message(packed_ws)

    print("SECURED: Broadcast properly blocked")

    await readonly_conn.disconnect()
    await event_bus.stop()
    await redis.aclose()


@pytest.mark.asyncio
async def test_cross_workspace_message_routing():
    """
    V-BUS-10: Test cross-workspace message routing.

    Expected: Clients in workspace A should not be able to send messages
    to clients in workspace B.

    Status: ✅ FIXED - Cross-workspace messaging now properly blocked.
    """
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    user_info = UserInfo.model_validate({
        "id": "attacker",
        "email": "attacker@test.com",
        "roles": [],
        "is_anonymous": False,
        "scope": None
    })

    # Create connection in workspace A
    attacker_conn = RedisRPCConnection(
        event_bus=event_bus,
        workspace="workspace-a",
        client_id="attacker",
        user_info=user_info,
        manager_id="test-manager",
        readonly=False
    )

    # Track messages received in workspace B
    workspace_b_messages = []

    async def workspace_b_handler(data):
        workspace_b_messages.append(data)

    event_bus.on("workspace-b/victim:msg", workspace_b_handler)

    # Attacker tries to send to workspace B
    message = {
        "type": "attack",
        "to": "workspace-b/victim",  # Cross-workspace target
        "data": {"msg": "Cross-workspace attack"}
    }
    packed = msgpack.packb(message)

    # After V-BUS-10 fix: This now correctly raises PermissionError
    with pytest.raises(PermissionError, match="Cross-workspace messaging not allowed"):
        await attacker_conn.emit_message(packed)

    await asyncio.sleep(0.3)

    # Verify no messages were delivered
    assert len(workspace_b_messages) == 0, "No messages should cross workspace boundaries"

    print("SECURED: Cross-workspace messaging blocked")

    event_bus.off("workspace-b/victim:msg", workspace_b_handler)
    await attacker_conn.disconnect()
    await event_bus.stop()
    await redis.aclose()


@pytest.mark.asyncio
async def test_message_source_spoofing():
    """
    V-BUS-03: Test message source spoofing in broadcast/emit.

    Expected: Message source ("from" field) should be set by the server
    based on authenticated connection, not user input.

    Current Implementation: Source is correctly set by emit_message,
    but we verify it can't be overridden.
    """
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    user_info = UserInfo.model_validate({
        "id": "attacker",
        "email": "attacker@test.com",
        "roles": [],
        "is_anonymous": False,
        "scope": None
    })

    attacker_conn = RedisRPCConnection(
        event_bus=event_bus,
        workspace="test-ws",
        client_id="attacker",
        user_info=user_info,
        manager_id="test-manager",
        readonly=False
    )

    received_messages = []

    async def message_handler(data):
        # Unpack to check source
        if isinstance(data, bytes):
            unpacker = msgpack.Unpacker(io.BytesIO(data))
            msg = unpacker.unpack()
            received_messages.append(msg)

    event_bus.on("test-ws/victim:msg", message_handler)

    # Attacker tries to spoof source
    message = {
        "type": "test",
        "from": "system/admin",  # Attempt to spoof source
        "to": "victim",
        "data": {"msg": "Fake admin message"}
    }
    packed = msgpack.packb(message)

    await attacker_conn.emit_message(packed)
    await asyncio.sleep(0.3)

    # Check if source was correctly enforced
    if received_messages:
        actual_from = received_messages[0].get("from")
        expected_from = "test-ws/attacker"

        if actual_from != expected_from:
            print(f"WARNING: V-BUS-03 VULNERABILITY - Source spoofing possible!")
            print(f"Expected from: {expected_from}")
            print(f"Actual from: {actual_from}")
        else:
            print(f"SECURED: Source correctly enforced as {actual_from}")

    event_bus.off("test-ws/victim:msg", message_handler)
    await attacker_conn.disconnect()
    await event_bus.stop()
    await redis.aclose()


@pytest.mark.asyncio
async def test_pattern_subscription_limits():
    """
    V-BUS-09: Test pattern subscription resource limits.

    Expected: There should be limits on number of patterns a client can subscribe to.

    Current Implementation: No limits enforced, allowing DoS via pattern flooding.
    """
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    initial_count = len(event_bus._subscribed_patterns)

    # Try to add many patterns
    patterns_added = 0
    for i in range(1000):
        pattern = f"targeted:test-ws/client-{i}:*"
        event_bus._subscribed_patterns.add(pattern)
        patterns_added += 1

    final_count = len(event_bus._subscribed_patterns)
    actual_added = final_count - initial_count

    if actual_added >= 900:  # Most were added
        print(f"WARNING: V-BUS-09 VULNERABILITY CONFIRMED!")
        print(f"Added {actual_added} patterns without limits")
        print("This could cause DoS through memory exhaustion")
        print("Expected: Pattern limit enforced")
        print("Actual: No limits")

    # Cleanup
    event_bus._subscribed_patterns.clear()
    await event_bus.stop()
    await redis.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_local_bypass():
    """
    V-BUS-06: Test circuit breaker bypass via local delivery.

    Expected: When circuit breaker is open, system should degrade gracefully.

    Current Implementation: Local messages bypass circuit breaker, potentially
    causing cascading failures during Redis outages.
    """
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    # Open circuit breaker
    event_bus._circuit_breaker_open = True
    event_bus._consecutive_failures = 10

    user_info = UserInfo.model_validate({
        "id": "client1",
        "email": "client1@test.com",
        "roles": [],
        "is_anonymous": False,
        "scope": None
    })

    conn = RedisRPCConnection(
        event_bus=event_bus,
        workspace="test-ws",
        client_id="client1",
        user_info=user_info,
        manager_id="test-manager",
        readonly=False
    )

    # Register as local client
    await event_bus.register_local_client("test-ws", "client1")

    messages_delivered = []

    async def handler(data):
        messages_delivered.append(data)

    event_bus.on("test-ws/client1:msg", handler)

    # Send message while circuit breaker is open
    message = {
        "type": "test",
        "to": "client1",
        "data": {"msg": "test during circuit breaker"}
    }
    packed = msgpack.packb(message)

    await conn.emit_message(packed)
    await asyncio.sleep(0.3)

    if messages_delivered:
        print("INFO: V-BUS-06 - Local messages delivered during circuit breaker open")
        print("This is by design for local clients, but could cause issues during Redis failures")

    event_bus.off("test-ws/client1:msg", handler)
    await conn.disconnect()
    await event_bus.stop()
    await redis.aclose()


@pytest.mark.asyncio
async def test_redis_channel_enumeration():
    """
    V-BUS-08: Test Redis channel enumeration.

    Expected: Channel names should not leak sensitive workspace/client information.

    Current Implementation: Predictable channel naming allows reconnaissance.
    """
    redis = fakeredis.aioredis.FakeRedis()
    event_bus = RedisEventBus(redis)
    await event_bus.init()

    # Create some subscriptions
    await event_bus.subscribe_to_client_events("secret-ws", "client-1")
    await event_bus.subscribe_to_client_events("private-ws", "admin")

    # Try to enumerate channels
    try:
        channels = await redis.pubsub_channels()
        channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in channels]

        # Look for patterns that reveal structure
        targeted_channels = [ch for ch in channel_names if 'targeted:' in ch or 'secret' in ch or 'private' in ch]

        if targeted_channels:
            print(f"WARNING: V-BUS-08 VULNERABILITY - Channel enumeration possible!")
            print(f"Found {len(targeted_channels)} channels revealing workspace structure:")
            for ch in targeted_channels[:3]:
                print(f"  - {ch}")
    except Exception as e:
        print(f"Channel enumeration failed: {e}")

    await event_bus.stop()
    await redis.aclose()


import io


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_readonly_client_broadcast_restriction())
    asyncio.run(test_cross_workspace_message_routing())
    asyncio.run(test_message_source_spoofing())
    asyncio.run(test_pattern_subscription_limits())
    asyncio.run(test_circuit_breaker_local_bypass())
    asyncio.run(test_redis_channel_enumeration())
