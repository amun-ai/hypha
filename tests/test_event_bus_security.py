"""Security tests for Redis event bus and message routing.

This module tests vulnerabilities in the Redis pub/sub event bus system,
including message interception, eavesdropping, targeted message hijacking,
broadcast injection, pattern subscription manipulation, and circuit breaker bypass.
"""

import pytest
import asyncio
import json
import msgpack
from hypha.core import RedisEventBus, RedisRPCConnection, UserInfo
from hypha_rpc import RPC
import io
import time


@pytest.mark.asyncio
async def test_cross_workspace_message_interception(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-01: Cross-workspace message interception via pattern subscription.

    Attack: A malicious client in workspace A subscribes to patterns that match
    messages in workspace B, allowing eavesdropping on private communications.

    Root Cause: Pattern subscription in RedisEventBus allows wildcard patterns
    without proper workspace isolation checks.
    """
    from hypha_rpc import connect_to_server

    # Connect victim client in workspace A
    victim_api = await connect_to_server(
        {"name": "victim", "server_url": fastapi_server_url, "workspace": "workspace-a", "token": test_user_token}
    )

    # Connect attacker client in workspace B
    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "workspace-b", "token": test_user_token}
    )

    # Attacker attempts to subscribe to victim's messages
    intercepted_messages = []

    async def intercept_handler(data):
        intercepted_messages.append(data)

    # Try to subscribe to victim's workspace pattern
    # This should NOT work due to workspace isolation
    try:
        # Attempt direct subscription to another workspace's events
        event_bus = fastapi_server.event_bus
        await event_bus.subscribe_to_client_events("workspace-a", "victim")

        # Register handler for victim's messages
        event_bus.on("workspace-a/victim:msg", intercept_handler)

        # Victim sends a private message to another client
        await victim_api.log("Secret message from victim")

        # Wait for potential interception
        await asyncio.sleep(0.5)

        # Check if attacker intercepted the message
        if len(intercepted_messages) > 0:
            pytest.fail("V-BUS-01: Attacker successfully intercepted cross-workspace messages!")

    finally:
        await victim_api.disconnect()
        await attacker_api.disconnect()


@pytest.mark.asyncio
async def test_targeted_message_hijacking(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-02: Targeted message hijacking via client ID spoofing.

    Attack: An attacker registers with a client_id that matches the pattern
    of legitimate clients, causing messages intended for others to be delivered
    to the attacker instead.

    Root Cause: Client registration doesn't validate uniqueness or prevent
    spoofing of client identifiers.
    """
    from hypha_rpc import connect_to_server

    # Connect legitimate client
    legit_api = await connect_to_server(
        {"name": "legit-service", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    received_by_legit = []

    def legit_handler(data):
        received_by_legit.append(data)

    # Register service
    await legit_api.register_service({
        "id": "legit-service",
        "config": {"visibility": "public"},
        "test": lambda x: f"legit: {x}"
    })

    # Attacker tries to register with same/similar client_id
    try:
        attacker_api = await connect_to_server(
            {"name": "legit-service", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
        )

        received_by_attacker = []

        def attacker_handler(data):
            received_by_attacker.append(data)

        # Send message to the service
        sender_api = await connect_to_server(
            {"name": "sender", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
        )

        # Try to call the service
        result = await sender_api.get_service("legit-service")
        response = await result.test("secret data")

        await asyncio.sleep(0.5)

        # Check if both received the message (vulnerability)
        if len(received_by_attacker) > 0:
            pytest.fail("V-BUS-02: Attacker hijacked messages via client ID spoofing!")

        await attacker_api.disconnect()
        await sender_api.disconnect()
    finally:
        await legit_api.disconnect()


@pytest.mark.asyncio
async def test_broadcast_message_injection(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-03: Broadcast message injection from unauthorized clients.

    Attack: A malicious client crafts broadcast messages that appear to come
    from system services or administrative sources.

    Root Cause: Broadcast message validation doesn't verify the source,
    allowing clients to impersonate system components.
    """
    from hypha_rpc import connect_to_server

    # Connect attacker client
    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    # Connect victim client
    victim_api = await connect_to_server(
        {"name": "victim", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    received_broadcasts = []

    async def broadcast_handler(data):
        received_broadcasts.append(data)

    # Victim subscribes to system broadcasts
    event_bus = fastapi_server.event_bus
    event_bus.on("test-ws/*:msg", broadcast_handler)

    try:
        # Attacker attempts to inject a fake system broadcast
        # Get the RPC connection to manipulate messages
        rpc = attacker_api._get_rpc()

        # Craft a malicious broadcast message
        fake_message = {
            "type": "system_alert",
            "from": "system/admin",  # Spoofed source
            "to": "*",
            "data": {"alert": "Please send your credentials to attacker-service"}
        }

        # Try to emit as broadcast
        packed = msgpack.packb(fake_message)
        await rpc._connection.emit_message(packed)

        await asyncio.sleep(0.5)

        # Check if spoofed message was accepted
        spoofed_broadcasts = [
            msg for msg in received_broadcasts
            if isinstance(msg, bytes) and b"system/admin" in msg
        ]

        if spoofed_broadcasts:
            pytest.fail("V-BUS-03: Attacker successfully injected spoofed broadcast messages!")

    finally:
        event_bus.off("test-ws/*:msg", broadcast_handler)
        await attacker_api.disconnect()
        await victim_api.disconnect()


@pytest.mark.asyncio
async def test_pattern_subscription_manipulation(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-04: Pattern subscription manipulation for unauthorized access.

    Attack: A client manipulates pattern subscriptions to receive messages
    from other clients' patterns, bypassing workspace isolation.

    Root Cause: Dynamic pattern subscription doesn't validate that the
    pattern belongs to the requesting client's workspace.
    """
    from hypha_rpc import connect_to_server

    # Connect attacker client
    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "attacker-ws", "token": test_user_token}
    )

    # Connect victim client in different workspace
    victim_api = await connect_to_server(
        {"name": "victim", "server_url": fastapi_server_url, "workspace": "victim-ws", "token": test_user_token}
    )

    intercepted = []

    async def malicious_handler(data):
        intercepted.append(data)

    try:
        # Attacker tries to subscribe to victim's patterns
        event_bus = fastapi_server.event_bus

        # Attempt to subscribe to victim workspace patterns
        malicious_pattern = "targeted:victim-ws/*:*"

        # Directly manipulate subscription (simulating vulnerability)
        if malicious_pattern not in event_bus._subscribed_patterns:
            event_bus._subscribed_patterns.add(malicious_pattern)
            if event_bus._pubsub:
                await event_bus._pubsub.psubscribe(malicious_pattern)

        # Register handler
        event_bus.on("victim-ws/victim:msg", malicious_handler)

        # Victim sends a message
        await victim_api.log("Secret victim message")

        await asyncio.sleep(0.5)

        # Check if attacker intercepted
        if intercepted:
            pytest.fail("V-BUS-04: Attacker manipulated patterns to intercept victim messages!")

    finally:
        await attacker_api.disconnect()
        await victim_api.disconnect()


@pytest.mark.asyncio
async def test_readonly_client_broadcast_bypass(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-05: Readonly client broadcast restriction bypass.

    Attack: A readonly client bypasses broadcast restrictions by crafting
    messages with workspace-prefixed targets instead of "*".

    Root Cause: Readonly check in emit_message only blocks "*" but doesn't
    check for workspace/* patterns which also broadcast to all clients.
    """
    from hypha_rpc import connect_to_server

    # Create a readonly connection
    event_bus = fastapi_server.event_bus
    user_info = UserInfo.model_validate({"id": "readonly-user", "email": "readonly@test.com"})

    # Create readonly RPC connection directly
    readonly_conn = RedisRPCConnection(
        event_bus=event_bus,
        workspace="test-ws",
        client_id="readonly-client",
        user_info=user_info,
        manager_id="test-manager",
        readonly=True  # Set readonly flag
    )

    received_broadcasts = []

    async def broadcast_handler(data):
        received_broadcasts.append(data)

    event_bus.on("test-ws/*:msg", broadcast_handler)

    try:
        # Readonly client tries to broadcast using "*" (should fail)
        message_star = {
            "type": "test",
            "to": "*",
            "data": {"msg": "broadcast via star"}
        }
        packed_star = msgpack.packb(message_star)

        try:
            await readonly_conn.emit_message(packed_star)
            pytest.fail("V-BUS-05: Readonly client should not be able to broadcast with '*'")
        except PermissionError:
            pass  # Expected

        # Now try to bypass using workspace/* pattern
        message_ws = {
            "type": "test",
            "to": "test-ws/*",  # Workspace-prefixed broadcast
            "data": {"msg": "broadcast via workspace/*"}
        }
        packed_ws = msgpack.packb(message_ws)

        try:
            await readonly_conn.emit_message(packed_ws)
            await asyncio.sleep(0.3)

            # Check if the broadcast was received
            if received_broadcasts:
                pytest.fail("V-BUS-05: Readonly client bypassed broadcast restriction via workspace/* pattern!")
        except PermissionError:
            pass  # This should be raised but currently might not be

    finally:
        event_bus.off("test-ws/*:msg", broadcast_handler)
        await readonly_conn.disconnect()


@pytest.mark.asyncio
async def test_circuit_breaker_bypass(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-06: Circuit breaker bypass for message delivery.

    Attack: Even when circuit breaker is open (indicating health issues),
    messages can still be delivered through local event bus, potentially
    causing cascading failures.

    Root Cause: Local message delivery bypasses circuit breaker checks,
    allowing operations to continue during degraded state.
    """
    from hypha_rpc import connect_to_server

    event_bus = fastapi_server.event_bus

    # Simulate circuit breaker opening
    event_bus._circuit_breaker_open = True
    event_bus._consecutive_failures = 10

    # Connect clients
    client1_api = await connect_to_server(
        {"name": "client1", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    client2_api = await connect_to_server(
        {"name": "client2", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    messages_received = []

    def message_handler(data):
        messages_received.append(data)

    try:
        # Register handler on event bus
        event_bus.on("test-ws/client2:msg", message_handler)

        # Client1 sends message while circuit breaker is open
        rpc = client1_api._get_rpc()
        message = {
            "type": "test",
            "to": "client2",
            "data": {"msg": "message during circuit breaker open"}
        }
        packed = msgpack.packb(message)

        # This should be blocked or at least logged when circuit breaker is open
        await rpc._connection.emit_message(packed)

        await asyncio.sleep(0.5)

        # Check if message was delivered despite circuit breaker
        if messages_received:
            print("WARNING: V-BUS-06: Messages delivered despite circuit breaker being open!")
            # This is actually expected behavior for local clients, but could cause issues
            # during Redis failures if not properly handled

    finally:
        event_bus.off("test-ws/client2:msg", message_handler)
        event_bus._circuit_breaker_open = False
        event_bus._consecutive_failures = 0
        await client1_api.disconnect()
        await client2_api.disconnect()


@pytest.mark.asyncio
async def test_message_replay_attack(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-07: Message replay attack via event bus.

    Attack: An attacker captures and replays messages from the event bus,
    potentially causing duplicate operations or state corruption.

    Root Cause: Messages don't include timestamps, nonces, or replay
    protection mechanisms.
    """
    from hypha_rpc import connect_to_server

    # Connect clients
    victim_api = await connect_to_server(
        {"name": "victim", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    captured_messages = []
    execution_count = []

    # Service that tracks execution
    def sensitive_operation(x):
        execution_count.append(x)
        return f"executed: {x}"

    await victim_api.register_service({
        "id": "sensitive-service",
        "config": {"visibility": "public"},
        "execute": sensitive_operation
    })

    # Capture messages
    event_bus = fastapi_server.event_bus

    async def capture_handler(data):
        captured_messages.append(data)

    event_bus.on("test-ws/victim:msg", capture_handler)

    try:
        # Execute legitimate operation
        svc = await attacker_api.get_service("sensitive-service")
        await svc.execute("operation-1")

        await asyncio.sleep(0.3)

        # Attacker replays captured message
        if captured_messages:
            for msg in captured_messages:
                # Replay the message
                await event_bus.emit("test-ws/victim:msg", msg)

            await asyncio.sleep(0.3)

            # Check if operation was executed multiple times
            if len(execution_count) > 1:
                pytest.fail(f"V-BUS-07: Message replay caused {len(execution_count)} executions (expected 1)!")

    finally:
        event_bus.off("test-ws/victim:msg", capture_handler)
        await victim_api.disconnect()
        await attacker_api.disconnect()


@pytest.mark.asyncio
async def test_redis_channel_enumeration(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-08: Redis channel enumeration for reconnaissance.

    Attack: An attacker enumerates Redis channels to discover active clients,
    workspaces, and communication patterns.

    Root Cause: Channel naming is predictable and Redis PUBSUB CHANNELS
    command can be used to list all active channels.
    """
    from hypha_rpc import connect_to_server

    # Connect some clients to create channels
    clients = []
    for i in range(3):
        client = await connect_to_server(
            {
                "name": f"client-{i}",
                "server_url": fastapi_server_url,
                "workspace": f"ws-{i}",
                "token": test_user_token
            }
        )
        clients.append(client)

    try:
        # Attacker tries to enumerate channels
        redis = fastapi_server.redis_client

        # List all pubsub channels
        channels = await redis.pubsub_channels()

        # Check if sensitive information is exposed
        channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in channels]

        # Look for patterns that reveal workspace/client structure
        targeted_channels = [ch for ch in channel_names if 'targeted:' in ch or 'broadcast:' in ch]

        if targeted_channels:
            print(f"WARNING: V-BUS-08: Enumerated {len(targeted_channels)} channels revealing system structure:")
            for ch in targeted_channels[:5]:  # Print first 5
                print(f"  - {ch}")

    finally:
        for client in clients:
            await client.disconnect()


@pytest.mark.asyncio
async def test_event_bus_denial_of_service(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-09: Event bus denial of service via pattern flooding.

    Attack: An attacker creates many pattern subscriptions to exhaust
    server resources and slow down message processing.

    Root Cause: No limits on number of patterns a client can subscribe to.
    """
    from hypha_rpc import connect_to_server

    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "test-ws", "token": test_user_token}
    )

    event_bus = fastapi_server.event_bus
    initial_pattern_count = len(event_bus._subscribed_patterns)

    try:
        # Attacker creates many subscriptions
        for i in range(1000):
            # Each subscription adds overhead
            pattern = f"targeted:test-ws/fake-client-{i}:*"
            event_bus._subscribed_patterns.add(pattern)

        final_pattern_count = len(event_bus._subscribed_patterns)
        added_patterns = final_pattern_count - initial_pattern_count

        if added_patterns >= 900:  # Most patterns were added
            print(f"WARNING: V-BUS-09: Added {added_patterns} patterns without limits!")
            print("This could cause DoS through memory exhaustion and slower message routing.")

        # Measure performance impact
        start_time = time.time()
        for i in range(100):
            await event_bus.emit_local(f"test-event-{i}", {"data": "test"})
        elapsed = time.time() - start_time

        if elapsed > 1.0:  # Arbitrary threshold
            print(f"WARNING: V-BUS-09: Performance degraded - 100 events took {elapsed:.2f}s")

    finally:
        # Cleanup
        event_bus._subscribed_patterns.clear()
        await attacker_api.disconnect()


@pytest.mark.asyncio
async def test_workspace_isolation_via_prefix_bypass(
    fastapi_server, fastapi_server_url, test_user_token
):
    """
    V-BUS-10: Workspace isolation bypass via prefix manipulation.

    Attack: An attacker crafts message targets with workspace prefixes
    to bypass workspace isolation and send messages to other workspaces.

    Root Cause: Message routing allows fully-qualified workspace/client_id
    targets without validating the sender's workspace matches the target.
    """
    from hypha_rpc import connect_to_server

    # Connect attacker in workspace A
    attacker_api = await connect_to_server(
        {"name": "attacker", "server_url": fastapi_server_url, "workspace": "workspace-a", "token": test_user_token}
    )

    # Connect victim in workspace B
    victim_api = await connect_to_server(
        {"name": "victim", "server_url": fastapi_server_url, "workspace": "workspace-b", "token": test_user_token}
    )

    messages_received = []

    def victim_handler(data):
        messages_received.append(data)

    event_bus = fastapi_server.event_bus
    event_bus.on("workspace-b/victim:msg", victim_handler)

    try:
        # Attacker tries to send message to victim in different workspace
        rpc = attacker_api._get_rpc()

        # Craft message with cross-workspace target
        message = {
            "type": "malicious",
            "to": "workspace-b/victim",  # Target in different workspace
            "data": {"msg": "Cross-workspace attack"}
        }
        packed = msgpack.packb(message)

        # This should be blocked by workspace isolation
        try:
            await rpc._connection.emit_message(packed)
            await asyncio.sleep(0.5)

            # Check if message crossed workspace boundary
            cross_workspace_messages = [
                msg for msg in messages_received
                if isinstance(msg, bytes) and b"Cross-workspace" in msg
            ]

            if cross_workspace_messages:
                pytest.fail("V-BUS-10: Attacker bypassed workspace isolation via prefix manipulation!")
        except PermissionError:
            pass  # Expected if properly secured

    finally:
        event_bus.off("workspace-b/victim:msg", victim_handler)
        await attacker_api.disconnect()
        await victim_api.disconnect()
