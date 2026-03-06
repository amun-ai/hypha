"""Test that unsubscribe_from_client_events uses exact match, not substring match.

Bug: `if client_key in pattern` uses Python substring containment, so disconnecting
client "ws/client" also removes subscriptions for "ws/client-test", "ws/client2", etc.
Pattern format: "targeted:{workspace}/{client_id}:*"

Fix: use `pattern == f"targeted:{client_key}:*"` for exact match.
"""
import asyncio
import pytest
import pytest_asyncio
from hypha.core import RedisEventBus

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def event_bus():
    from fakeredis import aioredis
    redis = aioredis.FakeRedis()
    bus = RedisEventBus(redis)
    await bus.init()
    yield bus
    await bus.stop()


async def test_unsubscribe_does_not_remove_prefix_match(event_bus):
    """Disconnecting 'ws/client' must NOT remove subscription for 'ws/client-test'.

    With the bug:
        client_key = "ws/client"
        pattern    = "targeted:ws/client-test:*"
        "ws/client" in "targeted:ws/client-test:*"  =>  True  (WRONG)
    """
    ws = "ws"
    # Subscribe both clients
    await event_bus.subscribe_to_client_events(ws, "client")
    await event_bus.subscribe_to_client_events(ws, "client-test")

    assert f"targeted:{ws}/client:*" in event_bus._subscribed_patterns
    assert f"targeted:{ws}/client-test:*" in event_bus._subscribed_patterns

    # Disconnect the shorter-named client
    await event_bus.unsubscribe_from_client_events(ws, "client")

    # Shorter client's subscription must be gone
    assert f"targeted:{ws}/client:*" not in event_bus._subscribed_patterns, (
        "subscription for 'ws/client' should have been removed"
    )
    # Longer client's subscription must survive
    assert f"targeted:{ws}/client-test:*" in event_bus._subscribed_patterns, (
        "unsubscribing 'ws/client' must NOT remove 'ws/client-test' subscription"
    )


async def test_unsubscribe_does_not_remove_numeric_suffix(event_bus):
    """Disconnecting 'ws/c1' must NOT remove subscription for 'ws/c10'."""
    ws = "workspace"
    await event_bus.subscribe_to_client_events(ws, "c1")
    await event_bus.subscribe_to_client_events(ws, "c10")
    await event_bus.subscribe_to_client_events(ws, "c100")

    await event_bus.unsubscribe_from_client_events(ws, "c1")

    assert f"targeted:{ws}/c1:*" not in event_bus._subscribed_patterns
    assert f"targeted:{ws}/c10:*" in event_bus._subscribed_patterns, (
        "c10 subscription must survive c1 disconnect"
    )
    assert f"targeted:{ws}/c100:*" in event_bus._subscribed_patterns, (
        "c100 subscription must survive c1 disconnect"
    )


async def test_unsubscribe_removes_only_exact_pattern(event_bus):
    """Only the exact pattern for the disconnecting client is removed."""
    ws = "myws"
    clients = ["alice", "alice-2", "alice-bot", "bob"]
    for c in clients:
        await event_bus.subscribe_to_client_events(ws, c)

    before = set(event_bus._subscribed_patterns)
    await event_bus.unsubscribe_from_client_events(ws, "alice")

    after = set(event_bus._subscribed_patterns)
    removed = before - after
    expected_removed = {f"targeted:{ws}/alice:*"}
    assert removed == expected_removed, (
        f"Expected only {expected_removed} to be removed, but got {removed}"
    )
