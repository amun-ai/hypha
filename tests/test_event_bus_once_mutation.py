"""Tests for EventBus.emit() mutation-during-iteration bug fix.

Root cause: EventBus.emit() iterated over self._callbacks[event_name] directly.
A once() handler calls off() to remove itself during the loop, which mutates the
underlying list. Python's list iterator then skips the element at the now-shifted
index, silently dropping every handler registered after the once() handler.

Fix: iterate over list(self._callbacks.get(event_name, [])) — a snapshot copy.
"""
import asyncio
import pytest

pytestmark = pytest.mark.asyncio


async def test_once_then_on_both_fire():
    """once() handler followed by on() handler: BOTH must be called.

    Before the fix, off() in once_wrapper mutated the list mid-iteration
    so the on() handler at the next index was silently skipped.
    """
    from hypha.utils import EventBus
    bus = EventBus()
    results = []

    bus.once("evt", lambda d: results.append("once"))
    bus.on("evt", lambda d: results.append("regular"))

    await bus.emit("evt", "payload")

    assert results == ["once", "regular"], (
        f"Expected ['once', 'regular'] but got {results}. "
        "The on() handler was silently skipped due to list mutation during iteration."
    )


async def test_two_once_handlers_both_fire():
    """Two consecutive once() registrations: both must fire on first emit."""
    from hypha.utils import EventBus
    bus = EventBus()
    results = []

    bus.once("evt", lambda d: results.append("once1"))
    bus.once("evt", lambda d: results.append("once2"))

    await bus.emit("evt", "payload")

    assert results == ["once1", "once2"], (
        f"Expected ['once1', 'once2'] but got {results}. "
        "Second once() handler was skipped due to list mutation."
    )


async def test_once_only_fires_once_after_fix():
    """once() must still only fire once — fix must not break that invariant."""
    from hypha.utils import EventBus
    bus = EventBus()
    results = []

    bus.once("evt", lambda d: results.append("once"))
    bus.on("evt", lambda d: results.append("regular"))

    await bus.emit("evt", "first")
    await bus.emit("evt", "second")

    # once fired on first emit; regular fired on both
    assert results == ["once", "regular", "regular"], (
        f"Expected ['once', 'regular', 'regular'] but got {results}."
    )


async def test_on_before_once_always_worked():
    """Regression: on() before once() already worked before fix — must still work."""
    from hypha.utils import EventBus
    bus = EventBus()
    results = []

    bus.on("evt", lambda d: results.append("regular"))
    bus.once("evt", lambda d: results.append("once"))

    await bus.emit("evt", "payload")

    assert results == ["regular", "once"], (
        f"Expected ['regular', 'once'] but got {results}."
    )


async def test_async_once_handler_does_not_skip_subsequent():
    """Async once() handler: subsequent on() handlers must still fire."""
    from hypha.utils import EventBus
    bus = EventBus()
    results = []

    async def async_once(data):
        results.append("async_once")

    bus.once("evt", async_once)
    bus.on("evt", lambda d: results.append("sync_regular"))

    await bus.emit("evt", "payload")

    assert "async_once" in results, "Async once() handler did not fire"
    assert "sync_regular" in results, (
        "sync_regular handler was skipped after async once() handler"
    )


async def test_redis_event_bus_once_both_fire():
    """RedisEventBus.once() also uses EventBus under the hood — verify fix propagates."""
    from fakeredis.aioredis import FakeRedis
    from hypha.core.store import RedisEventBus

    redis = FakeRedis()
    bus = RedisEventBus(redis=redis)
    await bus.init()

    results = []

    bus.once("evt", lambda d: results.append("once"))
    bus.on("evt", lambda d: results.append("regular"))

    await bus._redis_event_bus.emit("evt", "payload")

    assert results == ["once", "regular"], (
        f"RedisEventBus: expected ['once', 'regular'] but got {results}."
    )

    await bus.stop()
