"""Startup-safe orphan-service reaping (#0015).

Regression coverage for the 2026-07-29 hypha-server CrashLoop outage.

Root cause: a server crash leaves thousands of dead WebSocket clients' service
registrations in Redis. Startup used to reap them SYNCHRONOUSLY in the
readiness path (``init()`` awaited ``_cleanup_orphaned_client_services``), which
pinged every orphan SEQUENTIALLY with a 3s timeout. A dead client never
answers, so a ~1700-orphan pile cost ~1700 x 3s ~= 85 min of boot time —
blowing past the 15-min startup probe and CrashLooping the pod, which only made
the pile grow on the next boot.

The fix has two parts, both asserted here (docker-free, fakeredis):
  1. The reap is moved OUT of the readiness path into a post-startup background
     task, so a large pre-existing pile never delays ``init()``.
  2. The reap pings orphans CONCURRENTLY (concurrency-capped), so even one pass
     over a large pile is bounded, not O(N x timeout).
  3. A CONTINUOUS background reaper keeps the pile trimmed so it never grows to
     a restart-choking size.
"""
import asyncio
import time

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _seed_orphan(redis, workspace, client_id, extra=True):
    """Seed a dead client's service keys in Redis (values are never parsed by
    the reap — it scans + deletes by key pattern only)."""
    keys = [f"services:public|built-in:{workspace}/{client_id}:built-in@default"]
    if extra:
        keys.append(
            f"services:public|test:{workspace}/{client_id}:my-service@default"
        )
    for k in keys:
        await redis.set(k, b"{}")
    return keys


async def _orphan_keys(store, workspace, client_id):
    return await store._scan_keys(
        f"services:*|*:{workspace}/{client_id}:*@*"
    )


async def test_reap_removes_dead_clients_concurrently(monkeypatch):
    """A direct reap removes every dead client's service keys, and runs the
    pings CONCURRENTLY so a pile of orphans is bounded, not O(N x timeout)."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_PING_TIMEOUT", "1")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_CONCURRENCY", "50")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        n = 8
        for i in range(n):
            await _seed_orphan(store._redis, "ws-dead", f"deadclient{i}")
        # Sanity: seeded keys exist (2 per client).
        for i in range(n):
            assert await _orphan_keys(store, "ws-dead", f"deadclient{i}")

        t0 = time.perf_counter()
        await store._cleanup_orphaned_client_services()
        elapsed = time.perf_counter() - t0

        # All dead clients' keys (built-in AND the extra service) are gone.
        for i in range(n):
            assert not await _orphan_keys(store, "ws-dead", f"deadclient{i}"), (
                f"deadclient{i} keys should have been reaped"
            )

        # Concurrency proof: sequential would be ~n x 1s = ~8s; concurrent is
        # ~1s. Generous bound to stay non-flaky while still failing the old
        # sequential implementation.
        assert elapsed < 5, (
            f"reap took {elapsed:.1f}s for {n} orphans — expected concurrent "
            f"(~1s), got sequential-like timing"
        )
    finally:
        await store.teardown()


async def test_reap_is_not_in_readiness_path(monkeypatch):
    """A pre-existing orphan pile must NOT delay init(): the reap is deferred to
    a background task. Right after init() returns, the orphans are still present
    (init did not reap them inline) and the background reaper task exists."""
    # Keep the background reaper parked so this test observes the post-init
    # state deterministically (no race with the first background pass).
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)

    # Seed the pile BEFORE init (reset_redis=False → no flush).
    n = 10
    for i in range(n):
        await _seed_orphan(store._redis, "ws-crash", f"orphan{i}")

    t0 = time.perf_counter()
    await store.init(reset_redis=False)
    elapsed = time.perf_counter() - t0
    try:
        # Core reproduction: init did NOT block on the pile. With the old
        # sequential in-readiness reap this would be ~n x 3s = ~30s; deferred it
        # is a normal init.
        assert elapsed < 20, f"init() took {elapsed:.1f}s — reap not deferred?"

        # init did NOT reap inline — orphans survive because the background
        # reaper is still parked (60s initial delay).
        for i in range(n):
            assert await _orphan_keys(store, "ws-crash", f"orphan{i}"), (
                f"orphan{i} was reaped inline — reap is still in the readiness path"
            )

        # The deferred reaper task exists.
        assert store._orphan_reaper_task is not None
    finally:
        await store.teardown()


async def test_background_reaper_removes_pile(monkeypatch):
    """The post-startup background reaper eventually clears an orphan pile
    without any manual invocation."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "0.2")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INTERVAL", "0.5")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_PING_TIMEOUT", "1")
    store = RedisStore(None, redis_uri=None)

    n = 5
    for i in range(n):
        await _seed_orphan(store._redis, "ws-bg", f"bgorphan{i}")
    await store.init(reset_redis=False)
    try:
        # Poll for the background reaper to clear the pile.
        deadline = time.perf_counter() + 15
        cleared = False
        while time.perf_counter() < deadline:
            remaining = 0
            for i in range(n):
                if await _orphan_keys(store, "ws-bg", f"bgorphan{i}"):
                    remaining += 1
            if remaining == 0:
                cleared = True
                break
            await asyncio.sleep(0.5)
        assert cleared, "background reaper did not clear the orphan pile in time"
    finally:
        await store.teardown()


async def test_reaper_task_started_and_cancelled(monkeypatch):
    """The reaper task is created on init and finished (cancelled) on teardown,
    mirroring the malloc_trim task lifecycle."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        assert store._orphan_reaper_task is not None
    finally:
        await store.teardown()
    assert store._orphan_reaper_task.done()
