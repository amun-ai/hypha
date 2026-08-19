"""Issue #1052: the orphan reaper must CONFIRM death before deleting.

`RedisStore._cleanup_orphaned_client_services` (the #0015 continuous reaper)
used to delete ALL of a client's `services:*` keys after a SINGLE failed
cross-pod ping. The cross-pod ping is best-effort (Redis pub/sub has no
buffering), so a single dropped message during a reconnect / subscription-
convergence window made the reaper delete a **live** client's registration —
a silent, permanent outage of that client's services until manual restart.

The fix requires `HYPHA_ORPHAN_REAP_MIN_FAILURES` (default 3) CONSECUTIVE failed
probes, tracked across passes, before an irreversible delete. A client that
answers any probe has its counter reset; a client that disappears from the
candidate set has its counter pruned. This is defense-in-depth on top of the
#1053 subscription-convergence root-cause fix.

Docker-free / fakeredis, reproduce-before-fix: on the old single-ping code
`test_single_dropped_ping_does_not_reap_live_client`'s first assertion (keys
survive one failed pass) fails — the client is reaped immediately.
"""
import asyncio

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _seed_dead_client(redis, workspace, client_id):
    """Seed a client's built-in + a real service key (values are never parsed —
    the reaper scans + deletes by key pattern only). Makes (ws, client_id) a
    reap candidate that no live client answers for."""
    keys = [
        f"services:public|built-in:{workspace}/{client_id}:built-in@default",
        f"services:public|test:{workspace}/{client_id}:user-svc@default",
    ]
    for k in keys:
        await redis.set(k, b"{}")
    return keys


async def _svc_keys(store, workspace, client_id):
    return await store._scan_keys(
        f"services:*|*:{workspace}/{client_id}:*@*"
    )


async def test_single_dropped_ping_does_not_reap_live_client(monkeypatch):
    """One failed probe must NOT delete the client's services (reproduce-before-
    fix: old code reaped on the first failure). Only after MIN_FAILURES
    consecutive failures is the client reaped."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_PING_TIMEOUT", "1")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_MIN_FAILURES", "3")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")  # park bg reaper
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    ws, cid = "ws-live", "live-client"
    key = f"{ws}/{cid}"
    try:
        await _seed_dead_client(store._redis, ws, cid)
        assert await _svc_keys(store, ws, cid)

        # Pass 1: one failed probe — must NOT reap (this is the #1052 bug).
        await store._cleanup_orphaned_client_services()
        assert await _svc_keys(store, ws, cid), (
            "a live client was reaped after a SINGLE failed cross-pod ping (#1052)"
        )
        assert store._orphan_probe_failures.get(key) == 1

        # Pass 2: still below threshold — survives.
        await store._cleanup_orphaned_client_services()
        assert await _svc_keys(store, ws, cid)
        assert store._orphan_probe_failures.get(key) == 2

        # Pass 3: reaches MIN_FAILURES=3 — now (and only now) reaped.
        await store._cleanup_orphaned_client_services()
        assert not await _svc_keys(store, ws, cid), (
            "a genuinely dead client must be reaped after MIN_FAILURES passes"
        )
        # Counter is cleared once reaped.
        assert key not in store._orphan_probe_failures
    finally:
        await store.teardown()


async def test_recovered_client_resets_failure_counter(monkeypatch):
    """A client that answers a probe after previous failures has its counter
    reset, so accumulated transient failures never add up to a reap for a
    client that is intermittently reachable."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_PING_TIMEOUT", "1")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_MIN_FAILURES", "3")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    ws, cid = "ws-recover", "recover-client"
    key = f"{ws}/{cid}"
    await store.register_workspace(
        {
            "id": ws,
            "name": ws,
            "description": "recover test",
            "persistent": True,
            "owners": ["root"],
            "read_only": False,
        },
        overwrite=False,
    )
    try:
        # Phase A: dead — two failed probes accumulate.
        await _seed_dead_client(store._redis, ws, cid)
        await store._cleanup_orphaned_client_services()
        await store._cleanup_orphaned_client_services()
        assert store._orphan_probe_failures.get(key) == 2

        # Phase B: the client comes ALIVE (a real RPC client with the same id,
        # whose built-in service answers ping). One probe now succeeds ->
        # the counter must reset, and the client is not reaped.
        async with store.get_workspace_interface(
            store._root_user, ws, client_id=cid, silent=False
        ):
            await store._cleanup_orphaned_client_services()
            assert key not in store._orphan_probe_failures, (
                "a successful probe must reset the consecutive-failure counter"
            )
            assert await _svc_keys(store, ws, cid), (
                "a live, reachable client must never be reaped"
            )
    finally:
        await store.teardown()


async def test_disappeared_client_counter_is_pruned(monkeypatch):
    """A client that drops out of the candidate set (its keys already gone /
    reconnected away) must have its stale failure counter pruned, so counts
    never leak across unrelated client lifecycles."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_PING_TIMEOUT", "1")
    # High threshold so nothing is reaped during this test — we assert on the
    # counter bookkeeping, not on deletion.
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_MIN_FAILURES", "10")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    ws = "ws-prune"
    a, b = "client-a", "client-b"
    try:
        await _seed_dead_client(store._redis, ws, a)
        b_keys = await _seed_dead_client(store._redis, ws, b)

        # Pass 1: both fail -> both counted.
        await store._cleanup_orphaned_client_services()
        assert store._orphan_probe_failures.get(f"{ws}/{a}") == 1
        assert store._orphan_probe_failures.get(f"{ws}/{b}") == 1

        # b disappears entirely (keys removed by some other path / reconnect).
        for k in b_keys:
            await store._redis.delete(k)

        # Pass 2: only a is a candidate -> b's stale counter is pruned.
        await store._cleanup_orphaned_client_services()
        assert store._orphan_probe_failures.get(f"{ws}/{a}") == 2
        assert f"{ws}/{b}" not in store._orphan_probe_failures, (
            "a client no longer present must have its failure counter pruned"
        )
    finally:
        await store.teardown()
