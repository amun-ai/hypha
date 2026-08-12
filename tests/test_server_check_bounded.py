"""Startup ``check_and_cleanup_servers`` must be BOUNDED and CONCURRENT (Task A).

Sibling of #0015 (``test_orphan_reaper.py``). #0015 fixed the *orphaned client*
reap (moved it off the readiness path + made it concurrent). But the
*server-check* phase — ``RedisStore.check_and_cleanup_servers`` — was left on the
readiness path (``init()`` awaits it at ``store.py`` ~1026) and still had the
SAME O(N x timeout) shape that CrashLooped the pod:

  * ``list_servers()`` scans ``services:*|*:public/*:built-in@*`` — that is the
    built-in service of EVERY public-workspace client, not just live hypha
    instances. After a crash/rollout a pile of DEAD public built-in
    registrations survives in a non-reset (prod) Redis.
  * The old loop pinged each one SEQUENTIALLY; a dead client never answers, so
    ``get_remote_service(..., {"timeout": 2})`` burned the full timeout for each,
    and the ``svc.ping("ping")`` call itself had NO explicit timeout at all —
    a half-open peer that resolves but never replies could hang boot forever.

So N dead server registrations added O(N x 2s) — or unbounded — to the readiness
path, exactly the #0015 failure mode on a different phase.

The fix (asserted here, docker-free, fakeredis):
  1. Probe servers CONCURRENTLY (concurrency-capped), so one pass over a pile is
     ~ceil(N/cap) x timeout, not N x timeout.
  2. Bound EACH probe: both the ``get_remote_service`` resolution AND the
     ``svc.ping`` are given an explicit timeout (``HYPHA_SERVER_CHECK_TIMEOUT``).
  3. Bound the WHOLE phase with an overall deadline
     (``HYPHA_SERVER_CHECK_DEADLINE``); on deadline, log + continue rather than
     block readiness.

These are real tests: they seed genuine dead built-in registrations in fakeredis
and boot a real ``RedisStore`` WITHOUT resetting Redis (the prod-like condition —
a reset boot flushes the pile and can never reproduce it), then assert boot is
prompt and the dead registrations are cleaned.
"""

import asyncio
import time

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _seed_dead_server(redis, server_id):
    """Seed a dead server's public built-in registration + a couple of its
    services, exactly the shape ``list_servers()`` enumerates and
    ``check_and_cleanup_servers`` cleans. Values are never parsed (scan+delete by
    key pattern), so an empty JSON blob is sufficient."""
    keys = [
        f"services:public|built-in:public/{server_id}:built-in@default",
        f"services:public|test:public/{server_id}:svc-a@default",
        f"services:public|test:public/{server_id}-worker1:svc-b@default",
    ]
    for k in keys:
        await redis.set(k, b"{}")
    return keys


async def _keys_for_server(store, server_id):
    return await store._scan_keys(f"services:*|*:public/{server_id}:*@*")


async def test_dead_server_pile_does_not_block_boot_and_is_cleaned(monkeypatch):
    """A pile of dead server registrations in a non-reset Redis must NOT make
    boot O(N x timeout), and must be cleaned during boot.

    Old (serial, hardcoded 2s resolution timeout): N=12 dead servers add ~24s to
    init. New (concurrent, honoring HYPHA_SERVER_CHECK_TIMEOUT=1): ~1s.
    """
    monkeypatch.setenv("HYPHA_SERVER_CHECK_TIMEOUT", "1")
    monkeypatch.setenv("HYPHA_SERVER_CHECK_CONCURRENCY", "50")
    # Park the background orphan reaper so it can't interfere with timing.
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")

    store = RedisStore(None, redis_uri=None)

    n = 12
    for i in range(n):
        await _seed_dead_server(store._redis, f"deadsrv{i}")
    for i in range(n):
        assert await _keys_for_server(store, f"deadsrv{i}"), "seed failed"

    t0 = time.perf_counter()
    await store.init(reset_redis=False)
    elapsed = time.perf_counter() - t0
    try:
        # Boot was not serialized over the dead pile. Serial-old would be
        # ~n x 2s = ~24s on top of base init; concurrent is ~1s. Generous bound
        # that still fails the old sequential implementation.
        assert elapsed < 12, (
            f"init() took {elapsed:.1f}s for {n} dead servers — server-check "
            f"phase not concurrent/bounded (expected ~base+1s)"
        )

        # The dead servers' service keys were cleaned during boot.
        for i in range(n):
            assert not await _keys_for_server(store, f"deadsrv{i}"), (
                f"deadsrv{i} keys should have been cleaned by check_and_cleanup_servers"
            )
    finally:
        await store.teardown()


async def test_server_check_overall_deadline_does_not_hang_boot(monkeypatch):
    """Even if per-probe timeouts are large, the whole server-check phase is
    bounded by an overall deadline — boot proceeds (log + continue) instead of
    blocking on a slow/half-open pile.

    Per-probe timeout 30s but deadline 2s: the phase must return in ~2s, not 30s.
    """
    monkeypatch.setenv("HYPHA_SERVER_CHECK_TIMEOUT", "30")
    monkeypatch.setenv("HYPHA_SERVER_CHECK_DEADLINE", "2")
    monkeypatch.setenv("HYPHA_SERVER_CHECK_CONCURRENCY", "50")
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")

    store = RedisStore(None, redis_uri=None)

    n = 5
    for i in range(n):
        await _seed_dead_server(store._redis, f"slowsrv{i}")

    t0 = time.perf_counter()
    await store.init(reset_redis=False)
    elapsed = time.perf_counter() - t0
    try:
        # Deadline (2s) cuts the phase well before the 30s per-probe timeout.
        assert elapsed < 12, (
            f"init() took {elapsed:.1f}s — overall server-check deadline did not "
            f"bound the phase (per-probe timeout was 30s)"
        )
    finally:
        await store.teardown()


async def test_clean_boot_unaffected(monkeypatch):
    """Guard: with no dead pile, a normal boot still succeeds and registers this
    server (the hardening must not regress the happy path)."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        servers = await store.list_servers()
        assert store._server_id in servers, (
            f"this server should be registered after boot, got {servers}"
        )
    finally:
        await store.teardown()
