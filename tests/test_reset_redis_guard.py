"""F6/P0: reset-redis must not wipe a live multi-replica cluster.

Docker-free: a RedisStore on fakeredis. `_maybe_reset_redis` flushes only when
reset is requested AND no other hypha server is already registered (peers are
detected via their public built-in service keys, as list_servers() does).
"""
import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def test_reset_flushes_when_no_other_servers():
    store = RedisStore(None, redis_uri=None)
    r = store.get_redis()
    await r.set("some-key", b"v")

    await store._maybe_reset_redis(True)

    assert await r.get("some-key") is None  # flushed (first/sole server)


async def test_reset_skipped_when_a_peer_server_is_registered():
    store = RedisStore(None, redis_uri=None)
    r = store.get_redis()
    # A peer server's public built-in service key (what list_servers scans for).
    await r.hset(
        "services:public|x:public/server-peer:built-in@app", "id", b"x"
    )
    await r.set("some-key", b"v")

    await store._maybe_reset_redis(True)

    # The cluster's state must be preserved — no flush while a peer is alive.
    assert await r.get("some-key") == b"v"
    assert await r.hexists(
        "services:public|x:public/server-peer:built-in@app", "id"
    )


async def test_no_reset_when_not_requested():
    store = RedisStore(None, redis_uri=None)
    r = store.get_redis()
    await r.set("some-key", b"v")

    await store._maybe_reset_redis(False)

    assert await r.get("some-key") == b"v"  # untouched
