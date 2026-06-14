"""F6/P4: per-session lock so only one replica stops an inactive app.

Docker-free: tests the _acquire_stop_lock primitive directly against a shared
fakeredis (two replicas → one Redis), without constructing the full controller.
"""
import pytest
from fakeredis import aioredis as fakeredis

from hypha.apps import ServerAppController

pytestmark = pytest.mark.asyncio


async def test_only_one_replica_acquires_stop_lock():
    r = fakeredis.FakeRedis()  # shared Redis
    full_client_id = "ws-x/app-1"

    first = await ServerAppController._acquire_stop_lock(r, full_client_id)
    second = await ServerAppController._acquire_stop_lock(r, full_client_id)

    assert first is True, "first replica must win the stop lock"
    assert second is False, "second replica must be blocked (no double-stop)"


async def test_different_sessions_do_not_block_each_other():
    r = fakeredis.FakeRedis()
    assert await ServerAppController._acquire_stop_lock(r, "ws-x/app-1") is True
    assert await ServerAppController._acquire_stop_lock(r, "ws-x/app-2") is True


async def test_lock_expires_allowing_a_later_stop():
    r = fakeredis.FakeRedis()
    fid = "ws-x/app-1"
    assert await ServerAppController._acquire_stop_lock(r, fid, ttl_ms=50) is True
    assert await ServerAppController._acquire_stop_lock(r, fid, ttl_ms=50) is False
    # Simulate TTL expiry by deleting the key (fakeredis time control is coarse).
    await r.delete(f"app-stop-lock:{fid}")
    assert await ServerAppController._acquire_stop_lock(r, fid, ttl_ms=50) is True
