"""Tests for the Redis-backed LeaderLease single-leader election (F6, Phase 0).

Real tests against a shared in-process fakeredis (no mocks). Two LeaderLease
instances sharing ONE fakeredis client mirror two hypha replicas sharing one
Redis — exactly the topology the primitive exists for.
"""
import asyncio

import pytest
from fakeredis import aioredis as fakeredis

from hypha.core.leader import LeaderLease

pytestmark = pytest.mark.asyncio


def _redis():
    """A fresh shared fakeredis client (acts as the one Redis both replicas use)."""
    return fakeredis.FakeRedis()


async def test_single_instance_acquires_leadership():
    r = _redis()
    lease = LeaderLease(r, "server-a", ttl_ms=2000, renew_interval=0.5)
    assert lease.is_leader() is False  # not before start
    await lease.start()
    try:
        assert lease.is_leader() is True
    finally:
        await lease.stop()
    assert lease.is_leader() is False  # not after stop


async def test_only_one_leader_across_two_instances():
    r = _redis()  # shared Redis
    a = LeaderLease(r, "server-a", ttl_ms=2000, renew_interval=0.5)
    b = LeaderLease(r, "server-b", ttl_ms=2000, renew_interval=0.5)
    await a.start()
    await b.start()
    try:
        leaders = [x for x in (a, b) if x.is_leader()]
        assert len(leaders) == 1, "exactly one instance must be leader"
        assert a.is_leader() and not b.is_leader()  # a started first → a wins
    finally:
        await a.stop()
        await b.stop()


async def test_release_on_stop_lets_other_acquire():
    r = _redis()
    a = LeaderLease(r, "server-a", ttl_ms=2000, renew_interval=0.5)
    await a.start()
    assert a.is_leader()
    # Graceful handover: a releases its key on stop.
    await a.stop()
    assert (await r.get("hypha:leader")) is None  # freed by CAS-release
    # A follower can now take the free key.
    b = LeaderLease(r, "server-b", ttl_ms=2000, renew_interval=0.5)
    assert await b._contend() is True
    assert (await r.get("hypha:leader")) == b"server-b"


async def test_renew_extends_ttl():
    r = _redis()
    lease = LeaderLease(r, "server-a", ttl_ms=3000, renew_interval=1.0)
    assert await lease._acquire() is True
    lease._is_leader = True
    await asyncio.sleep(0.2)
    pttl_before = await r.pttl("hypha:leader")
    assert pttl_before < 3000  # has decayed
    assert await lease._contend() is True
    pttl_after = await r.pttl("hypha:leader")
    assert pttl_after > pttl_before  # extended back toward full ttl
    await lease.stop()


async def test_renew_returns_false_when_not_owner():
    r = _redis()
    a = LeaderLease(r, "server-a", ttl_ms=2000, renew_interval=0.5)
    b = LeaderLease(r, "server-b", ttl_ms=2000, renew_interval=0.5)
    assert await a._acquire() is True
    a._is_leader = True
    # b does not own the key → renew must NOT extend or steal it
    assert await b._contend() is False
    assert b.is_leader() is False
    # a still holds it
    assert (await r.get("hypha:leader")) == b"server-a"
    await a.stop()


async def test_release_does_not_drop_peer_lease():
    r = _redis()
    a = LeaderLease(r, "server-a", ttl_ms=2000, renew_interval=0.5)
    b = LeaderLease(r, "server-b", ttl_ms=2000, renew_interval=0.5)
    assert await a._acquire() is True
    # b never owned it; releasing must be a no-op and leave a's key intact.
    b._is_leader = True  # pretend b mistakenly thinks it leads
    await b._release()
    assert (await r.get("hypha:leader")) == b"server-a"
    await a.stop()


async def test_failover_after_expiry():
    r = _redis()
    a = LeaderLease(r, "server-a", ttl_ms=300, renew_interval=0.1)
    b = LeaderLease(r, "server-b", ttl_ms=300, renew_interval=0.1)
    assert await a._acquire() is True
    a._is_leader = True
    # Simulate a crashing without releasing: stop renewing and let the key lapse.
    await r.pexpire("hypha:leader", 1)
    await asyncio.sleep(0.05)
    assert (await r.get("hypha:leader")) is None  # lease expired
    # b acquires on its next attempt
    assert await b._acquire() is True
    b._is_leader = True
    assert b.is_leader() is True
    await b.stop()


async def test_failover_via_loop_after_leader_stops():
    """End-to-end: leader stops, follower's renew loop promotes it within a tick."""
    r = _redis()
    a = LeaderLease(r, "server-a", ttl_ms=400, renew_interval=0.1)
    b = LeaderLease(r, "server-b", ttl_ms=400, renew_interval=0.1)
    await a.start()
    await b.start()
    assert a.is_leader() and not b.is_leader()
    await a.stop()  # releases the key
    # b's loop should acquire within a couple of renew intervals
    for _ in range(20):
        await asyncio.sleep(0.05)
        if b.is_leader():
            break
    assert b.is_leader() is True, "follower must take leadership after leader stops"
    await b.stop()


async def test_ttl_must_exceed_two_renew_intervals():
    r = _redis()
    with pytest.raises(ValueError):
        LeaderLease(r, "server-a", ttl_ms=1000, renew_interval=1.0)  # 1000 <= 2000


async def test_custom_key_isolates_leases():
    r = _redis()
    a = LeaderLease(r, "server-a", key="lease:alpha", ttl_ms=2000, renew_interval=0.5)
    b = LeaderLease(r, "server-b", key="lease:beta", ttl_ms=2000, renew_interval=0.5)
    await a.start()
    await b.start()
    try:
        # Different keys → both can be leaders of their own lease.
        assert a.is_leader() and b.is_leader()
    finally:
        await a.stop()
        await b.stop()
