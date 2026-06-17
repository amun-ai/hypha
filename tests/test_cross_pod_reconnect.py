"""F6 multi-replica correctness: a client re-pinning to a sibling pod must not be
false-rejected by the old pod's per-pod recently-disconnected cache.

Regression test for the N>=2 prod incident (2026-06-17): the dead-peer check in
RedisRPCConnection rejected messages to a peer that had moved to another pod,
because the old pod kept it in `_recently_disconnected` (per-pod, in-memory,
120s TTL) for the full TTL even though the peer was alive on a sibling pod. The
fix broadcasts on (re)connect so every pod clears the stale entry.

Docker-free: two RedisEventBus instances share one fakeredis (= two pods on one
Redis), mirroring tests/test_event_bus_optimization.py.
"""
import asyncio

import pytest
from fakeredis import aioredis as fakeredis

from hypha.core import RedisEventBus

pytestmark = pytest.mark.asyncio


async def _make_bus(redis):
    bus = RedisEventBus(redis)
    await bus.init()
    return bus


async def test_reconnect_broadcast_clears_sibling_recently_disconnected():
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/1")
    pod_a = await _make_bus(redis)
    pod_b = await _make_bus(redis)
    try:
        # Pod A saw client X's WS drop -> tracks it as recently disconnected.
        pod_a.track_recently_disconnected("ws1", "X")
        assert pod_a.is_recently_disconnected("ws1", "X")

        # Client X re-pins to pod B -> pod B announces the (re)connection.
        await pod_b.notify_client_connected("ws1", "X")

        # The broadcast propagates over Redis -> pod A clears its stale entry.
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not pod_a.is_recently_disconnected("ws1", "X"):
                break
        assert not pod_a.is_recently_disconnected("ws1", "X"), (
            "old pod must clear recently-disconnected after the client re-pins "
            "to a sibling pod (otherwise it false-rejects messages to a live peer)"
        )
    finally:
        await pod_a.stop()
        await pod_b.stop()


async def test_genuine_disconnect_stays_cached_for_fast_fail():
    """A client that does NOT reconnect anywhere must stay cached, so the
    dead-peer fast-fail still works. A different client's reconnect must not
    spuriously clear it."""
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/2")
    pod_a = await _make_bus(redis)
    pod_b = await _make_bus(redis)
    try:
        pod_a.track_recently_disconnected("ws1", "gone")
        # A DIFFERENT client reconnects elsewhere — must not clear 'gone'.
        await pod_b.notify_client_connected("ws1", "other")
        await asyncio.sleep(0.3)
        assert pod_a.is_recently_disconnected("ws1", "gone"), (
            "an unrelated reconnect must not clear a genuinely-gone peer"
        )
    finally:
        await pod_a.stop()
        await pod_b.stop()


async def test_local_handler_clears_own_cache():
    """notify_client_connected also clears the announcing pod's own cache."""
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/3")
    pod = await _make_bus(redis)
    try:
        pod.track_recently_disconnected("ws1", "Z")
        assert pod.is_recently_disconnected("ws1", "Z")
        await pod.notify_client_connected("ws1", "Z")
        assert not pod.is_recently_disconnected("ws1", "Z")
    finally:
        await pod.stop()
