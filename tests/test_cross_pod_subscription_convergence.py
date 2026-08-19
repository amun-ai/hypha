"""Issue #1053: cross-pod RPC subscription-convergence race.

When a provider pod subscribes to a client's targeted events, it psubscribes the
Redis pattern ``targeted:<ws>/<client_id>:*``. The old code recorded the pattern
in ``_subscribed_patterns`` *optimistically* — BEFORE the psubscribe was
confirmed — and on a psubscribe timeout/error left the pattern recorded but NOT
wired to Redis. The ``if pattern not in self._subscribed_patterns`` guard then
turned every later subscribe call into a no-op, so the pattern was a **permanent
black hole**: cross-pod targeted RPC to that client was silently dropped forever
(masked for same-pod callers by the local short-circuit in ``emit``).

This reproduces the failure with two ``RedisEventBus`` instances sharing one
fakeredis (= two pods on one Redis), mirroring
``tests/test_cross_pod_reconnect.py`` — no Docker, deterministic.
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


async def _deliver(sender_bus, receiver_bus, ws, client_id, timeout=1.5):
    """Emit a targeted message from ``sender_bus`` and return the payload if it
    is delivered to a handler on ``receiver_bus`` within ``timeout`` (else None).

    ``sender_bus`` must NOT have the target client registered locally, otherwise
    ``emit`` short-circuits to a local delivery and never hits Redis pub/sub.
    """
    event = f"{ws}/{client_id}:msg"
    payload = {"hello": client_id}
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def handler(data):
        if not fut.done():
            fut.set_result(data)

    receiver_bus.on(event, handler)
    try:
        deadline = loop.time() + timeout
        while loop.time() < deadline and not fut.done():
            res = sender_bus.emit(event, payload)
            if asyncio.iscoroutine(res):
                await res
            try:
                return await asyncio.wait_for(asyncio.shield(fut), timeout=0.15)
            except asyncio.TimeoutError:
                continue
        return fut.result() if fut.done() else None
    finally:
        receiver_bus.off(event, handler)


def _break_psubscribe_for(bus, pattern):
    """Make ``bus``'s pubsub raise TimeoutError for exactly ``pattern`` (as the
    real ``asyncio.wait_for(psubscribe(...), 5.0)`` timeout would), reproducing
    the recorded-but-unwired state. Returns the original psubscribe to restore.
    """
    orig = bus._pubsub.psubscribe

    async def flaky(*channels, **kwargs):
        if channels and channels[0] == pattern:
            raise asyncio.TimeoutError()
        return await orig(*channels, **kwargs)

    bus._pubsub.psubscribe = flaky
    return orig


async def test_targeted_subscribe_timeout_is_reconciled_via_retry():
    """A psubscribe timeout at register time must NOT permanently black-hole the
    client: once Redis recovers, a subsequent subscribe call must re-wire the
    pattern and cross-pod delivery must succeed.

    Reproduce-before-fix: on old code the second subscribe is a no-op (guard on
    the desired set) so delivery stays broken -> this test fails.
    """
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/11")
    pod_a = await _make_bus(redis)  # provider pod
    pod_b = await _make_bus(redis)  # caller pod
    ws, client_id = "wsA", "clientA"
    pattern = f"targeted:{ws}/{client_id}:*"
    try:
        # Register clientA as local on pod_a so the receive path is realistic,
        # but keep pod_b unaware of it (forces pod_b -> Redis pub/sub).
        pod_a.register_local_client_sync(ws, client_id)

        # 1) psubscribe times out at register time -> recorded but not wired.
        restore = _break_psubscribe_for(pod_a, pattern)
        await pod_a.subscribe_to_client_events(ws, client_id)
        assert pattern in pod_a._subscribed_patterns  # desired recorded
        assert pattern not in pod_a._confirmed_patterns  # but not wired

        # Cross-pod delivery must fail while the pattern is unwired.
        assert await _deliver(pod_b, pod_a, ws, client_id, timeout=0.8) is None

        # 2) Redis recovers; a subsequent subscribe call must re-wire it.
        pod_a._pubsub.psubscribe = restore
        await pod_a.subscribe_to_client_events(ws, client_id)
        assert pattern in pod_a._confirmed_patterns

        # Now cross-pod delivery works.
        got = await _deliver(pod_b, pod_a, ws, client_id, timeout=2.0)
        assert got == {"hello": client_id}, "cross-pod targeted delivery must converge"
    finally:
        await pod_a.stop()
        await pod_b.stop()


async def test_reconcile_loop_rewires_without_a_second_subscribe_call():
    """subscribe_to_client_events is called ONCE per client (at register), so a
    register-time timeout needs an active reconciler to converge — not another
    subscribe call. The background reconcile must re-wire an unconfirmed desired
    pattern once Redis recovers."""
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/12")
    pod_a = await _make_bus(redis)
    pod_b = await _make_bus(redis)
    ws, client_id = "wsA", "clientB"
    pattern = f"targeted:{ws}/{client_id}:*"
    try:
        pod_a.register_local_client_sync(ws, client_id)
        restore = _break_psubscribe_for(pod_a, pattern)
        await pod_a.subscribe_to_client_events(ws, client_id)
        assert pattern not in pod_a._confirmed_patterns

        # Redis recovers; the ONLY convergence mechanism now is the reconciler.
        pod_a._pubsub.psubscribe = restore
        await pod_a._reconcile_subscriptions()
        assert pattern in pod_a._confirmed_patterns

        got = await _deliver(pod_b, pod_a, ws, client_id, timeout=2.0)
        assert got == {"hello": client_id}
    finally:
        await pod_a.stop()
        await pod_b.stop()


async def test_rewire_on_reconnect_preserves_desired_and_never_discards():
    """On a pubsub reconnect, a transiently-failing psubscribe for one desired
    pattern must NOT discard it (the old code did ``discard`` on failure ->
    permanent cross-pod loss). Every desired pattern must survive; the healthy
    one is confirmed and the flaky one is left unconfirmed for the reconciler."""
    redis = fakeredis.FakeRedis.from_url("redis://localhost:9997/13")
    pod_a = await _make_bus(redis)
    pod_b = await _make_bus(redis)
    ws = "wsA"
    good = "clientGood"
    flaky_client = "clientFlaky"
    good_pat = f"targeted:{ws}/{good}:*"
    flaky_pat = f"targeted:{ws}/{flaky_client}:*"
    try:
        pod_a.register_local_client_sync(ws, good)
        pod_a.register_local_client_sync(ws, flaky_client)
        await pod_a.subscribe_to_client_events(ws, good)
        await pod_a.subscribe_to_client_events(ws, flaky_client)
        assert {good_pat, flaky_pat} <= pod_a._confirmed_patterns

        # Simulate a reconnect: swap in a fresh pubsub where re-subscribing the
        # flaky client's pattern fails, then run the re-wire step directly.
        fresh = redis.pubsub()
        pod_a._pubsub = fresh
        restore = _break_psubscribe_for(pod_a, flaky_pat)
        await pod_a._rewire_desired_patterns()

        # Neither desired pattern is discarded; the good one is re-wired now,
        # the flaky one is left for the reconciler.
        assert {good_pat, flaky_pat} <= pod_a._subscribed_patterns, (
            "desired patterns must survive a re-subscribe failure"
        )
        assert good_pat in pod_a._confirmed_patterns
        assert flaky_pat not in pod_a._confirmed_patterns

        # Redis recovers -> the reconciler converges the flaky one too.
        pod_a._pubsub.psubscribe = restore
        await pod_a._reconcile_subscriptions()
        assert flaky_pat in pod_a._confirmed_patterns

        got = await _deliver(pod_b, pod_a, ws, good, timeout=2.0)
        assert got == {"hello": good}
    finally:
        await pod_a.stop()
        await pod_b.stop()
