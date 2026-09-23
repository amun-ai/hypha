"""Task #63 — the default ``hypha-login`` service must not be permanently
orphaned by a LIVE-HANDOFF race during a RollingUpdate.

Production symptom (kth-k8s, 09-23, pod wheat-accordion-70572137; cured only by a
manual rollout restart):

1. Rolling restart, ``maxSurge=1``: the NEW pod boots WHILE the login-owning OLD
   pod is still reachable.
2. The new pod's boot check (``RedisStore.init`` -> the login idempotency block in
   ``hypha/core/store.py``) resolves the old pod's ``hypha-login``, pings its
   owner, gets a ``pong``, logs ``Login service already registered and reachable
   (owner=<oldserver>)`` and SKIPS registering the default login.
3. The old pod terminates; its graceful shutdown ``_clear_all_server_services``
   deletes the ``hypha-login`` key it owned.
4. The new pod NEVER retries (registration was a one-shot at boot) -> ZERO
   ``hypha-login`` registrations in Redis -> ``GET /public/services/hypha-login``
   404s while the server is fully healthy.

This is DISTINCT from #0042 (``test_login_registration_stale_marker``): there the
skipped-over owner was a DEAD previous generation (a stale marker); the fix was to
prove liveness by pinging before trusting the marker. Here the skipped-over owner
was GENUINELY LIVE at boot and only died afterwards — a classic register-once
TOCTOU: the boot decision was correct at boot and invalidated later, and nothing
re-evaluates it.

Fix (task #63): a leader-gated periodic login guard
(``_login_guard_loop`` -> ``_ensure_login_service_registered``) re-runs the exact
boot check on an interval, so after any orphaning event a live server re-registers
the default login within one interval (bounded self-heal instead of a permanent
outage). ``_ensure_login_service_registered`` is the boot logic extracted so boot
and the guard share one implementation.

Real test (no mocks): two real ``RedisStore`` instances share one in-process
fakeredis (db 11) — a faithful two-pod simulation, exactly as
``test_cross_pod_reconnect`` relies on. Server A boots and registers login; server
B boots (prod-like, ``reset_redis=False``), defers to live A over the shared event
bus; A tears down and clears its login; we assert the orphan is reproduced and
then that the guard re-registers a live login owned by B.
"""

import asyncio

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _login_owners(store):
    """Return the set of client ids that own a public hypha-login registration."""
    keys = await store._scan_keys("services:*|*:public/*:hypha-login@*")
    owners = set()
    for k in keys:
        ks = k.decode("utf-8") if isinstance(k, bytes) else k
        # services:public|functions:public/<client_id>:hypha-login@*
        wsclient = ks.split("|", 1)[1].split(":", 1)[1]
        client_id = wsclient.split(":", 1)[0].split("/", 1)[1]
        owners.add(client_id)
    return owners


async def test_live_handoff_reregisters_login(monkeypatch):
    """Reproduce the live-handoff orphan, then prove the guard re-registers.

    Before the fix: after the live owner (A) tears down, ``hypha-login`` has ZERO
    owners and the deferring pod (B) never retries -> permanent 404. There is also
    no ``_ensure_login_service_registered`` to call.

    After the fix: a guard tick on B re-registers a live login owned by B, and
    ``start_login`` works end-to-end.
    """
    # Park the deferred background orphan reaper so it cannot mask the behavior by
    # reaping A's cleared services out from under us; the guard must stand alone.
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")
    # Keep the periodic guard from firing on its own during this deterministic
    # test — we drive a single guard tick by hand.
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")

    store_a = RedisStore(None, redis_uri=None)
    await store_a.init(reset_redis=True)  # A registers the default login.

    store_b = RedisStore(None, redis_uri=None)
    try:
        owners = await _login_owners(store_a)
        assert owners == {store_a._server_id}, (
            f"A should own exactly one login after boot, got {owners}"
        )

        # B boots prod-like (no reset) while A is still live. B must resolve A's
        # login over the shared event bus, ping A (pong), and DEFER — exactly the
        # production 'already registered and reachable (owner=A)' path.
        await store_b.init(reset_redis=False)
        owners = await _login_owners(store_b)
        assert store_b._server_id not in owners, (
            "B should have DEFERRED to live A, not registered its own login; "
            f"owners={owners}"
        )
        assert store_a._server_id in owners, (
            f"A's login should still be the sole registration, got {owners}"
        )

        # A terminates: graceful shutdown clears the services it owns, including
        # the login key. This is the orphaning event.
        await store_a.teardown()
        store_a = None

        # REPRODUCE THE BUG: login is now orphaned and nothing has re-registered.
        owners = await _login_owners(store_b)
        assert owners == set(), (
            "expected the live-handoff orphan (zero login owners after the live "
            f"owner tore down), got {owners} — the repro premise is wrong"
        )

        # THE FIX: a guard tick on the surviving server re-registers a live login.
        await store_b._ensure_login_service_registered()
        owners = await _login_owners(store_b)
        assert owners == {store_b._server_id}, (
            "the login guard did not re-register a live login owned by the "
            f"surviving server; owners={owners}"
        )

        # End-to-end: the re-registered login actually answers.
        api = await store_b.get_public_api()
        login_svc = await asyncio.wait_for(
            api.get_service("public/hypha-login", {"mode": "native:random"}),
            timeout=10,
        )
        result = await asyncio.wait_for(login_svc.start(), timeout=10)
        assert "login_url" in result, f"start_login returned unexpectedly: {result}"
    finally:
        if store_a is not None:
            await store_a.teardown()
        await store_b.teardown()


async def test_login_guard_loop_reregisters_when_leader(monkeypatch):
    """The periodic guard loop (not a hand-driven tick) re-registers after an
    orphaning event when this server is the leader."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "1")

    store_a = RedisStore(None, redis_uri=None)
    await store_a.init(reset_redis=True)

    store_b = RedisStore(None, redis_uri=None)
    await store_b.init(reset_redis=False)
    # Force B to consider itself leader so the guard is not gated out while we
    # wait on the loop (leader failover timing is exercised by LeaderLease's own
    # tests; here we test the guard action).
    monkeypatch.setattr(store_b, "is_leader", lambda: True)
    try:
        await store_a.teardown()
        store_a = None
        assert await _login_owners(store_b) == set(), "repro premise wrong"

        # Wait for the guard loop to fire (interval=1s). Poll up to ~8s.
        for _ in range(40):
            await asyncio.sleep(0.25)
            if await _login_owners(store_b) == {store_b._server_id}:
                break
        assert await _login_owners(store_b) == {store_b._server_id}, (
            "the periodic login guard loop did not re-register login within the "
            "interval while leader"
        )
    finally:
        if store_a is not None:
            await store_a.teardown()
        await store_b.teardown()


async def test_login_guard_is_leader_gated(monkeypatch):
    """A NON-leader must NOT re-register (prevents duplicate hypha-login keys /
    thundering-herd registration churn across replicas)."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "1")

    store_a = RedisStore(None, redis_uri=None)
    await store_a.init(reset_redis=True)

    store_b = RedisStore(None, redis_uri=None)
    await store_b.init(reset_redis=False)
    # Force B to be a non-leader for the whole test.
    monkeypatch.setattr(store_b, "is_leader", lambda: False)
    try:
        await store_a.teardown()
        store_a = None
        assert await _login_owners(store_b) == set(), "repro premise wrong"

        # Give the guard loop several ticks; a non-leader must leave it orphaned.
        await asyncio.sleep(3)
        assert await _login_owners(store_b) == set(), (
            "a non-leader re-registered login — the guard is not leader-gated, "
            "which risks duplicate registrations across replicas"
        )
    finally:
        if store_a is not None:
            await store_a.teardown()
        await store_b.teardown()


async def test_clean_boot_still_registers_login(monkeypatch):
    """Regression: a normal single-server boot still registers exactly one live
    login owned by this server (the guard/extraction must not change the happy
    path)."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        owners = await _login_owners(store)
        assert owners == {store._server_id}, (
            f"clean boot should register exactly the server's own login, got {owners}"
        )
    finally:
        await store.teardown()
