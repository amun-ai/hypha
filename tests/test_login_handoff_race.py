"""Task #63 — the default ``hypha-login`` service must not be permanently
orphaned by a LIVE-HANDOFF race when two server generations overlap.

What triggers the overlap in production (corrected 2026-09-30 by kth-k8s against
the live deployment — an earlier revision of this file asserted it was a
``maxSurge=1`` rolling update, which is WRONG for prod):

``hypha-server`` runs ``replicas=1`` with ``RollingUpdate maxSurge=0,
maxUnavailable=1``. That is deliberate — ``hypha-server/values.yaml`` records the
switch to no-overlap on 2026-06-17 for exactly this failure. So a **helm/rollout
update cannot produce overlap at all**: the old pod fully terminates (clearing
its login key, releasing the leader lease) before the new one starts, and the new
pod's boot check simply registers login with no competitor.

The overlap therefore arrives through restarts that ``maxSurge`` does not govern
— a bare ``kubectl delete pod``, a node eviction, an OOMKill restart, or the
hypha-health auto-heal restart — any of which can start the replacement while the
outgoing pod is still in graceful shutdown. The race below is reachable in prod,
just NOT via helm upgrades.

THIS RACE EXPLAINS ZERO CONFIRMED PRODUCTION INCIDENTS — do not cite one. Cluster
history (2026-09-30) shows every recent kth-k8s restart carried a
``kubectl.kubernetes.io/restartedAt`` annotation (all ``kubectl rollout restart``;
no bare pod delete, no OOMKill, no helm upgrade since 2026-08-20), and under
``maxSurge=0`` every one of those was no-overlap. The guard is justified as a
cause-agnostic invariant keeper, not as the fix for a diagnosed outage.

Note in particular that the 09-23 skip line does NOT prove a peer existed — see
``test_skip_line_can_be_a_self_match`` below.

Mechanism, as it would occur:

1. Two generations overlap: the NEW pod boots WHILE the login-owning OLD pod is
   still reachable.
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
import logging

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


async def test_login_guard_reregisters_promptly_on_leader_acquire(monkeypatch):
    """On the leader-ACQUIRE edge the guard re-registers immediately, not after a
    full interval — collapsing the post-handoff login-down window to ~election
    time. Uses a LARGE interval so a re-register within a few seconds can ONLY be
    the edge path, not the periodic backstop.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")
    # Large periodic interval: a prompt re-register cannot be the periodic tick.
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "600")

    store_a = RedisStore(None, redis_uri=None)
    await store_a.init(reset_redis=True)

    store_b = RedisStore(None, redis_uri=None)
    # B starts as a NON-leader (A is live and leads); its guard loop seeds
    # was_leader=False and polls leadership at min(600, 5)=5s.
    leader_flag = {"v": False}
    monkeypatch.setattr(store_b, "is_leader", lambda: leader_flag["v"])
    await store_b.init(reset_redis=False)
    try:
        await store_a.teardown()
        store_a = None
        assert await _login_owners(store_b) == set(), "repro premise wrong"

        # Simulate B winning leadership (the failover edge).
        leader_flag["v"] = True

        # The edge must be picked up within ~one poll (5s) + ensure, WELL under
        # the 600s periodic interval. Poll up to ~12s.
        for _ in range(48):
            await asyncio.sleep(0.25)
            if await _login_owners(store_b) == {store_b._server_id}:
                break
        assert await _login_owners(store_b) == {store_b._server_id}, (
            "the guard did not re-register on the leader-acquire edge (it should "
            "not have waited for the 600s periodic interval)"
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


async def test_single_server_runtime_deregistration_self_heals(monkeypatch):
    """Hypothesis (b): a SINGLE long-lived server whose login key disappears at
    RUNTIME — no peer, no handoff — must still self-heal.

    Prod 09-30 (hypha.aicell.io, 0.21.133) reported ``hypha-login`` 404 on a
    ``replicas=1`` pod that had been up 20 HOURS with a healthy readiness probe.
    Two causes could not be discriminated without pod logs: (a) the boot check
    deferred to the outgoing pod during the rolling update 20h earlier (the
    live-handoff race above), or (b) login registered fine at boot and was
    deregistered at runtime by some other mechanism (a reaper, a client-services
    clear, workspace churn).

    Since then (a) has become the LESS likely of the two for this incident: prod
    rolls with ``maxSurge=0``, so if the 09-29 15:04Z pod arrived via a helm
    rollout there was no live owner to defer to and the handoff race could not
    have applied. It would require that restart to have been a pod-delete /
    eviction / OOMKill / auto-heal instead. Nobody should state that the handoff
    race explains this outage.

    It stayed unproven. The dying pod's logs were captured before the restart,
    but the 21h-old kubelet buffer had already rotated past boot: 59k lines, zero
    startup markers, and no login-decision line anywhere — only a steady stream
    of failed ``public/*:hypha-login`` lookups across the whole window. The
    timing is consistent with (a) but that is circumstantial, so this incident
    must NOT be cited as proof of the handoff race. Which is the point: the guard
    has to cover both, because in practice you cannot find out which one you got.

    The guard must be robust to (b) WITHOUT knowing the mechanism, so this test
    deletes the registration directly out of Redis — deliberately mechanism-blind,
    standing in for whatever removed it — rather than reproducing any one cause.
    Unlike every other test here there is no second server: the orphan appears
    while this server keeps running, which is what makes it the (b) case.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "1")

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        assert await _login_owners(store) == {store._server_id}, (
            "boot should register a login owned by this server"
        )

        # The orphaning event: the registration vanishes while the server stays up.
        keys = await store._scan_keys("services:*|*:public/*:hypha-login@*")
        assert keys, "no login key to delete — repro premise wrong"
        for key in keys:
            await store._redis.delete(key)
        assert await _login_owners(store) == set(), (
            "expected zero login owners after the runtime deregistration"
        )

        # THE FIX: the periodic guard notices and re-registers, unattended. Before
        # #63 nothing re-evaluated the boot decision, so this stayed empty until a
        # manual pod restart — the 20h prod outage.
        for _ in range(40):
            await asyncio.sleep(0.25)
            if await _login_owners(store) == {store._server_id}:
                break
        assert await _login_owners(store) == {store._server_id}, (
            "the login guard did not heal a runtime deregistration on a single "
            "server — a replicas=1 deployment would stay 404 until a restart"
        )

        # End-to-end: the healed login actually answers, not just a key in Redis.
        api = await store.get_public_api()
        login_svc = await asyncio.wait_for(
            api.get_service("public/hypha-login", {"mode": "native:random"}),
            timeout=10,
        )
        result = await asyncio.wait_for(login_svc.start(), timeout=10)
        assert "login_url" in result, f"start_login returned unexpectedly: {result}"
    finally:
        await store.teardown()


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


async def test_skip_line_can_be_a_self_match(monkeypatch, caplog):
    """The 'already registered and reachable' skip line does NOT prove a peer.

    Forensic pin, added after kth-k8s established (2026-09-30, from ReplicaSet
    history) that EVERY recent prod restart was a ``kubectl rollout restart``.
    Under ``maxSurge=0`` those are all no-overlap, so on 09-23 there was no live
    peer to defer to — yet that pod logged the skip line. That looked like a
    liveness-check false positive.

    It is not. ``_ensure_login_service_registered`` runs AFTER startup functions
    (deliberately, so a custom login from a startup function is not clobbered),
    so anything that registers ``hypha-login`` earlier in THIS pod's own boot
    makes the check resolve it and ping its owner — itself, trivially alive.

    The line therefore means "a login resolved and its owner answered", not "a
    previous generation was still live". The discriminator is whether ``owner``
    equals this pod's own client id, which is what this test pins.

    This is also precisely the case where ``overwrite=True`` is load-bearing: the
    same pod's RPC peer already holds ``hypha-login`` in ``RPC._services``.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "2")

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        own_client = api.rpc.get_client_info()["id"]

        # Exactly one server exists; init() already registered login on it.
        assert await _login_owners(store) == {store._server_id}

        # Re-run the boot check, as a fresh pod would after its startup
        # functions had already registered a login.
        with caplog.at_level(logging.INFO, logger="redis-store"):
            live = await store._ensure_login_service_registered(source="boot")

        assert live is True, "the check should report a live login"

        skip_lines = [
            r.getMessage()
            for r in caplog.records
            if "is registered and reachable" in r.getMessage()
        ]
        assert skip_lines, f"expected the skip line, got: {caplog.messages}"

        # THE POINT: the owner is THIS pod, with no peer in existence.
        assert f"owner=public/{own_client}" in skip_lines[0], (
            "the skip line should name this pod's own client as the owner — "
            f"got {skip_lines[0]!r}"
        )

        # And the login is still registered exactly once (no duplicate).
        assert await _login_owners(store) == {store._server_id}
    finally:
        await store.teardown()
