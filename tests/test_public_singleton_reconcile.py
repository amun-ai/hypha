"""Issue #0010 — EVERY public singleton this server registers must self-heal,
not just ``hypha-login``.

Prod incident (bioimage.io, 2026-10-18): auto-login broke with
``KeyError: 'Service not found: public/*:artifact-manager@*'``. The pod was **13
days old with 0 restarts** and readiness/liveness both 200, so nothing in
Kubernetes was wrong and nothing restarted it; Redis held **zero**
``artifact-manager`` keys while ``hypha-login`` was present. A manual rollout
restart was the only cure.

``hypha-login`` survived because #63 gave it a guard. That guard is correct but
hardcoded to one service, and the failure class is not login-specific: it is
*"public singleton registered once at boot, no retry, no self-heal, N=1
deployment"*. On the live registry the server's own client owned **seven** such
singletons (``agent-skills``, ``artifact-manager``, ``built-in``,
``hypha-login``, ``queue``, ``s3-storage``, ``server-apps``) — one guarded, six
exposed.

These tests use real ``RedisStore`` instances and the real registry; the
orphaning event is a direct Redis delete, deliberately **mechanism-blind**,
because a reconcile loop must heal without knowing what removed the key (the
prod log buffer holds only minutes, so the cause was never recoverable).
"""

import asyncio

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _keys_for(store, client_id, service_id):
    return await store._scan_keys(
        f"services:*|*:public/{client_id}:{service_id}@*"
    )


async def test_reconcile_restores_a_deleted_own_service(monkeypatch):
    """The core repro: a public singleton's key vanishes at runtime and the
    reconcile brings it back — and the service still works afterwards.

    Before #0010 nothing re-registered anything but ``hypha-login``, so this
    stayed missing until a pod restart. ``queue`` stands in for
    ``artifact-manager`` here: both are registered by the server's own public
    client, which is the property that matters. (artifact-manager needs S3, so
    using queue keeps this Docker-free without changing the mechanism.)
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")  # drive by hand

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        own = api.rpc.get_client_info()["id"]

        assert await _keys_for(store, own, "queue"), "queue should be registered"

        for key in await _keys_for(store, own, "queue"):
            await store._redis.delete(key)
        assert not await _keys_for(store, own, "queue"), "repro premise wrong"

        repaired = await store._reconcile_own_public_services()

        assert "queue" in repaired, f"queue was not repaired; repaired={repaired}"
        assert await _keys_for(store, own, "queue"), (
            "the reconcile did not restore the queue registry entry"
        )

        # Not just a key in Redis — the service must actually serve.
        queue = await asyncio.wait_for(
            api.get_service("public/queue", {"mode": "native:random"}), timeout=10
        )
        await asyncio.wait_for(queue.push("reconcile-probe", {"v": 1}), timeout=10)
        assert await asyncio.wait_for(
            queue.pop("reconcile-probe"), timeout=10
        ) == {"v": 1}
    finally:
        await store.teardown()


async def test_reconcile_covers_every_own_public_service(monkeypatch):
    """Coverage is DERIVED, not hardcoded — delete them all, get them all back.

    This is the guard against the staleness that caused the incident: a
    hardcoded list of services goes stale the moment an eighth is added. Here
    the reconcile set comes from the RPC peer's own ``_services``, so whatever
    this server registered is covered by construction.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        own = api.rpc.get_client_info()["id"]
        own_services = sorted(api.rpc._services.keys())
        assert own_services, "server registered no public services"

        for key in await store._scan_keys(f"services:*|*:public/{own}:*@*"):
            await store._redis.delete(key)
        for service_id in own_services:
            assert not await _keys_for(store, own, service_id), (
                f"{service_id} should be deleted"
            )

        repaired = await store._reconcile_own_public_services()

        assert sorted(repaired) == own_services, (
            f"reconcile repaired {sorted(repaired)}, expected {own_services}"
        )
        for service_id in own_services:
            assert await _keys_for(store, own, service_id), (
                f"{service_id} was not restored"
            )
    finally:
        await store.teardown()


async def test_reconcile_never_touches_another_clients_service(monkeypatch):
    """THE SAFETY PROPERTY: never re-register a service we do not own.

    Public services belonging to external clients (model-runner,
    bioengine-worker, safe-colab-*, …) are not ours to restore, and
    re-registering them would be actively wrong — it would point the registry at
    a server that cannot serve them. Here a second store plays the external
    client; its service is deleted and must stay deleted.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")

    store_a = RedisStore(None, redis_uri=None)
    await store_a.init(reset_redis=True)
    store_b = RedisStore(None, redis_uri=None)
    await store_b.init(reset_redis=False)
    try:
        api_a = await store_a.get_public_api()
        api_b = await store_b.get_public_api()
        a_client = api_a.rpc.get_client_info()["id"]
        b_client = api_b.rpc.get_client_info()["id"]
        assert a_client != b_client

        # Pick a service B genuinely owns. Note B does NOT own `queue` or
        # `hypha-login`: both registrations are conditional and B correctly
        # defers to A's existing ones. `built-in` is registered by every RPC
        # peer, so B always owns its own.
        victim = "built-in"
        assert victim in api_b.rpc._services, f"B should own {victim}"
        assert await _keys_for(store_a, b_client, victim), f"B should own {victim}"

        for key in await _keys_for(store_a, b_client, victim):
            await store_a._redis.delete(key)
        assert not await _keys_for(store_a, b_client, victim), "repro premise wrong"

        repaired = await store_a._reconcile_own_public_services()

        # THE DECISIVE ASSERTION: B's entry must still be gone. A owns a service
        # of the same NAME, so a reconcile keyed on service id rather than
        # (client, service id) would wrongly "repair" B's — pointing the registry
        # at a server that cannot serve it.
        assert not await _keys_for(store_a, b_client, victim), (
            f"store A re-registered {victim} owned by store B — the reconcile is "
            "not ownership-scoped"
        )
        # A's own entry is untouched and still present.
        assert await _keys_for(store_a, a_client, victim), (
            "A's own service should still be registered"
        )
        assert victim not in repaired, (
            f"A reported repairing {victim}, but A's own copy was never missing; "
            f"repaired={repaired}"
        )
    finally:
        await store_a.teardown()
        await store_b.teardown()


async def test_controller_registered_service_is_covered(monkeypatch):
    """A service registered via ``register_public_service`` — the exact path
    ``artifact-manager`` uses — must be covered.

    ``artifact-manager``, ``s3-storage``, ``server-apps`` and ``agent-skills``
    all reach the registry through ``store.register_public_service()``, which
    appends to ``_public_services``; ``init`` then registers each one through the
    same public-API client the reconcile reads. This registers a stand-in by that
    same path so the prod shape is proven without needing S3 or the apps
    controller, then deletes its key and asserts the reconcile restores it.

    That is the assertion the incident needed: ``hypha-login`` was covered and
    ``artifact-manager`` was not, purely because the guard named one service.
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")

    store = RedisStore(None, redis_uri=None)
    # Must be declared BEFORE init: register_public_service asserts not ready.
    store.register_public_service(
        {
            "id": "pretend-artifact-manager",
            "name": "Pretend Artifact Manager",
            "config": {"visibility": "public"},
            "echo": lambda x: x,
        }
    )
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        own = api.rpc.get_client_info()["id"]

        assert "pretend-artifact-manager" in api.rpc._services, (
            "a service registered via register_public_service must appear in the "
            "public client's own _services — otherwise the reconcile cannot see it"
        )
        assert await _keys_for(store, own, "pretend-artifact-manager")

        for key in await _keys_for(store, own, "pretend-artifact-manager"):
            await store._redis.delete(key)
        assert not await _keys_for(store, own, "pretend-artifact-manager")

        repaired = await store._reconcile_own_public_services()

        assert "pretend-artifact-manager" in repaired, (
            f"controller-registered service not repaired; repaired={repaired}"
        )
        assert await _keys_for(store, own, "pretend-artifact-manager")
    finally:
        await store.teardown()


async def test_reconcile_is_a_noop_when_everything_is_present(monkeypatch):
    """Steady state must not churn the registry every interval."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "0")

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        assert await store._reconcile_own_public_services() == [], (
            "reconcile re-registered services that were already present"
        )
    finally:
        await store.teardown()


async def test_guard_loop_reconciles_without_being_leader(monkeypatch):
    """Own-service repair must NOT be leader-gated.

    A non-leader replica that loses its own ``artifact-manager`` key is the only
    party that can restore it — the leader cannot re-register another pod's
    services on its behalf. Leader-gating this would leave such a replica broken
    forever. (Login stays leader-gated; that one is a contended singleton.)
    """
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_GUARD_INTERVAL", "1")

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    # Force non-leader for the whole test.
    monkeypatch.setattr(store, "is_leader", lambda: False)
    try:
        api = await store.get_public_api()
        own = api.rpc.get_client_info()["id"]

        for key in await _keys_for(store, own, "queue"):
            await store._redis.delete(key)
        assert not await _keys_for(store, own, "queue"), "repro premise wrong"

        for _ in range(40):
            await asyncio.sleep(0.25)
            if await _keys_for(store, own, "queue"):
                break
        assert await _keys_for(store, own, "queue"), (
            "a NON-leader did not repair its own missing service — own-service "
            "reconcile must not be leader-gated"
        )
    finally:
        await store.teardown()
