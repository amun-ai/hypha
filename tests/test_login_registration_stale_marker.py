"""Task #42 — the default hypha-login service must (re)register on boot even when
a STALE hypha-login registration from a previous server generation survives in a
non-reset (production) Redis.

Production symptom (kth-k8s, deploying 0.21.132; rolled back to 0.21.107):
``hypha-login`` failed to register on the new pod. The boot log showed
``Login service already registered (likely from startup function)`` — i.e. the
startup idempotency check at ``RedisStore.init`` (``hypha/core/store.py``) decided
a login service was already present and SKIPPED registering the default one.
Every subsequent ``GET /public/services/hypha-login/start`` then returned
``KeyError 'Service not found: public/*:hypha-login@*'`` because the "already
registered" service was a DEAD registration owned by a client that no longer
exists.

Root cause: the check trusted ``get_service_info`` — a PURE Redis registry scan
(``hypha/core/workspace.py`` ~2477) with NO liveness check. Registration is not
proof of liveness (#0011/#0015 discipline). On a server that resets Redis (dev,
CI) the stale marker never survives, so this is invisible. On prod
(``HYPHA_RESET_REDIS=false``) a hypha-login marker left by a previous, now-dead
server generation survives ``init``. Before #0015 (0.21.126) the inline
``_cleanup_orphaned_client_services`` reap ran on the boot readiness path and
cleared that dead client's services BEFORE this check, masking the bug; #0015
correctly deferred that reap off the readiness path — which EXPOSED the latent
false-positive. That is why 0.21.107 (inline reap) works and 0.21.126/131/132 do
not, and why rolling back to 0.21.107 cured it.

This is a real test (no mocks): it seeds a genuine ``ServiceInfo`` for a dead
client via the real ``to_redis_dict`` serialization into fakeredis, boots a real
``RedisStore`` WITHOUT resetting Redis (the prod-like condition — a reset test
CANNOT catch this), and asserts that after boot a LIVE hypha-login owned by THIS
server is registered and the dead registration has been reaped.
"""

import asyncio

import pytest

from hypha.core import ServiceInfo, ServiceConfig, VisibilityEnum
from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


DEAD_CLIENT = "dead-server-gen-abc123"


async def _seed_stale_login(redis, client_id=DEAD_CLIENT):
    """Seed a stale hypha-login registration owned by a dead client.

    The value is a REAL ``ServiceInfo`` serialized exactly as ``register_service``
    would store it (``to_redis_dict`` -> ``hset``), so ``get_service_info`` will
    load and resolve it — reproducing the production false-positive where the
    startup check believes a login service "already exists".
    """
    svc = ServiceInfo(
        id=f"public/{client_id}:hypha-login",
        name="Hypha Login",
        type="functions",
        app_id="*",
        config=ServiceConfig(
            visibility=VisibilityEnum.public,
            workspace="public",
            require_context=False,
        ),
    )
    key = f"services:public|functions:public/{client_id}:hypha-login@*"
    await redis.hset(key, mapping=svc.to_redis_dict())
    return key


async def _login_owners(store):
    """Return the set of client ids that own a public hypha-login registration."""
    keys = await store._scan_keys("services:*|*:public/*:hypha-login@*")
    owners = set()
    for k in keys:
        ks = k.decode("utf-8") if isinstance(k, bytes) else k
        # services:public|functions:public/<client_id>:hypha-login@*
        wsclient = ks.split("|", 1)[1].split(":", 1)[1]  # public/<client_id>:hypha-login@*
        client_id = wsclient.split(":", 1)[0].split("/", 1)[1]
        owners.add(client_id)
    return owners


async def test_stale_login_marker_does_not_block_registration(monkeypatch):
    """A stale (dead-owner) hypha-login marker present in a non-reset Redis must
    NOT cause boot to skip registering the default login service.

    Before the fix: the startup check resolves the stale marker (pure registry
    scan), logs "already registered", and skips registration — so the only
    hypha-login is the DEAD one (owned by DEAD_CLIENT) and login is broken.

    After the fix: the check proves liveness by pinging the resolved owner; the
    unreachable dead owner is reaped and the default login (owned by THIS server)
    is registered.
    """
    # Keep the deferred background orphan reaper parked so this test observes the
    # boot-time behavior deterministically — the fix must work on the readiness
    # path itself, NOT rely on the post-startup reaper (that is #0015's job and
    # runs too late for the very first login request after a deploy).
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    monkeypatch.setenv("HYPHA_LOGIN_PING_TIMEOUT", "1")

    store = RedisStore(None, redis_uri=None)

    # Seed the stale marker BEFORE init, with reset_redis=False (prod-like: the
    # marker survives boot). This is the whole point — a reset-redis boot could
    # never reproduce it.
    stale_key = await _seed_stale_login(store._redis)
    assert await store._redis.exists(stale_key), "seed failed"

    await store.init(reset_redis=False)
    try:
        owners = await _login_owners(store)

        # The dead registration must be gone (reaped as an unreachable owner).
        assert DEAD_CLIENT not in owners, (
            "the stale dead-owner hypha-login registration was NOT reaped; "
            f"login owners after boot: {owners}"
        )

        # A live login owned by THIS server must exist.
        assert store._server_id in owners, (
            "the default hypha-login was not registered by this server — the "
            "stale marker false-positived the startup idempotency check "
            f"(login owners after boot: {owners})"
        )

        # End-to-end liveness: the resolved login service actually answers, i.e.
        # start_login works. Before the fix this raised the production
        # KeyError/timeout because the resolved owner was dead.
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
    """Guard: with NO stale marker, a normal boot still registers exactly one
    live login owned by this server (the fix must not regress the happy path)."""
    monkeypatch.setenv("HYPHA_ORPHAN_REAP_INITIAL_DELAY", "60")
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        owners = await _login_owners(store)
        assert owners == {store._server_id}, (
            f"clean boot should register exactly the server's own login, got {owners}"
        )
    finally:
        await store.teardown()
