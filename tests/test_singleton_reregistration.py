"""F6 Issue #3: public singleton services must be re-claimable after owner death.

In a multi-replica deployment the public singletons (hypha-login, queue) were
registered only at boot-if-absent, so when the owning pod died no survivor
re-claimed them and e.g. /public/services/hypha-login/start returned 404.
_ensure_public_singletons re-registers any missing singleton; the leader runs
it periodically.

Docker-free: a RedisStore on fakeredis; simulate owner death by deleting the
login service keys, then assert re-registration restores it.
"""
import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def _delete_keys_matching(redis, substr):
    deleted = 0
    cursor = 0
    while True:
        cursor, batch = await redis.scan(cursor=cursor, match="services:*", count=1000)
        for k in batch:
            ks = k.decode() if isinstance(k, bytes) else k
            if substr in ks:
                await redis.delete(k)
                deleted += 1
        if cursor == 0:
            break
    return deleted


async def test_login_singleton_reregistered_after_owner_loss():
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        # Login is registered at startup.
        info = await api.get_service_info(
            "public/hypha-login", {"mode": "native:random"}
        )
        assert info is not None

        # Simulate the owning pod dying: remove its hypha-login service keys.
        deleted = await _delete_keys_matching(store.get_redis(), "hypha-login")
        assert deleted > 0, "expected to remove the login service key(s)"

        # It is now gone from the cluster registry.
        with pytest.raises(Exception):
            await api.get_service_info(
                "public/hypha-login", {"mode": "native:random"}
            )

        # A survivor re-claims it.
        await store._ensure_public_singletons(api)
        info2 = await api.get_service_info(
            "public/hypha-login", {"mode": "native:random"}
        )
        assert info2 is not None
    finally:
        await store.teardown()


async def test_ensure_is_idempotent_when_present():
    """Calling ensure when the singletons already exist must not error or
    duplicate (no-op)."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        api = await store.get_public_api()
        await store._ensure_public_singletons(api)  # already present -> no-op
        info = await api.get_service_info(
            "public/hypha-login", {"mode": "native:random"}
        )
        assert info is not None
    finally:
        await store.teardown()
