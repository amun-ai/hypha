"""Store-level wiring of the F6 leader lease (Phase 1).

Docker-free: constructs a RedisStore on fakeredis (redis_uri=None), exercising
that init() starts a leader lease, is_leader() reflects it, and teardown()
stops it. A single instance is always the sole leader, preserving
single-replica behavior for the leader-gated control-plane singletons.
"""
import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def test_is_leader_defaults_true_before_init():
    """Before init() starts a lease, is_leader() defaults to True so leader-gated
    singletons run normally in single-instance / test contexts."""
    store = RedisStore(None, redis_uri=None)
    assert store._leader_lease is None
    assert store.is_leader() is True


async def test_init_starts_lease_and_single_instance_is_leader():
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        assert store._leader_lease is not None
        assert store.is_leader() is True  # sole instance → leader
    finally:
        await store.teardown()


async def test_teardown_releases_leadership():
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    assert store.is_leader() is True
    await store.teardown()
    # After teardown the lease loop is stopped and leadership released.
    assert store.is_leader() is False
