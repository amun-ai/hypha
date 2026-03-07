"""Tests for null value handling in _store_session_in_redis.

Reproduces the bug where setting a session field to None would leave the old
value in Redis (because the old code silently skipped None values instead of
explicitly deleting the hash field).
"""

import pytest
from fakeredis.aioredis import FakeRedis
from hypha.apps import ServerAppController


class MinimalController:
    """Minimal stand-in for ServerAppController with only Redis and cache."""

    def __init__(self, redis):
        self._redis = redis
        self._worker_cache = {}

    _store_session_in_redis = ServerAppController._store_session_in_redis
    _get_session_from_redis = ServerAppController._get_session_from_redis
    _scan_keys = ServerAppController._scan_keys


@pytest.mark.asyncio
async def test_null_field_deletes_stale_redis_value():
    """Setting a session field to None must remove it from Redis.

    Regression test for the bug where None values were silently skipped,
    causing old values to persist in Redis across updates.
    """
    redis = FakeRedis()
    c = MinimalController(redis)

    # First store: description has a value
    await c._store_session_in_redis("ws/client-1", {
        "status": "running",
        "description": "initial description",
        "worker_id": "worker-abc",
    })

    # Verify initial value is stored
    session = await c._get_session_from_redis("ws/client-1")
    assert session is not None
    assert session["description"] == "initial description"
    assert session["worker_id"] == "worker-abc"

    # Second store: description cleared to None
    await c._store_session_in_redis("ws/client-1", {
        "status": "running",
        "description": None,   # <-- explicitly cleared
        "worker_id": "worker-abc",
    })

    # After fix: description field must be gone from Redis (not stale old value)
    session = await c._get_session_from_redis("ws/client-1")
    assert session is not None
    assert "description" not in session, (
        f"description should have been deleted from Redis when set to None, "
        f"but got: {session.get('description')!r}"
    )
    # Other fields must survive
    assert session["status"] == "running"
    assert session["worker_id"] == "worker-abc"


@pytest.mark.asyncio
async def test_internal_fields_skipped():
    """Fields starting with '_' must never be written to Redis."""
    redis = FakeRedis()
    c = MinimalController(redis)

    await c._store_session_in_redis("ws/client-2", {
        "status": "running",
        "_worker": object(),   # internal: should be skipped
        "_private": "secret",  # internal: should be skipped
        "worker_id": "worker-xyz",
    })

    raw = await redis.hgetall("sessions:ws/client-2")
    keys = {k.decode() if isinstance(k, bytes) else k for k in raw}
    assert "_worker" not in keys
    assert "_private" not in keys
    assert "worker_id" in keys


@pytest.mark.asyncio
async def test_multiple_null_fields_all_deleted():
    """Multiple None fields in a single update must all be deleted."""
    redis = FakeRedis()
    c = MinimalController(redis)

    # Store initial values
    await c._store_session_in_redis("ws/client-3", {
        "name": "My App",
        "description": "A description",
        "docs": "Some docs",
        "status": "running",
    })

    # Clear multiple fields at once
    await c._store_session_in_redis("ws/client-3", {
        "name": "My App",
        "description": None,
        "docs": None,
        "status": "stopped",
    })

    session = await c._get_session_from_redis("ws/client-3")
    assert session is not None
    assert "description" not in session
    assert "docs" not in session
    assert session["name"] == "My App"
    assert session["status"] == "stopped"
