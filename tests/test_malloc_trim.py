"""Periodic malloc_trim background task lifecycle.

The server starts a periodic malloc_trim(0) loop to return glibc-retained free
memory to the OS (prod RSS plateaued ~2.1GB of which ~1.3GB was reclaimable
glibc-arena free memory, not live). This test verifies the task is started on
init and cancelled cleanly on teardown, and that the trim coroutine is safe on
any platform (it is a no-op off glibc/Linux).

Docker-free: RedisStore on fakeredis.
"""
import asyncio

import pytest

from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


async def test_malloc_trim_task_started_and_cancelled():
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        # The task is created on init regardless of platform (the coroutine
        # itself no-ops off glibc).
        assert store._malloc_trim_task is not None
    finally:
        await store.teardown()
    # After teardown the task must be finished (cancelled or completed), not
    # left running.
    assert store._malloc_trim_task.done()


async def test_periodic_malloc_trim_coroutine_is_safe():
    """The trim coroutine must never raise (no-op off glibc; trims on glibc)."""
    store = RedisStore(None, redis_uri=None)
    # Run the loop body briefly with a tiny interval, then cancel — it must
    # exit cleanly via CancelledError without surfacing any other error.
    import os

    os.environ["HYPHA_MALLOC_TRIM_INTERVAL"] = "0.05"
    try:
        task = asyncio.create_task(store._periodic_malloc_trim())
        await asyncio.sleep(0.2)
        if task.done():
            # Off glibc (e.g. macOS dev) the loop returns at the platform guard
            # — it must have completed without raising.
            assert task.exception() is None
        else:
            # On glibc (Linux/CI) it loops until cancelled.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        os.environ.pop("HYPHA_MALLOC_TRIM_INTERVAL", None)


async def test_malloc_trim_interval_zero_disables():
    """HYPHA_MALLOC_TRIM_INTERVAL<=0 returns immediately (disabled)."""
    import os

    store = RedisStore(None, redis_uri=None)
    os.environ["HYPHA_MALLOC_TRIM_INTERVAL"] = "0"
    try:
        # Should return promptly without needing cancellation.
        await asyncio.wait_for(store._periodic_malloc_trim(), timeout=2)
    finally:
        os.environ.pop("HYPHA_MALLOC_TRIM_INTERVAL", None)
