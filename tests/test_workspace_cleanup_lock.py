"""F6 Phase 1 (P3): per-workspace cleanup lock prevents concurrent multi-replica
cleanup of the same inactive workspace, without leaking workspaces.

Real fakeredis, no mocks of the system under test (the real
WorkspaceActivityManager). Minimal tracker/workspace-manager stands-ins supply
the two collaborators the cleanup callback touches.
"""
import pytest
from fakeredis import aioredis as fakeredis

from hypha.core.workspace import WorkspaceActivityManager

pytestmark = pytest.mark.asyncio


class _FakeTracker:
    """Non-None tracker so the activity manager is enabled; records reset calls."""

    def __init__(self):
        self.reset_calls = []

    async def reset_timer(self, entity_id, entity_type="default"):
        self.reset_calls.append(entity_id)


class _FakeWorkspaceManager:
    def __init__(self, client_keys=None):
        self._client_keys = client_keys or []

    async def _list_client_keys(self, workspace_id):
        return self._client_keys


def _make_mgr(redis, client_keys=None):
    return WorkspaceActivityManager(
        activity_tracker=_FakeTracker(),
        redis=redis,
        workspace_manager=_FakeWorkspaceManager(client_keys),
        inactive_period=300,
        check_interval=60,
    )


async def test_cleanup_deletes_empty_workspace_and_releases_lock():
    r = fakeredis.FakeRedis()
    await r.hset("workspaces", "ws-x", b"{}")
    mgr = _make_mgr(r, client_keys=[])

    await mgr._cleanup_inactive_workspace("ws-x")

    assert await r.hexists("workspaces", "ws-x") is False  # deleted
    assert await r.get("ws-cleanup-lock:ws-x") is None  # lock released


async def test_cleanup_skips_when_lock_held_by_other_replica():
    r = fakeredis.FakeRedis()
    await r.hset("workspaces", "ws-x", b"{}")
    mgr = _make_mgr(r, client_keys=[])

    # Simulate another replica mid-cleanup holding the per-workspace lock.
    assert await r.set("ws-cleanup-lock:ws-x", b"1", nx=True, px=30000)

    await mgr._cleanup_inactive_workspace("ws-x")
    # This replica must NOT delete the workspace while another holds the lock.
    assert await r.hexists("workspaces", "ws-x") is True

    # Once the lock frees, the retry cleans it up (no leak).
    await r.delete("ws-cleanup-lock:ws-x")
    await mgr._cleanup_inactive_workspace("ws-x")
    assert await r.hexists("workspaces", "ws-x") is False


async def test_two_replicas_only_one_deletes():
    r = fakeredis.FakeRedis()  # shared Redis
    await r.hset("workspaces", "ws-x", b"{}")
    a = _make_mgr(r, client_keys=[])
    b = _make_mgr(r, client_keys=[])

    # Both fire (sequentially in-process); the lock means the first deletes and
    # the second is a clean no-op — exactly one effective cleanup, no error.
    await a._cleanup_inactive_workspace("ws-x")
    await b._cleanup_inactive_workspace("ws-x")

    assert await r.hexists("workspaces", "ws-x") is False
    assert await r.get("ws-cleanup-lock:ws-x") is None


async def test_cleanup_resets_and_does_not_delete_when_clients_present():
    r = fakeredis.FakeRedis()
    await r.hset("workspaces", "ws-x", b"{}")
    mgr = _make_mgr(r, client_keys=["ws-x/client-1"])  # still has a client

    await mgr._cleanup_inactive_workspace("ws-x")

    assert await r.hexists("workspaces", "ws-x") is True  # NOT deleted
    assert mgr._tracker.reset_calls == ["ws-x"]  # activity timer reset
    assert await r.get("ws-cleanup-lock:ws-x") is None  # lock released even on this path
