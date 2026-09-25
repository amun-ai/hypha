"""Protected system workspaces must survive unload() (P1 prod incident 2026-09-14).

`WorkspaceManager.unload()` on a persistent, S3-backed workspace deletes the
workspace record from the Redis ``workspaces`` hash whenever
``WorkspaceActivityManager.register_for_cleanup(ws)`` returns ``False``. But that
method returns ``False`` for the *protected* system workspaces (``public``,
``ws-user-root``, ``ws-anonymous``) precisely BECAUSE they are protected::

    # WorkspaceActivityManager.register_for_cleanup
    if not self._enabled or workspace_id in self._protected_workspaces:
        return False

So being protected — the exact property that should shield these workspaces — is
what triggers their deletion in ``unload()``'s persistent+S3 branch::

    if await self._activity_manager.register_for_cleanup(ws):
        ...
    else:
        # Either activity manager disabled or protected workspace - delete immediately
        await self._redis.hdel("workspaces", ws)   # <-- deletes public/ws-user-root/ws-anonymous

In production (hypha.amun.ai, 2026-09-14) a Redis reschedule dropped the public
service keys; the ``public`` workspace then looked empty, a client disconnect
fired ``unload_if_empty("public")`` -> ``unload(force=False)``, and this branch
``hdel``'d ``public`` from Redis. Every subsequent public-service lookup then
``404``'d ("Failed to report token ... (404: )"), taking Svamp login down for
~3h until a manual restart.

These are REAL, no-mock tests: an in-process ``RedisStore`` (FakeRedis) wired to
a REAL ``S3Controller`` pointed at the ``minio_server`` fixture — the same
S3-enabled configuration ``hypha/server.py`` builds (``--s3-admin-type=generic``,
``--workspace-bucket=my-workspaces``) — with ``unload()`` invoked directly via
the manager. The destructive branch only runs when ``self._s3_controller`` is
truthy (the no-S3 path skips cleanup entirely and already preserves the
workspace), so a faithful reproduction MUST have S3 enabled.
"""

import uuid

import pytest

from . import (
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
    MINIO_SERVER_URL_PUBLIC,
)

pytestmark = pytest.mark.asyncio

PROTECTED_WORKSPACES = ["public", "ws-user-root", "ws-anonymous"]


async def _make_store_with_s3(server_id):
    """In-process RedisStore (FakeRedis) with a REAL S3Controller on minio.

    Mirrors how ``hypha/server.py`` wires S3: construct the store, construct the
    ``S3Controller`` (which self-registers via ``store.set_s3_controller``), then
    ``store.init(reset_redis=True)`` — which creates the protected system
    workspaces in the Redis ``workspaces`` hash.
    """
    from fastapi import FastAPI

    from hypha.core.store import RedisStore
    from hypha.s3 import S3Controller

    app = FastAPI()
    store = RedisStore(app, server_id=server_id, redis_uri=None)

    # Real S3 controller against the minio fixture (self-registers into store).
    S3Controller(
        store,
        endpoint_url=MINIO_SERVER_URL,
        access_key_id=MINIO_ROOT_USER,
        secret_access_key=MINIO_ROOT_PASSWORD,
        endpoint_url_public=MINIO_SERVER_URL_PUBLIC,
        s3_admin_type="generic",
        workspace_bucket="my-workspaces",
        executable_path="./bin",
    )

    await store.init(reset_redis=True)
    assert store._s3_controller is not None, "S3 controller must be wired for repro"
    return store


def _root_context(store, ws):
    """A root/admin context targeting workspace ``ws`` (shape from workspace.py)."""
    root = store._workspace_manager._root_user
    return {"user": root.model_dump(), "ws": ws}


@pytest.mark.parametrize("ws", PROTECTED_WORKSPACES)
async def test_unload_keeps_protected_workspace_in_redis(minio_server, ws):
    """unload() of a protected system workspace must NOT remove it from Redis.

    Reproduce-before-fix: on the buggy code the persistent+S3 branch ``hdel``s
    the workspace because ``register_for_cleanup`` returns ``False`` for
    protected workspaces. After the fix the record survives.
    """
    store = await _make_store_with_s3(f"protected-{ws.replace('_', '-')}")
    try:
        mgr = store._workspace_manager

        # The protected workspace exists after init.
        assert await store._redis.hexists("workspaces", ws), (
            f"protected workspace {ws} should exist after store.init()"
        )
        winfo = await mgr.load_workspace_info(ws, load=False)
        assert winfo.persistent, f"{ws} must be persistent for this repro"

        # Force unload (admin) — the exact call unload_if_empty makes, but
        # force=True so we exercise the deletion decision directly.
        await mgr.unload(context=_root_context(store, ws), force=True)

        # THE INVARIANT: a protected system workspace must remain in Redis.
        assert await store._redis.hexists("workspaces", ws), (
            f"unload() deleted protected system workspace {ws} from Redis — "
            f"this 404s every {ws}-service lookup (prod incident 2026-09-14)"
        )
    finally:
        await store.teardown()


async def test_unload_normal_persistent_workspace_registers_for_cleanup(minio_server):
    """Regression guard (production config: activity manager ENABLED).

    A normal (non-protected) persistent workspace must take the
    ``register_for_cleanup`` path — remaining in Redis and tracked by the
    activity manager — exactly as before the fix. The fix must change behavior
    ONLY for protected workspaces.
    """
    store = await _make_store_with_s3("normal-persistent")
    try:
        mgr = store._workspace_manager
        assert mgr._activity_manager.is_enabled(), (
            "in-process store should have activity tracking enabled (prod default)"
        )
        ws_id = f"ws-normal-{uuid.uuid4().hex[:8]}"

        await mgr.create_workspace(
            {"id": ws_id, "name": ws_id, "persistent": True},
            context=_root_context(store, ws_id),
        )
        assert await store._redis.hexists("workspaces", ws_id)
        assert not mgr._activity_manager.is_protected(ws_id)

        await mgr.unload(context=_root_context(store, ws_id), force=True)

        # Non-protected + activity manager enabled -> register_for_cleanup(True):
        # the workspace stays in Redis and is tracked for inactivity cleanup.
        assert await store._redis.hexists("workspaces", ws_id), (
            f"normal persistent workspace {ws_id} was dropped from Redis; it "
            f"should be registered for activity-based cleanup (unchanged behavior)"
        )
        assert ws_id in mgr._activity_manager._registrations, (
            f"normal persistent workspace {ws_id} should be registered with the "
            f"activity manager after unload"
        )
    finally:
        await store.teardown()


async def test_unload_disabled_activity_manager_evicts_normal_keeps_protected(
    minio_server,
):
    """The delete-immediately path (activity manager DISABLED — the
    no-activity-tracker deployment) must evict a NORMAL persistent workspace but
    still KEEP a protected one.

    This exercises the exact ``else`` branch the bug lived in: when
    ``register_for_cleanup`` returns False, deletion must be gated on
    ``not is_protected(ws)``. It confirms the fix neither over-protects ordinary
    workspaces nor deletes system ones.
    """
    store = await _make_store_with_s3("disabled-activity")
    try:
        mgr = store._workspace_manager
        # Put the activity manager into its real DISABLED mode (the state it has
        # when the store is constructed without an activity tracker). No behavior
        # is substituted — only the supported disabled configuration is selected.
        mgr._activity_manager._enabled = False
        assert not mgr._activity_manager.is_enabled()

        ws_id = f"ws-normal-{uuid.uuid4().hex[:8]}"
        await mgr.create_workspace(
            {"id": ws_id, "name": ws_id, "persistent": True},
            context=_root_context(store, ws_id),
        )
        assert await store._redis.hexists("workspaces", ws_id)

        # Normal persistent workspace -> register_for_cleanup(False, disabled) and
        # NOT protected -> evicted from Redis (reloaded from S3 on next access).
        await mgr.unload(context=_root_context(store, ws_id), force=True)
        assert not await store._redis.hexists("workspaces", ws_id), (
            f"with the activity manager disabled, normal persistent workspace "
            f"{ws_id} must be evicted from Redis (delete-immediately path)"
        )

        # Protected workspace -> register_for_cleanup(False) but IS protected ->
        # kept, even with the activity manager disabled.
        await mgr.unload(context=_root_context(store, "public"), force=True)
        assert await store._redis.hexists("workspaces", "public"), (
            "protected workspace public must be kept even when the activity "
            "manager is disabled"
        )
    finally:
        await store.teardown()


async def test_public_service_lookup_survives_public_unload(minio_server):
    """After unload() of ``public``, resolving the public workspace must still
    work (no ``KeyError`` -> no HTTP 404).

    This pins the exact downstream failure from the incident:
    ``load_workspace_info("public", load=False)`` raised
    ``KeyError('Workspace not found: public')`` once ``public`` was ``hdel``'d,
    and ``hypha/http.py`` maps that to a 404 on every public-service request.
    """
    store = await _make_store_with_s3("public-lookup")
    try:
        mgr = store._workspace_manager

        await mgr.unload(context=_root_context(store, "public"), force=True)

        # The operation that 404'd in prod must succeed: public is still resolvable.
        winfo = await mgr.load_workspace_info("public", load=False)
        assert winfo.id == "public", "public workspace must remain resolvable"
    finally:
        await store.teardown()
