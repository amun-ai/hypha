"""Test that worker_managed sessions persist on client disconnect.

Issue #899: client_disconnected kills worker-managed sessions on transient
network drops. Sessions with worker_managed=True in the ApplicationManifest
should NOT be stopped on disconnect — the worker controls their lifecycle.

These are unit tests using FakeRedis and a minimal controller fixture,
so they require no Docker, Postgres, or MinIO.
"""
import asyncio
import json
import pytest
from fakeredis.aioredis import FakeRedis

pytestmark = pytest.mark.asyncio


def _make_minimal_controller(fake_redis):
    """Build a minimal ServerAppController with the client_disconnected handler captured.

    Uses a MockEventBus to intercept the on_local("client_disconnected", ...)
    registration so the handler can be called directly in tests.
    """
    from hypha.apps import ServerAppController

    class MockEventBus:
        def __init__(self):
            self.handlers = {}

        def on_local(self, event_name, func):
            self.handlers.setdefault(event_name, []).append(func)

        def off_local(self, event_name, func=None):
            pass

    class MockRootUser:
        def model_dump(self):
            return {"id": "root", "email": "root@example.com", "is_anonymous": False}

    event_bus = MockEventBus()

    class MockStore:
        local_base_url = "http://localhost:8080"
        public_base_url = "http://localhost:8080"

        def get_redis(self):
            return fake_redis

        def get_event_bus(self):
            return event_bus

        def get_root_user(self):
            return MockRootUser()

        def register_public_service(self, svc):
            pass

        def set_server_app_controller(self, ctrl):
            pass

    ctrl = ServerAppController(MockStore(), False, 8080, None, True)
    handler = event_bus.handlers["client_disconnected"][0]
    return ctrl, handler


async def _seed_session(fake_redis, workspace, client_id, extra_fields=None):
    """Store a minimal session hash in FakeRedis, matching the encoding used by
    _store_session_in_redis (booleans stored as "1"/"0", lists/dicts as JSON with prefix).
    """
    key = f"sessions:{workspace}/{client_id}"
    data = {
        "worker_id": f"{workspace}/worker-abc:browser-worker",
        "app_id": "my-app",
        "status": "running",
    }
    if extra_fields:
        data.update(extra_fields)

    mapping = {}
    for k, v in data.items():
        if isinstance(v, bool):
            mapping[k] = "1" if v else "0"
        elif isinstance(v, list):
            mapping[f"json_list:{k}"] = json.dumps(v)
        elif isinstance(v, dict):
            mapping[f"json_dict:{k}"] = json.dumps(v)
        else:
            mapping[k] = str(v)

    await fake_redis.hset(key, mapping=mapping)


async def test_worker_managed_session_survives_disconnect():
    """Session with worker_managed=True must not be removed on client disconnect.

    This is the core fix for issue #899: the worker (not the disconnect event)
    controls the lifecycle of worker-managed sessions.
    """
    fake_redis = FakeRedis()
    await fake_redis.flushall()

    workspace = "test-ws"
    client_id = "client-123"
    # "1" is how booleans are stored in Redis (see _store_session_in_redis)
    await _seed_session(fake_redis, workspace, client_id, {"worker_managed": "1"})

    ctrl, handler = _make_minimal_controller(fake_redis)

    await handler({"id": client_id, "workspace": workspace})

    remaining = await fake_redis.hgetall(f"sessions:{workspace}/{client_id}")
    assert remaining, (
        "Session with worker_managed=True was incorrectly removed on client disconnect. "
        "Issue #899: worker-managed sessions must survive transient disconnects."
    )


async def test_daemon_session_survives_disconnect():
    """Session with daemon=True must not be removed on client disconnect.

    This validates existing daemon behaviour still works alongside the new
    worker_managed flag.
    """
    fake_redis = FakeRedis()
    await fake_redis.flushall()

    workspace = "test-ws"
    client_id = "client-daemon"
    await _seed_session(fake_redis, workspace, client_id, {"daemon": "1"})

    ctrl, handler = _make_minimal_controller(fake_redis)

    await handler({"id": client_id, "workspace": workspace})

    remaining = await fake_redis.hgetall(f"sessions:{workspace}/{client_id}")
    assert remaining, "Daemon session was incorrectly removed on client disconnect."


async def test_normal_session_removed_on_disconnect():
    """Normal session (no worker_managed, no daemon) is removed on disconnect.

    This ensures the fix does not accidentally suppress cleanup of regular sessions.
    """
    fake_redis = FakeRedis()
    await fake_redis.flushall()

    workspace = "test-ws"
    client_id = "client-normal"
    await _seed_session(fake_redis, workspace, client_id)

    stop_calls = []

    class MockWorker:
        async def stop(self, full_client_id, context=None):
            stop_calls.append(full_client_id)

    ctrl, handler = _make_minimal_controller(fake_redis)

    # Monkeypatch get_worker_by_id so the handler doesn't need a live server.
    async def fake_get_worker(wid):
        return MockWorker()

    ctrl.get_worker_by_id = fake_get_worker

    await handler({"id": client_id, "workspace": workspace})

    # Session must be removed from Redis
    remaining = await fake_redis.hgetall(f"sessions:{workspace}/{client_id}")
    assert not remaining, "Normal session should be removed after client disconnects."

    # worker.stop() must have been called exactly once
    assert stop_calls == [f"{workspace}/{client_id}"], (
        f"worker.stop() should have been called once with the full client ID, "
        f"got: {stop_calls}"
    )


async def test_worker_managed_flag_in_application_manifest():
    """ApplicationManifest accepts worker_managed field."""
    from hypha.core import ApplicationManifest

    # Default: worker_managed is False
    manifest = ApplicationManifest(
        id="test-app",
        name="Test App",
        type="window",
    )
    assert manifest.worker_managed is False

    # Explicit True
    manifest_wm = ApplicationManifest(
        id="test-app",
        name="Test App",
        type="window",
        worker_managed=True,
    )
    assert manifest_wm.worker_managed is True
