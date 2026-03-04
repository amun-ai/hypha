"""Test that _list_client_keys uses SCAN instead of blocking KEYS command.

redis.keys(pattern) is O(n) over ALL Redis keys and blocks the entire server
for the duration of the scan. This test verifies the fix uses cursor-based
SCAN, which is non-blocking and yields control between batches.
"""
import ast
import inspect
import pytest
import pytest_asyncio
import asyncio
from fakeredis.aioredis import FakeRedis

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def fake_redis():
    redis = FakeRedis()
    await redis.flushall()
    return redis


async def test_list_client_keys_uses_scan_not_keys(fake_redis):
    """Verify _list_client_keys uses SCAN (non-blocking) not KEYS (blocking)."""
    import textwrap
    import hypha.core.workspace as workspace_module

    source = textwrap.dedent(
        inspect.getsource(workspace_module.WorkspaceManager._list_client_keys)
    )
    tree = ast.parse(source)

    blocking_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            call = node.value
            if isinstance(call, ast.Call):
                func = call.func
                # Check for redis.keys() calls
                if isinstance(func, ast.Attribute) and func.attr == "keys":
                    blocking_calls.append(f"redis.keys() at line {getattr(node, 'lineno', '?')}")

    assert not blocking_calls, (
        f"_list_client_keys uses blocking redis.keys(): {blocking_calls}\n"
        "Use redis.scan() with cursor-based iteration instead — "
        "redis.keys() blocks the entire Redis server for its duration."
    )


async def test_list_client_keys_returns_correct_clients(fake_redis):
    """Verify _list_client_keys correctly finds clients via SCAN."""
    from hypha.core import RedisEventBus, UserInfo
    from hypha.core.store import RedisRPCConnection
    from hypha.core.workspace import WorkspaceManager

    # Seed fake Redis with service keys matching the expected pattern
    workspace = "test-ws"
    clients = ["client-alpha", "client-beta", "client-gamma"]

    for client_id in clients:
        # Format: services:{visibility}|{service_id}:{workspace}/{client_id}:{service_name}@{app}
        key = f"services:protected|svc-{client_id}:{workspace}/{client_id}:built-in@default"
        await fake_redis.hset(key, mapping={"id": f"{workspace}/{client_id}:built-in@default"})

    # WorkspaceManager._list_client_keys needs only _redis attribute
    class MinimalWM:
        _redis = fake_redis

    wm = MinimalWM()
    wm._list_client_keys = WorkspaceManager._list_client_keys.__get__(wm, MinimalWM)

    result = await wm._list_client_keys(workspace)

    assert set(result) == {f"{workspace}/{c}" for c in clients}, (
        f"Expected {clients}, got {result}"
    )


async def test_list_client_keys_handles_many_keys_without_blocking(fake_redis):
    """With 1000 service keys, SCAN returns all without blocking Redis.

    This would cause ~100ms+ stalls with redis.keys() on a busy server.
    """
    from hypha.core.workspace import WorkspaceManager

    workspace = "busy-ws"
    # Seed 200 clients × 5 services each = 1000 keys
    for i in range(200):
        client_id = f"client-{i:04d}"
        for j in range(5):
            key = f"services:protected|svc-{i}-{j}:{workspace}/{client_id}:svc-{j}@app"
            await fake_redis.hset(key, mapping={"id": f"{workspace}/{client_id}:svc-{j}@app"})

    class MinimalWM:
        _redis = fake_redis

    wm = MinimalWM()
    wm._list_client_keys = WorkspaceManager._list_client_keys.__get__(wm, MinimalWM)

    result = await wm._list_client_keys(workspace)

    assert len(result) == 200, f"Expected 200 unique clients, got {len(result)}"
    # Verify all client IDs are correct
    expected = {f"{workspace}/client-{i:04d}" for i in range(200)}
    assert set(result) == expected
