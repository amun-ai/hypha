"""Test that workspace operations use SCAN instead of blocking KEYS command.

redis.keys(pattern) is O(n) over ALL Redis keys and blocks the entire server
for the duration of the scan. This test verifies the fix uses cursor-based
SCAN, which is non-blocking and yields control between batches.

All redis.keys() calls in WorkspaceManager were replaced with _scan_keys()
which uses cursor-based SCAN iteration.
"""
import ast
import inspect
import textwrap
import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def fake_redis():
    redis = FakeRedis()
    await redis.flushall()
    return redis


def _find_redis_keys_calls(source: str) -> list:
    """Parse Python source and find any awaited redis.keys() calls."""
    tree = ast.parse(source)
    blocking_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            call = node.value
            if isinstance(call, ast.Call):
                func = call.func
                if isinstance(func, ast.Attribute) and func.attr == "keys":
                    blocking_calls.append(f"redis.keys() at line {getattr(node, 'lineno', '?')}")
    return blocking_calls


async def test_list_client_keys_uses_scan_not_keys(fake_redis):
    """Verify _list_client_keys uses SCAN (non-blocking) not KEYS (blocking)."""
    import hypha.core.workspace as workspace_module

    source = textwrap.dedent(
        inspect.getsource(workspace_module.WorkspaceManager._list_client_keys)
    )
    blocking_calls = _find_redis_keys_calls(source)
    assert not blocking_calls, (
        f"_list_client_keys uses blocking redis.keys(): {blocking_calls}\n"
        "Use redis.scan() with cursor-based iteration instead."
    )


async def test_scan_keys_helper_uses_scan_not_keys(fake_redis):
    """Verify the _scan_keys helper uses SCAN (non-blocking) not KEYS (blocking)."""
    import hypha.core.workspace as workspace_module

    source = textwrap.dedent(
        inspect.getsource(workspace_module.WorkspaceManager._scan_keys)
    )
    blocking_calls = _find_redis_keys_calls(source)
    assert not blocking_calls, (
        f"_scan_keys uses blocking redis.keys(): {blocking_calls}\n"
        "The helper MUST use redis.scan() with cursor-based iteration."
    )


async def test_workspace_manager_has_no_redis_keys_calls():
    """Verify WorkspaceManager has eliminated ALL redis.keys() calls.

    Every redis.keys() call blocks the entire Redis server for its duration.
    At 100K+ service entries, each call can stall for hundreds of milliseconds.
    All such calls must be replaced with _scan_keys() (cursor-based SCAN).
    """
    import hypha.core.workspace as workspace_module

    # Get the full module source
    source = inspect.getsource(workspace_module.WorkspaceManager)
    # Dedent to handle class-level indentation
    source = textwrap.dedent(source)
    blocking_calls = _find_redis_keys_calls(source)
    assert not blocking_calls, (
        f"WorkspaceManager still has blocking redis.keys() calls: {blocking_calls}\n"
        "Replace ALL redis.keys() calls with self._scan_keys() — "
        "redis.keys() is O(n_total_keys) and blocks the entire Redis server."
    )


async def test_scan_keys_returns_correct_results(fake_redis):
    """Verify _scan_keys correctly collects all keys matching a pattern."""
    from hypha.core.workspace import WorkspaceManager

    workspace = "test-ws"
    # Seed some service keys
    expected_keys = set()
    for i in range(5):
        key = f"services:protected|svc-{i}:{workspace}/client-{i}:svc-{i}@app"
        await fake_redis.hset(key, mapping={"id": f"svc-{i}"})
        expected_keys.add(key)

    # Also seed keys that should NOT match
    await fake_redis.hset("other:workspace:key", mapping={"id": "other"})

    class MinimalWM:
        _redis = fake_redis

    wm = MinimalWM()
    wm._scan_keys = WorkspaceManager._scan_keys.__get__(wm, MinimalWM)

    result = await wm._scan_keys(f"services:*:{workspace}/*")
    assert set(result) == expected_keys, (
        f"Expected {expected_keys}, got {set(result)}"
    )


async def test_scan_keys_handles_empty_result(fake_redis):
    """_scan_keys returns empty list when no keys match."""
    from hypha.core.workspace import WorkspaceManager

    class MinimalWM:
        _redis = fake_redis

    wm = MinimalWM()
    wm._scan_keys = WorkspaceManager._scan_keys.__get__(wm, MinimalWM)

    result = await wm._scan_keys("nonexistent:pattern:*")
    assert result == [], f"Expected empty list, got {result}"


async def test_scan_keys_handles_many_keys(fake_redis):
    """_scan_keys returns all 1000 keys, demonstrating non-blocking pagination."""
    from hypha.core.workspace import WorkspaceManager

    workspace = "busy-ws"
    # Seed 200 clients × 5 services each = 1000 keys
    expected_keys = set()
    for i in range(200):
        client_id = f"client-{i:04d}"
        for j in range(5):
            key = f"services:protected|svc-{i}-{j}:{workspace}/{client_id}:svc-{j}@app"
            await fake_redis.hset(key, mapping={"id": f"{workspace}/{client_id}:svc-{j}@app"})
            expected_keys.add(key)

    class MinimalWM:
        _redis = fake_redis

    wm = MinimalWM()
    wm._scan_keys = WorkspaceManager._scan_keys.__get__(wm, MinimalWM)

    result = await wm._scan_keys(f"services:*:{workspace}/*")
    assert len(result) == 1000, f"Expected 1000 keys, got {len(result)}"
    assert set(result) == expected_keys


async def test_list_client_keys_returns_correct_clients(fake_redis):
    """Verify _list_client_keys correctly finds clients via SCAN."""
    from hypha.core.workspace import WorkspaceManager

    workspace = "test-ws"
    clients = ["client-alpha", "client-beta", "client-gamma"]

    for client_id in clients:
        key = f"services:protected|svc-{client_id}:{workspace}/{client_id}:built-in@default"
        await fake_redis.hset(key, mapping={"id": f"{workspace}/{client_id}:built-in@default"})

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
    expected = {f"{workspace}/client-{i:04d}" for i in range(200)}
    assert set(result) == expected
