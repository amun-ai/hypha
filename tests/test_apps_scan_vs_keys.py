"""Test that ServerAppController uses SCAN instead of blocking KEYS.

redis.keys() is O(n_total_keys) and blocks the entire Redis server.
ServerAppController has 3 redis.keys() calls for sessions:* patterns that must
be replaced with cursor-based SCAN.
"""
import ast
import inspect
import textwrap
import pytest

pytestmark = pytest.mark.asyncio


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


async def test_server_app_controller_has_no_redis_keys_calls():
    """Verify ServerAppController has eliminated ALL redis.keys() calls."""
    import hypha.apps as apps_module

    source = inspect.getsource(apps_module.ServerAppController)
    source = textwrap.dedent(source)
    blocking_calls = _find_redis_keys_calls(source)
    assert not blocking_calls, (
        f"ServerAppController still has blocking redis.keys() calls: {blocking_calls}\n"
        "Replace ALL redis.keys() calls with self._scan_keys() — "
        "redis.keys() is O(n_total_keys) and blocks the entire Redis server."
    )


async def test_server_app_controller_scan_keys_helper_exists():
    """Verify ServerAppController has a _scan_keys helper method."""
    from hypha.apps import ServerAppController
    assert hasattr(ServerAppController, "_scan_keys"), (
        "ServerAppController must have a _scan_keys() cursor-based SCAN helper"
    )


async def test_is_worker_unused_uses_scan():
    """_is_worker_unused must not use redis.keys(); verify via AST inspection."""
    import hypha.apps as apps_module

    source = textwrap.dedent(
        inspect.getsource(apps_module.ServerAppController._is_worker_unused)
    )
    blocking = _find_redis_keys_calls(source)
    assert not blocking, (
        f"_is_worker_unused still uses blocking redis.keys(): {blocking}"
    )


async def test_get_all_sessions_uses_scan():
    """_get_all_sessions must not use redis.keys(); verify via AST inspection."""
    import hypha.apps as apps_module

    source = textwrap.dedent(
        inspect.getsource(apps_module.ServerAppController._get_all_sessions)
    )
    blocking = _find_redis_keys_calls(source)
    assert not blocking, (
        f"_get_all_sessions still uses blocking redis.keys(): {blocking}"
    )


async def test_cleanup_worker_sessions_uses_scan():
    """cleanup_worker_sessions must not use redis.keys(); verify via AST inspection."""
    import hypha.apps as apps_module

    source = textwrap.dedent(
        inspect.getsource(apps_module.ServerAppController.cleanup_worker_sessions)
    )
    blocking = _find_redis_keys_calls(source)
    assert not blocking, (
        f"cleanup_worker_sessions still uses blocking redis.keys(): {blocking}"
    )


async def test_scan_keys_functional():
    """_scan_keys returns decoded str keys matching the pattern."""
    from fakeredis.aioredis import FakeRedis
    from hypha.apps import ServerAppController

    fake_redis = FakeRedis()
    await fake_redis.flushall()

    # Seed session keys
    expected = set()
    for i in range(5):
        key = f"sessions:ws/client-{i}"
        await fake_redis.hset(key, mapping={"worker_id": f"worker-{i}"})
        expected.add(key)

    # Seed a non-matching key
    await fake_redis.hset("other:key", mapping={"id": "other"})

    class MinimalController:
        _redis = fake_redis

    c = MinimalController()
    c._scan_keys = ServerAppController._scan_keys.__get__(c, MinimalController)

    result = await c._scan_keys("sessions:*")
    assert set(result) == expected

    # Verify all results are str, not bytes
    for k in result:
        assert isinstance(k, str), f"Expected str, got {type(k)}: {k!r}"
