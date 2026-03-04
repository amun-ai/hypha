"""Test that RedisStore uses SCAN instead of blocking KEYS.

redis.keys() is O(n_total_keys) and blocks the entire Redis server.
RedisStore has 10 call sites that need to be replaced with cursor-based SCAN.
"""
import ast
import inspect
import textwrap
import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

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


async def test_redis_store_has_no_redis_keys_calls():
    """Verify RedisStore has eliminated ALL redis.keys() calls.

    Every redis.keys() call blocks the entire Redis server for its duration.
    All such calls must be replaced with _scan_keys() (cursor-based SCAN).
    """
    import hypha.core.store as store_module

    source = inspect.getsource(store_module.RedisStore)
    source = textwrap.dedent(source)
    blocking_calls = _find_redis_keys_calls(source)
    assert not blocking_calls, (
        f"RedisStore still has blocking redis.keys() calls: {blocking_calls}\n"
        "Replace ALL redis.keys() calls with self._scan_keys() — "
        "redis.keys() is O(n_total_keys) and blocks the entire Redis server."
    )


async def test_redis_store_scan_keys_helper_exists():
    """Verify RedisStore has a _scan_keys helper method."""
    from hypha.core.store import RedisStore
    assert hasattr(RedisStore, "_scan_keys"), (
        "RedisStore must have a _scan_keys() cursor-based SCAN helper"
    )


@pytest_asyncio.fixture
async def fake_redis():
    redis = FakeRedis()
    await redis.flushall()
    return redis


async def test_redis_store_scan_keys_returns_correct_results(fake_redis):
    """_scan_keys returns all matching keys as decoded strings."""
    from hypha.core.store import RedisStore

    # Seed some service keys
    expected = set()
    for i in range(5):
        key = f"services:public|*:ws/{i}:svc-{i}@app"
        await fake_redis.hset(key, mapping={"id": f"svc-{i}"})
        expected.add(key)

    # Seed a non-matching key
    await fake_redis.hset("other:key", mapping={"id": "other"})

    class MinimalStore:
        _redis = fake_redis

    s = MinimalStore()
    s._scan_keys = RedisStore._scan_keys.__get__(s, MinimalStore)

    result = await s._scan_keys("services:*:*/*:*@*")
    assert set(result) == expected


async def test_redis_store_scan_keys_returns_strings_not_bytes(fake_redis):
    """_scan_keys must return str keys, not bytes (unlike redis.keys())."""
    from hypha.core.store import RedisStore

    await fake_redis.hset("services:public|*:ws/c1:svc@app", mapping={"id": "x"})

    class MinimalStore:
        _redis = fake_redis

    s = MinimalStore()
    s._scan_keys = RedisStore._scan_keys.__get__(s, MinimalStore)

    result = await s._scan_keys("services:*")
    assert result, "Expected at least one result"
    for key in result:
        assert isinstance(key, str), f"Expected str but got {type(key)}: {key!r}"


async def test_client_exists_uses_scan(fake_redis):
    """client_exists must not use redis.keys(); verify via AST inspection."""
    import hypha.core.store as store_module

    source = textwrap.dedent(inspect.getsource(store_module.RedisStore.client_exists))
    blocking = _find_redis_keys_calls(source)
    assert not blocking, (
        f"client_exists still uses blocking redis.keys(): {blocking}"
    )


async def test_list_servers_uses_scan(fake_redis):
    """list_servers must not use redis.keys(); verify via AST inspection."""
    import hypha.core.store as store_module

    source = textwrap.dedent(inspect.getsource(store_module.RedisStore.list_servers))
    blocking = _find_redis_keys_calls(source)
    assert not blocking, (
        f"list_servers still uses blocking redis.keys(): {blocking}"
    )
