"""Test handling of corrupted service entries in Redis.

Reproduces https://github.com/amun-ai/hypha/issues/857
"""

import pytest
import asyncio

from hypha_rpc import connect_to_server
from . import SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def _create_in_process_server():
    """Create an in-process Hypha server with FakeRedis for testing."""
    from hypha.core.store import RedisStore
    from fastapi import FastAPI

    app = FastAPI()
    store = RedisStore(
        app,
        server_id="test-corrupted",
        redis_uri=None,  # Use fakeredis
    )
    await store.init(reset_redis=True)
    api = await store.get_public_api()
    redis = store.get_redis()
    return store, api, redis


async def test_corrupted_service_entry_does_not_break_list_services():
    """Test that a corrupted service entry (id=None) doesn't crash list_services.

    A corrupted service entry with id=None should be skipped and cleaned up,
    not crash the entire operation.
    """
    store, api, redis = await _create_in_process_server()

    workspace = "public"

    # Register a valid service first
    await api.register_service(
        {
            "id": "valid-service",
            "name": "Valid Service",
            "description": "A valid test service",
            "test_func": lambda x: x,
        }
    )

    # Verify list_services works before corruption
    services = await api.list_services()
    valid_ids = [s["id"] for s in services]
    assert any("valid-service" in sid for sid in valid_ids)

    # Inject a corrupted service entry directly into Redis
    # This simulates what happens when a client is killed and leaves behind
    # a corrupted entry with id=None
    corrupted_key = (
        f"services:protected|generic:{workspace}/fake-client:corrupted-svc@*"
    )
    corrupted_data = {
        "type": "generic",
        "name": "corrupted",
        # id is intentionally missing - this is the corruption
    }
    await redis.hset(corrupted_key, mapping=corrupted_data)

    # Verify the corrupted key exists
    assert await redis.exists(corrupted_key)

    # list_services should still work, skipping the corrupted entry
    services = await api.list_services()
    service_ids = [s["id"] for s in services]

    # The valid service should still be accessible
    assert any("valid-service" in sid for sid in service_ids)

    # The corrupted entry should have been cleaned up from Redis
    exists = await redis.exists(corrupted_key)
    assert not exists, "Corrupted service entry should have been cleaned up"

    await store.teardown()


async def test_corrupted_service_entry_does_not_break_register_service():
    """Test that a corrupted service entry doesn't break register_service.

    When registering a new service, the system checks for existing services
    with the same name. A corrupted entry should not crash this check.
    """
    store, api, redis = await _create_in_process_server()

    workspace = "public"

    # Inject a corrupted singleton service entry with the same name
    corrupted_key = (
        f"services:protected|generic:{workspace}/fake-client:my-service@*"
    )
    corrupted_data = {
        "type": "generic",
        "name": "my-service",
        "config": '{"singleton": true, "visibility": "protected"}',
        # id is missing - corrupted
    }
    await redis.hset(corrupted_key, mapping=corrupted_data)

    # Registering a service with the same name should still work
    # The corrupted entry should be skipped/cleaned up
    await api.register_service(
        {
            "id": "my-service",
            "name": "My Service",
            "description": "A test service",
            "test_func": lambda x: x,
        }
    )

    # Verify the service was registered
    services = await api.list_services()
    assert any("my-service" in s["id"] for s in services)

    await store.teardown()


async def test_corrupted_service_does_not_break_list_clients():
    """Test that corrupted service entries don't break list_clients."""
    store, api, redis = await _create_in_process_server()

    workspace = "public"

    # Inject a corrupted service entry that looks like a built-in service
    corrupted_key = (
        f"services:protected|generic:{workspace}/fake-client:built-in@*"
    )
    corrupted_data = {
        "type": "generic",
        "name": "built-in",
        # id is missing - corrupted
    }
    await redis.hset(corrupted_key, mapping=corrupted_data)

    # list_clients should still work, skipping the corrupted entry
    clients = await api.list_clients()
    # Should not crash - the corrupted entry should be skipped
    assert isinstance(clients, list)

    await store.teardown()


async def test_cleanup_with_many_stale_clients():
    """Test that cleanup handles many stale clients efficiently.

    Reproduces the issue where 24k+ stale clients caused cleanup() to time out.
    The cleanup should process clients in batches to avoid blocking.
    """
    store, api, redis = await _create_in_process_server()

    workspace = "public"

    # Inject many fake service entries (simulating stale clients)
    num_stale = 50
    for i in range(num_stale):
        fake_key = f"services:protected|generic:{workspace}/stale-client-{i}:built-in@*"
        fake_data = {
            "id": f"{workspace}/stale-client-{i}:built-in",
            "type": "generic",
            "name": "built-in",
            "config": '{"visibility": "protected"}',
        }
        await redis.hset(fake_key, mapping=fake_data)

    # Cleanup should complete without timing out
    result = await api.cleanup(timeout=2)
    # Should have removed the stale clients
    assert "removed_clients" in result
    assert len(result["removed_clients"]) >= num_stale

    await store.teardown()
