"""F6 production proof: real multi-replica integration tests.

Two REAL hypha-server processes (server-0 with --reset-redis, server-1 without)
share ONE real Redis (the fastapi_server_redis_1 / fastapi_server_redis_2
fixtures). These tests prove the multi-replica safety properties end-to-end —
not with fakeredis or stubs, but with live servers and a live Redis.

Requires Docker (redis-stack) + MinIO via the fixtures.
"""
import asyncio

import pytest
import redis.asyncio as aioredis
from hypha_rpc import connect_to_server

from . import SERVER_URL_REDIS_1, REDIS_PORT, find_item

pytestmark = pytest.mark.asyncio


async def test_leader_election_exactly_one_leader(
    fastapi_server_redis_1, fastapi_server_redis_2
):
    """Two servers sharing one Redis must elect exactly one leader (F6 Phase 0/1).

    The leader lease is a single shared key whose value is the leader's
    server_id — so a present, valid value proves exactly-one leadership across
    the two real replicas.
    """
    r = aioredis.from_url(f"redis://127.0.0.1:{REDIS_PORT}/0")
    try:
        leader = await r.get("hypha:leader")
        assert leader in (b"server-0", b"server-1"), (
            f"exactly one server must hold leadership; got {leader!r}"
        )
        # And the lease must be live (renewed), i.e. carry a positive TTL.
        pttl = await r.pttl("hypha:leader")
        assert pttl > 0, f"leader lease must have a live TTL, got {pttl}"
    finally:
        await r.aclose()


async def test_reset_guard_preserved_first_server_state(
    fastapi_server_redis_1, fastapi_server_redis_2
):
    """server-1 (booted after server-0) must NOT have wiped the shared Redis.

    server-0 starts with --reset-redis; server-1 starts afterward. Both servers
    being discoverable proves server-1's startup did not flush server-0's state
    (the F6/P0 reset guard) — otherwise server-0's registrations would be gone.
    """
    r = aioredis.from_url(f"redis://127.0.0.1:{REDIS_PORT}/0")
    try:
        # Both servers' built-in service keys must be present in the shared Redis.
        keys = []
        cursor = 0
        while True:
            cursor, batch = await r.scan(
                cursor, match="services:*|*:public/*:built-in@*", count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break
        client_ids = {k.decode().split("/")[1].split(":")[0] for k in keys}
        assert "server-0" in client_ids and "server-1" in client_ids, (
            f"both servers must be registered (state not wiped); got {client_ids}"
        )
    finally:
        await r.aclose()


async def test_cross_replica_service_call(
    fastapi_server_redis_1, fastapi_server_redis_2, test_user_token
):
    """A client on server-1 invokes a service registered by a client on server-0
    — a real cross-replica RPC over the Redis event bus."""
    server_url_2 = fastapi_server_redis_2

    api = await connect_to_server(
        {
            "client_id": "owner-x",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token,
        }
    )
    ws = await api.create_workspace(
        {
            "name": "mr-test-ws",
            "description": "multi-replica integration test workspace",
            "owners": ["user1@imjoy.io"],
        },
        overwrite=True,
    )
    assert ws["name"] == "mr-test-ws"
    token = await api.generate_token({"workspace": "mr-test-ws"})

    # Provider on server-0, caller on server-1.
    provider = await connect_to_server(
        {
            "client_id": "provider",
            "server_url": SERVER_URL_REDIS_1,
            "workspace": "mr-test-ws",
            "token": token,
            "method_timeout": 30,
        }
    )
    caller = await connect_to_server(
        {
            "client_id": "caller",
            "server_url": server_url_2,
            "workspace": "mr-test-ws",
            "token": token,
            "method_timeout": 30,
        }
    )
    await asyncio.sleep(0.2)

    clients = await caller.list_clients()
    assert find_item(clients, "id", "mr-test-ws/provider")
    assert find_item(clients, "id", "mr-test-ws/caller")

    await provider.register_service(
        {"id": "echo-svc", "echo": lambda x: x + 1},
        {"overwrite": True},
    )

    svc = await caller.get_service("provider:echo-svc")
    assert await svc.echo(41) == 42  # cross-replica call resolved & executed

    await provider.disconnect()
    await caller.disconnect()
    await api.disconnect()


async def test_cross_pod_repin_no_false_reject(
    fastapi_server_redis_1, fastapi_server_redis_2, test_user_token
):
    """A peer that re-pins from pod A to pod B must NOT be false-rejected by pod
    A's stale recently-disconnected cache.

    Regression for the N>=2 prod incident (2026-06-17): pod A kept the migrated
    client in `_recently_disconnected` (120s TTL) and rejected messages to it
    with "Target peer is not connected", even though it was alive on pod B. The
    fix broadcasts on (re)connect so pod A clears the stale entry.
    """
    server_a = SERVER_URL_REDIS_1
    server_b = fastapi_server_redis_2

    admin = await connect_to_server(
        {"client_id": "repin-admin", "server_url": server_a, "token": test_user_token}
    )
    await admin.create_workspace(
        {"name": "repin-ws", "persistent": True, "owners": ["user1@imjoy.io"]},
        overwrite=True,
    )
    token = await admin.generate_token({"workspace": "repin-ws"})

    # Caller stays on pod A for the whole test.
    caller = await connect_to_server(
        {"client_id": "caller", "server_url": server_a, "workspace": "repin-ws",
         "token": token, "method_timeout": 30}
    )
    # Provider 'migrant' starts on pod A and registers a service.
    provider_a = await connect_to_server(
        {"client_id": "migrant", "server_url": server_a, "workspace": "repin-ws",
         "token": token, "method_timeout": 30}
    )
    await provider_a.register_service(
        {"id": "svc", "echo": lambda x: x + 1}, {"overwrite": True}
    )
    svc = await caller.get_service("migrant:svc")
    assert await svc.echo(1) == 2  # works while migrant is on pod A

    # migrant disconnects from pod A -> pod A tracks it as recently-disconnected.
    await provider_a.disconnect()
    await asyncio.sleep(2)

    # migrant RE-PINS to pod B with the SAME client_id and re-registers — exactly
    # the topology-change re-pin that the nginx client_id hash ring does.
    provider_b = await connect_to_server(
        {"client_id": "migrant", "server_url": server_b, "workspace": "repin-ws",
         "token": token, "method_timeout": 30}
    )
    await provider_b.register_service(
        {"id": "svc", "echo": lambda x: x + 1}, {"overwrite": True}
    )
    await asyncio.sleep(1)  # let the reconnect broadcast + re-registration settle

    # Caller on pod A calls migrant's service. Within the 120s TTL window, the old
    # behavior would false-reject ("not connected"); with the fix it reaches pod B.
    svc2 = await caller.get_service("migrant:svc")
    assert await svc2.echo(10) == 11

    await provider_b.disconnect()
    await caller.disconnect()
    await admin.disconnect()
