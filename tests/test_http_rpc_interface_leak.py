"""Reproduce the prod HTTP /rpc memory leak (RedisRPCConnection + AsyncExitStack
never closed on the stateless POST /rpc path).

Prod evidence (valiant-goat, 0.21.103): a client polling POST /rpc drove
rpc_connections.created_total >> closed_total (844 vs 436), with live
RedisRPCConnection objects (~413) ~1:1 with retained AsyncExitStack (~401) and
~50 coroutine method-proxies each. The stateless POST /rpc handler builds a
workspace interface via store.get_workspace_interface(...) per request; if its
__aexit__ does not drive RedisRPCConnection.disconnect() to completion, the
connection (and its RPC peer + event-bus handlers) is retained.

This test exercises that exact enter/exit cycle in-process and asserts that the
connection registry, the created-minus-closed counter, and the live object count
all stay bounded.
"""
import asyncio
import gc

import pytest

from hypha.core import RedisRPCConnection, UserInfo, UserPermission
from hypha.core.auth import create_scope
from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


def _live_conn_objects():
    return sum(1 for o in gc.get_objects() if type(o).__name__ == "RedisRPCConnection")


async def test_stateless_rpc_interface_no_leak():
    """Each get_workspace_interface enter/exit must fully close its connection.

    Mirrors http_rpc._handle_stateless_call: `async with
    store.get_workspace_interface(user, ws) as api: ...`. After N cycles the
    connection registry, created-minus-closed, and live object count must not
    grow with N.
    """
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        # A normal authenticated user with admin on its own workspace (the shape
        # login_optional produces for a token-bearing caller).
        ws = "ws-user-leaktest"
        user_info = UserInfo(
            id="leaktest-user",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=[],
            scope=create_scope(
                f"{ws}#a",
                current_workspace=ws,
            ),
            expires_at=None,
        )
        await store.register_workspace(
            dict(
                name=ws,
                description="leak test",
                owners=["leaktest-user"],
                persistent=False,
                read_only=False,
            ),
            overwrite=True,
        )

        # Warm-up: first few interfaces may leave steady-state background state.
        for _ in range(3):
            async with store.get_workspace_interface(user_info, ws) as api:
                await api.echo("warmup")
        await asyncio.sleep(0.2)
        gc.collect()

        created_before = RedisRPCConnection._created_total_int
        closed_before = RedisRPCConnection._closed_total_int
        registry_before = len(RedisRPCConnection._connections)
        objects_before = _live_conn_objects()

        N = 40
        for i in range(N):
            async with store.get_workspace_interface(user_info, ws) as api:
                await api.echo(f"call-{i}")

        # Allow any async cleanup scheduled by disconnect() to run.
        await asyncio.sleep(0.5)
        gc.collect()
        await asyncio.sleep(0.1)

        created_after = RedisRPCConnection._created_total_int
        closed_after = RedisRPCConnection._closed_total_int
        registry_after = len(RedisRPCConnection._connections)
        objects_after = _live_conn_objects()

        created_delta = created_after - created_before
        closed_delta = closed_after - closed_before
        registry_delta = registry_after - registry_before
        objects_delta = objects_after - objects_before

        print(
            f"\n[leak] N={N} created_delta={created_delta} closed_delta={closed_delta} "
            f"unclosed={created_delta - closed_delta} registry_delta={registry_delta} "
            f"live_conn_objects={objects_before}->{objects_after} (delta={objects_delta})"
        )

        # We created N interfaces; each must be closed. The unclosed counter is
        # the exact prod leak signal (created_total - closed_total).
        assert created_delta >= N, (
            f"expected >= {N} connections created, got {created_delta}"
        )
        assert (created_delta - closed_delta) <= 2, (
            f"connection leak: {created_delta - closed_delta} connections created "
            f"but never closed (created={created_delta}, closed={closed_delta}). "
            f"This is the prod HTTP /rpc leak."
        )
        assert registry_delta <= 2, (
            f"registry leak: {registry_delta} connections left in registry"
        )
        assert objects_delta <= 2, (
            f"object leak: {objects_delta} RedisRPCConnection objects retained "
            f"after {N} enter/exit cycles"
        )
    finally:
        await store.teardown()


async def test_stateless_rpc_interface_no_leak_on_cancellation():
    """A request cancelled mid-flight (client disconnect) must still fully close.

    Reproduces the prod signature: __aexit__ does `await rpc.disconnect()` and
    catches only `except Exception`. CancelledError is BaseException, so if the
    request task is cancelled during the interface's lifetime, an interrupted
    cleanup would leak the connection (created_total >> closed_total).
    """
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        ws = "ws-user-leaktest2"
        user_info = UserInfo(
            id="leaktest-user2",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=[],
            scope=create_scope(f"{ws}#a", current_workspace=ws),
            expires_at=None,
        )
        await store.register_workspace(
            dict(name=ws, description="leak test", owners=["leaktest-user2"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        # warm up
        async with store.get_workspace_interface(user_info, ws) as api:
            await api.echo("warmup")
        await asyncio.sleep(0.2)
        gc.collect()

        created_before = RedisRPCConnection._created_total_int
        closed_before = RedisRPCConnection._closed_total_int
        registry_before = len(RedisRPCConnection._connections)

        async def one_request():
            async with store.get_workspace_interface(user_info, ws) as api:
                # simulate the request doing slow work (waiting on a target)
                await asyncio.sleep(30)

        N = 40
        for i in range(N):
            task = asyncio.create_task(one_request())
            # let it enter the interface (connection created), then cancel it
            # mid-flight — exactly what happens when the HTTP client drops.
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await asyncio.sleep(0.5)
        gc.collect()
        await asyncio.sleep(0.1)

        created_delta = RedisRPCConnection._created_total_int - created_before
        closed_delta = RedisRPCConnection._closed_total_int - closed_before
        registry_delta = len(RedisRPCConnection._connections) - registry_before
        unclosed = created_delta - closed_delta

        print(
            f"\n[leak-cancel] N={N} created_delta={created_delta} "
            f"closed_delta={closed_delta} unclosed={unclosed} "
            f"registry_delta={registry_delta}"
        )

        assert unclosed <= 2, (
            f"CANCELLATION LEAK: {unclosed} connections created but never closed "
            f"when requests were cancelled mid-flight (created={created_delta}, "
            f"closed={closed_delta}). This is the prod HTTP /rpc leak."
        )
        assert registry_delta <= 2, (
            f"registry leak on cancellation: {registry_delta} left in registry"
        )
    finally:
        await store.teardown()


async def test_interface_leak_when_aenter_fails():
    """If __aenter__ raises AFTER the connection is created, it must not leak.

    get_workspace_interface() creates the RedisRPCConnection synchronously in the
    factory, then __aenter__ makes an RPC call to the manager (get_manager_service,
    bounded by `timeout`). Under load that call can time out / fail, so __aenter__
    raises. Python does NOT call __aexit__ when __aenter__ raises — so the
    already-created connection (registered in the registry + event-bus handlers +
    RPC peer) is never disconnected. This is the prod HTTP /rpc leak: created_total
    climbs, closed_total does not.
    """
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        ws = "ws-user-leak4"
        user_info = UserInfo(
            id="leak4", is_anonymous=False, email=None, parent=None, roles=[],
            scope=create_scope(f"{ws}#a", current_workspace=ws), expires_at=None,
        )
        await store.register_workspace(
            dict(name=ws, description="x", owners=["leak4"], persistent=False,
                 read_only=False), overwrite=True,
        )
        async with store.get_workspace_interface(user_info, ws) as api:
            await api.echo("warmup")
        await asyncio.sleep(0.2); gc.collect()

        created_before = RedisRPCConnection._created_total_int
        closed_before = RedisRPCConnection._closed_total_int
        registry_before = len(RedisRPCConnection._connections)
        objects_before = _live_conn_objects()

        # Force __aenter__ to fail deterministically the way a manager-call timeout
        # would in prod, by making the manager-service lookup raise.
        from hypha.core.store import WorkspaceInterfaceContextManager
        orig = WorkspaceInterfaceContextManager._get_workspace_manager

        async def failing_get_manager(self):
            # The connection/rpc already exist (created in the factory). Simulate the
            # manager RPC call timing out under load.
            raise asyncio.TimeoutError("simulated manager timeout under load")

        WorkspaceInterfaceContextManager._get_workspace_manager = failing_get_manager
        try:
            N = 40
            failures = 0
            for i in range(N):
                try:
                    async with store.get_workspace_interface(user_info, ws) as api:
                        await api.echo(f"call-{i}")
                except asyncio.TimeoutError:
                    failures += 1
        finally:
            WorkspaceInterfaceContextManager._get_workspace_manager = orig

        await asyncio.sleep(0.5); gc.collect(); await asyncio.sleep(0.1)

        created_delta = RedisRPCConnection._created_total_int - created_before
        closed_delta = RedisRPCConnection._closed_total_int - closed_before
        registry_delta = len(RedisRPCConnection._connections) - registry_before
        objects_delta = _live_conn_objects() - objects_before
        unclosed = created_delta - closed_delta

        print(
            f"\n[leak-aenter] N={N} failures={failures} created_delta={created_delta} "
            f"closed_delta={closed_delta} unclosed={unclosed} "
            f"registry_delta={registry_delta} objects_delta={objects_delta}"
        )

        assert failures == N, f"expected all {N} __aenter__ to fail, got {failures}"
        assert unclosed <= 2, (
            f"AENTER LEAK: {unclosed} connections created but never closed when "
            f"__aenter__ failed (created={created_delta}, closed={closed_delta}). "
            f"Python does not call __aexit__ when __aenter__ raises, so the "
            f"connection created in the factory leaks. This is the prod HTTP /rpc leak."
        )
        assert registry_delta <= 2, f"registry leak: {registry_delta} left"
        assert objects_delta <= 2, f"object leak: {objects_delta} conns retained"
    finally:
        await store.teardown()
