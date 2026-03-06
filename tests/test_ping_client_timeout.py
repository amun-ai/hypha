#!/usr/bin/env python
"""Test that ping_client enforces asyncio.wait_for timeouts on svc calls.

No mocks: uses plain async functions and object.__new__ to wire only the
attributes ping_client / cleanup() actually access.
"""

import asyncio
import pytest

pytestmark = pytest.mark.asyncio


def _make_workspace_manager():
    """Return a minimal WorkspaceManager stub (no __init__ dependencies)."""
    from hypha.core.workspace import WorkspaceManager

    wm = object.__new__(WorkspaceManager)
    return wm


def _context(ws="test-ws"):
    return {
        "ws": ws,
        "from": f"{ws}/admin",
        "user": {"id": "admin", "is_anonymous": False},
    }


# ---------------------------------------------------------------------------
# 1. svc.ping() hanging must be cut off by the timeout
# ---------------------------------------------------------------------------


async def test_svc_ping_hang_is_bounded_by_timeout():
    """ping_client must not hang when svc.ping() never returns."""
    wm = _make_workspace_manager()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(60)  # simulate infinite hang

    class _HangingSvc:
        ping = _hang

    # Minimal RPC stub: get_remote_service returns the hanging service
    class _RPC:
        async def get_remote_service(self, service_id, opts=None):
            return _HangingSvc()

    wm._rpc = _RPC()

    # SCAN returns no keys so the fallback path exits immediately
    class _Redis:
        async def scan(self, cursor=0, match=None, count=None):
            return (0, [])

    wm._redis = _Redis()

    start = asyncio.get_event_loop().time()
    result = await wm.ping_client("test-ws/my-client", timeout=0.3, context=_context())
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 2.0, f"ping_client hung for {elapsed:.2f}s — no timeout guard"
    assert result != "pong", "Should not return 'pong' when svc hangs"


# ---------------------------------------------------------------------------
# 2. svc.echo() hanging must be cut off by the timeout (fallback path)
# ---------------------------------------------------------------------------


async def test_svc_echo_hang_is_bounded_by_timeout():
    """ping_client must not hang when svc.echo() never returns (fallback path)."""
    wm = _make_workspace_manager()

    # built-in service raises → fallback to SCAN path
    async def _builtin_fail(service_id, opts=None):
        raise ConnectionError("no built-in")

    async def _hang_echo(*_a, **_kw):
        await asyncio.sleep(60)

    class _EchoSvc:
        echo = _hang_echo
        # no 'ping' attribute

    async def _get_fallback_svc(service_id, opts=None):
        return _EchoSvc()

    # First call raises (built-in), subsequent calls return hanging svc
    _call_count = {"n": 0}

    class _RPC:
        async def get_remote_service(self, service_id, opts=None):
            _call_count["n"] += 1
            if _call_count["n"] == 1:
                raise ConnectionError("no built-in")
            return _EchoSvc()

    wm._rpc = _RPC()

    fake_key = b"services:public|test-ws/my-client:some-svc@app"

    class _Redis:
        async def scan(self, cursor=0, match=None, count=None):
            if cursor == 0:
                return (0, [fake_key])
            return (0, [])

    wm._redis = _Redis()

    start = asyncio.get_event_loop().time()
    result = await wm.ping_client("test-ws/my-client", timeout=0.3, context=_context())
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 2.0, f"ping_client hung for {elapsed:.2f}s on echo fallback"
    assert result != "pong"


# ---------------------------------------------------------------------------
# 3. Outer call in cleanup() is bounded by asyncio.wait_for
# ---------------------------------------------------------------------------


async def test_cleanup_ping_client_is_bounded():
    """cleanup() must not hang if ping_client itself hangs."""
    wm = _make_workspace_manager()

    async def _hang_ping(*_a, **_kw):
        await asyncio.sleep(60)

    wm.ping_client = _hang_ping

    # Minimal stubs for cleanup()
    class _WsInfo:
        id = "test-ws"
        persistent = False
        status = {"ready": True}

    async def _load_ws_info(*_a, **_kw):
        return _WsInfo()

    async def _list_clients(*_a, **_kw):
        return ["test-ws/dead-client"]

    async def _delete_client(*_a, **_kw):
        pass

    async def _delete_workspace(*_a, **_kw):
        pass

    def _validate_context(*_a, **_kw):
        pass

    wm.load_workspace_info = _load_ws_info
    wm._list_client_keys = _list_clients
    wm.delete_client = _delete_client
    wm.delete_workspace = _delete_workspace
    wm.validate_context = _validate_context

    context = {
        "ws": "test-ws",
        "from": "test-ws/admin",
        "user": {
            "id": "admin",
            "is_anonymous": False,
            "roles": [],
            "scope": {"workspaces": {"test-ws": "a"}},
        },
    }

    start = asyncio.get_event_loop().time()
    try:
        await asyncio.wait_for(
            wm.cleanup(workspace="test-ws", timeout=0.3, context=context),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        elapsed = asyncio.get_event_loop().time() - start
        pytest.fail(
            f"cleanup() timed out after {elapsed:.2f}s — "
            "outer ping_client call is not bounded by asyncio.wait_for"
        )

    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 4.0, f"cleanup() took {elapsed:.2f}s — ping timeout not enforced"
