#!/usr/bin/env python
"""Test that ping_client enforces asyncio.wait_for timeouts on svc calls."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPingClientTimeout:
    """Verify that svc.ping / svc.echo inside ping_client are bounded by timeout."""

    def _make_workspace_manager(self):
        """Return a minimal WorkspaceManager-like object with just enough wiring."""
        from hypha.core.workspace import WorkspaceManager

        wm = object.__new__(WorkspaceManager)
        # Provide minimal attributes so ping_client can run
        wm._rpc = AsyncMock()
        wm._redis = AsyncMock()
        return wm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _context(self, ws="test-ws"):
        return {"ws": ws, "from": "test-ws/admin", "user": {"id": "admin", "is_anonymous": False}}

    # ------------------------------------------------------------------
    # 1. svc.ping() hanging must be cut off by the timeout
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_svc_ping_hang_is_bounded_by_timeout(self):
        """ping_client must not hang when svc.ping() never returns."""
        wm = self._make_workspace_manager()

        hanging_svc = MagicMock()

        async def _hang(*_a, **_kw):
            await asyncio.sleep(60)  # simulate infinite hang

        hanging_svc.ping = _hang

        async def _get_svc(service_id, opts=None):
            return hanging_svc

        wm._rpc.get_remote_service = _get_svc
        # SCAN returns no keys so fallback path exits quickly
        wm._redis.scan = AsyncMock(return_value=(0, []))

        start = asyncio.get_event_loop().time()
        result = await wm.ping_client(
            "test-ws/my-client", timeout=0.3, context=self._context()
        )
        elapsed = asyncio.get_event_loop().time() - start

        # Must finish within ~0.6 s (timeout + small overhead, not 60 s)
        assert elapsed < 2.0, f"ping_client hung for {elapsed:.2f}s — no timeout guard"
        # Result should indicate failure, not "pong"
        assert result != "pong", "Should not return 'pong' when svc hangs"

    # ------------------------------------------------------------------
    # 2. svc.echo() hanging must be cut off by the timeout
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_svc_echo_hang_is_bounded_by_timeout(self):
        """ping_client must not hang when svc.echo() never returns (fallback path)."""
        wm = self._make_workspace_manager()

        # built-in service raises so we fall into the SCAN-fallback path
        async def _builtin_fail(service_id, opts=None):
            raise ConnectionError("no built-in")

        wm._rpc.get_remote_service = _builtin_fail

        # SCAN returns one fake key
        fake_key = b"services:public|test-ws/my-client:some-svc@app"

        async def _scan(cursor=0, match=None, count=None):
            if cursor == 0:
                return (0, [fake_key])
            return (0, [])

        wm._redis.scan = _scan

        # The fallback svc has echo but it hangs
        hanging_svc = MagicMock()

        async def _hang_echo(*_a, **_kw):
            await asyncio.sleep(60)

        hanging_svc.ping = None  # no ping attr → uses echo branch
        del hanging_svc.ping  # remove ping so hasattr is False
        hanging_svc.echo = _hang_echo

        async def _get_fallback_svc(service_id, opts=None):
            return hanging_svc

        # Override for fallback path only
        wm._rpc.get_remote_service = _get_fallback_svc

        start = asyncio.get_event_loop().time()
        result = await wm.ping_client(
            "test-ws/my-client", timeout=0.3, context=self._context()
        )
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 2.0, f"ping_client hung for {elapsed:.2f}s on echo fallback"
        assert result != "pong"

    # ------------------------------------------------------------------
    # 3. Outer call in cleanup() is bounded
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_cleanup_ping_client_is_bounded(self):
        """cleanup() must not hang if ping_client itself hangs."""
        wm = self._make_workspace_manager()

        # Make ping_client hang
        async def _hang_ping(*_a, **_kw):
            await asyncio.sleep(60)

        wm.ping_client = _hang_ping

        # Minimal stubs for cleanup()
        ws_info = MagicMock()
        ws_info.id = "test-ws"
        ws_info.persistent = False
        ws_info.status = {"ready": True}

        wm.load_workspace_info = AsyncMock(return_value=ws_info)
        wm._list_client_keys = AsyncMock(return_value=["test-ws/dead-client"])
        wm.delete_client = AsyncMock()
        wm.delete_workspace = AsyncMock()
        wm.validate_context = MagicMock()

        context = {
            "ws": "test-ws",
            "from": "test-ws/admin",
            "user": {
                "id": "admin",
                "is_anonymous": False,
                "roles": [],
                "scope": {"workspaces": {"test-ws": "a"}},  # "a" = admin
            },
        }

        start = asyncio.get_event_loop().time()
        # timeout=0.3 means cleanup should bound each ping to ~0.3 s
        try:
            result = await asyncio.wait_for(
                wm.cleanup(workspace="test-ws", timeout=0.3, context=context),
                timeout=5.0,  # outer safety net for test itself
            )
        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start
            pytest.fail(
                f"cleanup() timed out after {elapsed:.2f}s — "
                "outer ping_client call is not bounded by asyncio.wait_for"
            )

        elapsed = asyncio.get_event_loop().time() - start
        # Should complete within 5s (the outer guard for the test),
        # and specifically the internal per-client ping must have been cut off quickly.
        assert elapsed < 4.0, f"cleanup() took {elapsed:.2f}s — ping timeout not enforced"
