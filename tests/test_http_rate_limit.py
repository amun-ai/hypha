"""Tests for HTTP rate limiting middleware."""

import math
import os
import time

import pytest


class TestHTTPRateLimitMiddleware:
    """Tests for HTTPRateLimitMiddleware token bucket and behavior."""

    def _make_middleware(self, rate=100, burst=500):
        """Create a middleware instance with given rate/burst."""
        from hypha.http import HTTPRateLimitMiddleware

        # Temporarily set env vars
        old_rate = os.environ.get("HYPHA_HTTP_RATE_LIMIT")
        old_burst = os.environ.get("HYPHA_HTTP_BURST_LIMIT")
        os.environ["HYPHA_HTTP_RATE_LIMIT"] = str(rate)
        os.environ["HYPHA_HTTP_BURST_LIMIT"] = str(burst)
        try:
            mw = HTTPRateLimitMiddleware(app=None)
        finally:
            if old_rate is None:
                os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
            else:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is None:
                os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
            else:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst
        return mw

    def test_allows_within_burst(self):
        mw = self._make_middleware(rate=10, burst=5)
        for _ in range(5):
            allowed, _ = mw._consume("1.2.3.4")
            assert allowed is True

    def test_rejects_after_burst(self):
        mw = self._make_middleware(rate=10, burst=3)
        for _ in range(3):
            allowed, _ = mw._consume("1.2.3.4")
            assert allowed is True
        allowed, retry_after = mw._consume("1.2.3.4")
        assert allowed is False
        assert retry_after > 0

    def test_retry_after_value(self):
        mw = self._make_middleware(rate=10, burst=1)
        mw._consume("1.2.3.4")  # Use the single token
        allowed, retry_after = mw._consume("1.2.3.4")
        assert allowed is False
        # At rate=10, it takes 0.1s to refill 1 token
        assert 0 < retry_after <= 0.2

    def test_health_exempt_prefix(self):
        """Health check paths should be in exempt prefixes."""
        mw = self._make_middleware()
        assert any("/health".startswith(p) for p in mw._EXEMPT_PREFIXES)

    def test_default_values(self):
        """Default rate and burst should be 100 and 500."""
        from hypha.http import HTTPRateLimitMiddleware

        # Clear env vars to test defaults
        old_rate = os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
        old_burst = os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
        try:
            mw = HTTPRateLimitMiddleware(app=None)
            assert mw._rate == 100.0
            assert mw._burst == 500
        finally:
            if old_rate is not None:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is not None:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst

    def test_per_ip_isolation(self):
        mw = self._make_middleware(rate=10, burst=1)
        allowed1, _ = mw._consume("1.2.3.4")
        assert allowed1 is True
        # Same IP should be rejected
        allowed2, _ = mw._consume("1.2.3.4")
        assert allowed2 is False
        # Different IP should be allowed
        allowed3, _ = mw._consume("5.6.7.8")
        assert allowed3 is True

    def test_refills_over_time(self):
        mw = self._make_middleware(rate=1000, burst=1)
        allowed, _ = mw._consume("1.2.3.4")
        assert allowed is True
        allowed, _ = mw._consume("1.2.3.4")
        assert allowed is False
        # Simulate time passing
        mw._buckets["1.2.3.4"] = (
            mw._buckets["1.2.3.4"][0],
            mw._buckets["1.2.3.4"][1] - 0.01,  # 10ms ago at 1000/s = 10 tokens
        )
        allowed, _ = mw._consume("1.2.3.4")
        assert allowed is True

    def test_cleanup_removes_stale_entries(self):
        mw = self._make_middleware(rate=10, burst=5)
        mw._consume("old_ip")
        # Simulate the entry being old
        tokens, _ = mw._buckets["old_ip"]
        mw._buckets["old_ip"] = (tokens, time.monotonic() - 600)
        # Force cleanup
        mw._last_cleanup = time.monotonic() - 600
        mw._maybe_cleanup()
        assert "old_ip" not in mw._buckets


@pytest.mark.asyncio
async def test_websocket_upgrade_exempt():
    """WebSocket upgrade requests should bypass HTTP rate limiting."""
    from hypha.http import HTTPRateLimitMiddleware

    app_called = False

    async def mock_app(scope, receive, send):
        nonlocal app_called
        app_called = True

    mw = HTTPRateLimitMiddleware(app=mock_app)
    # Exhaust all tokens for this IP
    mw._buckets["10.0.0.1"] = (0.0, time.monotonic())

    # Simulate a WebSocket upgrade request
    scope = {
        "type": "http",
        "path": "/ws",
        "headers": [
            (b"upgrade", b"websocket"),
            (b"x-forwarded-for", b"10.0.0.1"),
        ],
        "client": ("10.0.0.1", 12345),
    }
    await mw(scope, None, None)
    assert app_called, "WebSocket upgrade should bypass rate limiting"


@pytest.mark.asyncio
async def test_429_response_has_retry_after():
    """429 responses should include Retry-After header."""
    from hypha.http import HTTPRateLimitMiddleware

    captured_response = {}

    async def mock_send(message):
        if message.get("type") == "http.response.start":
            captured_response["status"] = message.get("status")
            captured_response["headers"] = {
                k.decode(): v.decode()
                for k, v in message.get("headers", [])
            }

    async def mock_receive():
        return {}

    mw = HTTPRateLimitMiddleware(app=None)
    # Exhaust tokens
    mw._buckets["10.0.0.1"] = (0.0, time.monotonic())

    scope = {
        "type": "http",
        "path": "/api/test",
        "headers": [(b"x-forwarded-for", b"10.0.0.1")],
        "client": ("10.0.0.1", 12345),
    }
    await mw(scope, mock_receive, mock_send)
    assert captured_response.get("status") == 429
    assert "retry-after" in captured_response.get("headers", {})
