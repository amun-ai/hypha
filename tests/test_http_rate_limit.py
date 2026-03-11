"""Tests for HTTP rate limiting middleware.

Covers the rate limiter itself (unit tests) and simulates the real-world
scenario from issue #943: a daemon managing multiple sessions making
concurrent HTTP requests that previously triggered excessive 429 errors
with the old defaults (rate=20, burst=100).
"""

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


# ── Issue #943 regression tests ─────────────────────────────────────────
# Simulate the daemon scenario: one IP making many HTTP requests across
# multiple sessions (service calls, token refreshes, status checks).


class TestIssue943DaemonScenario:
    """Reproduce the rate limiting issue from #943.

    A daemon managing 10 sessions makes ~5-10 HTTP requests/second per
    session, totaling 50-100 req/s from one IP. With the old defaults
    (rate=20, burst=100), the burst would be exhausted in ~5 seconds and
    then only 20 req/s would be allowed — causing cascading 429 errors
    and client crashes.
    """

    def test_old_defaults_would_reject_daemon_traffic(self):
        """Prove that old defaults (rate=20, burst=100) fail under daemon load."""
        from hypha.http import HTTPRateLimitMiddleware

        old_rate = os.environ.get("HYPHA_HTTP_RATE_LIMIT")
        old_burst = os.environ.get("HYPHA_HTTP_BURST_LIMIT")
        os.environ["HYPHA_HTTP_RATE_LIMIT"] = "20"
        os.environ["HYPHA_HTTP_BURST_LIMIT"] = "100"
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

        ip = "192.168.1.100"
        rejected = 0
        # Simulate 10 sessions × 8 req/s = 80 req in first second
        # With burst=100, first 100 pass but subsequent at rate=20 fail
        for _ in range(150):  # 1.5 seconds of sustained traffic
            allowed, _ = mw._consume(ip)
            if not allowed:
                rejected += 1

        # With old rate=20, burst=100: first 100 pass, then only 20/s
        # so 50 out of 150 get rejected
        assert rejected > 0, "Old defaults should reject daemon-level traffic"

    def test_new_defaults_allow_daemon_traffic(self):
        """New defaults (rate=100, burst=500) handle daemon load gracefully."""
        from hypha.http import HTTPRateLimitMiddleware

        old_rate = os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
        old_burst = os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
        try:
            mw = HTTPRateLimitMiddleware(app=None)  # Uses new defaults
        finally:
            if old_rate is not None:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is not None:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst

        ip = "192.168.1.100"
        rejected = 0
        # 10 sessions × 8 req/s = 80 req/s — well within new burst of 500
        for _ in range(80):
            allowed, _ = mw._consume(ip)
            if not allowed:
                rejected += 1

        assert rejected == 0, (
            f"New defaults should handle 80 req/s without rejecting, but {rejected} were rejected"
        )

    def test_sustained_moderate_load_succeeds(self):
        """Simulate sustained moderate load over multiple time windows."""
        from hypha.http import HTTPRateLimitMiddleware

        old_rate = os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
        old_burst = os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
        try:
            mw = HTTPRateLimitMiddleware(app=None)
        finally:
            if old_rate is not None:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is not None:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst

        ip = "192.168.1.100"
        total_rejected = 0

        # Simulate 5 "seconds" of 80 req/s with time advancing between
        for second in range(5):
            for _ in range(80):
                allowed, _ = mw._consume(ip)
                if not allowed:
                    total_rejected += 1
            # Advance time by 1 second: refill 100 tokens
            tokens, last = mw._buckets[ip]
            mw._buckets[ip] = (tokens, last - 1.0)

        # With burst=500, rate=100: first 500 pass, then 100/s sustained
        # 400 total requests in 5 seconds (80×5) minus burst=500, all should pass
        # since 80 < 100 sustained rate
        assert total_rejected == 0, (
            f"Sustained 80 req/s should be within rate=100 limit, but {total_rejected} rejected"
        )

    def test_actual_burst_then_sustained_rate(self):
        """Test that traffic exceeding sustained rate after burst exhaustion gets throttled."""
        from hypha.http import HTTPRateLimitMiddleware

        old_rate = os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
        old_burst = os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
        try:
            mw = HTTPRateLimitMiddleware(app=None)
        finally:
            if old_rate is not None:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is not None:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst

        ip = "192.168.1.100"
        # Exhaust burst: send 500 requests instantly
        for _ in range(500):
            mw._consume(ip)

        # Now at sustained rate=100. Without time advancing, next request
        # should be rejected (bucket empty, no time to refill)
        allowed, retry_after = mw._consume(ip)
        assert allowed is False
        assert retry_after > 0

        # After enough time passes, should be allowed again
        tokens, last = mw._buckets[ip]
        mw._buckets[ip] = (tokens, last - 0.01)  # 10ms → 1 token at rate=100
        allowed, _ = mw._consume(ip)
        assert allowed is True


class TestConnectionLimitProtection:
    """Test that connection limits protect against resource exhaustion.

    Simulates the stress test scenario: many concurrent clients from
    one user trying to overwhelm the server.
    """

    @pytest.mark.asyncio
    async def test_connection_storm_rejected_after_limit(self):
        """Simulate a bad actor spawning unlimited WebSocket connections."""
        from hypha.resource_limits import ResourceLimitsManager

        rl = ResourceLimitsManager(redis=None)
        rl.max_connections_per_user = 200

        # Simulate 200 successful connections
        for i in range(200):
            assert await rl.check_connection_allowed("attacker", "10.0.0.1", "ws1") is None
            rl.on_client_connected(f"ws1/client{i}", "attacker", "ws1")

        # 201st connection should be rejected
        result = await rl.check_connection_allowed("attacker", "10.0.0.1", "ws1")
        assert result is not None
        assert "Connection limit exceeded" in result
        assert "200/200" in result

    @pytest.mark.asyncio
    async def test_workspace_flood_rejected(self):
        """Simulate many different users flooding one workspace."""
        from hypha.resource_limits import ResourceLimitsManager

        rl = ResourceLimitsManager(redis=None)
        rl.max_clients_per_workspace = 1000

        # 1000 different users connect to same workspace
        for i in range(1000):
            assert await rl.check_connection_allowed(f"user{i}", f"10.0.{i // 256}.{i % 256}", "target-ws") is None
            rl.on_client_connected(f"target-ws/client{i}", f"user{i}", "target-ws")

        # 1001st should be rejected regardless of user
        result = await rl.check_connection_allowed("new-user", "10.0.0.1", "target-ws")
        assert result is not None
        assert "workspace target-ws" in result

    @pytest.mark.asyncio
    async def test_legitimate_users_unaffected_by_attacker(self):
        """Attacker hitting their limit doesn't affect other users."""
        from hypha.resource_limits import ResourceLimitsManager

        rl = ResourceLimitsManager(redis=None)
        rl.max_connections_per_user = 200

        # Attacker maxes out their connections
        for i in range(200):
            rl.on_client_connected(f"ws1/attacker-{i}", "attacker", "ws1")

        # Attacker can't connect more
        assert await rl.check_connection_allowed("attacker", "10.0.0.1", "ws1") is not None

        # But legitimate users can still connect
        assert await rl.check_connection_allowed("legitimate-user", "10.0.0.2", "ws1") is None

    def test_message_flood_rate_limited(self):
        """Simulate a client sending messages as fast as possible."""
        from hypha.resource_limits import ResourceLimitsManager

        rl = ResourceLimitsManager(redis=None)
        rl.ws_msg_rate_limit = 200
        rl.ws_msg_burst_limit = 500

        allowed_count = 0
        rejected_count = 0
        # Try to send 1000 messages instantly (no time passing)
        for _ in range(1000):
            if rl.check_message_allowed("ws1/flood-client"):
                allowed_count += 1
            else:
                rejected_count += 1

        # First 500 (burst) should pass, rest rejected
        assert allowed_count == 500
        assert rejected_count == 500
