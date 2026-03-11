"""Tests for HTTP rate limiting middleware."""

import math

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from hypha.http import HTTPRateLimitMiddleware


def _make_app(rate: float = 10, burst: int = 5):
    """Create a minimal Starlette app with rate limiting."""
    import os

    os.environ["HYPHA_HTTP_RATE_LIMIT"] = str(rate)
    os.environ["HYPHA_HTTP_BURST_LIMIT"] = str(burst)

    app = Starlette()

    @app.route("/test")
    async def test_endpoint(request):
        return PlainTextResponse("ok")

    @app.route("/health")
    async def health_endpoint(request):
        return PlainTextResponse("healthy")

    app.add_middleware(HTTPRateLimitMiddleware)
    return app


class TestHTTPRateLimitMiddleware:
    """Tests for HTTPRateLimitMiddleware."""

    def test_allows_requests_within_burst(self):
        """Requests within burst capacity should succeed."""
        app = _make_app(rate=10, burst=5)
        client = TestClient(app, raise_server_exceptions=False)

        # First 5 requests should all succeed (burst=5)
        for i in range(5):
            resp = client.get("/test")
            assert resp.status_code == 200, f"Request {i+1} should succeed"

    def test_rejects_after_burst_exhausted(self):
        """Requests exceeding burst should get 429."""
        app = _make_app(rate=10, burst=3)
        client = TestClient(app, raise_server_exceptions=False)

        # Exhaust burst
        for _ in range(3):
            resp = client.get("/test")
            assert resp.status_code == 200

        # Next request should be rate limited
        resp = client.get("/test")
        assert resp.status_code == 429

    def test_429_includes_retry_after_header(self):
        """429 responses must include Retry-After header."""
        app = _make_app(rate=10, burst=1)
        client = TestClient(app, raise_server_exceptions=False)

        # Use the one token
        resp = client.get("/test")
        assert resp.status_code == 200

        # This should be rate limited
        resp = client.get("/test")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after >= 1  # At least 1 second (ceil)

    def test_429_response_body(self):
        """429 response should include detail message."""
        app = _make_app(rate=10, burst=1)
        client = TestClient(app, raise_server_exceptions=False)

        client.get("/test")  # exhaust burst
        resp = client.get("/test")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]

    def test_health_endpoint_exempt(self):
        """Health endpoints should never be rate limited."""
        app = _make_app(rate=10, burst=1)
        client = TestClient(app, raise_server_exceptions=False)

        # Exhaust the burst on normal endpoint
        client.get("/test")

        # Health endpoint should still work
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_default_limits_increased(self):
        """Default limits should be 100 req/s rate and 500 burst."""
        import os

        # Remove env overrides to test defaults
        old_rate = os.environ.pop("HYPHA_HTTP_RATE_LIMIT", None)
        old_burst = os.environ.pop("HYPHA_HTTP_BURST_LIMIT", None)
        try:
            app = Starlette()
            middleware = HTTPRateLimitMiddleware(app)
            assert middleware._rate == 100.0
            assert middleware._burst == 500
        finally:
            if old_rate is not None:
                os.environ["HYPHA_HTTP_RATE_LIMIT"] = old_rate
            if old_burst is not None:
                os.environ["HYPHA_HTTP_BURST_LIMIT"] = old_burst

    def test_consume_returns_retry_after(self):
        """_consume should return proper retry_after when rate limited."""
        app = Starlette()
        import os

        os.environ["HYPHA_HTTP_RATE_LIMIT"] = "10"
        os.environ["HYPHA_HTTP_BURST_LIMIT"] = "1"
        middleware = HTTPRateLimitMiddleware(app)

        # First consume should succeed
        allowed, retry_after = middleware._consume("1.2.3.4")
        assert allowed is True
        assert retry_after == 0.0

        # Second consume should fail with retry_after > 0
        allowed, retry_after = middleware._consume("1.2.3.4")
        assert allowed is False
        assert retry_after > 0

    def test_different_ips_have_separate_buckets(self):
        """Each IP should have its own rate limit bucket."""
        app = _make_app(rate=10, burst=1)
        client = TestClient(app, raise_server_exceptions=False)

        # Exhaust bucket for default IP
        resp = client.get("/test")
        assert resp.status_code == 200
        resp = client.get("/test")
        assert resp.status_code == 429

        # Different IP via X-Forwarded-For should have its own bucket
        resp = client.get("/test", headers={"X-Forwarded-For": "10.0.0.1"})
        assert resp.status_code == 200

    def test_websocket_upgrade_exempt(self):
        """WebSocket upgrade requests should bypass HTTP rate limiting."""
        app = _make_app(rate=10, burst=1)
        client = TestClient(app, raise_server_exceptions=False)

        # Exhaust the burst
        client.get("/test")
        # Verify rate limited
        resp = client.get("/test")
        assert resp.status_code == 429

        # WebSocket upgrade header should bypass (will get 400 since no WS
        # handler, but NOT 429)
        resp = client.get(
            "/test",
            headers={"Upgrade": "websocket", "Connection": "upgrade"},
        )
        assert resp.status_code != 429
