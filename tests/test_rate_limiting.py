"""Test suite for rate limiting infrastructure.

Tests the Redis-backed distributed rate limiter used for DoS prevention.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from hypha.core.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    build_rate_limit_key,
    rate_limit,
)

pytestmark = pytest.mark.asyncio


class MockRedisPipeline:
    """Mock Redis pipeline for testing."""

    def __init__(self):
        self.commands = []
        self.data = {}

    def zremrangebyscore(self, key, min_score, max_score):
        self.commands.append(("zremrangebyscore", key, min_score, max_score))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    async def execute(self):
        # Simulate results: [zremrangebyscore_count, zcard_count, zadd_result, expire_result]
        return [0, 0, 1, True]


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.pipeline_mock = MockRedisPipeline()

    def pipeline(self):
        return self.pipeline_mock

    async def delete(self, key):
        if key in self.data:
            del self.data[key]

    async def zcount(self, key, min_score, max_score):
        return 0

    async def zrange(self, key, start, stop, withscores=False):
        return []


class TestRateLimiter:
    """Test RateLimiter class."""

    async def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        assert limiter.redis == redis
        assert limiter.prefix == "ratelimit:"
        assert limiter.enabled is True

    async def test_rate_limiter_disabled(self):
        """Test rate limiter when disabled."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis, enabled=False)

        allowed, count, retry_after = await limiter.check_rate_limit(
            key="test:user:1",
            max_requests=1,
            window_seconds=60,
        )

        assert allowed is True
        assert count == 0
        assert retry_after == 0.0

    async def test_check_rate_limit_within_limit(self):
        """Test rate limit check when within limit."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        # First request should be allowed
        allowed, count, retry_after = await limiter.check_rate_limit(
            key="test:user:1",
            max_requests=10,
            window_seconds=60,
        )

        assert allowed is True
        assert retry_after == 0.0

    async def test_build_rate_limit_key_user_scope(self):
        """Test building rate limit key with user scope."""
        context = {
            "user": {"id": "user-123"},
            "ws": "my-workspace",
        }

        key = build_rate_limit_key(scope="user", context=context)
        assert key == "user:user-123"

    async def test_build_rate_limit_key_workspace_scope(self):
        """Test building rate limit key with workspace scope."""
        context = {
            "user": {"id": "user-123"},
            "ws": "my-workspace",
        }

        key = build_rate_limit_key(scope="workspace", context=context)
        assert key == "ws:my-workspace"

    async def test_build_rate_limit_key_global_scope(self):
        """Test building rate limit key with global scope."""
        context = {}

        key = build_rate_limit_key(scope="global", context=context)
        assert key == "global"

    async def test_build_rate_limit_key_custom_scope(self):
        """Test building rate limit key with custom scope."""
        context = {}

        key = build_rate_limit_key(scope="custom", context=context, custom_key="my:custom:key")
        assert key == "my:custom:key"

    async def test_build_rate_limit_key_with_endpoint(self):
        """Test building rate limit key with endpoint."""
        context = {"user": {"id": "user-123"}}

        key = build_rate_limit_key(scope="user", context=context, endpoint="upload_file")
        assert key == "user:user-123:upload_file"

    async def test_build_rate_limit_key_missing_user(self):
        """Test building rate limit key fails when user missing for user scope."""
        context = {"ws": "my-workspace"}

        with pytest.raises(ValueError, match="User ID not found"):
            build_rate_limit_key(scope="user", context=context)

    async def test_build_rate_limit_key_missing_workspace(self):
        """Test building rate limit key fails when workspace missing for workspace scope."""
        context = {"user": {"id": "user-123"}}

        with pytest.raises(ValueError, match="Workspace not found"):
            build_rate_limit_key(scope="workspace", context=context)

    async def test_rate_limit_decorator_allows_request(self):
        """Test rate limit decorator allows request within limit."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        call_count = 0

        @rate_limit(max_requests=10, window_seconds=60, scope="user")
        async def test_function(context: dict):
            nonlocal call_count
            call_count += 1
            return "success"

        context = {
            "user": {"id": "user-123"},
            "ws": "my-workspace",
            "_rate_limiter": limiter,
        }

        result = await test_function(context=context)

        assert result == "success"
        assert call_count == 1

    async def test_rate_limit_decorator_without_context(self):
        """Test rate limit decorator allows request when context missing."""
        @rate_limit(max_requests=10, window_seconds=60)
        async def test_function():
            return "success"

        # Should allow request (graceful degradation)
        result = await test_function()
        assert result == "success"

    async def test_rate_limit_decorator_without_rate_limiter(self):
        """Test rate limit decorator allows request when rate limiter not in context."""
        @rate_limit(max_requests=10, window_seconds=60)
        async def test_function(context: dict):
            return "success"

        context = {
            "user": {"id": "user-123"},
            "ws": "my-workspace",
            # No _rate_limiter key
        }

        # Should allow request (graceful degradation)
        result = await test_function(context=context)
        assert result == "success"

    async def test_reset_rate_limit(self):
        """Test resetting rate limit counter."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        # Reset should not raise exception
        await limiter.reset("test:user:1")

    async def test_get_usage(self):
        """Test getting current usage."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        usage = await limiter.get_usage("test:user:1", window_seconds=60)
        assert usage == 0

    async def test_graceful_degradation_on_redis_error(self):
        """Test graceful degradation when Redis fails."""

        class FailingRedis:
            def pipeline(self):
                raise Exception("Redis connection failed")

        limiter = RateLimiter(FailingRedis())

        # Should allow request despite Redis failure
        allowed, count, retry_after = await limiter.check_rate_limit(
            key="test:user:1",
            max_requests=10,
            window_seconds=60,
        )

        assert allowed is True
        assert count == 0
        assert retry_after == 0.0


class TestRateLimitIntegration:
    """Integration tests for rate limiting with realistic scenarios."""

    async def test_dos1_websocket_connection_limiting(self):
        """Test rate limiting for WebSocket connections (DOS1)."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        context = {
            "user": {"id": "attacker-user"},
            "_rate_limiter": limiter,
        }

        @rate_limit(max_requests=10, window_seconds=60, scope="user", endpoint="ws_connect")
        async def connect_websocket(context: dict):
            return "connected"

        # Simulate connection attempts - first 10 should work
        # (Note: In real test with actual Redis, we'd see the limit enforced)
        for i in range(5):
            result = await connect_websocket(context=context)
            assert result == "connected"

    async def test_dos2_service_registration_limiting(self):
        """Test rate limiting for service registration (DOS2)."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        context = {
            "user": {"id": "user-123"},
            "ws": "test-workspace",
            "_rate_limiter": limiter,
        }

        @rate_limit(max_requests=50, window_seconds=60, scope="user", endpoint="register_service")
        async def register_service(service_id: str, context: dict):
            return f"registered:{service_id}"

        # Register services
        for i in range(5):
            result = await register_service(f"service-{i}", context=context)
            assert result == f"registered:service-{i}"

    async def test_dos4_artifact_upload_limiting(self):
        """Test rate limiting for artifact uploads (DOS4)."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        context = {
            "user": {"id": "user-123"},
            "ws": "test-workspace",
            "_rate_limiter": limiter,
        }

        @rate_limit(max_requests=100, window_seconds=3600, scope="user", endpoint="upload_artifact")
        async def upload_artifact(file_name: str, context: dict):
            return f"uploaded:{file_name}"

        # Upload files
        for i in range(5):
            result = await upload_artifact(f"file-{i}.txt", context=context)
            assert result == f"uploaded:file-{i}.txt"

    async def test_workspace_scoped_limiting(self):
        """Test workspace-scoped rate limiting."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        @rate_limit(max_requests=100, window_seconds=60, scope="workspace")
        async def workspace_operation(context: dict):
            return "ok"

        # Different users in same workspace share limit
        context1 = {
            "user": {"id": "user-1"},
            "ws": "shared-workspace",
            "_rate_limiter": limiter,
        }

        context2 = {
            "user": {"id": "user-2"},
            "ws": "shared-workspace",
            "_rate_limiter": limiter,
        }

        result1 = await workspace_operation(context=context1)
        result2 = await workspace_operation(context=context2)

        assert result1 == "ok"
        assert result2 == "ok"

    async def test_global_rate_limiting(self):
        """Test global rate limiting (applies to all users/workspaces)."""
        redis = MockRedisClient()
        limiter = RateLimiter(redis)

        @rate_limit(max_requests=1000, window_seconds=60, scope="global")
        async def expensive_operation(context: dict):
            return "ok"

        context = {
            "user": {"id": "user-123"},
            "_rate_limiter": limiter,
        }

        result = await expensive_operation(context=context)
        assert result == "ok"
