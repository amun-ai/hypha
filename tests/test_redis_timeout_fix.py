#!/usr/bin/env python
"""Test Redis timeout fix and circuit breaker behavior."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from hypha.core import RedisEventBus


class TestRedisTimeoutFix:
    """Test the Redis timeout fix and circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_redis_subscription_timeout_handling(self):
        """Test that Redis subscription timeouts are handled gracefully."""
        
        # Create a mock Redis client that simulates timeout
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        
        # Configure psubscribe to timeout
        async def timeout_psubscribe(*args, **kwargs):
            await asyncio.sleep(6)  # Longer than 5 second timeout
            
        mock_pubsub.psubscribe = timeout_psubscribe
        mock_redis.pubsub.return_value = mock_pubsub
        
        # Create event bus with mocked Redis
        event_bus = RedisEventBus(mock_redis)
        event_bus._pubsub = mock_pubsub
        event_bus._circuit_breaker_open = False
        event_bus._consecutive_failures = 0
        event_bus._max_failures = 3
        
        # Test subscription with timeout
        workspace = "test-workspace"
        client_id = "test-client"
        
        # Should handle timeout gracefully without hanging
        start_time = asyncio.get_event_loop().time()
        await event_bus.subscribe_to_client_events(workspace, client_id)
        end_time = asyncio.get_event_loop().time()
        
        # Should return quickly (within timeout + small buffer)
        assert end_time - start_time < 7.0, "Subscription should timeout within reasonable time"
        
        # Should increment failure count
        assert event_bus._consecutive_failures == 1
        
        # Circuit breaker should not be open yet (max failures is 3)
        assert not event_bus._circuit_breaker_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """Test that circuit breaker activates after consecutive failures."""
        
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        
        # Configure psubscribe to always timeout
        async def timeout_psubscribe(*args, **kwargs):
            raise asyncio.TimeoutError("Subscription timeout")
            
        mock_pubsub.psubscribe = timeout_psubscribe
        mock_redis.pubsub.return_value = mock_pubsub
        
        # Create event bus with lower failure threshold
        event_bus = RedisEventBus(mock_redis)
        event_bus._pubsub = mock_pubsub
        event_bus._circuit_breaker_open = False
        event_bus._consecutive_failures = 0
        event_bus._max_failures = 2  # Lower threshold for testing
        
        workspace = "test-workspace"
        
        # First failure with client-1
        await event_bus.subscribe_to_client_events(workspace, "test-client-1")
        assert event_bus._consecutive_failures == 1
        assert not event_bus._circuit_breaker_open
        
        # Second failure with client-2 should trigger circuit breaker
        await event_bus.subscribe_to_client_events(workspace, "test-client-2")
        assert event_bus._consecutive_failures == 2
        assert event_bus._circuit_breaker_open

    @pytest.mark.asyncio
    async def test_successful_subscription_resets_failures(self):
        """Test that successful subscriptions reset the failure counter."""
        
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        
        # Configure psubscribe to succeed
        mock_pubsub.psubscribe = AsyncMock(return_value=None)
        mock_redis.pubsub.return_value = mock_pubsub
        
        # Create event bus with some existing failures
        event_bus = RedisEventBus(mock_redis)
        event_bus._pubsub = mock_pubsub
        event_bus._circuit_breaker_open = False
        event_bus._consecutive_failures = 2  # Start with failures
        event_bus._max_failures = 3
        
        workspace = "test-workspace"
        client_id = "test-client"
        
        # Successful subscription should reset failure count
        await event_bus.subscribe_to_client_events(workspace, client_id)
        
        assert event_bus._consecutive_failures == 0
        assert not event_bus._circuit_breaker_open
        
        # Verify psubscribe was called with the correct pattern
        expected_pattern = f"targeted:{workspace}/{client_id}:*"
        mock_pubsub.psubscribe.assert_called_once_with(expected_pattern)

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_subscriptions(self):
        """Test that circuit breaker prevents subscription attempts when open."""
        
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.psubscribe = AsyncMock()
        
        # Create event bus with open circuit breaker
        event_bus = RedisEventBus(mock_redis)
        event_bus._pubsub = mock_pubsub
        event_bus._circuit_breaker_open = True  # Circuit breaker is open
        event_bus._consecutive_failures = 5
        event_bus._max_failures = 3
        
        workspace = "test-workspace"
        client_id = "test-client"
        
        # Should not attempt subscription when circuit breaker is open
        await event_bus.subscribe_to_client_events(workspace, client_id)
        
        # psubscribe should not have been called
        mock_pubsub.psubscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pubsub_handling(self):
        """Test behavior when pubsub is not initialized."""
        
        # Create event bus without pubsub
        mock_redis = AsyncMock()
        event_bus = RedisEventBus(mock_redis)
        event_bus._pubsub = None  # No pubsub
        event_bus._circuit_breaker_open = False
        
        workspace = "test-workspace"
        client_id = "test-client"
        
        # Should handle gracefully without errors
        await event_bus.subscribe_to_client_events(workspace, client_id)
        
        # Should not affect failure counters
        assert event_bus._consecutive_failures == 0
        assert not event_bus._circuit_breaker_open


@pytest.mark.asyncio
async def test_integration_with_workspace_get_service():
    """Integration test ensuring workspace get_service_info raises clean KeyError on empty Redis.

    Verifies that the scan-based service lookup raises a clean "not found" error
    rather than a timeout or cancelled error when no services exist in Redis.
    Uses FakeRedis (no mocks) so the full _scan_keys code path executes.
    """
    from hypha.core.workspace import WorkspaceManager
    from hypha.core.auth import UserInfo
    from fakeredis.aioredis import FakeRedis

    # Use real FakeRedis so _scan_keys → redis.scan() executes correctly
    fake_redis = FakeRedis()
    await fake_redis.flushall()

    user_info = UserInfo(
        id="test-user",
        email="test@example.com",
        user_workspace="test-workspace",
        is_anonymous=False,
        roles=["user"],
        expires_at=None,
        parent=None,
    )

    # WorkspaceManager(store, redis, root_user, event_bus, server_info, client_id)
    workspace_mgr = WorkspaceManager(
        store=None,
        redis=fake_redis,
        root_user=user_info,
        event_bus=None,
        server_info={"public_base_url": "http://localhost"},
        client_id="test-client",
    )

    context = {
        "user": user_info.model_dump(),
        "ws": "test-workspace",
        "from": "test-workspace/test-client",
    }

    # get_service_info on an empty Redis must raise a clean KeyError ("not found"),
    # never a timeout or asyncio.CancelledError
    raised = False
    try:
        await workspace_mgr.get_service_info(
            "non-existent-service", context=context
        )
    except KeyError as e:
        raised = True
        assert "not found" in str(e).lower(), f"Expected 'not found' in error: {e}"
        assert "timeout" not in str(e).lower(), f"Got unexpected timeout: {e}"
        assert "cancelled" not in str(e).lower(), f"Got unexpected cancellation: {e}"
    assert raised, "Expected KeyError to be raised for non-existent service"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])